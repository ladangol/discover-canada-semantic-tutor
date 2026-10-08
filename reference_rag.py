from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
import time

import chromadb
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer


# =============================================================================
# Configuration
# =============================================================================

DB_PATH = Path("chroma_db")
COLLECTION_NAME = "discover-canada-window-minilm"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DISTANCE_SPACE = "cosine"

# One-line generation-model experiment:
GENERATION_MODEL = "zai-org-glm-4.7-flash"
ALTERNATIVE_GENERATION_MODEL = "openai-gpt-oss-120b"

VENICE_BASE_URL_DEFAULT = "https://api.venice.ai/api/v1"

DEFAULT_K = 5
GENERATION_TEMPERATURE = 0.0

# Synthesis questions can be much longer than direct fact questions. We still
# inspect finish_reason, so hitting this cap becomes a measured outcome rather
# than a silent source of incompleteness.
MAX_OUTPUT_TOKENS = 1200

UNKNOWN_ANSWER = "I don't know based on Discover Canada."

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
ABSTENTION_PATTERN = re.compile(
    r"^\s*i\s+(?:do\s+not|don['’]t)\s+know\s+based\s+on\s+discover\s+canada\b",
    flags=re.IGNORECASE,
)

CITATION_RUN_PATTERN = re.compile(
    r"\[S\d+\](?:\s*\[S\d+\])*"
)

CitationStatus = Literal[
    "ok",
    "no_citations",
    "invalid_labels",
]


# =============================================================================
# Data returned by retrieval / generation / RAG
# =============================================================================

@dataclass(frozen=True)
class RetrievedChunk:
    source_id: str
    chunk_id: str
    text: str
    page: str
    section: str
    block_type: str
    distance: float
    similarity: float


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass(frozen=True)
class CitationResolution:
    answer: str
    cited_chunks: list[RetrievedChunk]
    citation_status: CitationStatus
    invalid_citation_labels: list[str]


@dataclass(frozen=True)
class RAGResult:
    question: str
    raw_answer: str
    answer: str
    cited_chunks: list[RetrievedChunk]
    retrieved_chunks: list[RetrievedChunk]
    generation_model: str
    citation_status: CitationStatus
    invalid_citation_labels: list[str]
    abstained: bool
    finish_reason: str | None
    generation_truncated: bool
    prompt_tokens: int | None
    completion_tokens: int | None

    retrieval_seconds: float
    prompt_seconds: float
    generation_seconds: float
    citation_seconds: float
    total_seconds: float


# =============================================================================
# Chroma safety guards
# =============================================================================

def _configuration_as_dict(collection) -> dict[str, Any]:
    """
    Normalize common Chroma collection-configuration representations.

    We raise if we cannot inspect the persisted configuration. Silently assuming
    a distance metric would recreate the exact failure mode we are guarding.
    """
    config = getattr(collection, "configuration", None)

    if config is None:
        raise RuntimeError(
            f"Collection {collection.name!r} does not expose a configuration."
        )

    if isinstance(config, dict):
        return config

    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(config, method_name, None)

        if callable(method):
            value = method()

            if isinstance(value, dict):
                return value

    raise RuntimeError(
        f"Unsupported Chroma configuration object for "
        f"{collection.name!r}: {type(config).__name__}"
    )


def collection_distance_space(collection) -> str:
    config = _configuration_as_dict(collection)

    hnsw = config.get("hnsw")

    if isinstance(hnsw, dict):
        space = hnsw.get("space")

        if isinstance(space, str):
            return space

    # Defensive fallback for alternate serialized configuration shapes.
    space = config.get("hnsw:space")

    if isinstance(space, str):
        return space

    raise RuntimeError(
        f"Could not determine HNSW distance space for "
        f"{collection.name!r}."
    )


def assert_expected_distance_space(collection) -> None:
    actual = collection_distance_space(collection)

    if actual != EXPECTED_DISTANCE_SPACE:
        raise RuntimeError(
            f"Collection {collection.name!r} uses distance space "
            f"{actual!r}; this RAG code expects "
            f"{EXPECTED_DISTANCE_SPACE!r}. "
            "Refusing to interpret 1 - distance as cosine similarity."
        )


def assert_expected_embedding_model(collection) -> None:
    """
    Guard against a same-dimension, wrong-embedding-space failure.

    MiniLM and another embedding model can both emit 384-dimensional vectors.
    Chroma cannot detect that semantic-space mismatch from vector shape alone.

    Step 4 stored 'embedding_model' on every chunk, so inspect the persisted
    evidence once when the collection is opened. This also catches a partially
    re-ingested collection containing a mixture of embedding models.
    """
    result = collection.get(
        include=["metadatas"],
    )

    metadatas = result.get("metadatas")

    if not metadatas:
        raise RuntimeError(
            f"Collection {collection.name!r} has no metadata to verify "
            "the embedding model."
        )

    declared_models: set[str] = set()

    for metadata in metadatas:
        if metadata is None:
            raise RuntimeError(
                f"Collection {collection.name!r} contains a record with "
                "missing metadata."
            )

        model_name = metadata.get("embedding_model")

        if not isinstance(model_name, str):
            raise RuntimeError(
                f"Collection {collection.name!r} contains a record without "
                "'embedding_model' metadata."
            )

        declared_models.add(model_name)

    if declared_models != {EMBEDDING_MODEL_NAME}:
        raise RuntimeError(
            f"Collection {collection.name!r} declares embedding model(s) "
            f"{sorted(declared_models)!r}; query code uses "
            f"{EMBEDDING_MODEL_NAME!r}. "
            "Refusing to mix embedding spaces."
        )


def assert_collection_compatible(collection) -> None:
    assert_expected_distance_space(collection)
    assert_expected_embedding_model(collection)


# =============================================================================
# Retrieval
# =============================================================================

@lru_cache(maxsize=1)
def embedding_model() -> SentenceTransformer:
    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


@lru_cache(maxsize=1)
def retrieval_collection():
    """
    Open and validate the persistent collection once per process.

    The expensive embedding-model metadata scan therefore happens once, not
    twice per query. retrieve() always reaches the collection through this
    function, so importing retrieve() elsewhere still preserves the guard.
    """
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    assert_collection_compatible(collection)

    return collection


def retrieve(
    question: str,
    k: int = DEFAULT_K,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k chunks.

    There is intentionally NO hard similarity threshold in Step 5. Chroma
    returns the best available neighbours; Step 6 evaluates whether those
    neighbours contain sufficient evidence to answer.
    """
    collection = retrieval_collection()
    model = embedding_model()

    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    result = collection.query(
        query_embeddings=[
            query_vector.tolist()
        ],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    chunks: list[RetrievedChunk] = []

    for index, (
        chunk_id,
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):
        if metadata is None:
            raise RuntimeError(
                f"Retrieved chunk {chunk_id!r} has no metadata."
            )

        distance_value = float(distance)

        # Safe because retrieval_collection() verified cosine space.
        similarity = 1.0 - distance_value

        chunks.append(
            RetrievedChunk(
                source_id=f"S{index}",
                chunk_id=chunk_id,
                text=document,
                page=str(metadata["page"]),
                section=str(metadata["section"]),
                block_type=str(metadata["block_type"]),
                distance=distance_value,
                similarity=similarity,
            )
        )

    return chunks


# =============================================================================
# Prompt construction -- pure string work, no API calls
# =============================================================================

SYSTEM_PROMPT = f"""You answer questions using only the supplied excerpts from
Discover Canada.

Rules:
1. Use only facts that are supported by the supplied SOURCE blocks.
2. Do not answer from your prior knowledge.
3. If the supplied sources do not contain enough information to answer the
   question, reply exactly: "{UNKNOWN_ANSWER}"
4. Cite supporting sources inline using only their source labels, for example
   [S1] or [S1][S3].
5. Never invent a source label.
6. Do not write page numbers yourself. The application will turn source labels
   into page citations after generation.
7. A retrieved source being semantically similar to the question does not mean
   that it contains the answer. Check the actual text before answering.
"""


def format_source_block(
    chunk: RetrievedChunk,
) -> str:
    """
    The model sees source IDs plus provenance, but it is instructed to cite only
    the source ID. Page numbers in the final user-visible citation are resolved
    by code, not generated by the LLM.
    """
    return (
        f"SOURCE {chunk.source_id}\n"
        f"page: {chunk.page}\n"
        f"section: {chunk.section}\n"
        f"chunk_id: {chunk.chunk_id}\n"
        f"block_type: {chunk.block_type}\n"
        f"text:\n{chunk.text}"
    )


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    """
    System message = invariant behaviour/policy for this RAG application.
    User message   = dynamic question + dynamic retrieved evidence.
    """
    source_text = "\n\n---\n\n".join(
        format_source_block(chunk)
        for chunk in chunks
    )

    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED SOURCES:\n{source_text}"
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def build_no_rag_prompt(
    question: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the user's question accurately and concisely using "
                "your own knowledge. You have not been given Discover Canada "
                "as a source, so do not claim to quote or cite it."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]


# =============================================================================
# Generation -- the ONLY function that knows Venice exists
# =============================================================================

@lru_cache(maxsize=1)
def generation_client() -> OpenAI:
    load_dotenv()

    api_key = os.environ.get(
        "VENICE_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "VENICE_API_KEY is not set."
        )

    base_url = os.environ.get(
        "VENICE_BASE_URL",
        VENICE_BASE_URL_DEFAULT,
    )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

def generate(
    messages: list[dict[str, str]],
    *,
    model_id: str = GENERATION_MODEL,
) -> GenerationResult:
    """
    Call Venice through its OpenAI-compatible API.

    The API key is loaded here, never stored as a source-code literal, never
    printed, and never interpolated into our own exception messages.
    """
    client = generation_client()


    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except OpenAIError as exc:
        # Do not include SDK exception text; our own error path should not echo
        # request/authentication details.
        raise RuntimeError(
            "Venice generation request failed "
            f"({type(exc).__name__})."
        ) from None

    choice = response.choices[0]
    content = choice.message.content

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "Venice returned an empty text response."
        )

    usage = getattr(response, "usage", None)

    prompt_tokens = (
        getattr(usage, "prompt_tokens", None)
        if usage is not None
        else None
    )
    completion_tokens = (
        getattr(usage, "completion_tokens", None)
        if usage is not None
        else None
    )

    return GenerationResult(
        text=content.strip(),
        finish_reason=getattr(
            choice,
            "finish_reason",
            None,
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# =============================================================================
# Citation validation and rendering
# =============================================================================

def is_abstention(text: str) -> bool:
    """
    Tolerant semantic abstention detector.

    Accepts punctuation differences and additional explanation after the core
    abstention sentence. Strict exact compliance remains a separate Step 6
    metric.
    """
    return ABSTENTION_PATTERN.search(
        text
    ) is not None


def page_label(page: str) -> str:
    if "-" in page:
        start, end = page.split("-", 1)
        return f"pp. {start}\u2013{end}"

    return f"p. {page}"


def resolve_citations(
    raw_answer: str,
    chunks: list[RetrievedChunk],
) -> CitationResolution:
    """
    Resolve valid [S#] labels and report citation problems instead of raising.

    Important distinction:

    - `cited_chunks` preserves distinct source chunks for evaluation.
    - User-facing citations are deduplicated only within an adjacent citation
      run when multiple chunks render to the same page + section.

    Example:
        [S1][S3][S4]

    where S1, S3, and S4 are different chunks from page 41 / The Victoria Cross
    becomes:

        [Discover Canada, p. 41, The Victoria Cross]

    Invalid labels such as [S9] are preserved in the answer so the evaluation
    harness can inspect them.
    """

    by_source_id = {
        chunk.source_id: chunk
        for chunk in chunks
    }

    # Keep every citation occurrence for validation.
    cited_ids = [
        f"S{number}"
        for number in CITATION_PATTERN.findall(
            raw_answer
        )
    ]

    invalid_labels = sorted(
        {
            source_id
            for source_id in cited_ids
            if source_id not in by_source_id
        }
    )

    abstained = is_abstention(
        raw_answer
    )

    if invalid_labels:
        status: CitationStatus = (
            "invalid_labels"
        )
    elif not abstained and not cited_ids:
        status = "no_citations"
    else:
        status = "ok"

    def rendered_label(
        source_id: str,
    ) -> str:
        """
        Convert one source ID into its trusted user-facing citation.

        Invalid labels are deliberately preserved verbatim rather than
        discarded or raising an exception.
        """
        chunk = by_source_id.get(
            source_id
        )

        if chunk is None:
            return f"[{source_id}]"

        return (
            f"[Discover Canada, "
            f"{page_label(chunk.page)}, "
            f"{chunk.section}]"
        )

    def replace_citation_run(
        match: re.Match[str],
    ) -> str:
        """
        Resolve an adjacent citation run such as:

            [S1][S3][S4]

        Different source chunks may have identical user-facing provenance
        because overlapping chunks can come from the same page and section.

        Deduplicate identical rendered labels while preserving their
        first-occurrence order.
        """

        source_ids = [
            f"S{number}"
            for number
            in CITATION_PATTERN.findall(
                match.group(0)
            )
        ]

        rendered_labels: list[str] = []
        seen_labels: set[str] = set()

        for source_id in source_ids:
            label = rendered_label(
                source_id
            )

            if label in seen_labels:
                continue

            rendered_labels.append(
                label
            )
            seen_labels.add(
                label
            )

        return " ".join(
            rendered_labels
        )

    # Resolve citation runs rather than individual labels so identical
    # adjacent user-facing citations can be collapsed.
    resolved = CITATION_RUN_PATTERN.sub(
        replace_citation_run,
        raw_answer,
    )

    # Preserve distinct underlying source chunks for evaluation/provenance.
    cited_chunks: list[
        RetrievedChunk
    ] = []

    seen_source_ids: set[str] = set()

    for source_id in cited_ids:
        if (
            source_id
            in seen_source_ids
            or source_id
            not in by_source_id
        ):
            continue

        cited_chunks.append(
            by_source_id[source_id]
        )

        seen_source_ids.add(
            source_id
        )

    return CitationResolution(
        answer=resolved,
        cited_chunks=cited_chunks,
        citation_status=status,
        invalid_citation_labels=(
            invalid_labels
        ),
    )


# =============================================================================
# RAG orchestration
# =============================================================================

def rag_query(
    question: str,
    *,
    k: int = DEFAULT_K,
    model_id: str = GENERATION_MODEL,
) -> RAGResult:

    t0 = time.perf_counter()
    chunks = retrieve(
        question,
        k=k,
    )

    t1 = time.perf_counter()

    messages = build_prompt(
        question,
        chunks,
    )

    t2 = time.perf_counter()

    generation = generate(
        messages,
        model_id=model_id,
    )
    t3 = time.perf_counter()
    resolution = resolve_citations(
        generation.text,
        chunks,
    )
    t4 = time.perf_counter()

    retrieval_seconds = t1 - t0
    prompt_seconds = t2 - t1
    generation_seconds = t3 - t2
    citation_seconds = t4 - t3
    total_seconds = t4 - t0

    print('retrieval_seconds', retrieval_seconds)
    print('prompt_seconds', prompt_seconds)
    print('generation_seconds', generation_seconds)
    print('citation_seconds', citation_seconds)
    print('total_seconds', total_seconds)

    return RAGResult(
        question=question,
        raw_answer=generation.text,
        answer=resolution.answer,
        cited_chunks=(
            resolution.cited_chunks
        ),
        retrieved_chunks=chunks,
        generation_model=model_id,
        citation_status=(
            resolution.citation_status
        ),
        invalid_citation_labels=(
            resolution.invalid_citation_labels
        ),
        abstained=is_abstention(
            generation.text
        ),
        finish_reason=(
            generation.finish_reason
        ),
        generation_truncated=(
            generation.truncated
        ),
        prompt_tokens=(
            generation.prompt_tokens
        ),
        completion_tokens=(
            generation.completion_tokens
        ),
                retrieval_seconds=(
            retrieval_seconds
        ),
        prompt_seconds=(
            prompt_seconds
        ),
        generation_seconds=(
            generation_seconds
        ),
        citation_seconds=(
            citation_seconds
        ),
        total_seconds=(
            total_seconds
        ),
    )


def no_rag_query(
    question: str,
    *,
    model_id: str = GENERATION_MODEL,
) -> GenerationResult:
    return generate(
        build_no_rag_prompt(question),
        model_id=model_id,
    )


# =============================================================================
# Workshop output
# =============================================================================

def print_retrieval(
    chunks: list[RetrievedChunk],
) -> None:
    print("Retrieved:")

    for chunk in chunks:
        print(
            f"  {chunk.source_id}: "
            f"sim={chunk.similarity:.4f}  "
            f"{page_label(chunk.page)}  "
            f"{chunk.block_type:8s}  "
            f"{chunk.section}"
        )


def run_workshop_question(
    question: str,
    *,
    k: int = DEFAULT_K,
    compare_no_rag: bool = True,
) -> None:
    print("\n" + "=" * 88)
    print(f"QUESTION: {question}")
    print("=" * 88)

    result = rag_query(
        question,
        k=k,
    )

    print_retrieval(
        result.retrieved_chunks
    )

    print("\nRAG ANSWER:")
    print(result.answer)
    print(
        "\nRAG STATUS: "
        f"citation_status={result.citation_status}, "
        f"abstained={result.abstained}, "
        f"finish_reason={result.finish_reason!r}, "
        f"truncated={result.generation_truncated}"
    )

    if compare_no_rag:
        baseline = no_rag_query(
            question
        )

        print("\nNO-RAG ANSWER:")
        print(baseline.text)
        print(
            "\nNO-RAG STATUS: "
            f"finish_reason={baseline.finish_reason!r}, "
            f"truncated={baseline.truncated}"
        )


def main() -> None:
    questions = [
        "What does Confederation mean?",
        "Who was Sir Louis-Hippolyte La Fontaine?",
        "How does a bill become law?",
        "What is the capital of Nunavut?",
        "Which Victoria Cross recipients are named in Discover Canada?",
        "Who won the 2015 federal election?",
    ]

    for question in questions:
        run_workshop_question(
            question,
            k=DEFAULT_K,
            compare_no_rag=True,
        )


if __name__ == "__main__":
    main()
