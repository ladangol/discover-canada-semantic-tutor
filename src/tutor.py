"""Study behaviour: Ask, Explain, Quiz, Flashcards, and adaptive review.

Prompt building, source selection, grading and semantic-neighbour logic live
here. Retrieval comes from `retrieval.py`, the LLM from `generation.py`, and
stored embeddings from `corpus.py`. Nothing here imports Streamlit.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from src.config import DEFAULT_K
from src.corpus import Corpus, split_breadcrumb
from src.generation import GenerationResult, generate
from src.retrieval import RetrievedChunk, page_label, retrieve

Generate = Callable[..., GenerationResult]

UNKNOWN_ANSWER = "I don't know based on Discover Canada."
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
CITATION_RUN_PATTERN = re.compile(r"\[S\d+\](?:\s*\[S\d+\])*")
ABSTENTION_PATTERN = re.compile(
    r"^\s*i\s+(?:do\s+not|don['’]t)\s+know\s+based\s+on\s+discover\s+canada\b",
    flags=re.IGNORECASE,
)

MIN_QUIZ_WORDS = 30          # skip chunks too short to carry a testable fact
MAX_QUIZ_ATTEMPTS = 3
N_DISTRACTOR_CONTEXT = 4


class QuizGenerationError(RuntimeError):
    pass


# =============================================================================
# Sources + prompt/citation helpers
# =============================================================================

def sources_from_corpus(corpus: Corpus, indices: list[int]) -> list[RetrievedChunk]:
    """Wrap corpus chunks as labelled sources S1..Sn (no retrieval score)."""
    out = []
    for n, i in enumerate(indices, start=1):
        r = corpus.chunks.iloc[i]
        out.append(RetrievedChunk(
            source_id=f"S{n}", chunk_id=r.chunk_id, text=r.text, page=r.page,
            section=r.section, block_type=r.block_type,
        ))
    return out


def format_source_block(chunk: RetrievedChunk) -> str:
    return (
        f"SOURCE {chunk.source_id}\npage: {chunk.page}\nsection: {chunk.section}\n"
        f"block_type: {chunk.block_type}\ntext:\n{chunk.text}"
    )


def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(format_source_block(c) for c in chunks)


def citation_label(chunk: RetrievedChunk) -> str:
    return f"Discover Canada, {page_label(chunk.page)}, {chunk.section}"


def split_text(text: str) -> str:
    """Chunk text without its chapter/section breadcrumb (for display)."""
    return split_breadcrumb(text)[1]


def is_abstention(text: str) -> bool:
    return ABSTENTION_PATTERN.search(text) is not None


def resolve_citations(raw: str, chunks: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
    """Replace [S#] labels with page+section citations (resolved by code, not by
    the LLM). Unknown labels are left as-is. Returns (text, cited chunks)."""
    by_id = {c.source_id: c for c in chunks}

    def render_run(match: re.Match[str]) -> str:
        labels: list[str] = []
        for n in CITATION_PATTERN.findall(match.group(0)):
            chunk = by_id.get(f"S{n}")
            label = f"[{citation_label(chunk)}]" if chunk else f"[S{n}]"
            if label not in labels:
                labels.append(label)
        return " ".join(labels)

    cited, seen = [], set()
    for n in CITATION_PATTERN.findall(raw):
        sid = f"S{n}"
        if sid in by_id and sid not in seen:
            cited.append(by_id[sid])
            seen.add(sid)
    return CITATION_RUN_PATTERN.sub(render_run, raw), cited


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM reply (fences tolerated)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON found in model reply.")


def _ask_llm_json(messages: list[dict[str, str]], gen: Generate, *, max_tokens: int = 1800):
    return extract_json(gen(messages, max_tokens=max_tokens).text)


# =============================================================================
# Semantic selection: neighbours + MMR diversity
# =============================================================================

def mmr_select(
    embeddings: np.ndarray,
    candidates: list[int],
    n: int,
    *,
    lambda_: float = 0.5,
    query: np.ndarray | None = None,
) -> list[int]:
    """Greedy MMR: relevant to `query` (default: candidate centroid) but
    dissimilar to what is already chosen. Cosine on the original embeddings."""
    if len(candidates) <= n:
        return list(candidates)
    vecs = embeddings[candidates]
    if query is None:
        query = vecs.mean(axis=0)
    query = query / (np.linalg.norm(query) or 1.0)
    relevance = vecs @ query
    pair = vecs @ vecs.T

    chosen: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(chosen) < n:
        if chosen:
            scores = [lambda_ * relevance[j] - (1 - lambda_) * max(pair[j, c] for c in chosen)
                      for j in remaining]
        else:
            scores = [relevance[j] for j in remaining]
        best = remaining[int(np.argmax(scores))]
        chosen.append(best)
        remaining.remove(best)
    return [candidates[j] for j in chosen]


def neighbourhood(corpus: Corpus, chunk_id: str, n: int = 4) -> list[int]:
    """Semantically nearby chunks, at most one per OTHER section.

    Overlapping windows of the same section are near-duplicates, not new
    concepts, so the closest chunk of each distinct nearby section is kept.
    """
    out, seen = [], set()
    for j, _ in corpus.neighbours(chunk_id, len(corpus), exclude_same_section=True):
        section = corpus.chunks.section.iat[j]
        if section in seen:
            continue
        seen.add(section)
        out.append(j)
        if len(out) == n:
            break
    return out


# =============================================================================
# Ask / Explain
# =============================================================================

ASK_SYSTEM_PROMPT = f"""You answer questions using only the supplied excerpts from
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

EXPLAIN_SYSTEM_PROMPT = f"""You are a friendly study tutor for the Canadian citizenship
guide Discover Canada. Explain the supplied SOURCE blocks in simple, study-friendly
language for someone preparing for the citizenship test.

Rules:
1. Use only facts supported by the SOURCE blocks. Do not add outside knowledge.
2. Be concise: a short plain-language explanation, then 3-5 bullet "key points to remember".
3. Cite sources inline using only their labels, e.g. [S1] or [S2][S3]. Never invent a label.
4. Do not write page numbers yourself.
5. If the sources are unrelated to the request, reply exactly: "{UNKNOWN_ANSWER}"
"""


@dataclass(frozen=True)
class TutorAnswer:
    prompt_subject: str
    sources: list[RetrievedChunk]
    answer: str                     # citations rendered as page + section
    raw_answer: str
    cited: list[RetrievedChunk]
    abstained: bool
    truncated: bool


def _grounded_answer(
    system: str, user: str, subject: str, sources: list[RetrievedChunk], gen: Generate
) -> TutorAnswer:
    result = gen(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=1500,
    )
    rendered, cited = resolve_citations(result.text, sources)
    return TutorAnswer(subject, sources, rendered, result.text, cited,
                       is_abstention(result.text), result.truncated)


def ask(question: str, chunks: list[RetrievedChunk] | None = None,
        *, gen: Generate = generate) -> TutorAnswer:
    """Grounded RAG answer over the frozen top-5. Pass `chunks` to reuse a
    retrieval that was already done for the atlas."""
    chunks = chunks if chunks is not None else retrieve(question, k=DEFAULT_K)
    user = f"QUESTION:\n{question}\n\nRETRIEVED SOURCES:\n{format_sources(chunks)}"
    return _grounded_answer(ASK_SYSTEM_PROMPT, user, question, chunks, gen)


def explain_sources(subject: str, sources: list[RetrievedChunk],
                    *, gen: Generate = generate) -> TutorAnswer:
    user = f"WHAT TO EXPLAIN:\n{subject}\n\nSOURCES:\n{format_sources(sources)}"
    return _grounded_answer(EXPLAIN_SYSTEM_PROMPT, user, subject, sources, gen)


def explain_question(question: str, chunks: list[RetrievedChunk] | None = None,
                     *, gen: Generate = generate) -> TutorAnswer:
    chunks = chunks if chunks is not None else retrieve(question, k=DEFAULT_K)
    return explain_sources(question, chunks, gen=gen)


def explain_chunk(corpus: Corpus, chunk_id: str, *, gen: Generate = generate) -> TutorAnswer:
    """Selected chunk plus its two nearest neighbours as supporting context."""
    idx = [corpus.index_of(chunk_id)] + [j for j, _ in corpus.neighbours(chunk_id, 2)]
    row = corpus.row(chunk_id)
    return explain_sources(f"The passage in “{row.section}”", sources_from_corpus(corpus, idx), gen=gen)


MAX_EXPLAIN_SOURCES = 6


def scope_sources(corpus: Corpus, chapter: str | None, section: str | None,
                  n: int = MAX_EXPLAIN_SOURCES) -> list[int]:
    """Diverse, document-ordered chunks that cover a chapter/section scope."""
    pool = corpus.scope_indices(chapter, section)
    if not pool:
        raise ValueError("No chunks in that scope.")
    return sorted(mmr_select(corpus.embeddings, pool, n))


def explain_scope(corpus: Corpus, chapter: str | None, section: str | None,
                  *, gen: Generate = generate) -> TutorAnswer:
    sources = sources_from_corpus(corpus, scope_sources(corpus, chapter, section))
    return explain_sources(section or chapter or "Discover Canada", sources, gen=gen)


# =============================================================================
# Quiz
# =============================================================================

Kind = Literal["mcq", "free"]


@dataclass(frozen=True)
class QuizQuestion:
    kind: Kind
    question: str
    source: RetrievedChunk
    explanation: str
    options: list[str] = field(default_factory=list)   # MCQ, shuffled
    correct_index: int | None = None                   # MCQ
    reference_answer: str = ""
    distractor_sections: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return self.source.chunk_id


@dataclass(frozen=True)
class QuizResult:
    correct: bool
    feedback: str
    explanation: str
    source: RetrievedChunk


QUIZ_SYSTEM_PROMPT = """You write exam questions for the Canadian citizenship guide
Discover Canada. Base every question ONLY on the supplied SOURCE passage. Output
a single JSON object and nothing else."""


def pick_quiz_chunk(
    corpus: Corpus,
    *,
    chapter: str | None = None,
    section: str | None = None,
    exclude_ids: set[str] = frozenset(),
    candidates: list[int] | None = None,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random.Random()
    pool = candidates if candidates is not None else corpus.scope_indices(chapter, section)
    df = corpus.chunks
    usable = [i for i in pool
              if df.chunk_id.iat[i] not in exclude_ids and df.word_count.iat[i] >= MIN_QUIZ_WORDS]
    usable = usable or [i for i in pool if df.chunk_id.iat[i] not in exclude_ids] or list(pool)
    if not usable:
        raise QuizGenerationError("No chunks available in that scope.")
    return rng.choice(usable)


def _source_for(corpus: Corpus, index: int) -> RetrievedChunk:
    return sources_from_corpus(corpus, [index])[0]


def _mcq_prompt(source: RetrievedChunk, distractor_ctx: list[RetrievedChunk]) -> str:
    ctx = "\n\n".join(
        f"[D{n}] ({c.section}, {page_label(c.page)}): {' '.join(c.text.split())[:350]}"
        for n, c in enumerate(distractor_ctx, start=1)
    )
    return f"""SOURCE PASSAGE ({source.section}, {page_label(source.page)}):
{source.text}

NEARBY CONCEPTS (candidate distractors; semantically close but from other parts of the guide):
{ctx or '(none)'}

Write ONE multiple-choice question testing a specific fact stated in the SOURCE PASSAGE.
- The correct answer must be stated explicitly in the SOURCE PASSAGE.
- Write exactly 3 distractors. Prefer plausible ideas drawn from the NEARBY CONCEPTS, but
  use one only if it is clearly WRONG for this exact question. Otherwise invent a plausible
  wrong answer. A distractor must never also be supported by the SOURCE PASSAGE.
- Keep options short and similar in length and style.
- Do not mention "the passage" or "the source" in the question.

Return JSON: {{"question": str, "correct_answer": str, "distractors": [str, str, str],
"explanation": str (1-2 sentences, why the correct answer is right)}}"""


def _verify_mcq(source: RetrievedChunk, question: str, options: list[str], gen: Generate) -> list[int]:
    letters = "ABCD"
    listing = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    prompt = f"""SOURCE PASSAGE:
{source.text}

QUESTION: {question}
{listing}

Using ONLY the SOURCE PASSAGE, list every option that the passage clearly supports as a
correct answer to the question. Return JSON: {{"supported": ["A", ...]}} (use [] if none)."""
    reply = _ask_llm_json(
        [{"role": "system", "content": "You strictly check quiz answers against a source. Output JSON only."},
         {"role": "user", "content": prompt}], gen, max_tokens=600)
    supported = reply.get("supported", []) if isinstance(reply, dict) else []
    return sorted(letters.index(s.strip().upper()[0]) for s in supported
                  if isinstance(s, str) and s.strip() and s.strip().upper()[0] in letters)


def generate_quiz(
    corpus: Corpus,
    *,
    kind: Kind = "mcq",
    chapter: str | None = None,
    section: str | None = None,
    focus_chunk_id: str | None = None,
    candidate_ids: list[str] | None = None,
    exclude_ids: set[str] = frozenset(),
    rng: random.Random | None = None,
    gen: Generate = generate,
) -> QuizQuestion:
    """One question at a time.

    `focus_chunk_id` pins the source chunk (used for adaptive review);
    `candidate_ids` restricts the pool (e.g. a retrieved top-5).
    """
    rng = rng or random.Random()
    if focus_chunk_id:
        index = corpus.index_of(focus_chunk_id)
    else:
        cands = [corpus.index_of(c) for c in candidate_ids] if candidate_ids else None
        index = pick_quiz_chunk(corpus, chapter=chapter, section=section,
                                exclude_ids=exclude_ids, candidates=cands, rng=rng)
    source = _source_for(corpus, index)
    return _free_question(source, gen) if kind == "free" else _mcq_question(corpus, index, source, rng, gen)


def _free_question(source: RetrievedChunk, gen: Generate) -> QuizQuestion:
    prompt = f"""SOURCE PASSAGE ({source.section}, {page_label(source.page)}):
{source.text}

Write ONE short free-response question whose answer is stated explicitly in the passage.
Return JSON: {{"question": str, "reference_answer": str (1-2 sentences), "explanation": str (1 sentence)}}"""
    data = _ask_llm_json([{"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}], gen)
    try:
        return QuizQuestion("free", str(data["question"]).strip(), source,
                            str(data.get("explanation", "")).strip(),
                            reference_answer=str(data["reference_answer"]).strip())
    except (KeyError, TypeError) as exc:
        raise QuizGenerationError("Model returned a malformed question.") from exc


def _mcq_question(corpus: Corpus, index: int, source: RetrievedChunk,
                  rng: random.Random, gen: Generate) -> QuizQuestion:
    near = neighbourhood(corpus, source.chunk_id, N_DISTRACTOR_CONTEXT)
    distractor_ctx = sources_from_corpus(corpus, near)
    last_problem = "unknown"
    for _ in range(MAX_QUIZ_ATTEMPTS):
        try:
            data = _ask_llm_json(
                [{"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                 {"role": "user", "content": _mcq_prompt(source, distractor_ctx)}], gen)
            correct = str(data["correct_answer"]).strip()
            distractors = [str(d).strip() for d in data["distractors"]]
            question = str(data["question"]).strip()
        except (ValueError, KeyError, TypeError):
            last_problem = "malformed model output"
            continue
        options = [correct] + distractors
        if len(distractors) != 3 or len({o.lower() for o in options}) != 4 or not all(options):
            last_problem = "options were not 4 distinct answers"
            continue

        order = list(range(4))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        correct_index = order.index(0)

        # Independent check: exactly one option is supported by the source.
        try:
            supported = _verify_mcq(source, question, shuffled, gen)
        except (ValueError, KeyError, TypeError, AttributeError):
            last_problem = "verification output was malformed"
            continue
        if supported != [correct_index]:
            last_problem = "answer was not uniquely supported by the source"
            continue
        return QuizQuestion(
            "mcq", question, source, str(data.get("explanation", "")).strip(),
            options=shuffled, correct_index=correct_index,
            distractor_sections=[c.section for c in distractor_ctx],
        )
    raise QuizGenerationError(f"Could not produce a verified question ({last_problem}). Try again.")


def grade_mcq(q: QuizQuestion, chosen_index: int) -> QuizResult:
    ok = chosen_index == q.correct_index
    feedback = "Correct!" if ok else f"Not quite — the answer is: {q.options[q.correct_index]}"
    return QuizResult(ok, feedback, q.explanation, q.source)


def grade_free(q: QuizQuestion, answer: str, *, gen: Generate = generate) -> QuizResult:
    prompt = f"""SOURCE PASSAGE:
{q.source.text}

QUESTION: {q.question}
REFERENCE ANSWER: {q.reference_answer}
STUDENT ANSWER: {answer}

Judge whether the student's answer is correct according to the SOURCE PASSAGE. Accept
paraphrases; reject answers that miss or contradict the key fact.
Return JSON: {{"correct": true|false, "feedback": str (1-2 sentences, encouraging)}}"""
    data = _ask_llm_json([{"role": "system", "content": "You are a fair grader. Output JSON only."},
                          {"role": "user", "content": prompt}], gen, max_tokens=500)
    ok = bool(data.get("correct")) if isinstance(data, dict) else False
    feedback = str(data.get("feedback", "")) if isinstance(data, dict) else ""
    return QuizResult(ok, feedback, f"Reference answer: {q.reference_answer}", q.source)


# =============================================================================
# Flashcards
# =============================================================================

@dataclass(frozen=True)
class Flashcard:
    front: str
    back: str
    page: str
    section: str


def generate_flashcards(
    corpus: Corpus,
    *,
    n: int = 5,
    question: str | None = None,
    query_chunks: list[RetrievedChunk] | None = None,
    chapter: str | None = None,
    section: str | None = None,
    gen: Generate = generate,
) -> list[Flashcard]:
    """Cards from the current query (frozen top-5) or a chapter/section scope.

    Source chunks are chosen by MMR over the stored embeddings so five cards
    don't come from five overlapping windows of the same paragraph.
    """
    if question:
        chunks = query_chunks if query_chunks is not None else retrieve(question, k=DEFAULT_K)
        idx = [corpus.index_of(c.chunk_id) for c in chunks]
        picked = mmr_select(corpus.embeddings, idx, n, lambda_=0.5)
    else:
        picked = mmr_select(corpus.embeddings, corpus.scope_indices(chapter, section), n)
    sources = sources_from_corpus(corpus, picked)

    prompt = f"""{format_sources(sources)}

Write exactly one concise flashcard per SOURCE, each testing a different key fact stated in
that source. Front: a short question or term. Back: a short answer (max 25 words).
Return JSON: [{{"source": "S1", "front": str, "back": str}}, ...]"""
    data = _ask_llm_json(
        [{"role": "system", "content": "You write concise study flashcards using only the supplied sources. Output JSON only."},
         {"role": "user", "content": prompt}], gen, max_tokens=1800)
    by_id = {s.source_id: s for s in sources}
    cards = []
    for item in data if isinstance(data, list) else []:
        src = by_id.get(str(item.get("source", "")).strip()) if isinstance(item, dict) else None
        if src and item.get("front") and item.get("back"):
            cards.append(Flashcard(str(item["front"]).strip(), str(item["back"]).strip(),
                                   src.page, src.section))
    if not cards:
        raise QuizGenerationError("Model returned no usable flashcards. Try again.")
    return cards[:n]


# =============================================================================
# Adaptive review (session-only; the caller owns the history list)
# =============================================================================

@dataclass(frozen=True)
class AnswerRecord:
    chunk_id: str
    section: str
    question: str
    correct: bool


def weak_sections(history: list[AnswerRecord]) -> list[tuple[str, int]]:
    """Sections ranked by net misses (misses minus hits), worst first."""
    score: dict[str, int] = {}
    for r in history:
        score[r.section] = score.get(r.section, 0) + (-1 if r.correct else 1)
    return sorted(((s, v) for s, v in score.items() if v > 0), key=lambda t: -t[1])


def review_candidates(corpus: Corpus, missed_chunk_id: str, history: list[AnswerRecord],
                      n: int = 4) -> list[int]:
    """Neighbourhood of a missed concept, minus chunks already answered correctly."""
    done = {r.chunk_id for r in history if r.correct}
    near = neighbourhood(corpus, missed_chunk_id, n + len(done))
    return [j for j in near if corpus.chunks.chunk_id.iat[j] not in done][:n]


def next_review_question(
    corpus: Corpus, missed_chunk_id: str, history: list[AnswerRecord],
    *, kind: Kind = "mcq", rng: random.Random | None = None, gen: Generate = generate,
) -> QuizQuestion:
    """Quiz a semantic neighbour of the missed concept."""
    cands = review_candidates(corpus, missed_chunk_id, history)
    if not cands:
        raise QuizGenerationError("No new nearby concepts left to review.")
    rng = rng or random.Random()
    return generate_quiz(corpus, kind=kind, focus_chunk_id=corpus.chunks.chunk_id.iat[rng.choice(cands)],
                         rng=rng, gen=gen)


def explain_neighbourhood(corpus: Corpus, missed_chunk_id: str, *, gen: Generate = generate) -> TutorAnswer:
    idx = [corpus.index_of(missed_chunk_id)] + neighbourhood(corpus, missed_chunk_id, 3)
    row = corpus.row(missed_chunk_id)
    return explain_sources(f"“{row.section}” and closely related ideas",
                           sources_from_corpus(corpus, idx), gen=gen)
