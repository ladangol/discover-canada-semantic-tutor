"""Study behaviour: Ask, Explain, Quiz, Flashcards, and adaptive review.

Prompt building, source selection, grading and semantic-neighbour logic live
here. Retrieval comes from `retrieval.py`, the LLM from `generation.py`, and
stored embeddings from `corpus.py`. Nothing here imports Streamlit.
"""
from __future__ import annotations

import difflib
import json
import random
import re
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Literal

import numpy as np

from src.config import DEFAULT_K, query_cleanup_enabled
from src.corpus import Corpus, split_breadcrumb
from src.embeddings import encode_query
from src.generation import GenerationResult, generate
from src.query_understanding import retrieval_query
from src.retrieval import RetrievedChunk, page_label, retrieve
from src.suggestions import snap as snap_to_bank

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
N_DISTRACTORS = 3
N_DISTRACTOR_RESERVE = 2     # spare candidate passages in case one has no usable answer
MIN_DISTRACTOR_WORDS = 15
DEFAULT_DIFFICULTY = 0.5


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
8. Talk to the learner as a friendly tutor. Say "the guide"; never say "the
   retrieved sources", "the excerpts" or "the supplied text".
9. You only see some parts of the guide. If the question asks which or what
   things the guide mentions (a list), give what you found, then say there may
   be more in other parts of the guide. Never claim that the guide mentions no
   others.
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
class DistractorInfo:
    """A wrong option and the corpus passage it was drawn from."""
    option: str
    section: str
    page: str
    similarity: float     # cosine(source chunk, passage) on the stored embeddings


@dataclass(frozen=True)
class QuizQuestion:
    kind: Kind
    question: str
    source: RetrievedChunk
    explanation: str
    options: list[str] = field(default_factory=list)   # MCQ, shuffled
    correct_index: int | None = None                   # MCQ
    reference_answer: str = ""
    difficulty: float = DEFAULT_DIFFICULTY
    distractors: list["DistractorInfo"] = field(default_factory=list)
    option_similarity: float | None = None   # mean cosine(correct answer, each distractor) of the option TEXTS

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


def pick_distractor_passages(
    corpus: Corpus, source_chunk_id: str, difficulty: float,
    n: int = N_DISTRACTORS, reserve: int = N_DISTRACTOR_RESERVE,
) -> list[tuple[int, float]]:
    """Choose passages the wrong answers will be drawn from.

    `difficulty` in [0, 1] is the slider: candidates are every chunk of a
    different section, ranked by cosine similarity to the source chunk (original
    embeddings, one per section). 1.0 takes the closest ones (hard: wrong answers
    that are about nearly the same thing); 0.0 takes the most distant ones (easy).
    Returns n + reserve (index, similarity) pairs, the n nearest the target first.
    Euclidean distance on these unit vectors ranks identically to cosine.
    """
    ranked = corpus.neighbours(source_chunk_id, len(corpus), exclude_same_section=True)
    pool, seen = [], set()
    for j, sim in ranked:
        section = corpus.chunks.section.iat[j]
        if section in seen or corpus.chunks.word_count.iat[j] < MIN_DISTRACTOR_WORDS:
            continue
        seen.add(section)
        pool.append((j, sim))
    want = n + reserve
    center = round((1.0 - min(max(difficulty, 0.0), 1.0)) * (len(pool) - 1))
    start = min(max(center - want // 2, 0), max(len(pool) - want, 0))
    window = list(range(start, min(start + want, len(pool))))
    window.sort(key=lambda r: abs(r - center))
    return [pool[r] for r in window]


def _mcq_prompt(source: RetrievedChunk, passages: list[RetrievedChunk]) -> str:
    ctx = "\n\n".join(
        f"[D{n}] ({c.section}, {page_label(c.page)}): {' '.join(c.text.split())[:450]}"
        for n, c in enumerate(passages, start=1)
    )
    return f"""SOURCE PASSAGE ({source.section}, {page_label(source.page)}):
{source.text}

OTHER PASSAGES (wrong answers must be drawn from these):
{ctx}

Write ONE multiple-choice question testing a specific fact stated in the SOURCE PASSAGE.
- The correct answer must be stated explicitly in the SOURCE PASSAGE.
- Write exactly 3 wrong answers. EACH must be a short fact, name, number or phrase taken from a
  DIFFERENT one of the OTHER PASSAGES, of the same kind and similar length as the correct answer,
  so it is a plausible option. Tag each with the passage it came from ("D1", "D2", ...). Use
  only passages that really contain something usable; skip the rest.
- A wrong answer must be clearly wrong for this exact question and must not be supported by the
  SOURCE PASSAGE.
- Do not mention "the passage" or "the source" in the question.

Return JSON: {{"question": str, "correct_answer": str,
"distractors": [{{"from": "D2", "answer": str}}, {{"from": "D4", "answer": str}}, {{"from": "D1", "answer": str}}],
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
    difficulty: float = DEFAULT_DIFFICULTY,
    rng: random.Random | None = None,
    gen: Generate = generate,
) -> QuizQuestion:
    """One question at a time.

    `difficulty` (MCQ only, 0 easy .. 1 hard) sets how semantically close the
    wrong answers' source passages are to the correct answer's passage.

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
    return _free_question(source, gen) if kind == "free" else _mcq_question(corpus, index, source, difficulty, rng, gen)


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


def _mcq_question(corpus: Corpus, index: int, source: RetrievedChunk, difficulty: float,
                  rng: random.Random, gen: Generate) -> QuizQuestion:
    picks = pick_distractor_passages(corpus, source.chunk_id, difficulty)
    passages = sources_from_corpus(corpus, [j for j, _ in picks])
    sim_of = {f"D{n}": sim for n, (_, sim) in enumerate(picks, start=1)}
    last_problem = "unknown"
    for _ in range(MAX_QUIZ_ATTEMPTS):
        try:
            data = _ask_llm_json(
                [{"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                 {"role": "user", "content": _mcq_prompt(source, passages)}], gen)
            correct = str(data["correct_answer"]).strip()
            question = str(data["question"]).strip()
            raw = data["distractors"]
            tags = [str(d["from"]).strip().upper() for d in raw]
            wrong = [str(d["answer"]).strip() for d in raw]
        except (ValueError, KeyError, TypeError):
            last_problem = "malformed model output"
            continue
        options = [correct] + wrong
        if (len(wrong) != N_DISTRACTORS or len(set(tags)) != N_DISTRACTORS
                or not set(tags) <= set(sim_of) or len({o.lower() for o in options}) != 4
                or not all(options)):
            last_problem = "wrong answers were not 3 distinct options from 3 different passages"
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

        by_tag = {f"D{n}": passages[n - 1] for n in range(1, len(passages) + 1)}
        infos = [DistractorInfo(w, by_tag[t].section, by_tag[t].page, sim_of[t])
                 for w, t in zip(wrong, tags)]
        vectors = [encode_query(o) for o in options]
        option_sim = float(np.mean([vectors[0] @ v for v in vectors[1:]]))
        return QuizQuestion(
            "mcq", question, source, str(data.get("explanation", "")).strip(),
            options=shuffled, correct_index=correct_index, difficulty=difficulty,
            distractors=infos, option_similarity=option_sim,
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
    *, kind: Kind = "mcq", difficulty: float = DEFAULT_DIFFICULTY,
    rng: random.Random | None = None, gen: Generate = generate,
) -> QuizQuestion:
    """Quiz a semantic neighbour of the missed concept."""
    cands = review_candidates(corpus, missed_chunk_id, history)
    if not cands:
        raise QuizGenerationError("No new nearby concepts left to review.")
    rng = rng or random.Random()
    return generate_quiz(corpus, kind=kind, focus_chunk_id=corpus.chunks.chunk_id.iat[rng.choice(cands)],
                         difficulty=difficulty, rng=rng, gen=gen)


def explain_neighbourhood(corpus: Corpus, missed_chunk_id: str, *, gen: Generate = generate) -> TutorAnswer:
    idx = [corpus.index_of(missed_chunk_id)] + neighbourhood(corpus, missed_chunk_id, 3)
    row = corpus.row(missed_chunk_id)
    return explain_sources(f"“{row.section}” and closely related ideas",
                           sources_from_corpus(corpus, idx), gen=gen)


# =============================================================================
# One conversational turn (shared by every interface; voice is only an input/output adapter)
# =============================================================================

TurnKind = Literal["answer", "quiz_feedback", "clarify"]

_FILLER = {"the", "an", "option", "answer", "choice", "letter", "number", "is", "its", "it", "s", "i", "think", "pick",
           "choose", "going", "with", "go", "my", "would", "say", "please", "one", "that", "ill", "want", "take"}
_ORDINALS = {"first": 0, "1st": 0, "1": 0, "second": 1, "2nd": 1, "2": 1, "two": 1, "third": 2, "3rd": 2, "3": 2, "three": 2,
             "fourth": 3, "4th": 3, "4": 3, "four": 3, "last": -1}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("’", "'").replace("'", ""))).strip()


def match_option(options: list[str], spoken: str) -> int | None:
    """Which option did the learner mean? Handles 'B', 'option B', 'the second one', or the answer text itself.

    Returns None when it is unclear or ambiguous so the tutor can ask again instead of guessing a grade."""
    if not options:
        return None
    t = _norm(spoken)
    if not t:
        return None
    letters = "abcd"[: len(options)]
    rest = [w for w in t.split() if w not in _FILLER or w in _ORDINALS]
    if len(rest) == 1:
        w = rest[0]
        if len(w) == 1 and w in letters:
            return letters.index(w)
        if w in _ORDINALS:
            i = _ORDINALS[w]
            i = len(options) - 1 if i == -1 else i
            if 0 <= i < len(options):
                return i
    hits = [i for i, o in enumerate(options) if _norm(o) and (_norm(o) == t or _norm(o) in t or (len(t) >= 3 and t in _norm(o)))]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        scored = sorted(((difflib.SequenceMatcher(None, t, _norm(o)).ratio(), i) for i, o in enumerate(options)), reverse=True)
        if scored and scored[0][0] >= 0.85 and (len(scored) == 1 or scored[1][0] < scored[0][0] - 0.1):
            return scored[0][1]
    return None


@dataclass(frozen=True)
class TurnResult:
    kind: TurnKind
    text: str                                   # what the learner reads (markdown)
    speech_text: str                            # what is read aloud (before citation/markup stripping)
    cited: list[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False
    quiz_result: QuizResult | None = None
    chosen_index: int | None = None
    timings: dict[str, float] = field(default_factory=dict)   # measured seconds: retrieval, generation, total
    suggestions: list[str] = field(default_factory=list)      # "did you mean…" questions, only when the guide had no match
    retrieval_query: str = ""                                 # the string that was embedded for search (differs from the question only if cleaned)


SUGGEST_SYSTEM_PROMPT = """You help a learner preparing for the Canadian citizenship test. Their question could not be matched
to the guide's text, perhaps because it was worded differently or the guide uses other terms. Think about what they most
probably meant, using your general knowledge of Canada and the guide's table of contents (chapter and section titles only).
Propose up to 4 short, specific questions, closest to their intent first, that the guide may well answer. Write plain questions
only, not facts or answers; each is matched to the guide's own questions before the learner sees anything. Each question must be
under 100 characters and end with "?". Output JSON only: {"suggestions": ["...?", "...?"]}"""


def guide_outline(corpus: Corpus) -> str:
    return "\n".join(f"- {ch}: " + "; ".join(corpus.sections(ch)[:12]) for ch in corpus.chapters)


def suggest_questions(question: str, corpus: Corpus, *, gen: Generate = generate) -> list[str]:
    """Guesses at what the learner meant, for a question the grounded answer could not match (ONE LLM call). The model sees only the
    guide's section titles and may use general knowledge, so these are never shown as they are: `suggestions.snap` first replaces
    each with the nearest question from the pre-built bank of questions the guide is known to answer."""
    data = _ask_llm_json([{"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
                          {"role": "user", "content": f"LEARNER'S QUESTION: {question}\n\nGUIDE OUTLINE:\n{guide_outline(corpus)}"}],
                         gen, max_tokens=400)
    items = data.get("suggestions", []) if isinstance(data, dict) else []
    out: list[str] = []
    for it in items:
        t = re.sub(r"\s+", " ", str(it)).strip()
        if t.endswith("?") and 8 <= len(t) <= 120 and _norm(t) != _norm(question) and t not in out:
            out.append(t)
    return out[:4]



def search_text(question: str) -> str:
    """The string to embed for search in the learner/voice flows: the question without document-reference boilerplate (application layer,
    see query_understanding.py), or the question itself when QUERY_CLEANUP=0. Generation always gets the exact original question."""
    return retrieval_query(question).text if query_cleanup_enabled() else question


NO_MATCH = "I couldn't find that in the guide, so I won't guess. It may be worded differently there."


def answer_turn(ans: TutorAnswer, timings: dict[str, float] | None = None) -> TurnResult:
    """The learner-facing turn for a grounded answer."""
    return TurnResult("answer", ans.answer, ans.answer, ans.cited, abstained=ans.abstained, timings=timings or {})


def handle_turn(message: str, *, quiz: QuizQuestion | None = None, corpus: Corpus | None = None,
                retrieve_fn=retrieve, gen: Generate = generate, snap_fn=snap_to_bank) -> TurnResult:
    """Handle one confirmed learner message. With an open `quiz` question the message is an ANSWER to it (graded by the
    existing graders); otherwise it is a question answered from the guide. Blank text never reaches retrieval or the LLM.
    If the guide has no match and `corpus` is given, the reply offers up to 3 "did you mean…" questions from the guide's own question
    bank (one LLM call to guess the intent, then an embedding lookup; best effort)."""
    t0 = time.perf_counter()
    message = (message or "").strip()
    if not message:
        return TurnResult("clarify", "I didn't catch anything. Please say or type your question.",
                          "I didn't catch anything. Please say or type your question.")
    if quiz is not None:
        if quiz.kind == "mcq":
            idx = match_option(quiz.options, message)
            if idx is None:
                msg = "I wasn't sure which answer you meant. Say A, B, C or D, or say the answer itself."
                return TurnResult("clarify", msg, msg, timings={"total": time.perf_counter() - t0})
            res = grade_mcq(quiz, idx)
            gen_s = 0.0
        else:
            g0 = time.perf_counter()
            res, idx = grade_free(quiz, message, gen=gen), None
            gen_s = time.perf_counter() - g0
        head = "Correct!" if res.correct else (res.feedback or "Not quite.")
        body = f"{head}\n\n{res.explanation}".strip() if res.explanation else head
        spoken = f"{'Correct.' if res.correct else head} {res.explanation or ''}".strip()
        return TurnResult("quiz_feedback", body, spoken, [res.source], quiz_result=res, chosen_index=idx,
                          timings={"generation": gen_s, "total": time.perf_counter() - t0})
    r0 = time.perf_counter()
    query = search_text(message)
    chunks = retrieve_fn(query, k=DEFAULT_K)
    retrieval_s = time.perf_counter() - r0
    g0 = time.perf_counter()
    ans = ask(message, chunks, gen=gen)
    gen_s = time.perf_counter() - g0
    if not ans.abstained:
        return replace(answer_turn(ans, {"retrieval": retrieval_s, "generation": gen_s, "total": time.perf_counter() - t0}), retrieval_query=query)
    suggestions: list[str] = []
    if corpus is not None:
        try:
            suggestions = snap_fn(suggest_questions(message, corpus, gen=gen), asked=message)
        except Exception:                       # suggestions are a courtesy: never break the turn (budget, provider, parsing)
            suggestions = []
    text = NO_MATCH + ("\n\nDid you mean one of these? The guide answers them." if suggestions
                       else " Try a specific topic, such as a person, a place or an event.")
    spoken = NO_MATCH + (" Did you mean: " + " Or: ".join(suggestions) if suggestions
                         else " Try a specific topic, such as a person, a place or an event.")
    # cited=[]: nothing in the guide supports an answer here, so no source is shown as if it did
    return TurnResult("answer", text, spoken, [], abstained=True, suggestions=suggestions, retrieval_query=query,
                      timings={"retrieval": retrieval_s, "generation": gen_s, "total": time.perf_counter() - t0})
