# AGENTS.md — Discover Canada Atlas

Tool-neutral instructions for Claude Code, Codex, or any other coding agent.

## What this repo is

A polished local Streamlit demo built **on top of a frozen, already-evaluated
retrieval system** over the *Discover Canada* study guide. It is a product
repo, not a research repo.

## Frozen — do not touch

- Retrieval configuration is **frozen**: `sentence-transformers/all-MiniLM-L6-v2`,
  normalized embeddings, cosine space, **plain dense retrieval, k = 5**.
- Do **not** redo chunking. Do **not** redo embeddings. Do **not** tune retrieval.
- Do **not** run the held-out benchmark again or modify its results.
- Do **not** add BM25, RRF, rerankers, retrieval variants, LangChain,
  LangGraph, agent frameworks, React/Vue/FastAPI, or a database server.
- The Chroma collection is `discover-canada-window-minilm` in `chroma_db/`
  (236 chunks, 384-d). Embeddings are read from Chroma, never recomputed for
  the corpus. Only *queries* are embedded at runtime.

## Secrets — hard rules

- **Never open, read, print, cat, grep, search, parse, inspect, copy, or
  summarize `.env`.** It is an opaque secret store owned by the user. This
  includes recursive greps/finds, debug output, and "just checking the key
  exists".
- Application code may call `load_dotenv()` and read `VENICE_API_KEY` /
  `VENICE_MODEL` from `os.environ` at runtime. That is the only allowed use.
- Never log, print, test, document, or put an API key in an exception message.
  If unconfigured, say only: `VENICE_API_KEY is not configured.`
- `.env.example` holds placeholders only.

## Architecture rules

- Dependency direction: `app.py` → `tutor` / `visualization` → `retrieval` /
  `generation` → `embeddings` / `corpus` / `config`.
- **Keep UI code thin.** `app.py` orchestrates; no retrieval algorithms or
  prompt building inline.
- **Keep generation isolated.** Only `src/generation.py` knows Venice exists,
  behind `generate(messages, model_id=None)`.
- Retrieval, the atlas, and nearest-neighbour inspection must work with no
  API key.
- UMAP coordinates are for display only — never use 2-D distance as
  similarity. Neighbours/retrieval use the original 384-d cosine similarity.
- **Keep `docs/ARCHITECTURE.md` and `docs/FLOWS.md` synchronized** with the
  code whenever modules or call paths change. Diagrams must show only what is
  actually implemented.

## Analyze tab rules (hubs, timelines, chapter map)

- Pipeline is always retrieval → structured extraction (LLM, JSON only) → validation → rendering; keep the
  stages in separate modules (`learning_extractors`, `learning_validation`, `visualization`).
- The LLM sees only retrieved source text. Never infer facts, dates or relationships from embedding
  proximity or from UMAP/MDS positions. Omit unsupported items rather than guessing.
- Chapter statistics use the original embeddings (no LLM, no UMAP).
- Hub/timeline candidate retrieval (k=12) is separate from the frozen k=5 answer retrieval; do not change either.

## Infographic and learner-app rules

- Facts, dates, relationships and citations come only from validated JSON. The LLM never emits SVG, CSS
  or layout; all text is XML-escaped in `infographic_renderers.py`. Do not use an image model for factual graphics.
- Keep content (`infographic_models`), design system (`infographic_themes`), geometry (`infographic_layout`)
  and serialization (`infographic_renderers`) separate; add themes/layouts there, never per concept.
- `app.py` (research demo) and `study_app.py` (learner app) share the same `src/` and `ui/` code; do not
  duplicate retrieval, tutor, validation or rendering logic. The learner app must not show embeddings, UMAP,
  cosine scores, rankings or chunk ids (the one deliberate mention is the brand line "powered by AI embeddings").
- Learner-app “Did you know?” facts live in `src/facts.py`; each carries a verbatim quote that `tests/test_facts.py` checks
  against the guide. Mascots and the leaf mark are static files in `static/` (served via `enableStaticServing`).
- Exports must not contain interactive hooks, scripts, debug info or any secret. The Venice key stays server-side.

## Deployment rules

- Every AI call goes through `generation.generate`, which enforces the daily budget (`src/budget.py`) and maps provider
  failures to `GenerationUnavailable` with a learner-safe message. Never surface raw provider errors or log prompts/keys.
- Cache hits must not call `generate`. Keep `deploy/`, the `Dockerfile` and `docs/` in sync with the code (flags, env names).
- `deploy/deploy.sh` refuses to run on a droplet with under 1.8 GB RAM; never override that for the production box.

## Voice tutor rules

- Voice is only an input/output adapter (`src/voice/*`, `ui/voice_tutor.py`). Questions and quiz answers go through the same
  `tutor.handle_turn` / `ask` / `grade_mcq` / `grade_free` as typed ones; never add a second retrieval, grading or citation path.
- `src/voice/*` imports no Streamlit and nothing from `tutor`/`retrieval`. Models load once per process, lazily; failures become
  `VoiceError` messages that are safe to show, never tracebacks.
- Nothing reaches the tutor until the learner confirms the (editable) transcript. Raw audio and transcripts are never written to
  disk, logged, or sent to Venice (only the confirmed text is). Keep the original ASR text separate from the learner's correction.
- Do not add wake word, streaming/VAD or barge-in until Milestone 2/3. Tests use mocks only (no models, audio hardware or API key).
- The Talk page stays opt-in (`STUDY_VOICE=1`) and voice packages stay out of `requirements.txt`/the production image.

## Application layers around the frozen core

- The benchmarked retrieval core is frozen and stays untouched: ChromaDB, the chunk corpus, MiniLM and the Variant A retrieval function (plain dense, k = 5).
  Never re-run the held-out evaluation for product changes. Product behaviour lives in layers around it (see docs/APPLICATION_LAYERS.md).
- **Query understanding** (`src/query_understanding.py`) only DELETES document-reference boilerplate from a separate search string. It must never add, replace
  or rewrite words, never map synonyms or categories ("First Nations", "Aboriginal", "Indigenous", "Métis", "Inuit" are different), and the exact original question
  always goes to generation. The research Atlas does not use it.
- **Concept validation** (`src/concept_semantics.py`): types, direction and support are checked in code; a relationship the quote does not state is weakened to
  "is mentioned with" or rejected, never drawn as stated. Keep First Nations, Métis and Inuit apart.
- **Enumeration** (`src/enumeration.py`) is a separate workflow for "list everything" questions. Do not widen the normal retriever for it, do not call `retrieve` from
  it, and do not merge differently-worded items (dialect, word) into the requested kind. State the scope and cost.

## Running

```bash
conda activate canada-rag      # or any env with requirements.txt installed
python scripts/build_embedding_map.py   # once; writes artifacts/
streamlit run app.py        # Discover Canada Atlas (technical demo)
streamlit run study_app.py  # Discover Canada Study (learner app)
pytest tests/ -q
```
