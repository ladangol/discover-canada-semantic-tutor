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

## Running

```bash
conda activate canada-rag      # or any env with requirements.txt installed
python scripts/build_embedding_map.py   # once; writes artifacts/
streamlit run app.py
pytest tests/ -q
```
