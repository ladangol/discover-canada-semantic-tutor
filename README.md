# Discover Canada Atlas

An interactive study companion for the *Discover Canada* citizenship guide, built on a
frozen, already-evaluated retrieval index. Four connected ideas:

1. **Atlas** — every chunk plotted in 2-D (UMAP of the stored MiniLM embeddings), filterable,
   with a chunk inspector and its nearest semantic neighbours.
2. **Query view** — ask a question and see what the retriever sees: your query as a star,
   the top-5 retrieved chunks highlighted, with cosine scores.
3. **Study tutor** — Ask (grounded + cited), Explain, Quiz me (MCQ / free response),
   Flashcards.
4. **Semantic-neighbour study** — embedding neighbours supply MCQ distractors, diverse
   flashcard sources (MMR), and follow-up questions after a miss.

> UMAP positions are for **visualization only**. Retrieval, neighbours and diversity use cosine
> similarity on the original 384-d embeddings.

## Run

```bash
conda activate canada-rag               # or: pip install -r requirements.txt
cp .env.example .env                    # then add your own VENICE_API_KEY (optional)
python scripts/build_embedding_map.py   # one-off; writes artifacts/ (committed copy included)
streamlit run app.py
pytest tests -q                         # smoke tests, no API key needed
```

`.env` is yours and is never read by tooling in this repo except at runtime via
`python-dotenv`. Without `VENICE_API_KEY`, the atlas, query view, retrieval and neighbour
inspection all work; only generated answers, quizzes, flashcards and explanations are off.

## Frozen retrieval configuration

`sentence-transformers/all-MiniLM-L6-v2` · normalized embeddings · cosine · plain dense
retrieval · **k = 5** · Chroma collection `discover-canada-window-minilm` (236 chunks).
Nothing here re-chunks, re-embeds the corpus, or tunes retrieval.

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (components, modules, file tree) and
[docs/FLOWS.md](docs/FLOWS.md) (sequence diagrams for Ask, Quiz, and the atlas query flow).

## Decisions worth knowing

- **Chapter** = first breadcrumb line of a chunk (`Canada’s History`, …); chunks with a single
  heading use their own section (see `corpus.derive_chapter`). 11 chapters.
- **MCQ distractors come from embeddings**: wrong answers are drawn from passages ranked by cosine
  similarity to the source passage. The **difficulty slider** picks the band (hard = closest passages,
  easy = farthest). The LLM writes the question + wrong answers from those passages; code shuffles them; a second LLM call must find *exactly one* option supported by
  the source, otherwise the question is retried (max 3).
- **Citations** are rendered by code from `[S#]` labels, as in the reference implementation.
- **Neighbourhoods** keep one chunk per distinct nearby section, because windows of the same
  section overlap and are near-duplicates.
- **Session state only**: no accounts, database, or persistence. Closing the tab resets progress.
- File watching is disabled in `.streamlit/config.toml` (it crawls `transformers` and spams
  errors), so restart `streamlit run` after editing `src/`.
