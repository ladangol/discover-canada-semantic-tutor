# Discover Canada Atlas (+ Discover Canada Study)

An interactive study companion for the *Discover Canada* citizenship guide, built on a
frozen, already-evaluated retrieval index. Four connected ideas:

1. **Atlas** — every chunk plotted in 2-D (Plotly) or 3-D (three.js) from a UMAP of the stored MiniLM embeddings, filterable,
   with a chunk inspector and its nearest semantic neighbours.
2. **Query view** — ask a question and see what the retriever sees: your query as a star,
   the top-5 retrieved chunks highlighted, with cosine scores.
3. **Study tutor** — Ask (grounded + cited), Explain, Quiz me (MCQ / free response),
   Flashcards.
4. **Analyze** — a chapter semantic map (centroid cosine heatmap/network, semantic breadth; no LLM),
   source-grounded **concept/person hubs** and **timelines** (retrieve → LLM JSON → validate → render).
5. **Semantic-neighbour study** — embedding neighbours supply MCQ distractors, diverse
   flashcard sources (MMR), and follow-up questions after a miss.

> UMAP positions are for **visualization only**. Retrieval, neighbours and diversity use cosine
> similarity on the original 384-d embeddings.

## Two apps, one engine

| Command | Experience |
|---|---|
| `streamlit run app.py` | **Discover Canada Atlas**: explainable RAG demo (atlas, retrieval inspection, analysis, tutor). |
| `streamlit run study_app.py` | **Discover Canada Study**: simple citizenship-study app (Learn, Practice, Flashcards, Concept map, Timeline). |

The learner app has a bold, animated, responsive design (phone, tablet, desktop) with light and dark modes; use the ⋮ menu → Settings to switch theme (default: follow your system).

Concept hubs and timelines render as editorial infographics (interactive in the browser) and export as
**SVG** or **PNG** in landscape 16:9, square or portrait, for blog posts. Facts always come from validated
source text; the renderer only draws them.

## Run

```bash
conda activate canada-rag               # or: pip install -r requirements.txt
cp .env.example .env                    # then add your own VENICE_API_KEY (optional)
python scripts/build_embedding_map.py   # one-off; writes artifacts/ (committed copy included)
streamlit run app.py          # technical demo
streamlit run study_app.py    # learner app
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

## Analyze tab notes

- Hubs and timelines are generated from retrieved source text only; every node/event must carry a
  verbatim supporting quote, which is checked against the cited chunk. Failures are omitted and listed.
- Generated objects are cached in `artifacts/generated/` (git-ignored); tick *Rebuild* to regenerate.
  A whole-chapter timeline (e.g. Canada’s History) takes ~1.5 minutes the first time.
- Validation is mechanical (sources, quotes, years). It cannot prove a free-text description adds no
  paraphrase beyond the source, and recall depends on the model: it can miss events.

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
- **3D view** needs internet (three.js is loaded from jsDelivr) and the 3-D artifacts that
  `build_embedding_map.py` writes; without them the 2D/3D toggle is hidden.
- Run `streamlit run app.py` from the repo root so `.streamlit/config.toml` is picked up. (`app.py` also
  silences the watcher's harmless `torchvision` warnings, which otherwise flood the log.)
- Locally, Streamlit reloads `src/` and `ui/` when you save a file. If something odd happens right after a
  code change (for example a `TypeError` about an unexpected argument), restart `streamlit run`: a long-running
  server keeps old modules in memory. Production runs with the watcher off, so restart the service after updates.
