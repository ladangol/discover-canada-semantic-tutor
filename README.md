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

## Voice tutor (optional, Milestone 1)

The learner app has a **Talk** page: press the microphone, speak a question or a quiz answer, check and correct the transcript,
and the same tutor code as the text UI answers (with citations) and reads the reply aloud. Speech recognition (faster-whisper) and
speech output (Piper) run **locally on the machine that runs Streamlit**. Only the text you confirm goes to Venice; audio never does
and is never saved. Off by default: set `STUDY_VOICE=1` to show it, in the learner app (**Talk** tab) and in the technical Atlas (**Voice** tab, plain widgets, timings always shown). Design and diagrams: [docs/VOICE_ARCHITECTURE.md](docs/VOICE_ARCHITECTURE.md).

```bash
pip install -r requirements-voice.txt                       # faster-whisper, kokoro, soundfile, piper-tts (+ jiwer for the dev tool)
# Speech recognition: the Whisper model downloads itself on first use (ASR_MODEL=small ≈ 470 MB; "base" ≈ 140 MB is lighter).
# Speech output (default): Kokoro-82M downloads itself on first use (~330 MB model + ~0.5 MB per voice; needs network once).
#   KOKORO_VOICE=af_heart (default) — af_*/am_* American, bf_*/bm_* British.
# Alternative engine: TTS_ENGINE=piper with a Piper voice (.onnx + .onnx.json) from https://huggingface.co/rhasspy/piper-voices
#   e.g. en/en_US/lessac/medium/ (~60 MB) and TTS_VOICE_PATH=/absolute/path/to/en_US-lessac-medium.onnx
export STUDY_VOICE=1
streamlit run study_app.py                                  # learner app: the Talk tab
streamlit run app.py                                        # technical Atlas: the Voice tab
```

Optional environment (see `.env.example`): `ASR_MODEL`, `ASR_DEVICE`, `ASR_COMPUTE_TYPE`, `ASR_LANGUAGE`, `ASR_INITIAL_PROMPT`,
`TTS_ENGINE`, `KOKORO_VOICE`, `KOKORO_SPEED`, Piper-only `TTS_LENGTH_SCALE` / `TTS_NOISE_SCALE` / `TTS_NOISE_W`, `TTS_SENTENCE_PAUSE_MS`, `VOICE_MAX_SECONDS` (30), `VOICE_MAX_BYTES`, `VOICE_MAX_CONCURRENCY` (1), `STUDY_VOICE_DIAGNOSTICS=1` (per-stage timings).
**Licences:** Kokoro-82M is Apache-2.0. `piper-tts` is GPL-3.0 and every Piper voice has its own licence: read the voice's model card before
publishing a deployment. Models are downloaded, never part of this repository. If a model is missing the page says so and you can still type; nothing crashes.

The production droplet (2 GB) does not run voice; it needs roughly ≥ 4 GB RAM / 2 vCPU. Measure with `scripts/voice_proper_nouns.py`
(see [docs/VOICE_EVALUATION.md](docs/VOICE_EVALUATION.md)); no WER figures are claimed until real recordings are collected.

## Application layers (around the frozen core)

Retrieval is frozen. Three product layers sit around it, each removable without changing a retrieval result ([docs/APPLICATION_LAYERS.md](docs/APPLICATION_LAYERS.md)):
**query cleanup** (deletes "in Discover Canada"-style boilerplate from the search string only; `QUERY_CLEANUP=0` turns it off), **concept validation** (entity types, direction,
what the quote states, First Nations / Métis / Inuit kept apart) and **"scan the whole guide"** for list questions (opt-in `STUDY_ENUMERATION=1`, or
`python scripts/scan_guide.py languages [--mode thorough] [--focus "First Nations"]`).

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
