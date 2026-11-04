# Interview Q&A — Discover Canada Atlas

Practice answers for explaining the project. One paragraph each; say it in your own words. Where a number appears, it comes from this repo's docs or tests.

## 1. The big picture

**1. What is the project, in one breath?**
It is a study tutor for the official *Discover Canada* citizenship guide. The guide is split into 236 passages, embedded once, and stored in a local vector database. A learner's question is embedded, the five closest passages are retrieved, and a language model answers only from those passages, citing them. Around that core sit a learner app (chapter carousel, quizzes, concept maps, infographics) and a technical "Atlas" app for inspecting how retrieval behaves.

**2. Is it Python based? What runs on the backend?**
Yes, all Python (3.11). There is no separate backend service: Streamlit runs the Python code server-side, and that same process does retrieval, calls the language model, validates outputs and renders the page. The only persistent store is a local ChromaDB folder plus JSON caches on disk. I chose that deliberately, because for a single-document study tool a second API tier would add moving parts without adding capability.

**3. Why two apps (`app.py` and `study_app.py`)?**
They serve different audiences over one shared engine. `study_app.py` is the learner app that is deployed publicly. `app.py` is the technical Atlas, run locally, which shows the embedding map, nearest neighbours and experimental features like live voice. Keeping them separate lets me experiment freely without risking the deployed demo, and a test enforces that the learner app never imports the experimental voice code.

**4. Who is it for, and what problem does it solve?**
Newcomers studying for the Canadian citizenship test who want to ask questions in plain language and get answers grounded in the guide, not in a model's general memory. The second audience is me: it is a way to learn and demonstrate retrieval-augmented generation (RAG) end to end, including evaluation, rather than just calling a chat API.

**5. What does the request flow look like?**
The question is cleaned of document boilerplate, embedded with MiniLM, and compared by cosine similarity with the stored passages in ChromaDB. The top five passages are numbered and sent to the language model with an instruction to answer only from them and cite their numbers. The reply is checked, cited passages are shown as pills, and if nothing relevant was found the app abstains and offers suggestions instead of guessing. The call path is drawn in `docs/FLOWS.md`.

## 2. Technology choices

**6. Why Streamlit?**
Speed and fit. The project is Python-heavy (embeddings, vector search, validation), and Streamlit lets me build an interactive UI in the same language without a separate front-end codebase, API layer or build chain. For a demo and learning project that removes weeks of plumbing. The trade-off is limited control over layout and a rerun-on-interaction model, which I worked around with custom CSS and small custom components (the 3D carousel, flashcards, the infographic viewer).

**7. What are Streamlit's limits, and how did you deal with them?**
Every interaction reruns the script, so state must live in `st.session_state`, and heavy objects (the model, the Chroma client) must be cached. It also isn't built for real-time streams. For live voice I used `st.fragment` with polling plus a WebRTC component, and for the carousel I wrote a small JavaScript component. If this grew to many concurrent users, I would move the engine behind an API and use a proper front end; I know where that seam is because the engine modules do not import Streamlit.

**8. Why not FastAPI plus React?**
It would be the right choice for a product with many users, auth and a mobile app. Here it would double the surface area (two languages, an API contract, CORS, a build pipeline) for no benefit to the learner. The repo's rules explicitly forbid it for now, and the separation between `src/` (engine) and `ui/` (widgets) means the engine could be wrapped in FastAPI later without rewriting it.

**9. Why ChromaDB?**
It is an embedded vector database: no server to run, persisted to a folder, and fast enough for 236 vectors. I considered FAISS and pgvector. FAISS is a library without metadata handling; pgvector needs a database server. With a corpus this small, simplicity wins, and the Chroma folder deploys with the app. For millions of passages I would reconsider.

**10. Why the `all-MiniLM-L6-v2` embedding model?**
It is small (384 dimensions, runs on CPU), well understood, and good enough for English passage retrieval. That matters because the production server is a 2 GB droplet with no GPU. I did not pick it by guessing; the retrieval configuration was benchmarked on a held-out question set before the product was built, and then frozen.

**11. Which language model generates the answers, and why behind a wrapper?**
Answers come from a hosted model through an OpenAI-compatible API (Venice), called with the `openai` Python SDK. All calls go through one function in `src/generation.py`, so the rest of the code does not know which provider exists. That makes it easy to swap models, mock it in tests, and keep the API key out of every other module. The key is read from the environment at runtime and is never logged or committed.

**12. Why didn't you use LangChain or an agent framework?**
The pipeline is a handful of explicit steps (clean, embed, retrieve, generate, validate), and writing them directly makes each step testable and easy to explain. A framework would add abstractions I would then have to debug and justify. I can say what every line does, which matters more here than saving a few lines of glue.

**13. Why Plotly and UMAP?**
They are for the technical Atlas only. UMAP projects the 384-dimensional embeddings into 2-D so a person can see the guide's topics cluster; Plotly renders it interactively. I treat it strictly as a picture: retrieval and nearest-neighbour logic always use the original 384-D cosine similarity, never distance on the map, because 2-D projections distort distances.

**14. How is the app tested?**
With pytest, around 370 tests. Most are pure-logic tests with the language model mocked, so they run offline and fast. Streamlit's `AppTest` drives the UI pages, and there are integration-style checks such as one that runs the name fixer across the whole guide and requires zero changes to correct text. Real-model checks were done with scripts but are not part of the committed suite, because they need keys and downloads.

## 3. Retrieval and RAG design

**15. What is RAG and why use it here?**
Retrieval-augmented generation means retrieving relevant text first and making the model answer from it. The guide is a fixed, authoritative document, so I want answers that are traceable to it. A bare model might answer from memory, be outdated or invent details. With RAG every claim can point to a passage, and when no passage supports the question, the right behaviour is to say so.

**16. How did you chunk the guide?**
The guide is split into 236 overlapping windows of text, each tagged with page and section so citations can say where they came from. The chunking and its embeddings were computed once and frozen; at runtime only the learner's question is embedded. Keeping them fixed means the evaluation results stay valid for the product.

**17. Why k = 5?**
It was selected through evaluation, not by feel. Fewer passages risk missing the answer; more add noise and cost and make the model more likely to wander. Five gave the best trade-off in the benchmark. The setting is frozen on purpose: if I change it, the benchmark no longer describes what the product does.

**18. What is "Variant A"?**
It is the name of the winning retrieval configuration from the evaluation: plain dense retrieval with normalised embeddings and cosine similarity, no re-ranking and no keyword search. I tested fancier ideas (hybrid keyword search, rank fusion, re-rankers) and they did not justify their complexity on this corpus, so the simplest approach was kept.

**19. How did you evaluate retrieval?**
I built a set of questions with known answer passages, then measured whether the right passage appeared in the top-k for each variant. A held-out set, never used for tuning, gave the final numbers. After freezing, I do not re-run it, because repeatedly checking against the same set would overfit the product to it.

**20. What happens when the guide doesn't contain the answer?**
The app abstains. It says it could not find that in the guide rather than inventing something. Then it offers "did you mean…" suggestions. A language model proposes a guess at what the learner meant, and the app snaps it to the nearest question in a pre-built bank of 367 questions drawn from the guide, with a similarity floor, so tapping a suggestion always gets a grounded answer. That design cost two AI calls per abstention instead of six.

**21. How do you prevent hallucinations?**
Several layers. The prompt restricts the model to the supplied passages. Citations are numbered and checked against what was actually retrieved. For structured features (concept maps, timelines, lists), the model must return JSON with a verbatim quote, and code verifies that the quote really appears in the cited passage; unsupported items are dropped rather than repaired. And the app prefers abstaining to guessing.

**22. Why add a query-cleanup layer if retrieval is frozen?**
Questions like "What does Discover Canada say about Michif?" contain document boilerplate that pulls the embedding toward generic passages; the right passage ranked 78th and rose to 9th once cleaned. Cleanup only deletes boilerplate words. It never adds or swaps words, so it cannot invent a synonym. The retrieval function itself is unchanged, and the original question, not the cleaned one, goes to the model.

**23. Why did Michif still not appear, and what did you do?**
Even cleaned, rank 9 is outside the top five, so a question like "what languages does the guide mention?" cannot be answered by top-5 retrieval at all; the items are spread across many passages. So I built a separate enumeration workflow: it reads passages exhaustively (quick mode reads those containing the word stem; thorough mode reads all 236), extracts items with verbatim quotes, validates them, and reports how many passages and AI calls it used.

**24. What is the concept-map validation about?**
Concept maps are LLM-extracted graphs, and early versions produced plausible-looking but wrong links. Validation checks entity types, requires typed relationships with the right direction, requires the relationship's cue words to appear in the quoted text, and guards against mixing First Nations, Métis and Inuit. Anything unsupported is revised to a weaker "is mentioned with" or rejected. This came from real failures I observed, each with a regression test.

**25. Why keep the research Ask path pure Variant A?**
So the technical Atlas stays a faithful view of the benchmarked system. Product-layer improvements (cleanup, enumeration, validation) live in separate modules, with `docs/APPLICATION_LAYERS.md` stating which layer is benchmarked and which is not. That separation lets me be honest about which results apply to which behaviour.

## 4. Voice

**26. What does the voice feature do?**
A learner can speak a question and hear the answer. Milestone 1 is push-to-talk (record, transcribe, edit, send, hear the reply). Milestone 2, in the Atlas only, is hands-free: the app detects when you start and stop talking, sends the turn automatically, and shows listening, thinking and speaking states. Wake word and interruption are Milestone 3 and not built.

**27. How does speech recognition work?**
Locally, with faster-whisper (the `small` model, int8 on CPU). It runs on the same machine, so audio is not sent to a third party. I pass a hint list of Canadian names and terms (Papineau, Michif, Inuktitut) as an initial prompt, and a rule-based corrector fixes close sound-alikes, but only when the corrected word matches the guide's vocabulary; a guide-wide test requires it to change zero correct words.

**28. How does text-to-speech work?**
Kokoro (an open-source neural voice, 82M parameters) runs locally, with ONNX and Piper as alternatives. Before speaking, a spoken-form step rewrites written forms like "Queen Elizabeth II" to "Elizabeth the Second" and "1837–38" to "eighteen thirty-seven to thirty-eight". Pronunciation of names is guided with phoneme markup. I cannot hear it myself, so I validated by checking the phoneme output and by user listening tests.

**29. Why local voice models and not a cloud API?**
Privacy, cost and control: no audio leaves the machine and there is no per-minute fee. The price is latency and CPU load. I also tested browser-based TTS (WebGPU was fast, WASM was unusably slow), which is why voice stays out of the deployed 2 GB server for now.

**30. How does the hands-free mode know when you've finished talking?**
Silero VAD (a small neural voice-activity detector, run through ONNX) scores each 32 ms audio frame for speech. An endpointer applies hysteresis (separate start and stop thresholds), a minimum speech length, a pause allowance (0.8 s by default) and a lead-in buffer so the first syllable isn't clipped. In noise it kept correct turns up to a fairly high noise level, where a simple energy detector merged turns.

**31. What is half-duplex and why?**
While the tutor is thinking or speaking, the microphone gate is closed and anything heard is discarded. That prevents the tutor from hearing its own voice through the speakers and answering itself; in a playback test this produced zero false turns. The cost is you can't interrupt it, which is exactly what Milestone 3 (barge-in) would address with echo cancellation.

**32. Why streamlit-webrtc?**
Streamlit has no streaming-microphone API; its audio input only gives a finished recording. streamlit-webrtc streams audio frames to Python over WebRTC, where I resample to 16 kHz mono for the VAD. Its limits: it needs a live Streamlit session so tests use a stub, and on a remote server it would need HTTPS and a TURN server, which is why live voice is local-only.

**33. What was the hardest voice bug?**
Importing the Silero library silently set PyTorch to one thread for the whole process, which made Kokoro about five times slower: one 49-second reply took 38 seconds to generate. I found it by timing and fixed it by restoring the thread count after loading, with a test guarding it. It is a good example of a performance problem that no unit test of the feature itself would have found.

**34. How fast is the voice conversation?**
Not fast enough to feel natural. From the moment you stop talking to the reply starting takes roughly 6 to 20 seconds: recognition 1 to 2.5 s, the tutor 1 to 12 s, speech synthesis 2 to 8 s. The next improvement is to synthesise and play the reply sentence by sentence, so audio starts after the first sentence rather than after the whole reply. Spoken replies are also shortened by default, with the full text on screen.

## 5. Deployment and operations

**35. How is it deployed?**
The learner app runs on a 2 GB DigitalOcean droplet (Ubuntu 24.04) behind Caddy, which terminates HTTPS and proxies to Streamlit under a sub-path of my existing site. A systemd service keeps it running. The vector store and caches ship with the code, so the server needs no database service.

**36. Why Caddy?**
Automatic HTTPS certificates and a very small configuration. I had an existing site on the same droplet, so I added a path-scoped snippet that proxies only my app's prefix, leaving the rest untouched. It was a deliberate rule not to disturb the existing site.

**37. How does CI/CD work?**
A GitHub Actions workflow runs the test suite on every push and pull request, and deploys `main` to the droplet over SSH only if tests pass. Credentials live in GitHub secrets, and I set those up myself. This means a push to main is a deploy, so I only push when I mean to ship.

**38. How do you control cost and abuse on a public demo?**
A per-session budget limits the number of AI calls, voice is excluded from the public build, and results of expensive operations (concept maps, enumerations) are cached on disk keyed by model, prompt version and corpus size, so the same request isn't paid for twice. The API key lives only in the server's environment.

**39. How did you handle secrets?**
The `.env` file is never read, printed or committed by me or by the coding agent; code reads the key from the environment at runtime only, and error messages never include it. `.env.example` holds placeholders. In CI, secrets are GitHub secrets; on the server they sit in an environment file readable only by the service.

**40. What would break if this got a thousand users?**
The 2 GB server, because each session loads state and the embedding model is shared but CPU-bound, and generation calls would hit rate and cost limits. I would move the engine behind an API with a worker pool, cache embeddings of common questions, add authentication, and use a larger instance or serverless inference for the embedder. The module boundaries were chosen so that move is possible.

## 6. Challenges, trade-offs and reflection

**41. What was the biggest technical challenge?**
Making the answers trustworthy rather than merely fluent. Retrieval quality limits what the model can say, so I spent the effort on evaluation, abstention, quote verification and honest reporting of what each layer can and cannot do. Several features exist because a real question exposed a failure, such as Michif not appearing or "1982" dead-ending, and each got a fix plus a regression test.

**42. What did you decide not to build, and why?**
Hybrid search, re-ranking and fine-tuning, because the benchmark showed no gain worth the complexity; a separate API tier and front end; and wake word and barge-in voice. Saying no was deliberate: each would be additional surface to test and defend without improving the core promise, which is grounded answers from one document.

**43. What are the known limitations?**
Top-5 retrieval cannot answer list-everything questions without the enumeration workflow; answers depend on the hosted model's availability; voice has not been tested across microphones, accents and rooms, and is slow; speech-to-text can still mishear unusual names; and the guide is English with a fixed edition, so it can be out of date relative to the real test.

**44. How would you improve it next?**
Stream the voice reply sentence by sentence to cut latency, then add interruption with proper echo handling. For retrieval, I would grow the evaluation set before changing anything, so improvements are measured. For production, re-prewarm the concept-map caches after the prompt version bump, and add a small load test.

**45. What did you learn?**
That the model is the easy part: the quality comes from evaluation, validation and being honest about failure cases. I also learned to measure before optimising (the torch thread bug), to keep benchmarked components frozen and layer product features around them, and to write down architecture decisions as I go so I can explain them later.

**46. If you started over, what would you change?**
I would write the evaluation set earlier and larger, and set up the deploy pipeline sooner so I could test in a production-like environment. I would also separate the engine from Streamlit from day one, which I mostly did but would enforce with an import test.

**47. How do you know it works?**
Three kinds of evidence: the held-out retrieval benchmark for the search quality, the automated tests for behaviour and regressions, and hands-on challenge questions (RCMP, P.E.I., names, trap questions) in `docs/VOICE_CHALLENGE_QUESTIONS.md`. I am clear that the first is quantitative while the third is anecdotal.

**48. Is the answer quality measured, or only retrieval?**
Mostly retrieval, because that can be measured objectively against known passages. Answer quality is protected by structure (grounding, citations, abstention, quote checks) and spot-checked by hand, but I have not run a formal large-scale answer-quality evaluation, and I would say so if asked.

## 7. If they push on a specific word

**49. Embedding versus keyword search: when does each fail?**
Embeddings capture meaning but can miss exact names and rare terms (Michif ranked low for a generic question), while keyword search nails exact terms but misses paraphrases. I tested combining them and it did not beat plain dense retrieval on this corpus, so I handled the exact-term gap with the targeted enumeration workflow instead.

**50. Explain cosine similarity in one sentence.**
It measures the angle between two vectors, so two passages pointing the same direction in embedding space count as similar regardless of length; with normalised embeddings it reduces to a dot product.
