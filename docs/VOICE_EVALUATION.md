# Voice tutor — evaluation and reliability (Milestone 4)

Milestone 4 adds **no features**. It measures the voice experience that exists today (Milestones 1 and 2) and builds a harness that can
measure Milestone 3 (wake word, barge-in) the day it exists. Generated on 2026-11-03 by `python -m scripts.voice_eval.summarize --write-report`
from the result files in `artifacts/voice_eval/`: every number below comes from a results file, none is typed by hand.
The frozen ChromaDB, chunk corpus, MiniLM embeddings and Variant A top-5 retrieval were used as they are; the held-out benchmark was **not** run or tuned against.

## 1. What was and was not measured

| Area | Status | Honest scope |
|---|---|---|
| End-to-end latency (VAD endpoint, ASR, retrieval, LLM, TTS) | **Measured**, warm (24 turns) and cold (3 fresh processes) | One machine, Python pipeline only. Browser microphone/WebRTC and audio start-up are **not** in the numbers. |
| Wake-word detection latency | **Not performed** | Milestone 3 is not built. |
| ASR accuracy (WER, CER, names, retrieval impact) | **Measured on synthetic speech only** | Text-to-speech voices are cleaner and more regular than people. Treat every figure as a **lower bound** on the real error rate. No human recordings were available. |
| Wake-word reliability, false activations, tutor-speech activations | **Not performed** | Milestone 3 is not built. Protocol recorded in `run_wake_barge.py`. |
| Barge-in (time to interrupt, stop-and-listen, state after interruption) | **Not performed** | Milestone 3 is not built. |
| Quiet vs moderately noisy conditions | **ASR only**, clean and 15 dB pink noise | Pink noise is not a real room, a fan or other talkers. |
| Conversational task success | **Measured**, scripted, one synthetic voice | 17 attempted steps; a small, hand-written set, not a user study. |
| RAG grounding of the spoken answer | **Measured**, automatic checks | No human grading and no LLM judge; "supported" is a proxy (section 8). |
| Several browsers, several machines, human listening tests | **Not performed** | Only this laptop; no listening panel. |

## 2. Method

**Audio.** All speech is synthetic: four macOS `say` voices with different accents (Samantha US, Daniel UK, Karen AU, Rishi IN) plus the app's own Kokoro voice
`af_heart`. For latency and tasks the audio is a macOS voice (never Kokoro, which is the tutor's own voice) with 0.6 s of silence before and 2.5 s after, played into the
real `LiveEngine` (Silero VAD + endpointer) in 32 ms frames exactly as the microphone thread would. For ASR accuracy each question is also run with pink noise added at 15 dB SNR.

**Questions.** `scripts/voice_eval/questions.json`: 32 spoken questions about Discover Canada, 15 of them in the latency set. They include Michif, Inuktitut, Tecumseh,
Champlain, Louis-Joseph Papineau, Sir Wilfrid Laurier and Confederation, five questions the guide does not cover (the tutor must abstain), and four whose coverage by the guide was not checked in advance (used for ASR only).

**Pipeline under test.** Silero VAD → endpointer → faster-whisper (`small`, int8, CPU) with the app's vocabulary hint → `names.correct_names` → `tutor.handle_turn` (frozen retrieval,
real LLM) → Kokoro (PyTorch). Nothing in the app was changed to take these measurements; the harness reads the existing timers (`VoiceTimings`) and the sample clock.

**Definitions.**
* *Latency* = from the last sample of the learner's speech to the first sample of the tutor's reply. The app does not stream speech yet, so "time to first audio" equals synthesising the whole (≤350-character) reply.
* *Endpoint* is measured on the **audio sample clock**: how long after speech stops the VAD declares the turn over (this is the configured 0.8 s pause, by design, plus up to one frame).
* *Real-time factor* (seconds of compute per second of audio) is reported **separately** and is not a latency.
* *Cold* = a fresh Python process (models loaded from disk, files most likely in the OS cache). A reboot or never-opened-model cold disk start was **not** measured.
* *Percentiles.* With n this small p90 and p95 are interpolated between a handful of the slowest runs: use them as a rough guide, not a guarantee.
* *WER/CER* by `jiwer`, after lower-casing, folding accents and removing punctuation. A *name* counts as heard only if it appears in the transcript ignoring case, accents, spaces and hyphens (strict: "Jack Cartier" for "Jacques Cartier" is an error).
* *Meaning-affecting* has no human judge here. Two objective proxies: the name is wrong, **and** the frozen top-5 retrieval for the heard text no longer contains the passage the correctly spelled question retrieved first ("top-1 kept").

**Reproduce.**

```bash
python -m scripts.voice_eval.run_latency --warm 24 && python -m scripts.voice_eval.run_latency --cold 3
python -m scripts.voice_eval.run_asr                      # synthetic, clean + 15 dB noise (add --model medium --snr "" to compare sizes)
python -m scripts.voice_eval.run_asr --timing small,medium
python -m scripts.voice_eval.run_tasks && python -m scripts.voice_eval.run_grounding
python -m scripts.voice_eval.run_wake_barge               # records the protocol; NOT PERFORMED until Milestone 3
python -m scripts.voice_eval.summarize --write-report     # regenerates this file
python -m scripts.voice_eval.run_asr --recordings DIR     # REAL speech: DIR/*.wav + DIR/references.tsv ("file<TAB>reference text")
```

The task and grounding runs call the LLM (a few dozen calls each) and need `VENICE_API_KEY`; the others use no key. Run the latency commands with nothing else heavy running.

## 3. Hardware and software environment

```json
{
 "machine": "Apple M4 Pro",
 "memory_gb": 24,
 "os": "macOS-27.0.1-arm64-arm-64bit",
 "python": "3.11.16",
 "packages": {
  "faster-whisper": "1.2.1",
  "torch": "2.13.0",
  "kokoro": "0.9.4",
  "silero-vad": "6.2.3",
  "onnxruntime": "1.29.1",
  "sentence-transformers": "6.1.0",
  "chromadb": "1.5.9",
  "streamlit": "1.65.0",
  "streamlit-webrtc": "0.79.0",
  "jiwer": "4.0.0"
 },
 "asr": {
  "model": "small",
  "device": "cpu",
  "compute_type": "int8",
  "language": "en"
 },
 "tts_engine": "kokoro",
 "tts_voice": "af_heart",
 "llm_model": "z-ai-glm-5-3-flash",
 "vad": "silero",
 "hangover_ms": 800,
 "browser": "not used: the harness drives the Python pipeline directly; browser microphone/WebRTC path is NOT measured"
}
```

## 4. End-to-end latency

```mermaid
sequenceDiagram
    autonumber
    actor L as Learner (synthetic voice)
    participant V as Silero VAD + Endpointer
    participant A as Whisper + name fixer
    participant T as tutor.handle_turn
    participant R as Frozen retrieval k=5
    participant G as LLM
    participant S as Kokoro TTS
    L->>V: speech, then silence
    Note over L,V: t0 = last sample of speech
    V-->>A: utterance emitted after the pause
    Note over V,A: endpoint = sample clock
    A-->>T: text
    Note over A,T: asr = inference time
    T->>R: question
    R-->>T: 5 passages
    Note over T,R: retrieval
    T->>G: question + passages
    G-->>T: answer
    Note over T,G: generation; tutor total
    T-->>S: spoken reply
    S-->>L: first audio
    Note over T,S: tts = whole reply = time to first audio today
    Note over L,S: total = endpoint + asr + tutor + tts. Wake word (M3), browser transport and playback start are NOT measured.
```

**Warm** (24/24 turns; synthetic macOS `say` voices (US, UK, AU, IN accents), clean, 0.6 s lead + 2.5 s tail silence). Seconds.

| Stage | n | median | p90 | p95 | max |
|---|---|---|---|---|---|
| VAD endpoint decision (audio time) | 24 | 0.85 | 0.87 | 0.88 | 0.90 |
| Speech to text (Whisper) | 24 | 1.22 | 1.46 | 1.61 | 1.69 |
| Retrieval (frozen top-5) | 24 | 0.10 | 0.23 | 0.26 | 0.32 |
| LLM generation | 24 | 1.68 | 6.25 | 6.95 | 7.48 |
| Tutor total (retrieval + LLM + glue) | 24 | 1.84 | 6.49 | 7.19 | 7.55 |
| TTS, whole reply (= time to first audio today) | 24 | 4.19 | 5.04 | 5.13 | 5.18 |
| **End of speech → first audio of the reply** | 24 | 8.31 | 11.75 | 13.55 | 14.22 |
| (hypothetical) TTS of the first sentence only | 24 | 1.13 | 2.18 | 2.89 | 3.12 |
| (hypothetical) total if speech were streamed per sentence | 24 | 5.62 | 10.73 | 11.45 | 12.40 |

Not latencies (kept apart on purpose): ASR real-time factor median 0.49× (p95 0.69×); TTS real-time factor median 0.18× (p95 0.21×); VAD cost 0.12 ms per 32 ms frame.

**Cold** (3/3 fresh processes). A cold start = a fresh Python process (models loaded from disk, files likely in the OS page cache). A reboot/never-opened-model cold disk start was NOT measured. Seconds.

| Measure | n | median | p90 | p95 |
|---|---|---|---|---|
| Load VAD | 3 | 0.05 | 0.05 | 0.05 |
| Load Whisper | 3 | 0.94 | 0.97 | 0.97 |
| Load corpus + search model | 3 | 1.88 | 1.95 | 1.96 |
| Load Kokoro | 3 | 1.79 | 2.11 | 2.14 |
| All model loads | 3 | 4.65 | 5.04 | 5.09 |
| First turn after loading (end of speech → first audio) | 3 | 8.04 | 8.06 | 8.06 |
| Process start → first reply (loads + first turn) | 3 | 12.54 | 13.07 | 13.13 |


**Reading the numbers.**
* The learner waits a median of **8.31 s** (p90 11.75 s, p95 13.55 s; range 4.77–14.22 s over 24 warm turns) between finishing a sentence and hearing the first word of the reply. That is slow for a conversation.
* Speech synthesis is the largest fixed cost (median 4.19 s) because the whole reply is generated before playback starts, then the pause (0.85 s, by design), then Whisper (1.22 s). The LLM is usually quick (median 1.68 s) but has a long tail (p95 6.95 s), and the tail is what moves p90/p95 of the total.
* **TTS speed is not conversational latency.** The real-time factor is 0.18× (compute seconds per second of audio), which sounds fast, yet a 350-character reply still costs 4.19 s before the first sound because nothing is played until it is finished.
* Streaming the reply sentence by sentence **would** bring the median to about 5.62 s (p95 11.45 s) by starting playback after the first sentence (median 1.13 s to synthesise). That figure is a **projection from measured first-sentence synthesis, not a measurement of a streaming implementation.**
* Cold start: loading all models takes a median 4.65 s; the first turn afterwards took 8.04 s, about the same as a warm turn, so the penalty of a cold start is the load time (12.54 s from process start to first reply), not slower inference. The warm set contained 0 abstentions, so the abstain path (which adds an LLM call) is under-represented in the latency numbers.
* The VAD itself is negligible: 0.12 ms per 32 ms frame.

## 5. Speech recognition accuracy

Whisper `small` (the app's model), identical audio through three pipelines: **raw** (no hint), **hint** (the app's vocabulary hint) and **hint+fix** (the hint plus the rule-based name corrector — what the app does).

**asr_synthetic_small** — model `small`, 320 clips per pipeline. Audio: SYNTHETIC speech (macOS `say` voices + the app's Kokoro voice); pink noise where an SNR is given. Not people.

| Pipeline | Audio | clips | WER | CER | exact transcripts | names wrong | clips with a name wrong | top-1 passage kept | name wrong **and** top-1 lost |
|---|---|---|---|---|---|---|---|---|---|
| raw | clean | 160 | 10.1% | 4.0% | 62% | 57/135 | 53/120 | 86% | 22 |
| hint | clean | 160 | 4.2% | 1.7% | 80% | 24/135 | 24/120 | 98% | 3 |
| hint+fix | clean | 160 | 3.7% | 1.6% | 83% | 19/135 | 19/120 | 98% | 3 |
| raw | SNR 15 dB | 160 | 12.8% | 5.5% | 54% | 69/135 | 62/120 | 85% | 22 |
| hint | SNR 15 dB | 160 | 5.9% | 2.5% | 78% | 28/135 | 28/120 | 97% | 5 |
| hint+fix | SNR 15 dB | 160 | 5.4% | 2.4% | 81% | 23/135 | 23/120 | 97% | 5 |

Per name (misses/mentions, clean + noisy together):

| Name | raw | hint | hint+fix |
|---|---|---|---|
| Confederation | 0/20 | 0/20 | 0/20 |
| Haida Gwaii | 7/10 | 0/10 | 0/10 |
| Inuktitut | 14/20 | 0/20 | 0/20 |
| Jacques Cartier | 6/10 | 8/10 | 8/10 |
| Laura Secord | 8/10 | 0/10 | 0/10 |
| Louis Riel | 9/10 | 0/10 | 0/10 |
| Louis-Hippolyte La Fontaine | 5/10 | 1/10 | 1/10 |
| Louis-Joseph Papineau | 4/10 | 0/10 | 0/10 |
| Mi'kmaq | 9/10 | 1/10 | 1/10 |
| Michif | 20/20 | 13/20 | 13/20 |
| Métis | 7/10 | 2/10 | 2/10 |
| Nunavut | 8/20 | 0/20 | 0/20 |
| Prince Edward Island | 0/20 | 0/20 | 0/20 |
| RCMP | 0/10 | 0/10 | 0/10 |
| Samuel de Champlain | 4/10 | 6/10 | 6/10 |
| Saskatchewan | 0/10 | 0/10 | 0/10 |
| Sir George-Étienne Cartier | 8/10 | 9/10 | 9/10 |
| Sir Isaac Brock | 2/10 | 2/10 | 2/10 |
| Sir John A. Macdonald | 2/10 | 0/10 | 0/10 |
| Sir Wilfrid Laurier | 10/10 | 10/10 | 0/10 |
| Tecumseh | 3/10 | 0/10 | 0/10 |
| Terry Fox | 0/10 | 0/10 | 0/10 |

Per voice (clean, hint+fix): daniel 3.6%, karen 3.1%, kokoro 4.1%, rishi 3.6%, samantha 4.1%

Representative remaining errors (hint+fix):

| Said | Heard | Voice | SNR | name wrong | top-1 kept |
|---|---|---|---|---|---|
| What is Michif? | What is Métis? | samantha | clean | Michif | NO |
| What is Michif? | What is Métis? | samantha | 15.0 dB | Michif | NO |
| What is Michif? | What is Métis? | daniel | clean | Michif | NO |
| What is Michif? | What is Métis? | daniel | 15.0 dB | Michif | NO |
| What is Michif? | What is Métis? | karen | 15.0 dB | Michif | NO |
| What does the guide say about the Mi'kmaq? | What does the guide say about the McNac? | samantha | 15.0 dB | Mi'kmaq | NO |
| Who are the Métis? | Who are the mages? | karen | clean | Métis | NO |
| Who are the Métis? | Who are the majors? | karen | 15.0 dB | Métis | NO |
| Who was Samuel de Champlain? | Who was Samuel D. Champlain? | samantha | clean | Samuel de Champlain | yes |
| Who was Samuel de Champlain? | Who was Samuel D. Champlain? | samantha | 15.0 dB | Samuel de Champlain | yes |
| Who was Samuel de Champlain? | Who was Samuel D. Champlain? | karen | clean | Samuel de Champlain | yes |
| Who was Samuel de Champlain? | Who is Samuel D. Champlain? | karen | 15.0 dB | Samuel de Champlain | yes |
| Who was Samuel de Champlain? | Who was Samuel D. Champlain? | rishi | clean | Samuel de Champlain | yes |
| Who was Samuel de Champlain? | Who was Samuel D. Champlain? | rishi | 15.0 dB | Samuel de Champlain | yes |

**asr_synthetic_medium** — model `medium`, 160 clips per pipeline. Audio: SYNTHETIC speech (macOS `say` voices + the app's Kokoro voice); pink noise where an SNR is given. Not people.

| Pipeline | Audio | clips | WER | CER | exact transcripts | names wrong | clips with a name wrong | top-1 passage kept | name wrong **and** top-1 lost |
|---|---|---|---|---|---|---|---|---|---|
| raw | clean | 160 | 8.1% | 2.9% | 70% | 43/135 | 42/120 | 92% | 13 |
| hint | clean | 160 | 2.2% | 0.8% | 91% | 13/135 | 13/120 | 99% | 1 |
| hint+fix | clean | 160 | 1.8% | 0.7% | 92% | 10/135 | 10/120 | 99% | 1 |

Per name (misses/mentions, clean only):

| Name | raw | hint | hint+fix |
|---|---|---|---|
| Confederation | 0/10 | 0/10 | 0/10 |
| Haida Gwaii | 3/5 | 0/5 | 0/5 |
| Inuktitut | 2/10 | 0/10 | 0/10 |
| Jacques Cartier | 1/5 | 1/5 | 1/5 |
| Laura Secord | 1/5 | 0/5 | 0/5 |
| Louis Riel | 3/5 | 0/5 | 0/5 |
| Louis-Hippolyte La Fontaine | 3/5 | 0/5 | 0/5 |
| Louis-Joseph Papineau | 1/5 | 0/5 | 0/5 |
| Mi'kmaq | 5/5 | 0/5 | 0/5 |
| Michif | 7/10 | 0/10 | 0/10 |
| Métis | 3/5 | 1/5 | 1/5 |
| Nunavut | 0/10 | 0/10 | 0/10 |
| Prince Edward Island | 0/10 | 0/10 | 0/10 |
| RCMP | 0/5 | 0/5 | 0/5 |
| Samuel de Champlain | 2/5 | 3/5 | 3/5 |
| Saskatchewan | 0/5 | 0/5 | 0/5 |
| Sir George-Étienne Cartier | 5/5 | 4/5 | 4/5 |
| Sir Isaac Brock | 1/5 | 1/5 | 1/5 |
| Sir John A. Macdonald | 2/5 | 0/5 | 0/5 |
| Sir Wilfrid Laurier | 3/5 | 3/5 | 0/5 |
| Tecumseh | 1/5 | 0/5 | 0/5 |
| Terry Fox | 0/5 | 0/5 | 0/5 |

Per voice (clean, hint+fix): daniel 1.0%, karen 1.5%, kokoro 1.0%, rishi 2.1%, samantha 3.1%

Representative remaining errors (hint+fix):

| Said | Heard | Voice | SNR | name wrong | top-1 kept |
|---|---|---|---|---|---|
| Who are the Métis? | Who are the mages? | karen | clean | Métis | NO |
| Who was Samuel de Champlain? | Who was Samuel D. Champlain? | samantha | clean | Samuel de Champlain | yes |
| Who was Samuel de Champlain? | Who was Samuel D. Champlain? | karen | clean | Samuel de Champlain | yes |
| Who was Samuel de Champlain? | Who was Samuel D. Champlain? | rishi | clean | Samuel de Champlain | yes |
| What did Sir George-Étienne Cartier do for Confederation? | What did Sir George E.G.N. Cartier do for Confederation? | samantha | clean | Sir George-Étienne Cartier | yes |
| What did Sir George-Étienne Cartier do for Confederation? | What did Sir George Ichien Cartier do for Confederation? | daniel | clean | Sir George-Étienne Cartier | yes |
| What did Sir George-Étienne Cartier do for Confederation? | What did Sir George Chien Cartier do for Confederation? | karen | clean | Sir George-Étienne Cartier | yes |
| What did Sir George-Étienne Cartier do for Confederation? | What did Sir George Ichi and Cartier do for Confederation? | rishi | clean | Sir George-Étienne Cartier | yes |
| Who was Jacques Cartier? | Who was Jax Cartier? | rishi | clean | Jacques Cartier | yes |
| Who was Sir Isaac Brock? | Who was Isaac Brock? | daniel | clean | Sir Isaac Brock | yes |
| Where was the North West Mounted Police headquartered? | Where was the Northwest mounted police headquartered? | kokoro | clean | – | yes |
| Where was the North West Mounted Police headquartered? | Where was the Northwest Mounted Police headquartered? | samantha | clean | – | yes |


**Reading the numbers.**
* The vocabulary hint is what matters: on clean speech it cuts WER from 10.1% to 4.2% and name errors from 57/135 to 24/135; with 15 dB noise from 12.8% to 5.9%.
* The name corrector adds a little on top (WER 3.7% clean, 5.4% noisy; names wrong 19/135 clean). Its clearest effect is Sir Wilfrid Laurier (10 of 10 wrong without it, none with it); for every other name it changes little.
* What actually hurts is smaller than the name-error count suggests. The strict spelling check counts "Samuel D. Champlain" or "Jack Cartier" as wrong, yet the frozen top-5 retrieval still finds the same passage. The real damage is where a name is wrong **and** the correct passage is lost: 3/160 clips with the hint (22/160 without it) on clean speech and 5/160 with noise. The recurring cases are **Michif** (heard as "Métis"), **Mi'kmaq** and, with some accents, **Métis**.
* Passage retrieval kept the right top-1 passage for 98% of clean clips with the full pipeline, versus 86% with raw Whisper.
* Larger model (`medium`), clean speech only: WER 2.2% with the hint, 1.8% with hint+fix (names wrong 10/135, top-1 kept 99%); median transcription time on the same 12 clips, run alone: `small` 1.13 s, `medium` 3.52 s. On clean speech `medium` also fixes Michif (hint+fix: `small` 6/10 wrong, `medium` 0/10 wrong); Mi'kmaq was already right on clean speech with both (`small` 0/5 wrong, `medium` 0/5 wrong; the `small` model's Mi'kmaq miss appeared with noise). The price is about 2.39 s more on every turn. Whether that gain justifies the slower turn is a decision for the owner; this milestone does not change the model. The `medium` run used clean speech only (no noise comparison).
* **Caveat that bounds all of this:** the speakers are synthetic. Real accents, hesitations, speakers far from the microphone and room echo will produce more errors than these tables show.

## 6. Wake word and barge-in reliability

**Not performed.** Milestone 3 (wake word "Hey Moose" and barge-in) is not built, so wake-word detection rate, missed wake words, false activations in ordinary conversation, false activations from the tutor's own voice, time to interrupt TTS, whether the tutor stops and listens, and dialogue state after an interruption have **no measurements** in this report. `run_wake_barge.py` records the protocol (quiet and moderately noisy conditions, speakers plus microphone for the echo path) and writes `performed: false` so a later report cannot quote a number that was never taken.

What is established today, from Milestone 2: while the tutor is thinking or speaking the microphone gate is closed and everything heard is discarded, and playing the tutor's own reply back into the microphone produced no turn. That is a single check of the half-duplex gate with synthetic playback, not a reliability measurement.

## 7. Conversational task success

synthetic macOS `say` voice (samantha), clean. 21 steps: attempted 17 → passed 16, failed 1; not supported by the product 2; not performed (needs Milestone 3) 2.

| Task | Said | Heard | Result | Component / detail |
|---|---|---|---|---|
| ask a question, get a grounded spoken answer | What does RCMP stand for? | What does RCMP stand for? | pass |   |
| ask a question with a name | What did Chief Tecumseh do in the War of 1812? | What did Chief Tecumseh do in the War of 1812? | pass |   |
| ask a question with a name | Who was Samuel de Champlain? | Who was Samuel D. Champlain? | fail | asr name misheard: ['Samuel de Champlain'] |
| ask something the guide does not cover (must abstain) | Who was Louis-Joseph Papineau? | Who was Louis-Joseph Papineau? | pass |   |
| ask something the guide does not cover (must abstain) | Who is the current prime minister of Canada? | Who is the current Prime Minister of Canada? | pass |   |
| quiz answer by letter | Option A. | Option A. | pass |   |
| quiz answer by ordinal | The second one. | The second one. | pass |   |
| quiz answer by answer text | Anglophones | Anglophones. | pass |   |
| quiz answer by letter | Option B. | Option B. | pass |   |
| quiz answer by ordinal | The third one. | The third one. | pass |   |
| quiz answer by answer text | The Books of Remembrance | The Books of Remembrance | pass |   |
| quiz answer by letter | Option C. | Option C. | pass |   |
| quiz answer by ordinal | The fourth one. | The fourth one. | pass |   |
| quiz answer by answer text | Francophones | Francophones. | pass |   |
| quiz: mumbled non-answer is not graded | Hmm, I don't know. | I don't know. | pass |   |
| correct the transcript before sending | What did Laura Secord do in 1813? | What did Laura Secord do in 1813? | pass |   |
| request a quiz by voice | Quiz me about the RCMP. | Quiz me about the RCMP. | unsupported | intent (voice commands are not implemented; the quiz starts from a button) treated as an ordinary question → answered |
| ask for clarification of the previous answer | Can you explain that again in simpler words? | Can you explain that again in simpler words? | unsupported | dialogue memory (each turn is answered on its own; 'that' has no referent) outcome: answered with: Happy to help! I'm just not sure which part you'd like me to simplify, since I don't have  |
| interrupt an explanation | (speak over the tutor) |  | not performed | barge-in Milestone 3 (barge-in) is not built; there is nothing to interrupt yet |
| continue studying after an interruption | (resume the same topic) |  | not performed | barge-in Milestone 3 (barge-in) is not built; there is nothing to interrupt yet |
| dialogue state stays consistent over the whole session | 17 turns |  | pass |  ids increase, no draft left over |

Failures by component: asr × 1

**Reading the table.** 16 of 17 steps the product supports passed; 1 failed. Two requests are not supported by design and are reported as such rather than counted as passes: asking for a quiz by voice ("Quiz me about the RCMP" is treated as an ordinary question; the quiz starts from a button) and asking to clarify the previous answer (every turn is answered on its own, so "that" has no referent; the tutor asked which part to simplify rather than inventing one). The two interruption tasks need Milestone 3. The single failure is an ASR spelling ("Samuel D. Champlain") that did not change the passage retrieved. Dialogue state (turn ids, drafts, original ASR text kept separately from the learner's edit) stayed consistent over the 17-turn session. With one synthetic voice and about twenty steps this is a smoke test of the whole path, not an estimate of real-world task success.

## 8. RAG grounding and answer quality

synthetic macOS `say` voice (Samantha), clean. 28 questions (0 not captured).

| Check | Result |
|---|---|
| Guide covers it → answered (not abstained) | 23/23 |
| Guide does not cover it → abstained | 5/5 |
| Answer contains the expected facts | 23/23 |
| Citations accurate (shown sources ⊆ top-5; every written page is a shown source) | 23/23 |
| Every year/number and name in the answer appears in the retrieved passages | 21/23 |
| ASR text differed from the question | 4 |
| …and that changed answer/abstain | 0 |
| …and that changed the cited sources | 0 |
| Shown text vs text sent to the voice (WER) | n=23, median 0.0%, p95 0.0% |
| Round trip: Whisper hearing Kokoro vs words given (WER, numbers ignored) | n=23, median 1.7%, p95 9.8% |

Representative failures:

| Question | Heard | Problem |
|---|---|---|
| Who was Louis Riel? | Who was Louis Riel? | not in evidence: ['Hudson', 'Today'] |
| Who was Jacques Cartier? | Who was Choc Cartier? | not in evidence: ['Cartiers'] |

**Reading the table.** Out of 28 spoken questions, the guide-covered ones were answered (23/23), the five out-of-guide traps all abstained (5/5), answers contained the expected facts (23/23) and every citation shown was among the retrieved passages with every written page number matching a shown source (23/23). The numbers-and-names proxy passed 21/23; the two flags were checker artifacts on reading ("Hudson's" against "Hudsons", "Cartiers" as a plural), not invented facts. "Supported" is a proxy: it catches a made-up date or name, **not** a wrong paraphrase, so a human read of a sample of answers is still needed.
ASR changed the text on 4 questions and changed the answer/abstain decision on 0 and the cited sources on 0 of them (with the vocabulary hint on). The spoken text equalled the shown text in every case (the voice is given the same words, minus markup and citations), and Whisper hearing Kokoro read the reply got a median 1.7% word error (p95 9.8%) with numbers ignored, so what is spoken is what is shown.

A harness bug found and fixed during this milestone is worth recording: the grounding script switched the vocabulary hint off on the shared transcriber for its round-trip check and did not switch it back, so every later question was recognised without the hint. That version showed names such as "Louis Riel" and "Métis" mis-heard and three guide questions wrongly abstained — which is exactly the cost of running without the hint, and agrees with the ASR table above. The reported numbers are from the corrected run. LLM output at temperature 0 is also not perfectly repeatable: two otherwise identical grounding runs differed in whether one question was answered.

## 9. Representative failures

| Component | What happened | Evidence |
|---|---|---|
| ASR | **Michif** is heard as "Métis" most of the time, even with the hint; the question then retrieves a different passage. | Section 5, per-name table and failure list. |
| ASR | "Samuel de Champlain" is written "Samuel D. Champlain"; "Sir George-Étienne Cartier" is garbled. Retrieval still finds the right passage, so it is cosmetic. | Sections 5 and 7. |
| ASR | Mi'kmaq → "McNac"; Métis → "mages"/"majors" with some accents and noise: the question is lost. | Section 5. |
| Name fixer | Fixes Laurier completely but cannot recover words Whisper turns into other real words (Métis → "mages"). | Section 5. |
| Product | A spoken "quiz me" or "explain that again" is not understood as a command. | Section 7. |
| Latency | Median 8.31 s from finishing speaking to hearing the reply; the whole reply is synthesised before playback. | Section 4. |
| LLM | Occasional slow answer (p95 6.95 s) and small run-to-run differences at temperature 0. | Sections 4 and 8. |

## 10. Known limitations

* **No human speech, no real room, one machine, one browser path.** All audio is synthetic; the browser microphone (WebRTC, echo cancellation, device quirks) was not exercised; only an Apple M4 Pro laptop was used. Deployed-server hardware (2 GB droplet, no GPU) will be slower.
* Wake word, barge-in and everything that depends on them: not measured (Milestone 3).
* Small samples: 24 warm turns, 3 cold starts, 21 scripted task steps, 28 grounding questions. Percentiles beyond the median are rough.
* The "meaning-affecting" classification is an objective proxy (name wrong and top-1 passage lost), not a human judgement of meaning.
* The support check is a proxy; it does not grade the quality of a paraphrase. No LLM judge or human grading was used.
* Latency excludes browser transport, playback start and the learner's own time; the quoted cold start excludes a cold disk.
* Synthetic noise is pink noise at one level (15 dB SNR); no babble, music or reverberation.

## 11. What the measurements suggest (not done in this milestone)

1. **Stream the spoken reply sentence by sentence.** The largest single lever on perceived latency; projected, not yet measured as an implementation.
2. Add "Michif", "Mi'kmaq" and similar to the name fixer's sound-alike rules, or treat low-confidence key terms by asking the learner to confirm — only after checking them against real recordings.
3. Collect real recordings (`--recordings`) from several people and rooms before drawing any accuracy conclusion.
4. Re-run this harness after Milestone 3 and after any streaming change; the harness is written so the same commands produce the comparison.

## 12. Earlier instrumentation (Milestones 1–2)

Every spoken turn records processing time with `time.perf_counter` (`src/voice/models.py: VoiceTimings`): audio length, speech-to-text, retrieval, generation/grading, tutor total and text-to-speech.
Shown with `STUDY_VOICE_DIAGNOSTICS=1`. Processing time is not experienced time: it excludes recording, correcting the transcript and pressing Play.
`scripts/voice_proper_nouns.py` (Milestone 1) is the earlier single-purpose name check; `scripts/voice_eval/run_asr.py` supersedes it.
The `tests/test_voice_*.py` suite uses mocks only; `tests/test_voice_eval_helpers.py` covers the scoring helpers of this harness.
