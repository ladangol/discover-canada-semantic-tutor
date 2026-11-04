# Voice tutor — evaluation and reliability (Milestone 4)

Milestone 4 adds **no features**. It measures the voice experience that exists today (Milestones 1 and 2) and builds a harness that can
measure Milestone 3 (wake word, barge-in) the day it exists. Generated on [[generated_on]] by `python -m scripts.voice_eval.summarize --write-report`
from the result files in `artifacts/voice_eval/`: every number below comes from a results file, none is typed by hand.
The frozen ChromaDB, chunk corpus, MiniLM embeddings and Variant A top-5 retrieval were used as they are; the held-out benchmark was **not** run or tuned against.

## 1. What was and was not measured

| Area | Status | Honest scope |
|---|---|---|
| End-to-end latency (VAD endpoint, ASR, retrieval, LLM, TTS) | **Measured**, warm ([[warm_n]] turns) and cold ([[cold_n]] fresh processes) | One machine, Python pipeline only. Browser microphone/WebRTC and audio start-up are **not** in the numbers. |
| Wake-word detection latency | **Not performed** | Milestone 3 is not built. |
| ASR accuracy (WER, CER, names, retrieval impact) | **Measured on synthetic speech only** | Text-to-speech voices are cleaner and more regular than people. Treat every figure as a **lower bound** on the real error rate. No human recordings were available. |
| Wake-word reliability, false activations, tutor-speech activations | **Not performed** | Milestone 3 is not built. Protocol recorded in `run_wake_barge.py`. |
| Barge-in (time to interrupt, stop-and-listen, state after interruption) | **Not performed** | Milestone 3 is not built. |
| Quiet vs moderately noisy conditions | **ASR only**, clean and 15 dB pink noise | Pink noise is not a real room, a fan or other talkers. |
| Conversational task success | **Measured**, scripted, one synthetic voice | [[tasks_attempted]] attempted steps; a small, hand-written set, not a user study. |
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

{{ENV}}

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

{{LATENCY}}

**Reading the numbers.**
* The learner waits a median of **[[warm_total_med]] s** (p90 [[warm_total_p90]] s, p95 [[warm_total_p95]] s; range [[warm_total_min]]–[[warm_total_max]] s over [[warm_n]] warm turns) between finishing a sentence and hearing the first word of the reply. That is slow for a conversation.
* Speech synthesis is the largest fixed cost (median [[warm_tts_med]] s) because the whole reply is generated before playback starts, then the pause ([[warm_endpoint_med]] s, by design), then Whisper ([[warm_asr_med]] s). The LLM is usually quick (median [[warm_gen_med]] s) but has a long tail (p95 [[warm_gen_p95]] s), and the tail is what moves p90/p95 of the total.
* **TTS speed is not conversational latency.** The real-time factor is [[tts_rtf_med]]× (compute seconds per second of audio), which sounds fast, yet a 350-character reply still costs [[warm_tts_med]] s before the first sound because nothing is played until it is finished.
* Streaming the reply sentence by sentence **would** bring the median to about [[warm_streamed_med]] s (p95 [[warm_streamed_p95]] s) by starting playback after the first sentence (median [[warm_first_med]] s to synthesise). That figure is a **projection from measured first-sentence synthesis, not a measurement of a streaming implementation.**
* Cold start: loading all models takes a median [[cold_load_med]] s; the first turn afterwards took [[cold_turn_med]] s, about the same as a warm turn, so the penalty of a cold start is the load time ([[cold_all_med]] s from process start to first reply), not slower inference. The warm set contained [[warm_abstained]] abstentions, so the abstain path (which adds an LLM call) is under-represented in the latency numbers.
* The VAD itself is negligible: [[vad_ms_med]] ms per 32 ms frame.

## 5. Speech recognition accuracy

Whisper `small` (the app's model), identical audio through three pipelines: **raw** (no hint), **hint** (the app's vocabulary hint) and **hint+fix** (the hint plus the rule-based name corrector — what the app does).

{{ASR}}

**Reading the numbers.**
* The vocabulary hint is what matters: on clean speech it cuts WER from [[small_raw_wer_clean]] to [[small_hint_wer_clean]] and name errors from [[small_raw_names_clean]] to [[small_hint_names_clean]]; with 15 dB noise from [[small_raw_wer_noisy]] to [[small_hint_wer_noisy]].
* The name corrector adds a little on top (WER [[small_fix_wer_clean]] clean, [[small_fix_wer_noisy]] noisy; names wrong [[small_fix_names_clean]] clean). Its clearest effect is Sir Wilfrid Laurier (10 of 10 wrong without it, none with it); for every other name it changes little.
* What actually hurts is smaller than the name-error count suggests. The strict spelling check counts "Samuel D. Champlain" or "Jack Cartier" as wrong, yet the frozen top-5 retrieval still finds the same passage. The real damage is where a name is wrong **and** the correct passage is lost: [[small_hint_lost_clean]] clips with the hint ([[small_raw_lost_clean]] without it) on clean speech and [[small_fix_lost_noisy]] with noise. The recurring cases are **Michif** (heard as "Métis"), **Mi'kmaq** and, with some accents, **Métis**.
* Passage retrieval kept the right top-1 passage for [[small_fix_top1_clean]] of clean clips with the full pipeline, versus [[small_raw_top1_clean]] with raw Whisper.
* Larger model (`medium`), clean speech only: WER [[medium_hint_wer_clean]] with the hint, [[medium_fix_wer_clean]] with hint+fix (names wrong [[medium_fix_names_clean]], top-1 kept [[medium_fix_top1_clean]]); median transcription time on the same 12 clips, run alone: `small` [[small_asr_time]] s, `medium` [[medium_asr_time]] s. On clean speech `medium` also fixes Michif (hint+fix: `small` [[small_michif]], `medium` [[medium_michif]]); Mi'kmaq was already right on clean speech with both (`small` [[small_mikmaq]], `medium` [[medium_mikmaq]]; the `small` model's Mi'kmaq miss appeared with noise). The price is about [[medium_extra]] s more on every turn. Whether that gain justifies the slower turn is a decision for the owner; this milestone does not change the model. The `medium` run used clean speech only (no noise comparison).
* **Caveat that bounds all of this:** the speakers are synthetic. Real accents, hesitations, speakers far from the microphone and room echo will produce more errors than these tables show.

## 6. Wake word and barge-in reliability

**Not performed.** Milestone 3 (wake word "Hey Moose" and barge-in) is not built, so wake-word detection rate, missed wake words, false activations in ordinary conversation, false activations from the tutor's own voice, time to interrupt TTS, whether the tutor stops and listens, and dialogue state after an interruption have **no measurements** in this report. `run_wake_barge.py` records the protocol (quiet and moderately noisy conditions, speakers plus microphone for the echo path) and writes `performed: false` so a later report cannot quote a number that was never taken.

What is established today, from Milestone 2: while the tutor is thinking or speaking the microphone gate is closed and everything heard is discarded, and playing the tutor's own reply back into the microphone produced no turn. That is a single check of the half-duplex gate with synthetic playback, not a reliability measurement.

## 7. Conversational task success

{{TASKS}}

**Reading the table.** [[tasks_passed]] of [[tasks_attempted]] steps the product supports passed; [[tasks_failed]] failed. Two requests are not supported by design and are reported as such rather than counted as passes: asking for a quiz by voice ("Quiz me about the RCMP" is treated as an ordinary question; the quiz starts from a button) and asking to clarify the previous answer (every turn is answered on its own, so "that" has no referent; the tutor asked which part to simplify rather than inventing one). The two interruption tasks need Milestone 3. The single failure is an ASR spelling ("Samuel D. Champlain") that did not change the passage retrieved. Dialogue state (turn ids, drafts, original ASR text kept separately from the learner's edit) stayed consistent over the 17-turn session. With one synthetic voice and about twenty steps this is a smoke test of the whole path, not an estimate of real-world task success.

## 8. RAG grounding and answer quality

{{GROUNDING}}

**Reading the table.** Out of [[g_questions]] spoken questions, the guide-covered ones were answered ([[g_answered]]), the five out-of-guide traps all abstained ([[g_abstained]]), answers contained the expected facts ([[g_expected]]) and every citation shown was among the retrieved passages with every written page number matching a shown source ([[g_citations]]). The numbers-and-names proxy passed [[g_support]]; the two flags were checker artifacts on reading ("Hudson's" against "Hudsons", "Cartiers" as a plural), not invented facts. "Supported" is a proxy: it catches a made-up date or name, **not** a wrong paraphrase, so a human read of a sample of answers is still needed.
ASR changed the text on [[g_asr_differs]] questions and changed the answer/abstain decision on [[g_asr_decision]] and the cited sources on [[g_asr_sources]] of them (with the vocabulary hint on). The spoken text equalled the shown text in every case (the voice is given the same words, minus markup and citations), and Whisper hearing Kokoro read the reply got a median [[g_roundtrip_med]] word error (p95 [[g_roundtrip_p95]]) with numbers ignored, so what is spoken is what is shown.

A harness bug found and fixed during this milestone is worth recording: the grounding script switched the vocabulary hint off on the shared transcriber for its round-trip check and did not switch it back, so every later question was recognised without the hint. That version showed names such as "Louis Riel" and "Métis" mis-heard and three guide questions wrongly abstained — which is exactly the cost of running without the hint, and agrees with the ASR table above. The reported numbers are from the corrected run. LLM output at temperature 0 is also not perfectly repeatable: two otherwise identical grounding runs differed in whether one question was answered.

## 9. Representative failures

| Component | What happened | Evidence |
|---|---|---|
| ASR | **Michif** is heard as "Métis" most of the time, even with the hint; the question then retrieves a different passage. | Section 5, per-name table and failure list. |
| ASR | "Samuel de Champlain" is written "Samuel D. Champlain"; "Sir George-Étienne Cartier" is garbled. Retrieval still finds the right passage, so it is cosmetic. | Sections 5 and 7. |
| ASR | Mi'kmaq → "McNac"; Métis → "mages"/"majors" with some accents and noise: the question is lost. | Section 5. |
| Name fixer | Fixes Laurier completely but cannot recover words Whisper turns into other real words (Métis → "mages"). | Section 5. |
| Product | A spoken "quiz me" or "explain that again" is not understood as a command. | Section 7. |
| Latency | Median [[warm_total_med]] s from finishing speaking to hearing the reply; the whole reply is synthesised before playback. | Section 4. |
| LLM | Occasional slow answer (p95 [[warm_gen_p95]] s) and small run-to-run differences at temperature 0. | Sections 4 and 8. |

## 10. Known limitations

* **No human speech, no real room, one machine, one browser path.** All audio is synthetic; the browser microphone (WebRTC, echo cancellation, device quirks) was not exercised; only an Apple M4 Pro laptop was used. Deployed-server hardware (2 GB droplet, no GPU) will be slower.
* Wake word, barge-in and everything that depends on them: not measured (Milestone 3).
* Small samples: [[warm_n]] warm turns, [[cold_n]] cold starts, 21 scripted task steps, 28 grounding questions. Percentiles beyond the median are rough.
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
