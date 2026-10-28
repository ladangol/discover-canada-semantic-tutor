# Voice tutor — evaluation plan and Milestone 1 instrumentation

**Nothing here is a result.** No WER, latency or accuracy figure is claimed until it has been measured on real recordings.
The frozen RAG held-out evaluation is **not** re-run.

## Measured now (Milestone 1)

Every spoken turn records *processing* time with `time.perf_counter` (`src/voice/models.py: VoiceTimings`):

| Stage | Where it is measured |
|---|---|
| Audio length | `audio.read_wav` (from the WAV header) |
| Speech to text | `asr.FasterWhisperTranscriber.transcribe` (includes consuming the lazy segment generator) |
| Retrieval | `tutor.handle_turn` around `retrieve` (questions only) |
| Generation / grading | `tutor.handle_turn` around `ask` / `grade_free` (0 for deterministic MCQ grading) |
| Tutor total | `tutor.handle_turn` |
| Text to speech | `tts.PiperSynthesizer.synthesize` |

Shown only with `STUDY_VOICE_DIAGNOSTICS=1` ("Developer diagnostics" under the latest reply), never in the learner UI.

**Processing time is not experienced time.** The learner's wait also includes recording, correcting the transcript and pressing Play,
none of which is included. Quote the two separately.

## Canadian proper nouns (development tool)

`scripts/voice_proper_nouns.py` scores names from the guide (Sir John A. Macdonald, Louis-Hippolyte La Fontaine, Laura Secord,
James FitzGibbon, Tecumseh, Nunavut, Iqaluit, Confederation, …) with `jiwer` (WER and CER on lower-cased, punctuation-free text):

* `--voice <piper .onnx>`: **synthetic** speech round-trip. A smoke check that the toolchain and the Whisper `initial_prompt`
  hint work; synthetic voices are cleaner than people, so do not report these numbers as accuracy.
* `--recordings DIR`: **real** recordings (`*.wav` + `references.tsv`). Only this mode produces reportable WER/CER.

## Automated tests (Milestone 1)

`tests/test_voice_*.py`, mocks only (no models, microphone or API key): valid/empty/corrupt/too-long/oversized audio; lazy-segment
transcription; busy host and slot release; ASR/TTS missing or failing; speech-text preparation; MCQ option matching ("B", "the
second one", the answer text); quiz answers graded by the existing graders (not sent to RAG); no tutor call without confirmed text;
once-only processing across Streamlit reruns; audio/text tied to the right turn; the Talk page is hidden by default and renders
without models.

## Later milestones (not implemented)

| Metric | Milestone |
|---|---|
| ASR WER/CER overall and on Canadian names; ASR latency vs model size | 4 (collect real recordings first) |
| End-to-end latency split (ASR, retrieval, LLM, TTS) at p50/p95 | 4 |
| Quiz-answer grading accuracy for spoken answers; grounded-answer quality | 4 |
| VAD endpoint latency | 2 |
| Wake-word misses and false activations; TTS echo not triggering the wake word | 3 |
| Interruption responsiveness (barge-in) | 3 |
