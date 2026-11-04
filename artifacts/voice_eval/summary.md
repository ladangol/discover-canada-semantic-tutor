## Environment

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

## Latency

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

## ASR

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

## Tasks

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

## Grounding

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