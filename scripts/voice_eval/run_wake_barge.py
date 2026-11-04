"""Wake word and barge-in reliability (Milestone 4, section 3): NOT PERFORMED, because Milestone 3 is not built.

  python -m scripts.voice_eval.run_wake_barge

This records the protocol so the measurement can be run, unchanged, the day the features exist. It measures nothing and says so in the result
file, so the report cannot quote a number that was never taken. Protocol (quiet room and moderate noise, ~15 dB SNR pink noise or a real room):

  wake word        detection rate  = detections / N spoken "Hey Moose" (varied voices);  missed = N - detections
                   false activation, ordinary conversation = triggers per hour of speech that does not contain the phrase
                   false activation, tutor's own speech    = triggers while the tutor's replies play through the speakers
  barge-in         time to stop    = from the first sample of the learner's speech over TTS to the last sample of tutor audio
                   stops + listens = tutor audio stops AND the learner's utterance is transcribed (not lost)
                   state           = after the interruption the turn list, the open quiz question and the draft are consistent
Audio: needs a real room with speakers and a microphone (echo path); synthetic playback is only a smoke test.
"""
from scripts.voice_eval import common as C

if __name__ == "__main__":
    p = C.save("wake_barge", {"performed": False, "reason": "Milestone 3 (wake word and barge-in) is not implemented, so there is nothing to measure.",
                              "metrics_defined": ["wake-word detection rate", "false activations in conversation", "false activations from the tutor's own speech",
                                                  "missed wake words", "time to interrupt TTS", "tutor stops and listens", "dialogue state after interruption"]})
    print("NOT PERFORMED: Milestone 3 is not built. Protocol recorded ->", p)
