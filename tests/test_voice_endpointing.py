"""Turn-taking logic with scripted detectors: no microphone, no model."""
import numpy as np
import pytest

from src.voice.endpointing import EndpointConfig, Endpointer
from src.voice.vad import FRAME, FRAME_MS, EnergyVad


class Scripted:
    """A detector whose probability for frame k is probs[k]. Frames carry their index (frame i = i * 1e-3) so surviving audio can be identified."""
    name = "scripted"

    def __init__(self, probs):
        self.probs, self.k, self.resets = list(probs), 0, 0

    def probability(self, frame):
        p = self.probs[self.k] if self.k < len(self.probs) else 0.0
        self.k += 1
        return p

    def reset(self):
        self.resets += 1


class ContentVad:
    """Speech where the frame is loud (>0.5): lets the audio itself decide, so gates and chunking can be tested without a script."""
    name = "content"

    def __init__(self):
        self.resets = 0

    def probability(self, frame):
        return 0.95 if float(np.mean(frame)) > 0.5 else 0.0

    def reset(self):
        self.resets += 1


def indexed(n, start=0):
    return np.concatenate([np.full(FRAME, (start + i) * 1e-3, np.float32) for i in range(n)])


def speech(n):
    return np.full(n * FRAME, 0.9, np.float32)


def silence(n):
    return np.zeros(n * FRAME, np.float32)


def cat(*parts):
    return np.concatenate(parts)


def kinds(ev):
    return [e.kind for e in ev]


def scripted_run(probs, cfg):
    return Endpointer(Scripted(probs), cfg).feed(indexed(len(probs)))


# 32 ms frames: min speech 3, hangover 10, lead-in 3 (+2 spare), tail 2
CFG = EndpointConfig(min_speech_ms=96, hangover_ms=320, preroll_ms=96, tail_ms=64)


def test_a_turn_ends_after_the_hangover_and_returns_lead_in_speech_and_a_short_tail():
    ev = scripted_run([0.0] * 5 + [0.9] * 20 + [0.0] * 30, CFG)
    assert kinds(ev) == ["speech_start", "utterance"]
    # 8 lead-in frames kept (5 silent + the 3 that confirmed the start) + 17 more speech frames + a 2-frame tail
    assert len(ev[1].samples) == (8 + 17 + 2) * FRAME and ev[1].seconds == pytest.approx(27 * FRAME_MS / 1000)


def test_a_blip_shorter_than_the_minimum_never_starts_a_turn():
    assert scripted_run([0.0] * 3 + [0.9] * 2 + [0.0] * 30, CFG) == []                 # two speech-like frames (64 ms < 96 ms)


def test_a_dip_shorter_than_the_hangover_does_not_split_the_turn():
    ev = scripted_run([0.9] * 10 + [0.2] * 4 + [0.9] * 10 + [0.0] * 30, CFG)
    assert kinds(ev) == ["speech_start", "utterance"] and len(ev[1].samples) > 24 * FRAME


def test_a_pause_longer_than_the_hangover_makes_two_turns():
    ev = scripted_run([0.9] * 10 + [0.0] * 14 + [0.9] * 10 + [0.0] * 20, CFG)
    assert kinds(ev) == ["speech_start", "utterance", "speech_start", "utterance"]


def test_hysteresis_probabilities_between_the_two_thresholds_keep_the_turn_open():
    cfg = EndpointConfig(threshold=0.5, end_threshold=0.35, min_speech_ms=96, hangover_ms=320, preroll_ms=96, tail_ms=64)
    ev = scripted_run([0.9] * 6 + [0.4] * 40 + [0.0] * 20, cfg)                         # 0.4: below the start threshold, above the end threshold
    assert kinds(ev) == ["speech_start", "utterance"] and len(ev[1].samples) > 40 * FRAME


def test_the_lead_in_before_the_speech_is_kept():
    ev = scripted_run([0.0] * 8 + [0.9] * 12 + [0.0] * 20, CFG)
    assert ev[1].samples[0] == pytest.approx(3e-3)                                      # frame 3: five silent frames before the speech at frame 8


def test_the_tail_is_trimmed_to_a_short_silence_not_the_whole_hangover():
    ev = scripted_run([0.9] * 10 + [0.0] * 40, CFG)
    assert len(ev[1].samples) == (3 + 7 + 2) * FRAME


def test_a_speaker_who_never_stops_is_cut_at_the_maximum_length():
    cfg = EndpointConfig(min_speech_ms=96, hangover_ms=320, preroll_ms=96, max_utterance_s=1.0)
    ev = scripted_run([0.9] * 200, cfg)
    cut = next(e for e in ev if e.kind == "max_length")
    assert kinds(ev)[0] == "speech_start" and cut.seconds == pytest.approx(1.0, abs=0.1)


def test_long_silence_reports_an_idle_timeout_so_the_mic_can_be_stopped():
    ev = scripted_run([0.0] * 100, EndpointConfig(idle_timeout_s=1.0))
    assert kinds(ev) == ["idle_timeout"] * 3                                            # once per second of silence (3.2 s of audio)


def test_audio_is_accepted_in_any_chunk_size():
    probs = [0.0] * 3 + [0.9] * 15 + [0.0] * 30
    whole = scripted_run(probs, CFG)
    pieces, got, x = Endpointer(Scripted(probs), CFG), [], indexed(len(probs))
    for i in range(0, len(x), 301):                                                      # a chunk size that never lines up with 512
        got += pieces.feed(x[i:i + 301])
    assert kinds(got) == kinds(whole) and got[-1].samples.shape == whole[-1].samples.shape


# --- the gate: the tutor must never hear itself ------------------------------------------------------------------------------------------

def test_a_closed_gate_discards_audio_and_forgets_a_half_heard_turn():
    ep = Endpointer(ContentVad(), CFG)
    assert kinds(ep.feed(speech(6))) == ["speech_start"] and ep.state == "speech"
    ep.close_gate()
    assert ep.state == "idle" and ep.feed(speech(300)) == []                             # the tutor is speaking: nothing is heard or remembered
    ep.open_gate()
    assert ep.state == "idle" and ep._frames == [] and ep._carry.size == 0


def test_after_the_gate_reopens_a_new_turn_starts_cleanly_without_the_old_audio():
    ep = Endpointer(ContentVad(), CFG)
    ep.feed(speech(6))                                                                    # half a turn heard...
    ep.close_gate(), ep.open_gate()                                                       # ...then the tutor spoke
    ev = ep.feed(cat(speech(10), silence(20)))
    assert kinds(ev) == ["speech_start", "utterance"]
    assert len(ev[1].samples) <= (3 + 10 + 2) * FRAME                                     # only the new speech; nothing from before the gate


def test_the_detector_state_is_reset_after_every_turn_and_with_the_gate():
    vad = ContentVad()
    ep = Endpointer(vad, CFG)
    ep.feed(cat(speech(10), silence(20)))
    after_turn = vad.resets
    ep.close_gate()
    assert after_turn >= 1 and vad.resets > after_turn


def test_a_gate_that_never_reopens_yields_nothing_however_much_is_said():
    ep = Endpointer(ContentVad(), CFG)
    ep.close_gate()
    assert all(ep.feed(speech(50)) == [] for _ in range(10))


# --- the energy fallback -----------------------------------------------------------------------------------------------------------------

def test_the_energy_detector_separates_a_quiet_room_from_loud_sound():
    vad, rng = EnergyVad(), np.random.default_rng(0)
    quiet = [vad.probability((rng.standard_normal(FRAME) * 0.002).astype(np.float32)) for _ in range(60)]
    loud = [vad.probability((np.sin(np.arange(FRAME) * 0.3) * 0.3).astype(np.float32)) for _ in range(5)]
    assert max(quiet) < 0.5 and min(loud) > 0.9 and vad.probability(np.zeros(FRAME, np.float32)) < 0.1


def test_the_energy_detector_drives_the_endpointer_end_to_end():
    rng = np.random.default_rng(1)
    noise = lambda n: (rng.standard_normal(n * FRAME) * 0.002).astype(np.float32)
    voice = lambda n: (np.sin(np.arange(n * FRAME) * 0.3) * 0.3).astype(np.float32)
    ep = Endpointer(EnergyVad(), EndpointConfig())
    ev = ep.feed(noise(40)) + ep.feed(voice(30)) + ep.feed(noise(40))
    assert kinds(ev) == ["speech_start", "utterance"]
