"""Browser audio frames (48 kHz, stereo or mono, 20 ms) -> the live engine. Uses real PyAV frames and a fake detector."""
import av
import numpy as np
import pytest

from src.voice import live
from src.voice.endpointing import EndpointConfig
from ui import live_audio as la


class LevelVad:
    name = "level"

    def probability(self, frame):
        return 0.95 if float(np.sqrt(np.mean(frame ** 2))) > 0.05 else 0.0

    def reset(self):
        pass


def frame(samples, rate=48000, layout="stereo", fmt="s16"):
    """A real av.AudioFrame holding `samples` (mono float array) at `rate`, like a browser sends."""
    ch = 2 if layout == "stereo" else 1
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    packed = np.repeat(pcm[:, None], ch, axis=1).reshape(1, -1)
    f = av.AudioFrame.from_ndarray(packed, format=fmt, layout=layout)
    f.sample_rate = rate
    return f


def tone(sec, rate=48000, amp=0.3):
    return np.sin(np.arange(int(rate * sec)) * 2 * np.pi * 220 / rate) * amp


def test_a_48khz_stereo_frame_becomes_16khz_mono_with_the_same_loudness():
    rs, out = la.new_resampler(), []
    for i in range(10):                                           # ten 20 ms frames
        out.append(la.frame_to_pcm16k(frame(tone(0.02)[:960]), rs))
    x = np.concatenate(out)
    assert x.dtype == np.float32 and abs(len(x) - 3200) <= 320 and abs(np.max(np.abs(x)) - 0.3) < 0.05      # 200 ms at 16 kHz (the resampler buffers a little)
    assert abs(np.sqrt(np.mean(x ** 2)) - 0.3 / np.sqrt(2)) < 0.03


@pytest.mark.parametrize("rate,layout", [(48000, "mono"), (44100, "stereo"), (16000, "mono"), (8000, "mono")])
def test_other_sample_rates_and_layouts_are_normalised(rate, layout):
    rs = la.new_resampler()
    x = np.concatenate([la.frame_to_pcm16k(frame(tone(0.02, rate)[: int(rate * 0.02)], rate, layout), rs) for _ in range(25)])
    assert abs(len(x) - 8000) <= 640 and 0.15 < np.max(np.abs(x)) < 0.35


def test_a_streamed_turn_goes_through_the_processor_into_an_utterance():
    engine = live.LiveEngine(LevelVad(), EndpointConfig(min_speech_ms=96, hangover_ms=320, preroll_ms=96, tail_ms=64), margin_s=0.3)
    proc = la.LiveAudioProcessor(engine)
    quiet = np.zeros(960)
    for samples in [quiet] * 20 + [tone(0.02)[:960]] * 100 + [quiet] * 60:        # 0.4 s quiet, 2 s of voice, 1.2 s quiet
        assert proc.recv(frame(samples)) is not None
    u = engine.pop_utterance()
    assert u is not None and 1.8 < u.seconds < 2.8                                # about the 2 s that was spoken, plus lead-in and a short tail
    assert engine.pop_utterance() is None


def test_nothing_is_kept_when_nobody_speaks():
    engine = live.LiveEngine(LevelVad(), EndpointConfig(), margin_s=0.3)
    proc = la.LiveAudioProcessor(engine)
    for _ in range(200):
        proc.recv(frame(np.zeros(960)))
    assert engine.pop_utterance() is None and engine.status().state == live.LISTENING


def test_the_gate_keeps_the_tutors_own_voice_out():
    engine = live.LiveEngine(LevelVad(), EndpointConfig(min_speech_ms=96, hangover_ms=320, preroll_ms=96, tail_ms=64), margin_s=0.3)
    proc = la.LiveAudioProcessor(engine)
    engine.hold(), engine.release(speak_seconds=30.0)                             # the tutor is speaking
    for samples in [tone(0.02)[:960]] * 100 + [np.zeros(960)] * 60:               # ...and its voice comes back through the microphone
        proc.recv(frame(samples))
    assert engine.pop_utterance() is None and engine.status().state == live.SPEAKING
