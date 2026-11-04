"""Live (hands-free) engine: status, half-duplex gate, utterance queue, and utterance -> draft. Fake clock, scripted detector; no models."""
import threading

import numpy as np
import pytest

from src import config
from src.voice import audio, live, session as vs
from src.voice.endpointing import EndpointConfig
from src.voice.vad import FRAME
from tests.voice_helpers import FakeTranscriber

CFG = EndpointConfig(min_speech_ms=96, hangover_ms=320, preroll_ms=96, tail_ms=64)


class ContentVad:
    name = "content"

    def probability(self, frame):
        return 0.95 if float(np.mean(frame)) > 0.5 else 0.0

    def reset(self):
        pass


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def speech(n):
    return np.full(n * FRAME, 0.9, np.float32)


def silence(n):
    return np.zeros(n * FRAME, np.float32)


def engine(clock=None, margin=0.5, cfg=CFG):
    clock = clock or Clock()
    return live.LiveEngine(ContentVad(), cfg, margin_s=margin, clock=clock), clock


def say_one_turn(e):
    e.feed(speech(12)), e.feed(silence(15))


def test_one_spoken_turn_becomes_one_utterance():
    e, _ = engine()
    say_one_turn(e)
    u = e.pop_utterance()
    assert u is not None and u.reason == "utterance" and 0.3 < u.seconds < 1.2 and e.pop_utterance() is None


def test_status_follows_the_conversation():
    e, _ = engine()
    assert e.status().state == live.LISTENING
    e.feed(speech(6))
    assert e.status().state == live.HEARING
    e.feed(silence(15))
    assert e.status().state == live.LISTENING and e.status().queued == 1
    e.hold()
    assert e.status().state == live.THINKING


def test_while_the_tutor_is_busy_nothing_is_heard_and_nothing_is_remembered():
    e, _ = engine()
    e.hold()
    say_one_turn(e)                                                  # the learner (or the speaker's echo) makes noise while the tutor works
    assert e.pop_utterance() is None
    e.release(0.0)
    assert e.status().state == live.LISTENING and e.pop_utterance() is None


def test_the_microphone_stays_closed_while_the_tutor_speaks_and_for_a_margin_after():
    e, clock = engine(margin=0.5)
    e.hold()
    e.release(speak_seconds=2.0)
    assert e.status().state == live.SPEAKING
    say_one_turn(e)                                                  # the tutor's own voice coming back through the speakers
    assert e.pop_utterance() is None
    clock.t += 2.2
    assert e.status().state == live.SPEAKING                         # the audio has ended but the margin has not
    clock.t += 0.4
    assert e.status().state == live.LISTENING
    say_one_turn(e)
    assert e.pop_utterance() is not None                             # the learner can answer now


def test_a_silent_reply_reopens_the_microphone_at_once():
    e, _ = engine()
    e.hold(), e.release(0.0)
    assert e.status().state == live.LISTENING


def test_a_second_turn_can_follow_the_first_without_any_button():
    e, clock = engine(margin=0.3)
    say_one_turn(e)
    assert e.pop_utterance() is not None
    e.hold(), e.release(1.0)
    clock.t += 1.5
    say_one_turn(e)
    assert e.pop_utterance() is not None


def test_a_speaker_who_never_stops_is_cut_off_and_reported():
    e, _ = engine(cfg=EndpointConfig(min_speech_ms=96, hangover_ms=320, preroll_ms=96, max_utterance_s=0.5))
    e.feed(speech(40))
    u = e.pop_utterance()
    assert u is not None and u.reason == "max_length"


def test_long_silence_flags_the_idle_timeout_so_the_page_can_stop_listening():
    e, _ = engine(cfg=EndpointConfig(idle_timeout_s=1.0))
    e.feed(silence(40))
    assert e.status().idle_expired
    e.reset_idle()
    assert not e.status().idle_expired


def test_changing_the_pause_length_takes_effect_immediately():
    e, _ = engine()
    e.set_config(EndpointConfig(min_speech_ms=96, hangover_ms=1600, preroll_ms=96, tail_ms=64))      # a slower talker: a longer pause before the turn ends
    e.feed(speech(12)), e.feed(silence(20))                          # 640 ms of silence: not enough any more
    assert e.pop_utterance() is None
    e.feed(silence(40))
    assert e.pop_utterance() is not None


def test_the_audio_thread_and_the_page_thread_can_run_together():
    e, _ = engine()
    got, stop = [], threading.Event()

    def page():
        while not stop.is_set():
            e.status()
            if (u := e.pop_utterance()):
                got.append(u)
    t = threading.Thread(target=page)
    t.start()
    for _ in range(5):
        say_one_turn(e)
    stop.set(), t.join(timeout=5)
    got += [u for u in iter(e.pop_utterance, None)]
    assert len(got) == 5


# --- one utterance -> the draft ------------------------------------------------------------------------------------------------------

def utterance(seconds=1.0):
    x = (np.sin(np.arange(int(16000 * seconds)) * 0.2) * 0.3).astype(np.float32)
    return live.Utterance(x, seconds, "utterance", 0.0)


def test_an_utterance_becomes_a_draft_with_names_fixed_and_the_original_kept():
    s = vs.VoiceState()
    ok = live.ingest_utterance(s, utterance(), lambda: FakeTranscriber("Who was Louis-Joseph Papino?"))
    assert ok and s.heard == "Who was Louis-Joseph Papineau?" and s.raw_heard == "Who was Louis-Joseph Papino?" and s.fixes


def test_an_utterance_the_recogniser_could_not_read_is_not_sent():
    s = vs.VoiceState()
    assert live.ingest_utterance(s, utterance(), lambda: FakeTranscriber("   ")) is False and s.notice


def test_the_cut_out_audio_is_a_valid_16khz_wav():
    wav = audio.encode_wav(utterance(1.5).samples)
    x, info = audio.read_wav(wav)
    assert info.sample_rate == 16000 and info.channels == 1 and info.duration_seconds == pytest.approx(1.5, abs=0.01) and len(x) == 24000
    assert audio.wav_seconds(wav) == pytest.approx(1.5, abs=0.01) and audio.wav_seconds(b"junk") == 0.0


def test_live_settings_are_clamped_and_private_by_default(monkeypatch):
    monkeypatch.setenv("VOICE_LIVE_HANGOVER_MS", "99999"), monkeypatch.setenv("VOICE_LIVE_THRESHOLD", "nope"), monkeypatch.delenv("VOICE_LIVE_STUN", raising=False)
    s = config.live_settings()
    assert s["hangover_ms"] == 2500 and s["threshold"] == 0.5 and s["stun"] is False        # no public STUN server unless asked
    assert s["vad"] == "silero"
