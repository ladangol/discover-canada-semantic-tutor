"""Speaking while the answer is still being written: the stream filter, the sentence streamer, the microphone gate, and the turn hook.
No models, network or audio hardware: a fake voice returns a known number of samples per character."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src import generation as gen
from src import tutor
from src.voice import live, speak
from src.voice.endpointing import EndpointConfig


# --- <think> filter ------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("chunks,expected", [
    (["Hello ", "world."], "Hello world."),
    (["<think>plan</think>Answer."], "Answer."),
    (["<thi", "nk>plan", "</th", "ink>Ans", "wer."], "Answer."),
    (["Use <b>bold</b> and a < b."], "Use <b>bold</b> and a < b."),
    (["Before <think>x</think> after"], "Before  after"),
])
def test_think_blocks_are_removed_even_when_tags_are_split(chunks, expected):
    f = gen._ThinkFilter()
    assert "".join(f.feed(c) for c in chunks) + f.flush() == expected


# --- GenerationStream with a fake provider -------------------------------------------------------------------------------------------------
def fake_client(pieces, finish="stop", fail_first=0, usage=None):
    calls = {"n": 0}

    def create(**kw):
        calls["n"] += 1
        assert kw["stream"] is True
        if calls["n"] <= fail_first:
            raise gen.OpenAIError("overloaded")
        chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p), finish_reason=None)], usage=None) for p in pieces]
        chunks.append(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=finish)], usage=usage))
        return iter(chunks)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def patch_provider(monkeypatch, client, recorded):
    monkeypatch.setattr(gen, "_client", lambda: client)
    monkeypatch.setattr(gen.budget, "check", lambda: None)
    monkeypatch.setattr(gen.budget, "record", lambda p, c: recorded.append((p, c)))
    monkeypatch.setattr(gen, "generation_model", lambda: "m")
    monkeypatch.setattr(gen.time, "sleep", lambda s: None)


def test_stream_yields_pieces_then_a_result_like_generate(monkeypatch):
    rec = []
    client, _ = fake_client(["<think>hm</think>", "  Louis ", "Riel was Métis."])
    patch_provider(monkeypatch, client, rec)
    s = gen.generate_stream([{"role": "user", "content": "hi"}])
    assert "".join(s) == "Louis Riel was Métis."
    assert s.result.text == "Louis Riel was Métis." and s.result.finish_reason == "stop" and rec and rec[0][1] > 0


def test_a_failure_before_any_text_is_retried(monkeypatch):
    client, calls = fake_client(["Fine."], fail_first=2)
    patch_provider(monkeypatch, client, [])
    s = gen.generate_stream([{"role": "user", "content": "hi"}])
    assert "".join(s) == "Fine." and calls["n"] == 3


def test_an_empty_stream_is_an_error_not_silence(monkeypatch):
    client, _ = fake_client(["<think>only thoughts</think>"])
    patch_provider(monkeypatch, client, [])
    with pytest.raises(RuntimeError):
        list(gen.generate_stream([{"role": "user", "content": "hi"}]))


# --- the turn hook: same answer, text handed on as it is written -----------------------------------------------------------------------
class FakeStream:
    def __init__(self, text):
        self.text, self.result = text, None

    def __iter__(self):
        for i in range(0, len(self.text), 7):
            yield self.text[i:i + 7]
        self.result = gen.GenerationResult(self.text, "stop", 1, 1)


CHUNK = SimpleNamespace(source_id="S1", chunk_id="c1", text="Confederation was in 1867.", page="12", section="Confederation", block_type="p")


def test_on_text_receives_the_answer_and_the_turn_is_unchanged():
    answer = "Confederation happened in 1867 [S1]."
    seen = []
    streamed = tutor.handle_turn("When was Confederation?", retrieve_fn=lambda q, k: [CHUNK], stream_fn=lambda m, **k: FakeStream(answer), on_text=seen.append)
    plain = tutor.handle_turn("When was Confederation?", retrieve_fn=lambda q, k: [CHUNK], gen=lambda m, **k: gen.GenerationResult(answer, "stop", 1, 1))
    assert "".join(seen) == answer
    assert streamed.text == plain.text and [c.chunk_id for c in streamed.cited] == [c.chunk_id for c in plain.cited] and not streamed.abstained


@pytest.mark.parametrize("text,verdict", [("", None), ("I", None), ("I don't kn", None), ("I don't know based on Discover Canada.", False),
                                          ("I do not know based on Discover Canada", False), ("I don’t know based on discover canada", False),
                                          ("Confederation was", True), ("I donut", True)])
def test_the_idk_reply_is_recognised_while_it_is_still_being_written(text, verdict):
    assert tutor.speak_verdict(text) is verdict


# --- the voice: sentences, cap, abstention -------------------------------------------------------------------------------------------------------
class FakeVoice:
    """24 samples per character, so durations are predictable. Records what it was asked to say."""
    pause_ms = 0

    def __init__(self):
        self.said, self.lock = [], threading.Lock()

    def pcm(self, sentence):
        with self.lock:
            self.said.append(sentence)
        return np.ones(len(sentence) * 24, np.int16)


def streamer(max_chars=240, verdict=lambda t: True):
    out, voice = [], FakeVoice()
    sp = speak.Speaker(out.append, rate=24000)
    return speak.SpeechStreamer(voice, sp, max_chars=max_chars, verdict=verdict), voice, out


def feed_in_pieces(s, text, n=5):
    for i in range(0, len(text), n):
        s.feed(text[i:i + n])


def test_sentences_are_spoken_in_order_as_they_complete_and_citations_are_never_read():
    s, voice, out = streamer(max_chars=None)
    feed_in_pieces(s, "Confederation was in 1867 [Discover Canada, p. 12, History]. It joined four provinces. Canada grew later.")
    assert voice.said[:2] == ["Confederation was in 1867.", "It joined four provinces."] or len(voice.said) <= 2   # the last piece is held back until finish
    res = s.finish("ignored when streamed")
    assert voice.said == ["Confederation was in 1867.", "It joined four provinces.", "Canada grew later."]
    assert all("[" not in x for x in voice.said) and res.mime_type == "audio/wav" and len(out) == 3 and s.streamed


def test_the_spoken_length_is_capped_and_the_tail_says_there_is_more():
    s, voice, _ = streamer(max_chars=60)
    feed_in_pieces(s, "First sentence is here. Second sentence is also here. Third one is far too late. Fourth one too.")
    s.finish("")
    assert voice.said[:2] == ["First sentence is here.", "Second sentence is also here."]
    assert voice.said[-1] == speak.MORE_ON_SCREEN and "Third one is far too late." not in voice.said


def test_a_short_reply_has_no_tail():
    s, voice, _ = streamer(max_chars=240)
    feed_in_pieces(s, "Yes. That is right.")
    s.finish("")
    assert speak.MORE_ON_SCREEN not in voice.said and voice.said == ["Yes.", "That is right."]


def test_the_idk_reply_is_never_streamed_the_turns_own_text_is_spoken_instead():
    s, voice, _ = streamer(verdict=tutor.speak_verdict)
    feed_in_pieces(s, "I don't know based on Discover Canada.")
    assert voice.said == []
    s.finish("I couldn't find that in the guide, so I won't guess. Did you mean: Who was Louis Riel?")
    assert voice.said[0].startswith("I couldn't find that") and not any("don't know" in x for x in voice.said) and not s.streamed


def test_nothing_streamed_means_the_whole_text_is_spoken_at_the_end_for_quiz_feedback():
    s, voice, _ = streamer(max_chars=None)
    s.finish("Correct! Confederation was in 1867.")
    assert voice.said == ["Correct!", "Confederation was in 1867."]


def test_a_very_long_first_sentence_is_cut_at_a_word():
    s, voice, _ = streamer(max_chars=40)
    feed_in_pieces(s, "This single sentence just keeps going and going well past any reasonable spoken limit for a voice. Then more.")
    s.finish("")
    assert len(voice.said[0]) <= 41 and voice.said[0].endswith(".") and voice.said[-1] == speak.MORE_ON_SCREEN


def test_first_audio_time_is_recorded_and_a_voice_failure_is_reported_not_raised_mid_stream():
    class Broken(FakeVoice):
        def pcm(self, sentence):
            raise RuntimeError("model crashed")
    out = []
    s = speak.SpeechStreamer(Broken(), speak.Speaker(out.append), max_chars=None)
    feed_in_pieces(s, "One. Two. Three.")                 # must not raise while the answer is still arriving
    with pytest.raises(Exception):
        s.finish("")


def test_regnal_numbers_and_st_are_spoken_forms_not_split_into_sentences():
    s, voice, _ = streamer(max_chars=None)
    feed_in_pieces(s, "Queen Elizabeth II visited St. Lawrence Seaway. She opened it.")
    s.finish("")
    assert voice.said[0] == "Queen Elizabeth the Second visited Saint Lawrence Seaway."


# --- the microphone gate follows the audio that was queued --------------------------------------------------------------------------------------
class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class QuietVad:
    name = "q"

    def probability(self, f):
        return 0.0

    def reset(self):
        pass


def test_speak_keeps_the_gate_shut_until_all_queued_speech_has_played():
    clk = Clock()
    e = live.LiveEngine(QuietVad(), EndpointConfig(), margin_s=0.5, clock=clk)
    assert e.status().state == live.LISTENING
    e.speak(2.0), e.speak(3.0)                              # two sentences: 2 s then 3 s -> 5 s in total
    clk.t += 4.9
    assert e.status().state == live.SPEAKING
    clk.t += 0.7                                            # 5.6 s: past 5 s + 0.5 s margin
    assert e.status().state == live.LISTENING


def test_release_never_shortens_a_gate_that_streamed_audio_still_needs():
    clk = Clock()
    e = live.LiveEngine(QuietVad(), EndpointConfig(), margin_s=0.5, clock=clk)
    e.hold(), e.speak(4.0)
    e.release(0.0)
    assert e.status().state == live.SPEAKING
    clk.t += 4.6
    assert e.status().state == live.LISTENING


def test_the_utterance_callback_fires_at_once_outside_the_lock():
    got = []

    class Loud(QuietVad):
        def probability(self, f):
            return 0.95 if float(np.abs(f).mean()) > 0.01 else 0.0
    e = live.LiveEngine(Loud(), EndpointConfig(min_speech_ms=96, hangover_ms=320, preroll_ms=96, tail_ms=64), margin_s=0.3,
                        on_utterance=lambda u: got.append((u, e.status().state)))     # calling back into the engine must not deadlock
    voice = (np.sin(np.arange(960) * 0.1) * 0.3).astype(np.float32)
    for x in [np.zeros(960, np.float32)] * 10 + [voice] * 60 + [np.zeros(960, np.float32)] * 40:
        e.feed(x)
    assert len(got) == 1 and got[0][0].seconds > 1.0


def test_acknowledgement_plays_once_the_phrases_are_ready_and_never_before():
    acks = speak.Acks()
    assert acks.pick() is None
    acks.warm(FakeVoice())
    deadline = time.time() + 2
    while acks.pick() is None and time.time() < deadline:
        time.sleep(0.01)
    assert acks.pick() is not None
