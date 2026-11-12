"""Spoken commands on the Live page: starting and leaving a quiz question by voice, ignoring hesitations, and the "still looking" follow-up.
No models, network or audio hardware."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.voice import intents, live, speak
from src.voice.endpointing import EndpointConfig


@pytest.mark.parametrize("text,kind", [
    ("ask me a multiple choice question", "mcq"), ("Ask me a question", "mcq"), ("quiz me", "mcq"), ("Quiz me please", "mcq"),
    ("give me an open question", "free"), ("Can you ask me an open-ended question?", "free"), ("another question", "mcq"),
    ("next question please", "mcq"), ("test me", "mcq"), ("a multiple choice question please", "mcq"), ("give me another one", "mcq")])
def test_requests_to_be_quizzed_are_recognised(text, kind):
    assert intents.quiz_request(text) == kind


@pytest.mark.parametrize("text", [
    "Who was Papineau?", "What are the questions on the citizenship test?", "What is a multiple choice question?", "B", "the second one",
    "Tell me about the quiz show", "When did Confederation happen",
    "What was the next question on the exam about the rebellions in Lower Canada and Upper Canada in 1837"])
def test_real_questions_and_answers_are_never_taken_for_commands(text):
    assert intents.quiz_request(text) is None and not intents.quiz_exit(text)


def test_another_question_keeps_the_kind_of_the_last_one():
    assert intents.quiz_request("another question", last_kind="free") == "free"
    assert intents.quiz_request("another multiple choice question", last_kind="free") == "mcq"


@pytest.mark.parametrize("text", ["stop the quiz", "never mind", "cancel the question", "back to normal questions"])
def test_leaving_a_question_by_voice(text):
    assert intents.quiz_exit(text)


@pytest.mark.parametrize("text", ["um", "Umm...", "Hmm.", "you", "Thank you.", "", "uh, um"])
def test_hesitations_and_invented_filler_words_are_noise(text):
    assert intents.is_noise(text)


@pytest.mark.parametrize("text", ["Who was Papineau?", "B", "the second one", "Yes, I meant that.", "okay so who was he"])
def test_real_speech_is_not_noise(text):
    assert not intents.is_noise(text)


def test_yes_counts_only_when_a_reply_is_expected():
    assert intents.is_noise("yes") and not intents.is_noise("yes", expecting_reply=True)


# --- the page's routing (the reported bug: after answering a question, normal questions stopped working) ----------------------------------------
@pytest.fixture
def page(monkeypatch):
    from ui import voice_live as vl
    calls = []
    state = {}
    monkeypatch.setattr(vl, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(vl, "_answer", lambda pipe, c, vs, text, mode, q, quiz_state, open_q, record, voice_ok, heard_at=None:
                        calls.append(("answer", text, mode, open_q)) or SimpleNamespace(id=1))
    monkeypatch.setattr(vl, "_ask_for_question", lambda pipe, new_question, kind, voice_ok, heard_at=None: calls.append(("ask_for", kind)))
    monkeypatch.setattr(vl, "_say", lambda pipe, voice_ok, text: calls.append(("say", text)))
    monkeypatch.setattr(vl.vsession, "discard_draft", lambda vs: calls.append(("discard",)))
    pipe = SimpleNamespace(engine=SimpleNamespace(hold=lambda: None, release=lambda s=0.0: None))
    vs = SimpleNamespace(turns=[])
    q = SimpleNamespace(question="Who?", kind="mcq", options=["a", "b", "c", "d"])
    return vl, state, calls, pipe, vs, q


def handle(vl, pipe, vs, text):
    return vl._handle_text(pipe, None, vs, text, None, None, True)


def test_after_answering_a_question_a_normal_question_goes_to_the_tutor(page):
    vl, state, calls, pipe, vs, q = page
    state[vl.QUIZ_KEY] = {"q": q, "res": object(), "pick": 1, "id": 1}          # the question was answered: no question is open
    handle(vl, pipe, vs, "Who was Papineau?")
    assert calls == [("answer", "Who was Papineau?", vl.vt.ASK, False)]


def test_while_a_question_is_open_the_words_are_the_answer(page):
    vl, state, calls, pipe, vs, q = page
    state[vl.QUIZ_KEY] = {"q": q, "res": None, "pick": None, "id": 1}
    handle(vl, pipe, vs, "B")
    assert calls == [("answer", "B", vl.vt.QUIZ, True)]


def test_ask_me_a_question_starts_one_without_any_click(page):
    vl, state, calls, pipe, vs, q = page
    handle(vl, pipe, vs, "ask me a multiple choice question")
    handle(vl, pipe, vs, "ask me an open question")
    assert [c for c in calls if c[0] == "ask_for"] == [("ask_for", "mcq"), ("ask_for", "free")]
    assert not [c for c in calls if c[0] == "answer"]


def test_another_question_works_while_one_is_open(page):
    vl, state, calls, pipe, vs, q = page
    state[vl.QUIZ_KEY] = {"q": q, "res": None, "pick": None, "id": 1}
    handle(vl, pipe, vs, "another question")
    assert ("ask_for", "mcq") in calls and not [c for c in calls if c[0] == "answer"]


def test_stop_the_quiz_leaves_the_question_and_normal_questions_work_again(page):
    vl, state, calls, pipe, vs, q = page
    state[vl.QUIZ_KEY] = {"q": q, "res": None, "pick": None, "id": 1}
    handle(vl, pipe, vs, "stop the quiz")
    assert state[vl.QUIZ_KEY] is None and any(c[0] == "say" for c in calls)
    handle(vl, pipe, vs, "Who was Papineau?")
    assert calls[-1] == ("answer", "Who was Papineau?", vl.vt.ASK, False)


def test_a_breath_or_an_um_does_nothing_even_with_a_question_open(page):
    vl, state, calls, pipe, vs, q = page
    state[vl.QUIZ_KEY] = {"q": q, "res": None, "pick": None, "id": 1}
    for t in ("um", "Hmm.", "you", ""):
        assert handle(vl, pipe, vs, t) is None
    assert calls == [("discard",)] * 4


# --- engine: Stop talking ------------------------------------------------------------------------------------------------------------------------
class Clock:
    t = 100.0

    def __call__(self):
        return self.t


class QuietVad:
    name = "q"

    def probability(self, f):
        return 0.0

    def reset(self):
        pass


def test_stop_talking_reopens_the_microphone_quickly():
    clk = Clock()
    e = live.LiveEngine(QuietVad(), EndpointConfig(), margin_s=0.8, clock=clk)
    e.speak(60.0)
    assert e.status().state == live.SPEAKING
    e.stop_speaking()
    clk.t += 0.4
    assert e.status().state == live.LISTENING


# --- "I'm still looking" ---------------------------------------------------------------------------------------------------------------------------
class FakeVoice:
    pause_ms = 0

    def pcm(self, s):
        return np.ones(240, np.int16)


def ready_acks():
    a = speak.Acks()
    a.warm(FakeVoice())
    deadline = time.time() + 2
    while (a.pick() is None or a.pick_still() is None) and time.time() < deadline:
        time.sleep(0.01)
    return a


def test_still_looking_is_said_when_the_reply_is_slow_and_not_more_than_the_limit():
    out = []
    sp = speak.Speaker(out.append)
    with speak.Patience(ready_acks(), sp, spoke=lambda: False, first_after=0.05, every=0.05, limit=2):
        time.sleep(0.5)
    assert len(out) == 2


def test_still_looking_is_never_said_once_the_reply_has_begun():
    out = []
    sp = speak.Speaker(out.append)
    with speak.Patience(ready_acks(), sp, spoke=lambda: True, first_after=0.05, every=0.05, limit=3):
        time.sleep(0.3)
    assert out == []


def test_still_looking_stops_the_moment_the_work_is_done():
    out = []
    sp = speak.Speaker(out.append)
    with speak.Patience(ready_acks(), sp, spoke=lambda: False, first_after=5.0):
        pass
    time.sleep(0.2)
    assert out == []


def test_a_phrase_is_not_repeated_back_to_back():
    a = ready_acks()
    picks = [id(a.pick_still()) for _ in range(30)]
    assert all(x != y for x, y in zip(picks, picks[1:]))


def test_the_acknowledgement_goes_to_the_speaker_as_a_short_phrase_that_cannot_be_talked_over():
    from ui import live_audio as la
    pushed = []
    pipe = la.LiveAudioPipe(acks=ready_acks())
    pipe.speaker = speak.Speaker(lambda x: None, on_speak=lambda sec, interruptible: pushed.append(interruptible))
    pipe.acknowledge(SimpleNamespace(seconds=2.0))
    assert pushed == [False]
    pipe.ack_enabled = False
    pipe.acknowledge(SimpleNamespace(seconds=2.0))
    assert pushed == [False]


def test_the_page_acknowledges_only_after_real_words_were_understood(page):
    vl, state, calls, pipe, vs, q = page
    acked = []
    for text in ("um", "uuuuum", "Hmm.", "you"):
        vl._handle_text(pipe, None, vs, text, None, None, True, ack=lambda: acked.append(1))
    assert acked == []                                                  # "cool, uuuum, let me think" gets no "one moment"
    vl._handle_text(pipe, None, vs, "Who was Papineau?", None, None, True, ack=lambda: acked.append(1))
    assert acked == [1]


# --- talking over the tutor ---------------------------------------------------------------------------------------------------------------------------
class Loud:
    name = "loud"

    def probability(self, f):
        return 0.95 if float(np.abs(f).mean()) > 0.01 else 0.0

    def reset(self):
        pass


VOICE = (np.sin(np.arange(512) * 0.2) * 0.3).astype(np.float32)
QUIET = np.zeros(512, np.float32)


class Steppy:
    """A clock that advances 32 ms per audio frame, so feeding frames moves time forward like a real stream."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def frames(self, engine, chunk, n):
        for _ in range(n):
            self.t += 0.032
            engine.feed(chunk)


def barge_engine(on, clock):
    hits = []
    cfg = EndpointConfig(min_speech_ms=96, hangover_ms=320, preroll_ms=96, tail_ms=64)
    e = live.LiveEngine(Loud(), cfg, margin_s=0.3, clock=clock, barge_in=on, barge_min_ms=480, on_interrupt=lambda: hits.append(clock.t))
    return e, hits


def test_speaking_over_the_tutor_cuts_it_off_and_keeps_your_words():
    clk = Steppy()
    e, hits = barge_engine(True, clk)
    e.hold(), e.speak(20.0), e.release(0.0)                       # a reply is playing (20 s) while the page's turn has finished
    clk.frames(e, QUIET, 20)                                      # nothing said: no interruption
    assert not hits and e.status().state == live.SPEAKING
    clk.frames(e, VOICE, 40)                                      # 1.3 s of clear speech
    assert len(hits) == 1
    clk.frames(e, VOICE, 20), clk.frames(e, QUIET, 20)            # the learner finishes their sentence
    u = e.pop_utterance()
    assert u is not None and u.seconds > 1.5                      # the words from before the cut-off are in it, not only what came after


def test_a_cough_or_breath_does_not_interrupt():
    clk = Steppy()
    e, hits = barge_engine(True, clk)
    e.speak(20.0)
    clk.frames(e, VOICE, 8)                                       # 0.26 s: shorter than the 0.48 s needed
    clk.frames(e, QUIET, 20)
    assert not hits and e.pop_utterance() is None and e.status().state == live.SPEAKING


def test_without_barge_in_the_tutor_is_never_interrupted():
    clk = Steppy()
    e, hits = barge_engine(False, clk)
    e.speak(20.0)
    clk.frames(e, VOICE, 60)
    assert not hits and e.pop_utterance() is None


def test_a_short_phrase_like_one_moment_cannot_be_talked_over_but_the_reply_after_it_can():
    clk = Steppy()
    e, hits = barge_engine(True, clk)
    e.speak(1.0, interruptible=False)                            # "One moment."
    clk.frames(e, VOICE, 20)
    assert not hits
    clk.frames(e, QUIET, 30)                                     # it finishes
    e.speak(10.0)                                                # the reply
    clk.frames(e, VOICE, 40)
    assert len(hits) == 1


def test_the_page_is_not_held_shut_after_an_interruption_even_if_the_turn_is_still_running():
    clk = Steppy()
    e, hits = barge_engine(True, clk)
    e.hold(), e.speak(20.0)                                      # the turn is still being worked on AND a first sentence is playing
    clk.frames(e, VOICE, 40)
    assert hits and e.status().state in (live.HEARING, live.LISTENING)


# --- stale speech from an interrupted reply never leaks out ---------------------------------------------------------------------------------------------
def test_clearing_the_speaker_drops_later_pushes_from_the_old_reply():
    out = []
    sp = speak.Speaker(out.append)
    old = sp.epoch
    sp.push(np.ones(10, np.int16), epoch=old)
    sp.clear()
    sp.push(np.ones(10, np.int16), epoch=old)                    # a sentence that finished synthesising after the interruption
    sp.push(np.ones(10, np.int16), epoch=sp.epoch)              # a new reply
    assert len(out) == 2


def test_an_aborted_reply_stops_speaking_and_keeps_what_was_heard():
    class Voice:
        pause_ms = 0

        def __init__(self):
            self.n = 0

        def pcm(self, sentence):
            self.n += 1
            return np.ones(2400, np.int16)

    out = []
    sp = speak.Speaker(out.append)
    s = speak.SpeechStreamer(Voice(), sp, max_chars=None)
    s.feed("One is here. Two is here. ")
    deadline = time.time() + 2
    while not out and time.time() < deadline:
        time.sleep(0.01)
    s.abort()
    s.feed("Three is here. Four is here. ")
    n_before = len(out)
    res = s.finish("")
    assert len(out) == n_before and res.mime_type == "audio/wav" and s.aborted


@pytest.mark.parametrize("text", ["Cool, uuuuuum.", "uuuuuuum", "cool", "Hmmmm... let me think", "Cool, uuuuuum, let me think about it",
                                  "I'm thinking about my next question", "give me a second", "hold on, let me think", "nice", "Okay, cool."])
def test_thinking_out_loud_and_reactions_get_no_reply(text):
    assert intents.is_noise(text) and intents.quiz_request(text) is None


@pytest.mark.parametrize("text", ["let me think about who was Papineau", "Cool, who was Papineau?", "What is my next question", "Umm what is Confederation"])
def test_a_real_question_after_a_filler_is_still_a_question(text):
    assert not intents.is_noise(text)


def test_next_question_is_still_a_command_even_though_its_words_are_soft(page):
    vl, state, calls, pipe, vs, q = page
    handle(vl, pipe, vs, "next question")
    handle(vl, pipe, vs, "I'm thinking about my next question")
    assert [c for c in calls if c[0] == "ask_for"] == [("ask_for", "mcq")] and not [c for c in calls if c[0] == "answer"]
