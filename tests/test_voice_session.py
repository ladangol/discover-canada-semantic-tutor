"""Turn routing and the 'process each thing exactly once' guarantees — mock adapters only, no models, no API key."""
import pytest

from src import tutor
from src.voice import session as vs
from src.voice.models import VoiceTimings, VoiceUnavailable
from tests.voice_helpers import FakeSynth, FakeTranscriber, Gen, NoCalls, free_q, make_wav, mcq, chunk


def ask_turn(gen=None, retrieved=None):
    gen = gen or Gen()
    return lambda text: tutor.handle_turn(text, retrieve_fn=lambda q, k=5: retrieved or [chunk()], gen=gen)


# --- handle_turn routing ----------------------------------------------------------------------------------------

def test_question_goes_through_retrieval_then_grounded_answer():
    gen = Gen()
    r = tutor.handle_turn("When was Confederation?", retrieve_fn=lambda q, k=5: [chunk()], gen=gen)
    assert r.kind == "answer" and gen.calls == 1
    assert r.cited and "Confederation" in r.cited[0].section
    assert {"retrieval", "generation", "total"} <= set(r.timings)


def test_blank_text_never_reaches_retrieval_or_llm():
    r = tutor.handle_turn("   ", retrieve_fn=NoCalls(), gen=NoCalls())
    assert r.kind == "clarify"


def test_spoken_mcq_answer_is_graded_deterministically_not_asked_as_a_question():
    r = tutor.handle_turn("Option B", quiz=mcq(correct_index=1), retrieve_fn=NoCalls(), gen=NoCalls())
    assert r.kind == "quiz_feedback" and r.quiz_result.correct and r.chosen_index == 1


def test_wrong_spoken_mcq_answer_explains():
    r = tutor.handle_turn("the second one", quiz=mcq(correct_index=0), retrieve_fn=NoCalls(), gen=NoCalls())
    assert not r.quiz_result.correct and "1867" in r.text and "It was 1867." in r.text
    assert r.speech_text.startswith("Not quite")


def test_answer_by_option_text():
    r = tutor.handle_turn("1867.", quiz=mcq(correct_index=0), retrieve_fn=NoCalls(), gen=NoCalls())
    assert r.quiz_result.correct and r.speech_text.startswith("Correct.")


def test_unclear_option_asks_again_without_grading_or_calls():
    r = tutor.handle_turn("hmm I don't know", quiz=mcq(), retrieve_fn=NoCalls(), gen=NoCalls())
    assert r.kind == "clarify" and r.quiz_result is None


def test_open_answer_uses_the_existing_free_grader_once():
    gen = Gen('{"correct": true, "feedback": "Yes, 1867!"}')
    r = tutor.handle_turn("1867", quiz=free_q(), retrieve_fn=NoCalls(), gen=gen)
    assert r.kind == "quiz_feedback" and r.quiz_result.correct and gen.calls == 1


@pytest.mark.parametrize("said,expect", [
    ("A", 0), ("a.", 0), ("option c", 2), ("I think it's D", 3), ("the last one", 3), ("number 2", 1),
    ("Ontario", 2), ("Nova Scotia please", 3), ("1812", 1), ("banana", None), ("", None),
])
def test_match_option(said, expect):
    assert tutor.match_option(["1867", "1812", "Ontario and Quebec", "Nova Scotia"], said) == expect


def test_ambiguous_text_does_not_guess():
    assert tutor.match_option(["Upper Canada", "Lower Canada", "Quebec", "Nova Scotia"], "Canada") is None


# --- reruns and confirmation -------------------------------------------------------------------------------------

def test_a_recording_is_transcribed_once_across_reruns():
    s, t, wav = vs.VoiceState(), FakeTranscriber(), make_wav()
    assert vs.ingest_recording(s, wav, lambda: t) is True
    for _ in range(5):                                       # Streamlit reruns with the same widget value
        assert vs.ingest_recording(s, wav, lambda: t) is False
    assert t.calls == 1 and s.has_draft and s.heard == t.text
    assert s.draft_timings.asr == 0.25 and s.draft_timings.audio_seconds == pytest.approx(1.0)


def test_new_recording_replaces_the_draft_and_reseeds_the_textbox():
    s, t = vs.VoiceState(), FakeTranscriber()
    vs.ingest_recording(s, make_wav(freq=300), lambda: t)
    v1 = s.draft_version
    t.text = "Who was Tecumseh?"
    vs.ingest_recording(s, make_wav(freq=500), lambda: t)
    assert s.heard == "Who was Tecumseh?" and s.draft_version > v1 and t.calls == 2


def test_nothing_reaches_the_tutor_without_a_confirmed_draft():
    s, calls = vs.VoiceState(), []
    assert vs.submit_turn(s, "Who was Tecumseh?", mode="ask", turn_fn=lambda t: calls.append(t), synth_fn=None) is None
    assert calls == []


def test_blank_confirmation_is_ignored_and_keeps_the_draft():
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber())
    assert vs.submit_turn(s, "  ", mode="ask", turn_fn=NoCalls(), synth_fn=None) is None
    assert s.has_draft


def test_confirmed_text_is_used_exactly_once_even_if_the_script_reruns():
    s, gen, synth = vs.VoiceState(), Gen(), FakeSynth()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("Can you explain responsible government"))
    first = vs.submit_turn(s, "Can you explain responsible government?", mode="ask", turn_fn=ask_turn(gen), synth_fn=synth.synthesize)
    again = vs.submit_turn(s, "Can you explain responsible government?", mode="ask", turn_fn=ask_turn(gen), synth_fn=synth.synthesize)
    assert first is not None and again is None
    assert gen.calls == 1 and len(synth.calls) == 1 and len(s.turns) == 1
    assert not s.has_draft


def test_the_correction_is_sent_and_the_original_asr_text_is_kept_apart():
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("who was tea come say"))
    sent = []
    turn = vs.submit_turn(s, "Who was Tecumseh?", mode="ask",
                          turn_fn=lambda t: (sent.append(t), ask_turn()(t))[1], synth_fn=None)
    assert sent == ["Who was Tecumseh?"]                      # never silently replaced by the ASR text
    assert turn.heard == "who was tea come say" and turn.said == "Who was Tecumseh?" and turn.edited


def test_audio_and_text_stay_with_their_own_turn():
    s, synth = vs.VoiceState(), FakeSynth()
    for q in ("First question?", "Second question?"):
        vs.ingest_recording(s, make_wav(freq=300 + len(s.turns) * 100), lambda: FakeTranscriber(q))
        vs.submit_turn(s, q, mode="ask", turn_fn=ask_turn(Gen(f"Answer to {q} [S1].")), synth_fn=synth.synthesize)
    a, b = s.turns
    assert (a.said, b.said) == ("First question?", "Second question?")
    assert a.audio.audio_bytes != b.audio.audio_bytes
    assert "First question?" in a.result.text and "Second question?" in b.result.text


def test_old_turns_drop_audio_bytes_but_keep_text():
    s, synth = vs.VoiceState(), FakeSynth()
    for i in range(vs.KEEP_AUDIO_TURNS + 2):
        vs.ingest_recording(s, make_wav(freq=200 + i * 37), lambda: FakeTranscriber("q"))
        vs.submit_turn(s, "q", mode="ask", turn_fn=ask_turn(), synth_fn=synth.synthesize)
    assert s.turns[0].audio is None and s.turns[0].result.text and s.turns[-1].audio is not None


def test_recorder_widget_is_reset_after_submit_and_after_record_again():
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber())
    n = s.rec_n
    vs.submit_turn(s, "q", mode="ask", turn_fn=ask_turn(), synth_fn=None)
    assert s.rec_n == n + 1
    vs.ingest_recording(s, make_wav(freq=900), lambda: FakeTranscriber())
    vs.discard_draft(s)
    assert s.rec_n == n + 2 and not s.has_draft


# --- graceful failure --------------------------------------------------------------------------------------------

def test_corrupted_recording_gives_a_notice_and_a_typeable_draft():
    s, t = vs.VoiceState(), FakeTranscriber()
    assert vs.ingest_recording(s, b"garbage" * 50, lambda: t) is True
    assert t.calls == 0 and s.has_draft and s.heard == "" and "couldn't read" in s.notice


def test_asr_failure_is_a_notice_not_a_traceback():
    class Down:
        def transcribe(self, *a):
            raise VoiceUnavailable("Speech recognition isn't installed on this server. You can still type your question.")
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: Down())
    assert s.has_draft and "still type" in s.notice


def test_asr_that_hears_nothing_says_so():
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("   "))
    assert "couldn't make out" in s.notice


def test_tts_failure_still_returns_the_written_answer():
    def broken(_text):
        raise VoiceUnavailable("No voice model is set up, so I can't speak. The written answer is below.")
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("q"))
    turn = vs.submit_turn(s, "q", mode="ask", turn_fn=ask_turn(), synth_fn=broken)
    assert turn.audio is None and "written answer" in turn.audio_note and turn.result.text


def test_tutor_failure_keeps_the_draft_so_the_learner_can_retry():
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("q"))
    with pytest.raises(RuntimeError):
        vs.submit_turn(s, "q", mode="ask", turn_fn=lambda t: (_ for _ in ()).throw(RuntimeError("boom")), synth_fn=None)
    assert s.has_draft and not s.turns


def test_timings_are_measured_per_stage_and_separate_from_correction_time():
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("q"))
    turn = vs.submit_turn(s, "q", mode="ask", turn_fn=ask_turn(), synth_fn=FakeSynth().synthesize)
    t: VoiceTimings = turn.timings
    assert t.asr == 0.25 and t.tts == 0.1 and t.retrieval is not None and t.generation is not None
    assert t.processing_total() == pytest.approx(t.asr + t.tutor_total + t.tts)
    assert dict(t.rows())["Text to speech"] == "0.10 s"


# --- "did you mean…": one guess call, then the guide's own question bank -----------------------------------------------------

import json  # noqa: E402

from tests.voice_helpers import FakeCorpus, SeqGen  # noqa: E402

ABSTAIN = tutor.UNKNOWN_ANSWER
GOOD = "Confederation was in 1867 [S1]."


def guesses(*qs):
    return json.dumps({"suggestions": list(qs)})


def run_turn(gen, snap=lambda cands, asked="": cands[:3], message="What happened in 1982 in Canada?", corpus=FakeCorpus()):
    return tutor.handle_turn(message, corpus=corpus, retrieve_fn=lambda q, k=5: [chunk()], gen=gen, snap_fn=snap)


def test_no_match_costs_one_extra_llm_call_and_shows_the_snapped_questions():
    gen = SeqGen(ABSTAIN, guesses("What is the Charter of Rights?", "Which king ruled during the American Revolution?"))
    seen = {}

    def snap(cands, asked=""):
        seen.update(cands=cands, asked=asked)
        return ["When was the Canadian Charter of Rights and Freedoms entrenched in the Constitution?"]   # a bank question
    r = run_turn(gen, snap)
    assert gen.calls == 2                                                      # the learner's question + ONE guess call; nothing else
    assert seen["cands"] == ["What is the Charter of Rights?", "Which king ruled during the American Revolution?"]
    assert seen["asked"] == "What happened in 1982 in Canada?"
    assert r.abstained and r.suggestions == ["When was the Canadian Charter of Rights and Freedoms entrenched in the Constitution?"]
    assert "won't guess" in r.text and "Did you mean" in r.text and "I don't know based" not in r.text
    assert "Did you mean: When was the Canadian Charter" in r.speech_text and r.cited == []


def test_when_nothing_is_close_the_tutor_asks_for_a_specific_topic():
    r = run_turn(SeqGen(ABSTAIN, guesses("What is a pizza?")), snap=lambda cands, asked="": [])
    assert r.abstained and r.suggestions == [] and "Did you mean" not in r.text and "specific topic" in r.text


def test_a_failing_guess_call_never_breaks_the_turn():
    class Boom(SeqGen):
        def __call__(self, *a, **k):
            if self.calls >= 1:
                raise RuntimeError("provider down")
            return super().__call__(*a, **k)
    r = run_turn(Boom(ABSTAIN))
    assert r.abstained and r.suggestions == [] and "specific topic" in r.text


def test_a_failing_snap_never_breaks_the_turn():
    def boom(cands, asked=""):
        raise RuntimeError("embedding model unavailable")
    r = run_turn(SeqGen(ABSTAIN, guesses("What is the Charter?")), snap=boom)
    assert r.abstained and r.suggestions == [] and "specific topic" in r.text


def test_garbage_guess_json_gives_no_suggestions_and_no_crash():
    got = {}
    r = run_turn(SeqGen(ABSTAIN, "sorry, I can't"), snap=lambda cands, asked="": got.setdefault("c", cands) and [])
    assert r.abstained and r.suggestions == [] and got.get("c", []) == [] and "specific topic" in r.text


def test_guesses_are_cleaned_before_snapping():
    got = {}
    run_turn(SeqGen(ABSTAIN, guesses("What is the Charter of Rights?", "What is the Charter of Rights?", "not a question", "Who?")),
             snap=lambda cands, asked="": got.setdefault("c", cands) and [])
    assert got["c"] == ["What is the Charter of Rights?"]                       # deduped; no non-questions; no too-short ones


def test_an_abstention_never_shows_sources_even_if_the_model_attached_labels():
    r = tutor.handle_turn("zzz", retrieve_fn=lambda q, k=5: [chunk()], gen=SeqGen(ABSTAIN + " [S1][S2]"))
    assert r.abstained and r.cited == []


def test_no_guess_call_without_a_corpus_or_when_the_guide_answered():
    g1 = SeqGen(ABSTAIN)
    tutor.handle_turn("zzz", retrieve_fn=lambda q, k=5: [chunk()], gen=g1)
    g2 = SeqGen(GOOD)
    r = tutor.handle_turn("When?", corpus=FakeCorpus(), retrieve_fn=lambda q, k=5: [chunk()], gen=g2)
    assert g1.calls == 1 and g2.calls == 1 and r.suggestions == [] and r.cited


def test_a_tapped_suggestion_is_a_normal_confirmed_turn_recorded_once():
    q = "When was the Canadian Charter of Rights and Freedoms entrenched in the Constitution?"
    s, gen = vs.VoiceState(), Gen()
    vs.begin_text(s, q)
    first = vs.submit_turn(s, q, mode="ask", turn_fn=ask_turn(gen), synth_fn=None)
    again = vs.submit_turn(s, q, mode="ask", turn_fn=ask_turn(gen), synth_fn=None)
    assert first is not None and again is None and gen.calls == 1 and not first.edited


def test_duplicate_source_labels_are_shown_once():
    from ui import study_theme
    html = study_theme.pills(["p. 15 · United empire loyalists"] * 3 + ["p. 8 · The Charter"], "#fff")
    assert html.count("United empire loyalists") == 1 and html.count("The Charter") == 1


def test_the_tutor_prompt_keeps_jargon_out_and_never_claims_the_guide_lacks_others():
    p = tutor.ASK_SYSTEM_PROMPT
    assert 'never say "the\n   retrieved sources"' in p and "there may\n   be more in other parts of the guide" in p
    assert "Never claim that the guide mentions no\n   others" in p
