"""Automatic fixes for misrecognised Canadian names ("Papino" -> "Papineau"). Rule-based; no model needed."""
import pytest

from src import tutor
from src.voice import names, session as vs
from tests.voice_helpers import FakeTranscriber, Gen, chunk, make_wav


@pytest.mark.parametrize("heard,fixed", [
    ("Who was Louis-Joseph Papino?", "Who was Louis-Joseph Papineau?"),
    ("Tell me about Papino and the Rebellions.", "Tell me about Papineau and the Rebellions."),
    ("Who was Papinow?", "Who was Papineau?"),
    ("Who is Mackenzy?", "Who is Mackenzie?"),
    ("Who was Champlin?", "Who was Champlain?"),
    ("Who was Tecumsay?", "Who was Tecumseh?"),
])
def test_misheard_names_are_fixed(heard, fixed):
    assert names.correct_names(heard)[0] == fixed


@pytest.mark.parametrize("text", [
    "What is the Charter?", "Who can vote in Canada?", "Tell me about the Rebellions of 1837.", "What about Montreal and Quebec City?",
    "Who was Papadopoulos?",                 # a real, different name: not forced into a Canadian one
    "Who was Papineau?",                     # already right
    "What do the Métis speak?", "Who was the Prince of Wales?", "Tell me about Princess Louise.", "Is Welsh spoken in Wales?", "What about England and Scotland?", "Where is Saskatchewan?", "Tell me about Prince Edward Island.", "Who is the Governor General?", "",
])
def test_correct_text_ordinary_words_and_other_names_are_left_alone(text):
    assert names.correct_names(text) == (text, [])


def test_only_capitalised_non_words_are_changed():
    assert names.correct_names("who was papino?") == ("who was papino?", [])             # lower case is not treated as a name
    assert names.correct_names("Rebellions and Responsible government")[1] == []         # words the guide uses in lower case are ordinary words


def test_the_changes_are_reported_and_the_text_is_only_edited_at_the_name():
    fixed, changes = names.correct_names("Who was Louis-Joseph Papino and who was Champlin?")
    assert changes == [("Papino", "Papineau"), ("Champlin", "Champlain")]
    assert fixed == "Who was Louis-Joseph Papineau and who was Champlain?"


def test_the_skeleton_ignores_vowels_accents_and_doubles():
    assert names.skeleton("Papineau") == names.skeleton("Papino") == names.skeleton("Papinow") == "pn"
    assert names.skeleton("Métis") == names.skeleton("Metis")


def test_no_guess_when_two_names_fit_equally():
    assert names._vocabulary()                                                               # vocabulary loaded (guide + curated)
    text, changes = names.correct_names("Who was Mackenz?")
    assert changes == [] or changes[0][1].startswith("Mackenz")                              # either unchanged or a clear single match


# --- in the session: the draft shows the fix, the original is kept, the tutor gets the confirmed text -------------------------------

def test_the_draft_shows_the_fix_and_keeps_the_original_recognised_text():
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("Who was Louis-Joseph Papino?"))
    assert s.heard == "Who was Louis-Joseph Papineau?" and s.raw_heard == "Who was Louis-Joseph Papino?" and s.fixes == [("Papino", "Papineau")]


def test_accepting_the_automatic_fix_is_not_counted_as_a_manual_correction():
    s, sent = vs.VoiceState(), []
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("Who was Louis-Joseph Papino?"))
    turn = vs.submit_turn(s, "Who was Louis-Joseph Papineau?", mode="ask",
                          turn_fn=lambda t: (sent.append(t), tutor.handle_turn(t, retrieve_fn=lambda q, k=5: [chunk()], gen=Gen()))[1], synth_fn=None)
    assert sent == ["Who was Louis-Joseph Papineau?"]                                       # the tutor never sees "Papino"
    assert turn.heard == "Who was Louis-Joseph Papino?" and turn.proposed == "Who was Louis-Joseph Papineau?" and turn.fixes == [("Papino", "Papineau")]
    assert not turn.edited


def test_changing_the_draft_by_hand_is_still_an_edit():
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("Who was Louis-Joseph Papino?"))
    turn = vs.submit_turn(s, "Who was Papineau's ally?", mode="ask",
                          turn_fn=lambda t: tutor.handle_turn(t, retrieve_fn=lambda q, k=5: [chunk()], gen=Gen()), synth_fn=None)
    assert turn.edited and turn.heard == "Who was Louis-Joseph Papino?"


def test_text_without_names_to_fix_has_no_fixes():
    s = vs.VoiceState()
    vs.ingest_recording(s, make_wav(), lambda: FakeTranscriber("Can you explain responsible government?"))
    assert s.fixes == [] and s.heard == s.raw_heard


def test_the_speech_hint_list_names_papineau():
    from src import config
    assert "Papineau" in config.ASR_INITIAL_PROMPT_DEFAULT


def test_the_guides_own_text_is_never_changed():
    """The guide is correct by definition: running the fixer over all of it must change nothing (an earlier, looser version changed 24 words,
    e.g. Bell -> Billy, Laura -> Laurier, Paris -> Prairies, Carleton -> Charlottetown)."""
    import json
    import re
    from src.config import ROOT
    changed = []
    for chunk in json.loads((ROOT / "chunks_window.json").read_text()):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk["text"]):
            changed += names.correct_names(sentence)[1]
    assert changed == []


@pytest.mark.parametrize("text", ["Who was Bell?", "Who is Laura?", "Where is Paris?", "Tell me about Carleton.", "What about Gray and Hill?", "Who was Smith?"])
def test_short_and_ordinary_words_are_never_turned_into_names(text):
    assert names.correct_names(text) == (text, [])


def test_possessives_keep_their_s_and_ordinary_possessives_are_left_alone():
    assert names.correct_names("Papino's rebellion") == ("Papineau's rebellion", [("Papino", "Papineau")])
    assert names.correct_names("Canada’s and America’s history and France’s kings") == ("Canada’s and America’s history and France’s kings", [])


@pytest.mark.parametrize("heard,fixed", [
    ("Who was Lee Joseph Papineau?", "Who was Louis-Joseph Papineau?"),
    ("Who was Lee-Joseph Papino?", "Who was Louis-Joseph Papineau?"),
    ("Tell me about Lee Hippolyte La Fontaine.", "Tell me about Louis-Hippolyte La Fontaine."),
])
def test_whole_phrase_slips_are_fixed_only_when_the_name_is_unmistakable(heard, fixed):
    assert names.correct_names(heard)[0] == fixed


def test_lee_joseph_alone_is_left_alone():
    assert names.correct_names("Is Lee Joseph a student?") == ("Is Lee Joseph a student?", [])
