"""The scoring helpers of the voice evaluation harness (scripts/voice_eval). No models, microphone or API key."""
from __future__ import annotations

import numpy as np

from scripts.voice_eval import common as C
from scripts.voice_eval.run_asr import PIPELINES, per_entity, score
from scripts.voice_eval.run_grounding import checkable, drop_numbers


def test_percentiles_report_n_and_ignore_missing():
    p = C.pct([1.0, 2.0, 3.0, 4.0, None])
    assert p["n"] == 4 and p["median"] == 2.5 and p["max"] == 4.0 and p["p95"] > p["p90"] > p["median"]
    assert C.pct([]) == {"n": 0}


def test_entity_match_ignores_case_accents_spaces_and_hyphens():
    assert C.entity_found("Louis-Hippolyte La Fontaine", "who was louis hippolyte lafontaine")
    assert C.entity_found("Métis", "the Metis people") and C.entity_found("Mi'kmaq", "the mikmaq")
    assert not C.entity_found("Tecumseh", "who was tea come see")


def test_noise_is_added_at_the_requested_snr():
    rng = np.random.default_rng(0)
    x = (0.3 * np.sin(np.linspace(0, 2000, 16000))).astype(np.float32)
    noisy = C.add_noise(x, 10.0, rng)
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean((noisy - x) ** 2))
    assert 8.0 < snr < 12.0


def test_stream_reports_the_last_speech_sample():
    speech = np.concatenate([np.zeros(100), 0.2 * np.ones(1600), np.zeros(100)]).astype(np.float32)
    sig, end = C.stream(speech, lead_s=0.5, tail_s=1.0)
    assert end == int(0.5 * C.SR) + 1600 and len(sig) == end + C.SR
    assert np.abs(sig[end - 1]) > 0.01 and np.abs(sig[end:]).max() == 0


def test_capture_returns_none_without_an_utterance():
    class Quiet:
        def feed(self, x): pass
        def pop_utterance(self): return None
    utt, at, _ = C.capture(Quiet(), np.zeros(4096, np.float32))
    assert utt is None and at == 4096


def test_scoring_counts_name_errors_and_retrieval_loss():
    def row(heard, missed, kept, p):
        return {"id": "q", "voice": "v", "snr_db": None, "pipeline": p, "ref": "Who was Tecumseh?", "heard": heard, "entities": ["Tecumseh"],
                "entities_missed": missed, "audio_s": 2.0, "asr_s": 1.0, "ref_top1_kept": kept, "top1_same": kept, "overlap5": 1.0 if kept else 0.2}
    rows = [row("Who was Tecumseh?", [], True, "raw"), row("Who was tea come see?", ["Tecumseh"], False, "hint"), row("Who was Tecumseh?", [], True, "hint+fix")]
    s = score(rows)
    assert s["raw | clean"]["wer"] == 0 and s["hint | clean"]["name_errors"] == 1 and s["hint | clean"]["name_wrong_and_retrieval_lost"] == 1
    assert per_entity(rows)["Tecumseh"] == {"raw": "0/1", "hint": "1/1", "hint+fix": "0/1"} and set(PIPELINES) == {"raw", "hint", "hint+fix"}


def test_grounding_checks_find_numbers_and_inner_names_but_not_citations():
    nums, names = checkable("Samuel de Champlain built a fortress in 1608 [Discover Canada, p. 14, Exploring]. He met Hurons in 1615.")
    assert nums == {"1608", "1615"} and "Hurons" in names and "Discover" not in names and "Samuel" not in names
    assert drop_numbers("In eighteen sixty-seven, 1867 provinces") == "in provinces"


def test_a_possessive_is_the_name_not_a_new_name():
    assert checkable("Tecumseh's alliance with Brock was decisive.")[1] == {"Brock"}


def test_words_the_learner_said_and_contractions_are_not_invented_facts():
    assert checkable("I'm not sure about 'Laura Seekert'. Laura Secord crossed in 1813.", said="Who was Laura Seekert?")[1] == {"Secord"}
