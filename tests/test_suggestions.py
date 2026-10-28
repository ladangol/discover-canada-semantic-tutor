"""The question bank: snapping a guess to the nearest guide-answerable question (fake embedder, no model) and bank integrity."""
import json
import re

import numpy as np
import pytest

from src import suggestions as S

VOCAB = ["charter", "rights", "constitution", "1982", "loyalists", "king", "hockey", "bill", "rebellion", "1837", "pizza", "tecumseh", "shawnee"]
BANK = [
    {"q": "When was the Charter of Rights entrenched in the Constitution?", "chunk_id": "a", "section": "Rights", "page": "8"},
    {"q": "What words does the Charter of Rights begin with?", "chunk_id": "b", "section": "Rights", "page": "8"},     # same section as above
    {"q": "Who were the Loyalists?", "chunk_id": "c", "section": "Loyalists", "page": "15"},
    {"q": "Why did reformers rebel in 1837?", "chunk_id": "d", "section": "Rebellions", "page": "18"},
    {"q": "Who led the Shawnee in the War of 1812?", "chunk_id": "e", "section": "War of 1812", "page": "17"},
]


def fake_embed(texts):
    """Bag-of-words over a tiny vocabulary, L2-normalised: enough to make similarity meaningful without a model."""
    out = np.zeros((len(texts), len(VOCAB) + 1), np.float32)
    for i, t in enumerate(texts):
        words = re.findall(r"[a-z0-9]+", t.lower())
        for j, w in enumerate(VOCAB):
            out[i, j] = sum(w in x for x in words)
        out[i, -1] = 0.05                                    # keeps the norm non-zero
    return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def tiny_bank(monkeypatch):
    monkeypatch.setattr(S, "_embed", fake_embed)
    monkeypatch.setattr(S, "load_bank", lambda: BANK)
    monkeypatch.setattr(S, "_matrix", lambda: fake_embed([b["q"] for b in BANK]))


def test_a_guess_is_replaced_by_the_nearest_bank_question():
    assert S.snap(["What is the Charter of Rights and the Constitution in 1982?"], min_similarity=0.5)[0] == BANK[0]["q"]


def test_best_matches_come_first_and_weak_ones_are_dropped():
    out = S.snap(["Tell me about a pizza", "Who were the Loyalists?", "Who led the Shawnee, Tecumseh?"], min_similarity=0.6)
    assert out[0] == "Who were the Loyalists?" and "pizza" not in " ".join(out).lower()
    assert BANK[4]["q"] in out


def test_nothing_close_means_no_suggestions():
    assert S.snap(["Tell me about pizza"], min_similarity=0.6) == []


def test_one_bank_question_is_offered_once_and_one_per_section():
    out = S.snap(["charter rights constitution", "charter rights 1982", "rights charter begin"], min_similarity=0.3)
    assert len(out) == len(set(out)) and sum(q in (BANK[0]["q"], BANK[1]["q"]) for q in out) == 1


def test_the_learners_own_question_is_never_offered_back():
    assert S.snap(["Who were the Loyalists?"], asked="who were the loyalists", min_similarity=0.3) == []


def test_at_most_n_and_empty_inputs():
    assert len(S.snap(["charter", "loyalists", "rebellion 1837", "shawnee tecumseh"], n=2, min_similarity=0.2)) == 2
    assert S.snap([], min_similarity=0.0) == [] and S.snap(["", "  "], min_similarity=0.0) == []


def test_a_missing_bank_gives_nothing_instead_of_an_error(monkeypatch):
    monkeypatch.setattr(S, "load_bank", lambda: [])
    monkeypatch.setattr(S, "_matrix", lambda: np.zeros((0, 4), np.float32))
    assert S.snap(["anything"], min_similarity=0.0) == []


# --- the real bank file -----------------------------------------------------------------------------------------------

def test_the_shipped_bank_is_well_formed_and_every_question_retrieves_its_own_passage(monkeypatch):
    monkeypatch.undo()                                           # use the real file and the real frozen retrieval here
    S.load_bank.cache_clear()
    data = json.loads(S.BANK_PATH.read_text())
    bank = data["questions"]
    assert len(bank) >= 200 and data["version"] == 1
    for b in bank:
        assert set(b) == {"q", "chunk_id", "section", "page"} and b["q"].endswith("?") and len(b["q"]) <= 110
    assert len({b["q"].lower() for b in bank}) == len(bank)
    from src.retrieval import retrieve
    step = max(1, len(bank) // 12)
    for b in bank[::step]:                                       # a spread of ~12: each must come back in the frozen top-5
        assert b["chunk_id"] in [c.chunk_id for c in retrieve(b["q"], k=5)], b["q"]
