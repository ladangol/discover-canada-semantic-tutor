"""Every “Did you know?” fact must be backed by a verbatim quote in the guide, on the stated page."""
from __future__ import annotations

import random

from src.corpus import load_corpus
from src.facts import FACTS, HAND_WRITTEN, draw, new_deck
from src.learning_validation import norm


def test_every_fact_quote_is_in_the_guide_on_that_page():
    corpus = load_corpus()
    for fact in FACTS:
        start = fact.page.split("-")[0]
        texts = [norm(r.body) for r in corpus.chunks.itertuples()
                 if r.page == fact.page or r.page.split("-")[0] == start or fact.page.split("-")[0] in r.page.split("-")]
        assert any(norm(fact.quote) in t for t in texts), f"quote not found on p.{fact.page}: {fact.quote}"


def test_deck_shows_every_fact_once_before_any_repeat():
    rng = random.Random(1)
    deck = new_deck(rng)
    seen = [draw(rng, deck) for _ in range(len(FACTS))]
    assert sorted(seen) == list(range(len(FACTS)))


def test_a_refilled_deck_never_starts_with_the_fact_just_shown():
    for seed in range(100):
        rng, deck = random.Random(seed), []
        last = len(FACTS) - 1
        assert draw(rng, deck, avoid=last) != last


def test_fact_texts_are_questions_and_not_duplicated():
    assert len(FACTS) > len(HAND_WRITTEN)
    texts = [f.text.lower() for f in FACTS]
    assert all(t.endswith("?") for t in texts) and len(set(texts)) == len(texts)


def test_bank_has_a_fact_for_every_province_and_territory():
    places = ["Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland", "Northwest Territories", "Nova Scotia",
              "Nunavut", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Yukon"]
    blob = " ".join(f.text + " " + f.section for f in FACTS).lower()
    assert [p for p in places if p.lower() not in blob] == []
