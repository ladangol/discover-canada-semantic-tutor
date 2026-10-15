"""Every “Did you know?” fact must be backed by a verbatim quote in the guide, on the stated page."""
from __future__ import annotations

import random

from src.corpus import load_corpus
from src.facts import FACTS, pick
from src.learning_validation import norm


def test_every_fact_quote_is_in_the_guide_on_that_page():
    corpus = load_corpus()
    for fact in FACTS:
        start = fact.page.split("-")[0]
        texts = [norm(r.body) for r in corpus.chunks.itertuples()
                 if r.page == fact.page or r.page.split("-")[0] == start or fact.page.split("-")[0] in r.page.split("-")]
        assert any(norm(fact.quote) in t for t in texts), f"quote not found on p.{fact.page}: {fact.quote}"


def test_pick_avoids_the_current_fact_and_is_in_range():
    rng = random.Random(1)
    for _ in range(50):
        i = pick(rng, avoid=3)
        assert i != 3 and 0 <= i < len(FACTS)
