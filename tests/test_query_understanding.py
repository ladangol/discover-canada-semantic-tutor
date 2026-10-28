"""Regression tests for the application-layer retrieval-query cleanup (written BEFORE the cleanup was enabled).

The cleanup is NOT part of the frozen, benchmarked retrieval core: it only builds a separate string that is embedded for search.
The exact original question always goes to generation. It never swaps one group term for another."""
import re
from pathlib import Path

import pytest

from src import tutor
from src.query_understanding import retrieval_query

ROOT = Path(__file__).resolve().parent.parent

# --- what is removed ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("asked,cleaned", [
    ("What first-nation languages are mentioned in Discover Canada?", "What first-nation languages are mentioned?"),
    ("What languages are mentioned in the guide?", "What languages are mentioned?"),
    ("What does the guide say about the Charter, according to the guide?", "What about the Charter?"),
    ("Who was Champlain according to the study guide?", "Who was Champlain?"),
    ("According to Discover Canada, who can vote?", "who can vote?"),
    ("Who can vote, according to the official Discover Canada guide?", "Who can vote?"),
    ("Based on the guide, what is a constitutional monarchy?", "what is a constitutional monarchy?"),
    ("What is the RCMP in the Discover Canada study guide?", "What is the RCMP?"),
    ("Which provinces are listed in this guide?", "Which provinces are listed?"),
    ("What First Nation dialects Discover Canada talks about?", "What First Nation dialects?"),
    ("What languages does Discover Canada mention?", "What languages?"),
    ("Who is Tecumseh as described in the guide?", "Who is Tecumseh?"),
    ("what are the rights of citizens in discover canada", "what are the rights of citizens"),
    ("WHAT LANGUAGES ARE MENTIONED IN DISCOVER CANADA?", "WHAT LANGUAGES ARE MENTIONED?"),
])
def test_document_reference_boilerplate_is_removed(asked, cleaned):
    q = retrieval_query(asked)
    assert q.text == cleaned and q.original == asked and q.removed, q


# --- what is NOT touched -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("asked", [
    "What is the RCMP?", "Who was Samuel de Champlain?", "When did Prince Edward Island join Canada?",
    "What is Discover Canada?",                            # the guide itself is the topic
    "Who wrote Discover Canada?", "How long is the Discover Canada guide?", "What is the guide's last chapter?",
    "What is in the guide?",                                # removing the reference would leave nothing to search for
    "In the guide?", "According to the guide", "Tell me about the Métis.", "What languages do First Nations speak?",
    "Is Michif a language?", "What happened in 1982 in Canada?", "", "   ",
])
def test_questions_without_removable_boilerplate_are_returned_unchanged(asked):
    q = retrieval_query(asked)
    assert q.text == asked and q.removed == ()


@pytest.mark.parametrize("term", ["First Nations", "First Nation", "Aboriginal", "Indigenous", "Métis", "Inuit", "Inuktitut", "Michif"])
def test_group_terms_are_never_replaced_or_added(term):
    asked = f"What do the guide and {term} languages mention in Discover Canada according to the guide about {term} peoples?"
    q = retrieval_query(asked)
    assert q.text.count(term) == asked.count(term)                 # kept exactly as the learner wrote them
    for other in {"First Nations", "Aboriginal", "Indigenous"} - {term}:
        if other not in asked:
            assert other not in q.text                              # no synonym substitution, no category change


def test_only_words_are_deleted_never_added():
    for asked in ["What first-nation languages are mentioned in Discover Canada?", "According to the guide, who can vote?",
                  "What languages does Discover Canada mention?", "Who is Tecumseh as described in the guide?"]:
        out = retrieval_query(asked).text
        assert set(re.findall(r"[\wéÉ'-]+", out.lower())) <= set(re.findall(r"[\wéÉ'-]+", asked.lower()))


def test_cleanup_is_idempotent():
    for asked in ["What first-nation languages are mentioned in Discover Canada?", "Who can vote, according to the guide?"]:
        once = retrieval_query(asked).text
        assert retrieval_query(once).text == once


# --- the exact original question still reaches generation ---------------------------------------------------------------

class Gen:
    def __init__(self):
        self.prompts = []

    def __call__(self, messages, *a, **k):
        from src.generation import GenerationResult
        self.prompts.append(messages[-1]["content"])
        return GenerationResult("It is mentioned [S1].", "stop", 1, 1)


def test_search_uses_the_cleaned_string_and_generation_uses_the_exact_original(monkeypatch):
    from tests.voice_helpers import chunk
    monkeypatch.setenv("QUERY_CLEANUP", "1")
    seen, gen = [], Gen()
    asked = "What first-nation languages are mentioned in Discover Canada?"
    r = tutor.handle_turn(asked, retrieve_fn=lambda q, k=5: (seen.append(q), [chunk()])[1], gen=gen)
    assert seen == ["What first-nation languages are mentioned?"]                 # what was embedded
    assert f"QUESTION:\n{asked}\n" in gen.prompts[0]                                # what the model was asked: untouched
    assert r.retrieval_query == "What first-nation languages are mentioned?"


def test_the_switch_turns_it_off(monkeypatch):
    from tests.voice_helpers import chunk
    monkeypatch.setenv("QUERY_CLEANUP", "0")
    seen = []
    asked = "What languages are mentioned in the guide?"
    r = tutor.handle_turn(asked, retrieve_fn=lambda q, k=5: (seen.append(q), [chunk()])[1], gen=Gen())
    assert seen == [asked] and r.retrieval_query == asked


def test_an_unchanged_question_is_searched_exactly_as_asked(monkeypatch):
    from tests.voice_helpers import chunk
    monkeypatch.setenv("QUERY_CLEANUP", "1")
    seen = []
    tutor.handle_turn("What is the RCMP?", retrieve_fn=lambda q, k=5: (seen.append(q), [chunk()])[1], gen=Gen())
    assert seen == ["What is the RCMP?"]


# --- the benchmarked core is untouched and the research app stays pure ---------------------------------------------------

def test_the_frozen_retrieval_core_does_not_know_about_the_cleanup():
    for f in ("src/retrieval.py", "src/embeddings.py", "src/corpus.py", "reference_rag.py"):
        assert "query_understanding" not in (ROOT / f).read_text(), f


def test_the_research_atlas_still_searches_with_the_raw_question():
    for f in ("app.py", "src/visualization.py"):
        assert "query_understanding" not in (ROOT / f).read_text(), f


# --- effect on the real, frozen retrieval (local embeddings, no LLM): never worse, and better where the boilerplate hurt --------

CASES = [  # (question with boilerplate, a phrase that the answering passage contains)
    ("What languages are mentioned in Discover Canada, such as Michif?", "Michif"),
    ("What dialect do the Métis speak according to the guide?", "Michif"),
    ("Who was Samuel de Champlain according to Discover Canada?", "1608 Champlain"),
    ("What does the RCMP do, according to the study guide?", "Mounted Police"),
    ("What is the smallest province in Canada in the Discover Canada guide?", "smallest province"),
    ("Who discovered insulin, according to the guide?", "insulin"),
    ("Who was Terry Fox according to Discover Canada?", "Marathon of Hope"),
    ("What is the capital of Nunavut according to the guide?", "Iqaluit"),
    ("What happened in the War of 1812 in Discover Canada?", "War of 1812"),
]


def test_cleanup_never_ranks_the_answering_passage_lower_on_real_retrieval():
    import numpy as np
    from src.corpus import load_corpus
    from src.embeddings import encode_query
    c = load_corpus()
    texts = [re.sub(r"\s+", " ", t) for t in c.chunks.text]

    def best_rank(q, phrase):
        sims = c.embeddings @ (encode_query(q) / np.linalg.norm(encode_query(q)))
        order = list(np.argsort(-sims))
        return min(order.index(i) + 1 for i, t in enumerate(texts) if phrase in t)
    worse = []
    for asked, phrase in CASES:
        cleaned = retrieval_query(asked).text
        assert cleaned != asked, asked
        a, b = best_rank(asked, phrase), best_rank(cleaned, phrase)
        if b > a + 1:                                          # allow ±1 noise; anything larger is a regression
            worse.append((asked, a, b))
    assert not worse, worse
