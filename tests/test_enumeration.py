"""Document-wide enumeration is a SEPARATE workflow from Q&A. Tests use the real guide text and a fake model (no API key)."""
import json
import re
from pathlib import Path

import pytest

from src import enumeration as en
from src.corpus import load_corpus
from src.generation import GenerationResult

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "GENERATED_DIR", tmp_path / "generated")        # never write into the real artifacts/generated


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def blocks(prompt: str) -> dict[str, str]:
    """{'S1': text, …} from a prompt built by enumeration._prompt (what the model would see)."""
    return {m.group(1): m.group(2) for m in re.finditer(r"SOURCE (S\d+)\n.*?text:\n(.*?)(?=\n\n---|\n\nList every item|\Z)", prompt, flags=re.S)}


class FakeReader:
    """Plays the model: reports Inuktitut (wording 'language'), Michif (wording 'dialect') and a fabricated item, wherever the passage has them."""
    def __init__(self, extra=()):
        self.calls, self.extra = 0, list(extra)

    def __call__(self, messages, **kw):
        self.calls += 1
        self.temperatures = getattr(self, "temperatures", []) + [kw.get("temperature")]
        items = []
        for label, text in blocks(messages[-1]["content"]).items():
            flat = " ".join(text.split())
            if "in the Inuktitut language" in flat:
                items.append({"name": "Inuktitut", "stated_as": "language", "source": label, "evidence_quote": "in the Inuktitut language"})
            if "speak their own dialect, Michif" in flat:
                items.append({"name": "Michif", "stated_as": "dialect", "source": label, "evidence_quote": "speak their own dialect, Michif"})
            m = re.search(r"(French[^.]{0,80}?languages)", flat)
            if m:
                items.append({"name": "French", "stated_as": "english", "source": label, "evidence_quote": m.group(1)})   # a sloppy model label: ignored
        items += [dict(x, source=x.get("source", "S1")) for x in self.extra]
        return GenerationResult(json.dumps({"items": items}), "stop", 1, 1)


# --- scope: the learner's own word, no synonyms ------------------------------------------------------------------------------------

def test_quick_mode_reads_exactly_the_passages_that_contain_the_learners_word(corpus):
    stem = en.head_stem("languages")
    assert stem == "langua"
    quick = set(en._indices(corpus, "languages", "quick"))
    for i, t in enumerate(corpus.chunks.text):
        assert (i in quick) == bool(re.search(r"\blangua", t, re.I)), i              # a literal match, in both directions
    assert 0 < len(quick) < len(corpus.chunks) / 5


def test_no_synonym_is_ever_added_to_the_scope(corpus):
    quick = set(en._indices(corpus, "languages", "quick"))
    dialect_only = [i for i, t in enumerate(corpus.chunks.text) if re.search(r"\bdialect", t, re.I) and not re.search(r"\blangua", t, re.I)]
    assert not (quick & set(dialect_only))                                           # "dialect" is not treated as "language"
    assert en.head_stem("First Nations languages") == "langua" and en.head_stem("the list of prime ministers") == "minist"


def test_thorough_mode_reads_every_passage_and_the_plan_states_the_cost(corpus):
    thorough = en.plan_scan(corpus, "languages", "thorough")
    quick = en.plan_scan(corpus, "languages", "quick")
    assert thorough.passages == len(corpus.chunks) and thorough.calls == -(-len(corpus.chunks) // en.BATCH)
    assert quick.passages == len(en._indices(corpus, "languages", "quick")) and quick.calls < thorough.calls
    assert "literal match, no synonyms" in quick.scope and "every passage" in thorough.scope


# --- extraction, validation and grouping -----------------------------------------------------------------------------------------

def test_items_are_verified_and_grouped_by_the_guides_own_wording(corpus):
    gen = FakeReader(extra=[
        {"name": "Klingon", "stated_as": "language", "evidence_quote": "a language of the stars"},          # not in the passage
        {"name": "Cree", "stated_as": "language", "evidence_quote": "in the Inuktitut language"},           # quote real, but it does not name Cree
        {"name": "Elvish", "stated_as": "language", "source": "S99", "evidence_quote": "in the Inuktitut language"},   # fabricated source
    ])
    res, cached = en.scan_guide("languages", corpus, mode="quick", use_cache=False, gen=gen)
    assert not cached and res.calls == gen.calls and res.scanned == len(en._indices(corpus, "languages", "quick"))
    names = {i.name: i for i in res.items}
    assert {"Inuktitut", "Michif", "French"} <= set(names) and not ({"Klingon", "Cree", "Elvish"} & set(names))
    assert names["Inuktitut"].exact and names["French"].exact
    assert not names["Michif"].exact and names["Michif"].stated_as == "dialect"        # listed separately, with the guide's word
    assert [i.name for i in res.worded_differently] == ["Michif"] and "Michif" not in [i.name for i in res.exact]
    assert any("Klingon" in o and "not in the cited passage" in o for o in res.omitted)
    assert not any("too short" in o for o in res.omitted)
    assert any(o.startswith("Cree") for o in res.omitted)
    assert any("Elvish" in o for o in res.omitted)
    assert all(e.quote and e.page for i in res.items for e in i.evidence)               # every item carries a verbatim quote and a page


def test_the_quote_must_use_the_kind_or_the_guides_own_word():
    rows, omitted = en._validate([{"name": "Michif", "stated_as": "tongue", "source": "S1", "evidence_quote": "speak their own dialect, Michif"}],
                                 "languages", "", {"S1": type("C", (), {"text": "They speak their own dialect, Michif.", "page": "11", "section": "x", "chunk_id": "c"})()})
    assert rows == [] and "neither" in omitted[0]


def test_the_same_item_from_two_passages_is_merged_with_both_quotes(corpus):
    ev = lambda q, p: en.Evidence(q, p, "s", "c" + p)
    rows = [("Inuktitut", "language", True, ev("in the Inuktitut language", "11")), ("the Inuktitut", "language", True, ev("Inuktitut is an official language", "51")),
            ("Inuktitut language", "language", True, ev("the Inuktitut language is spoken", "51"))]
    items = en._group(rows, "langua")
    assert len(items) == 1 and items[0].name == "Inuktitut" and [e.page for e in items[0].evidence] == ["11", "51", "51"] and items[0].exact


def test_an_item_is_the_requested_kind_if_the_guide_words_any_mention_that_way():
    ev = lambda q: en.Evidence(q, "11", "s", "c")
    once = [("French", "languages", True, ev("the official languages are English and French")), ("French", "", False, ev("French-speaking Canadians"))]
    g = en._group(once, "langua")
    assert len(g) == 1 and g[0].exact and len(g[0].evidence) == 2                  # one explicit mention is enough; both quotes are kept
    never = [("Michif", "dialect", False, ev("their own dialect, Michif")), ("Michif", "dialect", False, ev("Michif is still spoken"))]
    assert not en._group(never, "langua")[0].exact and en._group(never, "langua")[0].stated_as == "dialect"


# --- the peoples stay apart --------------------------------------------------------------------------------------------------------

def test_a_first_nations_focus_excludes_inuit_and_metis_items_and_says_why(corpus):
    res, _ = en.scan_guide("languages", corpus, focus="First Nations", mode="quick", use_cache=False, gen=FakeReader())
    assert "Inuktitut" not in [i.name for i in res.items] and "Michif" not in [i.name for i in res.items]
    assert any("Inuktitut" in o and "not the First Nations" in o for o in res.omitted)
    assert any("Michif" in o and "Métis" in o for o in res.omitted)


def test_an_inuit_focus_keeps_inuktitut_but_not_michif(corpus):
    res, _ = en.scan_guide("languages", corpus, focus="Inuit", mode="quick", use_cache=False, gen=FakeReader())
    assert "Inuktitut" in [i.name for i in res.items] and "Michif" not in [i.name for i in res.items]


# --- cost, caching, progress ------------------------------------------------------------------------------------------------------

def test_results_are_cached_and_a_cache_hit_makes_no_calls(corpus):
    gen = FakeReader()
    first, c1 = en.scan_guide("languages", corpus, mode="quick", gen=gen)
    n = gen.calls
    again, c2 = en.scan_guide("languages", corpus, mode="quick", gen=gen)
    assert not c1 and c2 and gen.calls == n and again == first


def test_progress_is_reported(corpus):
    seen = []
    res, _ = en.scan_guide("languages", corpus, mode="quick", use_cache=False, gen=FakeReader(), progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (res.calls, res.calls) and len(seen) == res.calls


def test_an_empty_request_is_refused_before_any_call(corpus):
    with pytest.raises(ValueError):
        en.scan_guide("the list of", corpus, gen=FakeReader())


# --- separation from ordinary Q&A and from the benchmarked core -------------------------------------------------------------------------

def test_enumeration_is_not_reachable_from_the_answer_path_or_the_frozen_core():
    for f in ("src/tutor.py", "src/retrieval.py", "src/embeddings.py", "src/corpus.py", "src/visualization.py", "app.py", "reference_rag.py"):
        assert "enumeration" not in (ROOT / f).read_text(), f
    assert "retrieve(" not in (ROOT / "src/enumeration.py").read_text().replace("retrieve(…", "")        # it reads passages; it never queries the retriever


def test_a_quote_that_is_too_short_is_refused_with_that_reason():
    chunk = type("C", (), {"text": "French and English are the official languages.", "page": "5", "section": "x", "chunk_id": "c"})()
    rows, omitted = en._validate([{"name": "French", "stated_as": "languages", "source": "S1", "evidence_quote": "official languages"}], "languages", "", {"S1": chunk})
    assert rows == [] and "too short" in omitted[0]


def test_exactness_comes_from_the_quote_not_from_the_models_label():
    chunk = type("C", (), {"text": "English and French are Canada's official languages. They speak their own dialect, Michif.", "page": "5", "section": "x", "chunk_id": "c"})()
    rows, omitted = en._validate([
        {"name": "French", "stated_as": "english", "source": "S1", "evidence_quote": "English and French are Canada's official languages"},   # sloppy label, quote is right
        {"name": "Michif", "stated_as": "dialect", "source": "S1", "evidence_quote": "speak their own dialect, Michif"},                       # the guide's own word, in the quote
        {"name": "Michif", "stated_as": "language", "source": "S1", "evidence_quote": "They speak their own dialect, Michif."},               # claims "language"; the quote says "dialect"
    ], "languages", "", {"S1": chunk})
    got = {(r[0], r[2], r[1]) for r in rows}
    assert ("French", True, "languages") in got                                   # exact, decided from the quote
    assert ("Michif", False, "dialect") in got                                    # worded differently, with the guide's own word
    assert len(rows) == 2 and any("neither" in o for o in omitted)                # the mislabelled one is dropped, not guessed at


def test_extraction_runs_at_temperature_zero(corpus):
    gen = FakeReader()
    en.scan_guide("languages", corpus, mode="quick", use_cache=False, gen=gen)
    assert gen.temperatures and set(gen.temperatures) == {0.0}
