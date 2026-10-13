"""Chapter statistics, evidence validation, extractors and cache. No Venice key needed."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from src import config, learning_cache
from src import learning_extractors as ex
from src import learning_validation as lv
from src.chapter_stats import analyze_chapters, mds_layout, strongest_edges
from src.corpus import load_corpus
from src.generation import GenerationResult
from src.retrieval import RetrievedChunk


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GENERATED_DIR", tmp_path / "generated")
    monkeypatch.delenv("VENICE_API_KEY", raising=False)


# --- chapter statistics (original embeddings only) ---------------------------------------

def test_chapter_statistics_use_original_embeddings(corpus):
    a = analyze_chapters(corpus)
    sim = np.array(a.similarity)
    assert sim.shape == (len(a.chapters),) * 2 and np.allclose(sim, sim.T) and np.allclose(np.diag(sim), 1)
    # recompute one chapter by hand from the stored vectors
    ch = "Canada’s History"
    i = a.chapters.index(ch)
    emb = corpus.embeddings[(corpus.chunks.chapter == ch).to_numpy()]
    centroid = emb.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    assert a.stats[i].breadth == pytest.approx(float(np.mean(1 - emb @ centroid)), abs=1e-6)
    assert a.stats[i].n_chunks == len(emb)
    assert all(0 <= s.breadth <= 1 and 0 <= s.same_chapter_nn_rate <= 1 for s in a.stats)


def test_network_rule_keeps_strongest_links_per_chapter(corpus):
    sim = np.array(analyze_chapters(corpus).similarity)
    edges = strongest_edges(sim, 2)
    for i in range(len(sim)):
        row = sim[i].copy(); row[i] = -np.inf
        for j in np.argsort(-row)[:2]:
            assert (min(i, int(j)), max(i, int(j))) in edges
    assert mds_layout(sim).shape == (len(sim), 2)


# --- validation -----------------------------------------------------------------------------

def chunk(label, text, page="18", section="Confederation"):
    return RetrievedChunk(label, f"id-{label}", text, page, section, "body")


SRC = {"S1": chunk("S1", "Canada’s History\nConfederation\n\nIn 1867 the Fathers of Confederation created two levels "
                          "of government: federal and provincial. Rebellions followed in 1837–38.")}


def event(**kw):
    base = dict(date_label="1867", start_year=1867, end_year=1867, title="Confederation",
                description="Two levels of government were created.", importance="", sources=["S1"],
                evidence_quote="the Fathers of Confederation created two levels of government")
    return base | kw


def test_event_validation_accepts_supported_and_rejects_fabrications():
    good, _, _ = lv.validate_events([event()], SRC)
    assert [e.date_label for e in good] == ["1867"] and good[0].source_chunk_ids == ["id-S1"]
    bad = [
        event(sources=["S9"]),                                   # fabricated source label
        event(evidence_quote="a quote that is not in the source"),
        event(date_label="1868", start_year=1868, end_year=1868),  # year not in source
        event(description="Founded in 1901."),                     # unsupported year in text
    ]
    kept, _, omitted = lv.validate_events(bad, SRC)
    assert kept == [] and len(omitted) == 4


def test_ranges_abbreviated_ranges_and_broad_periods():
    ok, undated, _ = lv.validate_events([
        event(date_label="1837–38", start_year=1837, end_year=1838, title="Rebellions",
              evidence_quote="Rebellions followed in 1837–38"),
        event(date_label="late 1700s", start_year=None, end_year=None, title="Slave trade debate",
              evidence_quote="the Fathers of Confederation created two levels"),
    ], SRC)
    assert ok[0].start_year == 1837 and ok[0].end_year == 1838
    assert len(undated) == 1 and undated[0].start_year is None   # preserved, not given a made-up year


def test_duplicates_merge_and_events_sort_chronologically():
    kept, _, omitted = lv.validate_events([
        event(),
        event(title="The Confederation", sources=["S1"]),
        event(date_label="1837–38", start_year=1837, end_year=1838, title="Rebellions",
              evidence_quote="Rebellions followed in 1837–38"),
    ], SRC)
    assert [e.start_year for e in kept] == [1837, 1867] and any("merged" in o for o in omitted)


def test_hub_validation_drops_unsupported_and_duplicate_nodes():
    raw = [
        {"label": "Provincial government", "relationship": "created", "explanation": "x", "sources": ["S1"],
         "evidence_quote": "created two levels of government federal and provincial"},
        {"label": "Provincial government", "relationship": "again", "explanation": "x", "sources": ["S1"],
         "evidence_quote": "created two levels of government federal and provincial"},
        {"label": "Made up", "relationship": "r", "explanation": "x", "sources": ["S1"], "evidence_quote": "nothing like this"},
        {"label": "Ghost", "relationship": "r", "explanation": "x", "sources": ["S7"], "evidence_quote": "created two levels of government"},
    ]
    nodes, omitted = lv.validate_hub_nodes(raw, "Confederation", SRC, max_nodes=6)
    assert [n.label for n in nodes] == ["Provincial government"] and len(omitted) == 3


# --- extractors with a fake LLM + cache ---------------------------------------------------------

def quote(prompt: str, label: str) -> str:
    text = re.search(rf"SOURCE {label}\n.*?text:\n(.*?)(?:\n\n---|\Z)", prompt, flags=re.S).group(1)
    body = text.split("\n\n", 1)[-1]
    return " ".join(body.split()[:10])


class FakeHubLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, messages, **_):
        self.calls += 1
        prompt = messages[-1]["content"]
        reply = {"central_concept": "Prime Minister", "summary": "A head of government.", "summary_sources": ["S1"],
                 "nodes": [{"label": "Cabinet", "relationship": "leads", "explanation": "Leads it.",
                            "sources": ["S1"], "evidence_quote": quote(prompt, "S1")},
                           {"label": "Invented", "relationship": "r", "explanation": "x",
                            "sources": ["S99"], "evidence_quote": "not there at all"}]}
        return GenerationResult(json.dumps(reply), "stop", 1, 1)


def test_hub_pipeline_retrieves_validates_and_caches(corpus):
    fake = FakeHubLLM()
    hub, cached = ex.build_concept_hub("Prime Minister", corpus, gen=fake)
    assert not cached and [n.label for n in hub.nodes] == ["Cabinet"] and len(hub.omitted) == 1
    assert hub.nodes[0].source_chunk_ids[0] in set(corpus.chunks.chunk_id)   # real chunk id, not "S1"
    again, cached2 = ex.build_concept_hub("Prime Minister", corpus, gen=fake)
    assert cached2 and fake.calls == 1 and again.nodes == hub.nodes
    _, cached3 = ex.build_concept_hub("Prime Minister", corpus, gen=fake, use_cache=False)
    assert not cached3 and fake.calls == 2


def test_cache_key_changes_with_model_and_request(monkeypatch):
    base = {"concept": "x"}
    p1 = learning_cache.cache_path("concept_hub", "x", base | {"model": "a"})
    p2 = learning_cache.cache_path("concept_hub", "x", base | {"model": "b"})
    p3 = learning_cache.cache_path("concept_hub", "x", base | {"model": "a", "max_nodes": 8})
    assert len({p1, p2, p3}) == 3 and p1.parent == config.GENERATED_DIR


def test_json_is_retried_exactly_once():
    replies = iter(["not json", "still not json", '{"nodes": []}'])
    calls = []

    def gen(messages, **_):
        calls.append(1)
        return GenerationResult(next(replies), "stop", 1, 1)

    with pytest.raises(ex.QuizGenerationError):
        ex._json_with_one_retry([{"role": "user", "content": "x"}], gen)
    assert len(calls) == 2


def test_timeline_for_chapter_batches_all_chunks(corpus, monkeypatch):
    monkeypatch.setattr(config, "TIMELINE_BATCH_CHUNKS", 40)
    seen = []

    def gen(messages, **_):
        prompt = messages[-1]["content"]
        seen.append(prompt.count("SOURCE S"))
        return GenerationResult(json.dumps({"events": []}), "stop", 1, 1)

    data, _ = ex.build_timeline(corpus, chapter="Canada’s History", gen=gen)
    assert sum(seen) == int((corpus.chunks.chapter == "Canada’s History").sum()) and data.events == []
