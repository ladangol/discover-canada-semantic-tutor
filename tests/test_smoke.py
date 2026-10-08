"""Smoke tests mirroring the build-order validation. None need a Venice key."""
from __future__ import annotations

import json
import random
import re
import sys

import numpy as np
import pytest

from src import tutor
from src.config import (
    DEFAULT_K,
    EMBEDDING_DIM,
    EMBEDDING_MAP_PATH,
    NOT_CONFIGURED_MESSAGE,
    ROOT,
    UMAP_REDUCER_PATH,
)
from src.corpus import derive_chapter, load_corpus
from src.embeddings import encode_query
from src.generation import GenerationNotConfigured, GenerationResult, generate
from src.retrieval import RetrievedChunk, retrieve, retrieval_collection

QUESTION = "What happened during the First World War?"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(autouse=True)
def no_venice_key(monkeypatch):
    """Everything below must work with generation unconfigured."""
    monkeypatch.delenv("VENICE_API_KEY", raising=False)


# 1-4: Chroma loads, count, embeddings retrievable, dimension --------------------

def test_chroma_loads_and_matches_reference_chunks(corpus):
    assert retrieval_collection().count() == len(corpus) == 236
    reference = json.loads((ROOT / "chunks_window.json").read_text())
    assert set(corpus.chunks["chunk_id"]) == {c["chunk_id"] for c in reference}


def test_embeddings_have_expected_shape_and_are_normalized(corpus):
    assert corpus.embeddings.shape == (len(corpus), EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(corpus.embeddings, axis=1), 1.0, atol=1e-4)


# 5-6: frozen dense query at k=5, identical to the reference ---------------------

def test_dense_retrieval_k5_matches_reference_rag(monkeypatch):
    monkeypatch.chdir(ROOT)  # reference_rag opens the relative path "chroma_db"
    sys.path.insert(0, str(ROOT))
    import reference_rag

    ours = retrieve(QUESTION)
    theirs = reference_rag.retrieve(QUESTION, k=5)
    assert len(ours) == DEFAULT_K == 5
    assert [c.chunk_id for c in ours] == [c.chunk_id for c in theirs]
    assert [round(c.similarity, 6) for c in ours] == [round(c.similarity, 6) for c in theirs]
    sims = [c.similarity for c in ours]
    assert sims == sorted(sims, reverse=True)


# 7-8: UMAP artifacts and query transform ----------------------------------------

def test_umap_artifacts_and_query_projection(corpus):
    if not (EMBEDDING_MAP_PATH.exists() and UMAP_REDUCER_PATH.exists()):
        sys.path.insert(0, str(ROOT / "scripts"))
        import build_embedding_map

        build_embedding_map.build()

    from src import visualization as viz

    atlas = viz.load_map()
    assert len(atlas) == len(corpus)
    assert {"chunk_id", "x", "y", "page", "section", "block_type", "chapter", "preview"} <= set(atlas.columns)

    view = viz.analyze_query(QUESTION)
    assert len(view.chunks) == 5 and np.isfinite(view.xy).all()
    # The retrieved chunks come from the original embedding, not the 2-D map.
    assert [c.chunk_id for c in view.chunks] == [c.chunk_id for c in retrieve(QUESTION)]


# corpus helpers -------------------------------------------------------------------

def test_chapter_derivation():
    assert derive_chapter("Canada’s History\nThe First World War\n\nbody", "The First World War") == "Canada’s History"
    assert derive_chapter("Understanding the Oath\n\nbody", "Understanding the Oath") == "Understanding the Oath"


def test_neighbours_use_cosine_on_original_embeddings(corpus):
    cid = corpus.chunks.chunk_id.iat[10]
    got = corpus.neighbours(cid, 5)
    sims = corpus.embeddings @ corpus.embeddings[10]
    sims[10] = -np.inf
    assert [j for j, _ in got] == list(np.argsort(-sims)[:5])
    other = corpus.neighbours(cid, 5, exclude_same_section=True)
    assert all(corpus.chunks.section.iat[j] != corpus.chunks.section.iat[10] for j, _ in other)


def test_neighbourhood_has_distinct_sections(corpus):
    cid = corpus.chunks[corpus.chunks.section == "Responsible government"].chunk_id.iloc[0]
    sections = [corpus.chunks.section.iat[j] for j in tutor.neighbourhood(corpus, cid, 5)]
    assert len(sections) == len(set(sections)) == 5


def test_mmr_is_diverse_and_unique(corpus):
    pool = corpus.scope_indices("Canada’s History")
    picked = tutor.mmr_select(corpus.embeddings, pool, 5)
    assert len(picked) == len(set(picked)) == 5
    centroid = corpus.embeddings[pool].mean(axis=0)
    top_relevant = sorted(pool, key=lambda i: -corpus.embeddings[i] @ centroid)[:5]

    def mean_pairwise(idx):
        m = corpus.embeddings[idx] @ corpus.embeddings[idx].T
        return (m.sum() - len(idx)) / (len(idx) * (len(idx) - 1))

    assert mean_pairwise(picked) < mean_pairwise(top_relevant)


# generation is optional ---------------------------------------------------------------

def test_missing_key_reports_only_a_generic_message():
    with pytest.raises(GenerationNotConfigured) as err:
        generate([{"role": "user", "content": "hi"}])
    assert str(err.value) == NOT_CONFIGURED_MESSAGE


def test_retrieval_works_without_generation_key():
    assert len(retrieve("How does a bill become law?")) == 5


# tutor logic with a fake LLM ------------------------------------------------------------

def test_citations_are_resolved_by_code():
    chunks = [
        RetrievedChunk("S1", "a", "t", "21", "The First World War", "body"),
        RetrievedChunk("S2", "b", "t", "21", "The First World War", "body"),
    ]
    text, cited = tutor.resolve_citations("Fact [S1][S2]. Bad [S9].", chunks)
    assert text == "Fact [Discover Canada, p. 21, The First World War]. Bad [S9]."
    assert [c.source_id for c in cited] == ["S1", "S2"]


class FakeLLM:
    """Writes a question whose correct answer is 'RIGHT'; the verifier reports
    whichever options `supported_texts` names."""

    def __init__(self, supported_texts=("RIGHT",)):
        self.supported_texts = supported_texts
        self.prompts: list[str] = []

    def __call__(self, messages, **_):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "list every option" in prompt:
            options = dict(re.findall(r"^([A-D])\. (.+)$", prompt, flags=re.M))
            letters = [k for k, v in options.items() if v in self.supported_texts]
            return GenerationResult(json.dumps({"supported": letters}), "stop", 1, 1)
        reply = {"question": "Q?", "correct_answer": "RIGHT",
                 "distractors": ["w1", "w2", "w3"], "explanation": "because"}
        return GenerationResult(json.dumps(reply), "stop", 1, 1)


def test_mcq_has_one_verified_answer_and_semantic_distractor_context(corpus):
    fake = FakeLLM()
    q = tutor.generate_quiz(corpus, kind="mcq", section="Responsible government",
                            rng=random.Random(0), gen=fake)
    assert q.options[q.correct_index] == "RIGHT" and len(q.options) == 4
    assert q.source.section == "Responsible government"
    # distractor candidates from embedding neighbours were handed to the LLM
    assert "NEARBY CONCEPTS" in fake.prompts[0] and q.distractor_sections
    assert tutor.grade_mcq(q, q.correct_index).correct
    assert not tutor.grade_mcq(q, (q.correct_index + 1) % 4).correct


def test_mcq_rejected_when_not_uniquely_supported(corpus):
    fake = FakeLLM(supported_texts=("RIGHT", "w1"))
    with pytest.raises(tutor.QuizGenerationError):
        tutor.generate_quiz(corpus, kind="mcq", section="Confederation", gen=fake)


def test_adaptive_review_targets_neighbours_not_the_missed_chunk(corpus):
    missed = corpus.chunks[corpus.chunks.section == "Responsible government"].chunk_id.iloc[0]
    history = [tutor.AnswerRecord(missed, "Responsible government", "q", False)]
    cands = tutor.review_candidates(corpus, missed, history)
    assert cands and corpus.index_of(missed) not in cands
    assert tutor.weak_sections(history) == [("Responsible government", 1)]
    q = tutor.next_review_question(corpus, missed, history, rng=random.Random(0), gen=FakeLLM())
    assert corpus.index_of(q.chunk_id) in cands
