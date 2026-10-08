"""Frozen plain dense retrieval (Variant A, k=5).

Mechanics are copied from reference_rag.retrieve(): encode the question with
normalized MiniLM, run Chroma's cosine query, similarity = 1 - distance.
No thresholds, no re-ranking, no hybrid search.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import chromadb
import numpy as np

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_K,
    EMBEDDING_MODEL_NAME,
    EXPECTED_DISTANCE_SPACE,
)
from src.embeddings import encode_query


@dataclass(frozen=True)
class RetrievedChunk:
    """A source passage. `distance`/`similarity` are None for passages that
    were selected from the corpus directly (e.g. a whole section) rather than
    retrieved by a query."""

    source_id: str
    chunk_id: str
    text: str
    page: str
    section: str
    block_type: str
    distance: float | None = None
    similarity: float | None = None


def page_label(page: str) -> str:
    if "-" in page:
        start, end = page.split("-", 1)
        return f"pp. {start}–{end}"
    return f"p. {page}"


def _collection_distance_space(collection) -> str:
    config: Any = collection.configuration
    if not isinstance(config, dict):
        config = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    hnsw = config.get("hnsw")
    if isinstance(hnsw, dict) and isinstance(hnsw.get("space"), str):
        return hnsw["space"]
    raise RuntimeError(f"Could not determine distance space for {collection.name!r}.")


@lru_cache(maxsize=1)
def retrieval_collection():
    """Open the persisted collection once and refuse to run if it is not the
    cosine / MiniLM index that `1 - distance == cosine similarity` assumes."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME, embedding_function=None)

    space = _collection_distance_space(collection)
    if space != EXPECTED_DISTANCE_SPACE:
        raise RuntimeError(
            f"Collection uses distance space {space!r}, expected "
            f"{EXPECTED_DISTANCE_SPACE!r}."
        )

    metadatas = collection.get(include=["metadatas"])["metadatas"]
    models = {(m or {}).get("embedding_model") for m in metadatas}
    if models != {EMBEDDING_MODEL_NAME}:
        raise RuntimeError(
            f"Collection declares embedding model(s) {sorted(map(str, models))}, "
            f"expected {EMBEDDING_MODEL_NAME!r}."
        )
    return collection


def retrieve(
    question: str,
    k: int = DEFAULT_K,
    *,
    query_vector: np.ndarray | None = None,
) -> list[RetrievedChunk]:
    """Top-k chunks by cosine similarity.

    `query_vector` lets a caller that already embedded the question (the atlas
    also projects it through UMAP) reuse that embedding. It must come from
    `embeddings.encode_query`.
    """
    collection = retrieval_collection()
    vector = encode_query(question) if query_vector is None else query_vector

    result = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    rows = zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    )
    for rank, (chunk_id, document, metadata, distance) in enumerate(rows, start=1):
        distance = float(distance)
        chunks.append(
            RetrievedChunk(
                source_id=f"S{rank}",
                chunk_id=chunk_id,
                text=document,
                page=str(metadata["page"]),
                section=str(metadata["section"]),
                block_type=str(metadata["block_type"]),
                distance=distance,
                similarity=1.0 - distance,
            )
        )
    return chunks
