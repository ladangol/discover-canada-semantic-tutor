"""Chapter-level semantic statistics from the ORIGINAL 384-d embeddings.

No LLM and no UMAP: centroids, cosine similarity and breadth are computed on the
stored MiniLM vectors. These describe closeness in embedding space only, not
historical causation or dependence.
"""
from __future__ import annotations

import numpy as np

from src.corpus import Corpus
from src.learning_models import ChapterAnalysis, ChapterSemanticStats


def analyze_chapters(corpus: Corpus, n_nearest: int = 3) -> ChapterAnalysis:
    chapters = corpus.chapters
    emb = corpus.embeddings
    owner = corpus.chunks["chapter"].to_numpy()

    centroids = np.stack([emb[owner == c].mean(axis=0) for c in chapters])
    units = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    sim = units @ units.T

    # nearest chunk (excluding itself) -> is it in the same chapter?
    chunk_sim = emb @ emb.T
    np.fill_diagonal(chunk_sim, -np.inf)
    nearest_owner = owner[np.argmax(chunk_sim, axis=1)]

    stats = []
    for i, chapter in enumerate(chapters):
        mine = owner == chapter
        breadth = max(0.0, float(np.mean(1.0 - emb[mine] @ units[i])))
        order = [j for j in np.argsort(-sim[i]) if j != i][:n_nearest]
        stats.append(ChapterSemanticStats(
            chapter=chapter,
            n_chunks=int(mine.sum()),
            words=int(corpus.chunks.loc[mine, "word_count"].sum()),
            breadth=breadth,
            same_chapter_nn_rate=float(np.mean(nearest_owner[mine] == chapter)),
            nearest=[(chapters[j], float(sim[i, j])) for j in order],
        ))
    return ChapterAnalysis(chapters=chapters, similarity=sim.tolist(), stats=stats)


def mds_layout(similarity: np.ndarray) -> np.ndarray:
    """2-D positions for the chapter network via classical MDS on cosine distance
    (1 - centroid cosine). Positions are approximate; the edges carry the numbers."""
    d2 = (1.0 - similarity) ** 2
    n = len(d2)
    j = np.eye(n) - np.ones((n, n)) / n
    vals, vecs = np.linalg.eigh(-0.5 * j @ d2 @ j)
    top = np.argsort(-vals)[:2]
    return vecs[:, top] * np.sqrt(np.maximum(vals[top], 0.0))


def strongest_edges(similarity: np.ndarray, per_chapter: int) -> dict[tuple[int, int], float]:
    """Display rule: each chapter keeps its N strongest links (union, undirected)."""
    edges: dict[tuple[int, int], float] = {}
    for i in range(len(similarity)):
        row = similarity[i].copy()
        row[i] = -np.inf
        for j in np.argsort(-row)[:per_chapter]:
            edges[(min(i, int(j)), max(i, int(j)))] = float(similarity[i, j])
    return edges
