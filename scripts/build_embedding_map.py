"""Fit UMAP on the stored Chroma embeddings and save the atlas artifacts.

    python scripts/build_embedding_map.py

Writes artifacts/embedding_map.json (2-D coordinates + metadata) and
artifacts/umap_reducer.joblib (the fitted reducer, so queries can be
projected into the same space). Nothing is re-embedded.

UMAP is for display only; it is never used for retrieval or similarity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import umap

from src.config import (
    ARTIFACTS_DIR,
    EMBEDDING_MAP_PATH,
    EMBEDDING_MODEL_NAME,
    UMAP_PARAMS,
    UMAP_REDUCER_PATH,
)
from src.corpus import load_corpus


def build() -> None:
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} chunks, embeddings {corpus.embeddings.shape}")

    reducer = umap.UMAP(**UMAP_PARAMS)
    coords = reducer.fit_transform(corpus.embeddings)

    points = corpus.chunks[
        ["chunk_id", "page", "section", "block_type", "chapter", "preview"]
    ].copy()
    points["x"] = coords[:, 0].astype(float)
    points["y"] = coords[:, 1].astype(float)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    payload = {
        "meta": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "umap_params": UMAP_PARAMS,
            "n_chunks": len(points),
            "note": "UMAP coordinates are for visualization only.",
        },
        "points": points.to_dict(orient="records"),
    }
    EMBEDDING_MAP_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    joblib.dump(reducer, UMAP_REDUCER_PATH)
    print(f"Wrote {EMBEDDING_MAP_PATH.name} and {UMAP_REDUCER_PATH.name} in {ARTIFACTS_DIR}")


if __name__ == "__main__":
    build()
