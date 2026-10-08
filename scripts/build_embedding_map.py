"""Fit UMAP on the stored Chroma embeddings and save the atlas artifacts.

    python scripts/build_embedding_map.py

Writes artifacts/embedding_map.json + umap_reducer.joblib (2-D, for the Plotly
atlas) and embedding_map_3d.json + umap_reducer_3d.joblib (3-D, for the three.js
atlas); the fitted reducers let queries be projected into the same spaces. Nothing is re-embedded.

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
    EMBEDDING_MAP_3D_PATH,
    EMBEDDING_MAP_PATH,
    EMBEDDING_MODEL_NAME,
    UMAP_PARAMS,
    UMAP_PARAMS_3D,
    UMAP_REDUCER_3D_PATH,
    UMAP_REDUCER_PATH,
)
from src.corpus import load_corpus


def _fit_and_save(corpus, params: dict, map_path: Path, reducer_path: Path, axes: list[str]) -> None:
    reducer = umap.UMAP(**params)
    coords = reducer.fit_transform(corpus.embeddings)

    points = corpus.chunks[
        ["chunk_id", "page", "section", "block_type", "chapter", "preview"]
    ].copy()
    for i, axis in enumerate(axes):
        points[axis] = coords[:, i].astype(float)

    payload = {
        "meta": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "umap_params": params,
            "n_chunks": len(points),
            "note": "UMAP coordinates are for visualization only.",
        },
        "points": points.to_dict(orient="records"),
    }
    map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    joblib.dump(reducer, reducer_path)
    print(f"Wrote {map_path.name} and {reducer_path.name}")


def build() -> None:
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} chunks, embeddings {corpus.embeddings.shape}")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    _fit_and_save(corpus, UMAP_PARAMS, EMBEDDING_MAP_PATH, UMAP_REDUCER_PATH, ["x", "y"])
    _fit_and_save(corpus, UMAP_PARAMS_3D, EMBEDDING_MAP_3D_PATH, UMAP_REDUCER_3D_PATH, ["x", "y", "z"])


if __name__ == "__main__":
    build()
