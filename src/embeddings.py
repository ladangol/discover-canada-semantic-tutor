"""Query embedding with the frozen MiniLM encoder (corpus vectors come from Chroma)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def encode_query(text: str) -> np.ndarray:
    """Return one L2-normalized 384-d float32 vector (same call as the reference)."""
    vector = get_encoder().encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vector.astype(np.float32)
