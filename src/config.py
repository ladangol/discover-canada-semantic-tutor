"""Central configuration. Retrieval settings here are FROZEN (see AGENTS.md)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Runtime-only: values are read from the process environment, never printed.
load_dotenv(ROOT / ".env")

# --- Frozen retrieval configuration -----------------------------------------
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "discover-canada-window-minilm"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EXPECTED_DISTANCE_SPACE = "cosine"
DEFAULT_K = 5

# --- Visualization artifacts -------------------------------------------------
ARTIFACTS_DIR = ROOT / "artifacts"
EMBEDDING_MAP_PATH = ARTIFACTS_DIR / "embedding_map.json"
UMAP_REDUCER_PATH = ARTIFACTS_DIR / "umap_reducer.joblib"
EMBEDDING_MAP_3D_PATH = ARTIFACTS_DIR / "embedding_map_3d.json"
UMAP_REDUCER_3D_PATH = ARTIFACTS_DIR / "umap_reducer_3d.joblib"
UMAP_PARAMS = {
    "n_components": 2,
    "metric": "cosine",
    "n_neighbors": 15,
    "min_dist": 0.1,
    "random_state": 42,
}

UMAP_PARAMS_3D = {**UMAP_PARAMS, "n_components": 3}

# --- Generation (Venice, OpenAI-compatible) ---------------------------------
DEFAULT_GENERATION_MODEL = "zai-org-glm-4.7-flash"
VENICE_BASE_URL_DEFAULT = "https://api.venice.ai/api/v1"
NOT_CONFIGURED_MESSAGE = "VENICE_API_KEY is not configured."


def generation_model() -> str:
    return os.environ.get("VENICE_MODEL") or DEFAULT_GENERATION_MODEL


def venice_base_url() -> str:
    return os.environ.get("VENICE_BASE_URL") or VENICE_BASE_URL_DEFAULT


def generation_configured() -> bool:
    """True/False only — the key itself is never returned or displayed."""
    return bool(os.environ.get("VENICE_API_KEY"))
