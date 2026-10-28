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

# --- Learning visualizations (NOT the frozen RAG answer config) --------------
# Candidate retrieval to gather evidence for hubs/timelines; separate from DEFAULT_K.
HUB_CANDIDATES_K = 12
TIMELINE_CANDIDATES_K = 12
TIMELINE_BATCH_CHUNKS = 10      # chunks per extraction call when a whole chapter/section is used
LEARNING_PROMPT_VERSION = "v3"  # bump to invalidate cached generated objects
HUB_PROMPT_VERSION = "v4"       # concept hubs only (typed entities + validated relationships); timelines keep v3 and their cache

# --- Visualization artifacts -------------------------------------------------
ARTIFACTS_DIR = ROOT / "artifacts"
GENERATED_DIR = ARTIFACTS_DIR / "generated"
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
DEFAULT_GENERATION_MODEL = "z-ai-glm-5-3-flash"
VENICE_BASE_URL_DEFAULT = "https://api.venice.ai/api/v1"
NOT_CONFIGURED_MESSAGE = "VENICE_API_KEY is not configured."


def generation_model() -> str:
    return os.environ.get("VENICE_MODEL") or DEFAULT_GENERATION_MODEL


def venice_base_url() -> str:
    return os.environ.get("VENICE_BASE_URL") or VENICE_BASE_URL_DEFAULT


def generation_configured() -> bool:
    """True/False only — the key itself is never returned or displayed."""
    return bool(os.environ.get("VENICE_API_KEY"))


# --- Voice tutor (Milestone 1; local ASR/TTS, see docs/VOICE_ARCHITECTURE.md) ------------------------------
# Everything is read from the environment at call time. The page is hidden unless STUDY_VOICE=1.
def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


def voice_enabled() -> bool:
    return os.environ.get("STUDY_VOICE", "0") == "1"


def voice_diagnostics() -> bool:
    """Show the per-stage timing panel on the Talk page (developers only)."""
    return os.environ.get("STUDY_VOICE_DIAGNOSTICS", "0") == "1"


ASR_MODEL_DEFAULT = "small"
ASR_INITIAL_PROMPT_DEFAULT = ("Discover Canada. Sir John A. Macdonald, Louis-Hippolyte La Fontaine, Laura Secord, "
                              "James FitzGibbon, Tecumseh, Nunavut, Iqaluit, Confederation, Louis Riel, Métis, Michif, Inuktitut, Inuit, "
                              "Mi'kmaq, Haida Gwaii, Champlain, Cartier, Laurier, RCMP, P.E.I.")


def asr_settings() -> dict:
    return {"model": os.environ.get("ASR_MODEL") or ASR_MODEL_DEFAULT,
            "device": os.environ.get("ASR_DEVICE") or "cpu",
            "compute_type": os.environ.get("ASR_COMPUTE_TYPE") or "int8",
            "language": os.environ.get("ASR_LANGUAGE") or "en",
            "initial_prompt": os.environ.get("ASR_INITIAL_PROMPT", ASR_INITIAL_PROMPT_DEFAULT) or None}


def tts_voice_path() -> Path | None:
    p = os.environ.get("TTS_VOICE_PATH")
    return Path(p).expanduser() if p else None


def voice_limits() -> dict:
    return {"max_seconds": _env_int("VOICE_MAX_SECONDS", 30), "max_bytes": _env_int("VOICE_MAX_BYTES", 3_000_000),
            "concurrency": _env_int("VOICE_MAX_CONCURRENCY", 1)}


def tts_settings() -> dict:
    """Piper voice character. None = use the voice's own defaults. Lower noise_scale = steadier; length_scale > 1 = slower;
    pause_ms = silence between sentences (Piper speaks sentence by sentence)."""
    def f(name):
        try:
            return float(os.environ[name]) if os.environ.get(name) else None
        except ValueError:
            return None
    try:
        pause = max(0, int(os.environ["TTS_SENTENCE_PAUSE_MS"])) if os.environ.get("TTS_SENTENCE_PAUSE_MS") else None   # None = engine default
    except ValueError:
        pause = None
    return {"length_scale": f("TTS_LENGTH_SCALE"), "noise_scale": f("TTS_NOISE_SCALE"), "noise_w_scale": f("TTS_NOISE_W"),
            "pause_ms": pause}


def tts_engine() -> str:
    """'kokoro' (default, PyTorch), 'kokoro-onnx' (smaller, no torch; see docs) or 'piper'."""
    e = (os.environ.get("TTS_ENGINE") or "kokoro").lower()
    return e if e in ("piper", "kokoro-onnx") else "kokoro"


def kokoro_onnx_paths() -> tuple[Path, Path]:
    """Model (int8 ≈ 92 MB) and voices file for TTS_ENGINE=kokoro-onnx, from the kokoro-onnx 'model-files-v1.0' release."""
    d = ROOT / "models" / "kokoro-onnx"
    return (Path(os.environ.get("KOKORO_ONNX_MODEL") or d / "kokoro-v1.0.int8.onnx").expanduser(),
            Path(os.environ.get("KOKORO_ONNX_VOICES") or d / "voices-v1.0.bin").expanduser())


def kokoro_settings() -> dict:
    """Kokoro-82M voice id (af_* / am_* American, bf_* / bm_* British) and speaking speed (1.0 = normal)."""
    try:
        speed = float(os.environ.get("KOKORO_SPEED") or 1.0)
    except ValueError:
        speed = 1.0
    return {"voice": os.environ.get("KOKORO_VOICE") or "af_heart", "speed": min(max(speed, 0.6), 1.4)}


def query_cleanup_enabled() -> bool:
    """Application-layer retrieval-query cleanup (src/query_understanding.py) for the learner/voice flows. QUERY_CLEANUP=0 turns it off."""
    return os.environ.get("QUERY_CLEANUP", "1") != "0"
