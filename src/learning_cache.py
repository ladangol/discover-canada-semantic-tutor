"""File cache for LLM-generated learning objects (artifacts/generated/).

The key hashes everything that changes the result (kind, request, model id,
prompt version, candidate k). Only generated content is stored, never secrets.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src import config


def cache_path(kind: str, label: str, params: dict[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")[:40] or "untitled"
    return config.GENERATED_DIR / f"{kind}_{slug}_{digest}.json"


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())["object"]
    except (OSError, ValueError, KeyError):
        return None


def save(path: Path, params: dict[str, Any], obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params, "object": obj}, ensure_ascii=False, indent=1))
