"""'Did you mean…' questions drawn from a pre-built bank of questions the guide is known to answer.

`scripts/build_suggestion_bank.py` (run once, offline) writes 2 questions per passage, generated from that passage's own text and kept
only if the frozen retrieval returns the passage for its own question. So every bank question is answerable by construction.
At runtime an LLM guesses what the learner meant (it understands names and dates, which MiniLM does not), and `snap` replaces each
guess with the nearest bank question by embedding, so the learner only ever sees questions the guide answers. Nothing here ranks
passages or changes retrieval; it only looks up questions."""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np

from src.config import ARTIFACTS_DIR
from src.embeddings import encode_query, get_encoder

BANK_PATH = ARTIFACTS_DIR / "suggestion_bank.json"
SNAP_MIN_SIMILARITY = 0.70     # a guess whose nearest bank question is less similar than this is dropped (the guide has nothing close)
MAX_SUGGESTIONS = 3


@lru_cache(maxsize=1)
def load_bank() -> list[dict]:
    try:
        return json.loads(BANK_PATH.read_text())["questions"]
    except (OSError, ValueError, KeyError):
        return []


def _embed(texts: list[str]) -> np.ndarray:
    return get_encoder().encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


@lru_cache(maxsize=1)
def _matrix() -> np.ndarray:
    bank = load_bank()
    return _embed([b["q"] for b in bank]) if bank else np.zeros((0, 384), np.float32)


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def snap(candidates: list[str], asked: str = "", n: int = MAX_SUGGESTIONS, min_similarity: float = SNAP_MIN_SIMILARITY) -> list[str]:
    """Replace each guessed question with the nearest bank question (cosine on MiniLM), best matches first. Drops guesses with
    nothing close in the bank, repeats, questions from the same section, and the learner's own question."""
    bank, m = load_bank(), _matrix()
    cands = [c for c in candidates if (c or "").strip()]
    if not bank or not cands:
        return []
    sims = _embed(cands) @ m.T                                   # candidates x bank
    ranked = sorted(((float(sims[i].max()), int(sims[i].argmax())) for i in range(len(cands))), reverse=True)
    out, seen_q, seen_sections = [], set(), set()
    for score, j in ranked:
        b = bank[j]
        if score < min_similarity or len(out) >= n:
            break
        if j in seen_q or b["section"] in seen_sections or _norm(b["q"]) == _norm(asked):
            continue
        seen_q.add(j), seen_sections.add(b["section"])
        out.append(b["q"])
    return out


_YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "right", "please", "exactly", "that", "this", "those", "one", "it", "i", "meant",
        "mean", "the", "thats", "that's", "is", "was", "did", "do", "go", "ahead", "tell", "me", "about", "more", "so"}
_ORDINALS = {"first": 0, "1st": 0, "1": 0, "one": 0, "second": 1, "2nd": 1, "2": 1, "two": 1, "third": 2, "3rd": 2, "3": 2, "three": 2}


def pick_offered(message: str, offered: list[str]) -> str | None:
    """If `message` is just an answer to "Did you mean one of these?" (yes / that one / the second one / I meant that), return the
    offered question it points at; otherwise None. Rule-based: a longer message is a new question and is never touched."""
    if not offered:
        return None
    words = re.sub(r"[^a-z0-9' ]", " ", (message or "").lower()).split()
    if not words or len(words) > 8:
        return None
    for i, w in enumerate(words):                               # "the second one", "number 2", "the last one"
        if w in ("last", "final") and "one" in words:
            return offered[-1]
        if w in _ORDINALS and (w not in ("one", "1", "2", "3", "two", "three") or "number" in words or "option" in words):
            return offered[_ORDINALS[w]] if _ORDINALS[w] < len(offered) else None
        if w in ("first", "second", "third", "1st", "2nd", "3rd") and _ORDINALS[w] < len(offered):
            return offered[_ORDINALS[w]]
    if all(w in _YES for w in words) and any(w in ("yes", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "right", "that", "this", "those", "exactly", "please") for w in words):
        return offered[0]                                       # the closest guess comes first
    return None
