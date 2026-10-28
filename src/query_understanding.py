"""Application-layer query understanding. NOT part of the frozen, benchmarked retrieval core.

The benchmarked configuration (Variant A: plain dense, MiniLM, cosine, k=5; `src/retrieval.py`) is untouched. This module builds a
SEPARATE retrieval-query string for the learner/voice question flows:

  * it only DELETES document-reference boilerplate ("in Discover Canada", "according to the guide", "does the guide mention");
  * it never adds, replaces or rewrites words. In particular "First Nations", "Aboriginal", "Indigenous", "Métis" and "Inuit" are
    different terms and are left exactly as the learner wrote them (no synonym or category mapping);
  * the exact original question still goes to the answering model, so the answer is about what the learner asked.

Why: embedding "...mentioned in Discover Canada?" pulls in generic passages about the guide and about languages, because those words
carry weight in a short query. Removing them lets the topical words decide the search. If too little would remain, the original is used.
Switch off with QUERY_CLEANUP=0. The research Atlas (app.py) does not use this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# "the guide", "this guide", "the study guide", "Discover Canada", "the official Discover Canada study guide", "the booklet" ...
_DOC = r"(?:(?:the|this|that)\s+)?(?:official\s+|citizenship\s+|study\s+)*(?:discover\s+canada(?:\s+(?:study\s+)?guide)?|guide|booklet)"

_PATTERNS = [
    # "according to the guide," / "based on Discover Canada," / "as described in the guide"
    re.compile(rf"(?:,\s*)?\b(?:according\s+to|based\s+on|as\s+per|per|as\s+(?:stated|described|mentioned|written|explained|outlined|discussed)\s+in)\s+{_DOC}\b\s*,?", re.I),
    # "in the guide", "in Discover Canada", "from the study guide"
    re.compile(rf"\b(?:in|from|within|inside|throughout|across)\s+{_DOC}\b", re.I),
    # "does Discover Canada mention" / "does the guide say" (a following "about …" is kept)
    re.compile(rf"\b(?:does|do|did)\s+{_DOC}\s+(?:say|state|mention|discuss|cover|describe|explain|teach|tell(?:\s+us)?|talk|speak)\b", re.I),
    # "…dialects Discover Canada talks about?"
    re.compile(rf"\b{_DOC}\s+(?:talks?|speaks?|says?|mentions?|discusses?|covers?|describes?|states?|tells?)\s+(?:about|of)\b", re.I),
]

_STOP = {"the", "and", "for", "are", "was", "were", "who", "what", "when", "where", "why", "how", "does", "did", "that", "this", "about",
         "which", "with", "from", "into", "its", "has", "have", "can", "you", "tell", "give", "any", "some"}
MIN_CONTENT_WORDS = 1


@dataclass(frozen=True)
class RetrievalQuery:
    original: str          # exactly what the learner asked (this is what generation sees)
    text: str              # what is embedded for search (== original when nothing was removed)
    removed: tuple[str, ...] = ()


def _tidy(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\s+([?.!,;:])", r"\1", t)
    t = re.sub(r",\s*([?.!])", r"\1", t)
    return t.lstrip(" ,;:").strip()


def _content_words(text: str) -> int:
    return sum(1 for w in re.findall(r"[\wÀ-ÿ'-]+", text.lower()) if len(w) > 2 and w not in _STOP)


def retrieval_query(question: str) -> RetrievalQuery:
    original = question or ""
    text, removed = original, []
    for pattern in _PATTERNS:
        for m in pattern.finditer(text):
            removed.append(m.group(0).strip(" ,"))
        text = pattern.sub(" ", text)
    cleaned = _tidy(text)
    if not removed or _content_words(cleaned) < MIN_CONTENT_WORDS:          # nothing to remove, or too little would be left
        return RetrievalQuery(original, original)
    return RetrievalQuery(original, cleaned, tuple(removed))
