"""Fix speech-recognition mistakes on Canadian proper names ("Papino" -> "Papineau") after transcription. Rule-based, no model, no network.

Only a capitalised word that is NOT an ordinary word and sounds like a known name is changed:
  * targets are a curated list of names learners ask about (some are not in the guide on purpose: the tutor then says so, instead of answering
    about a stranger called "Papino") plus distinctive long names from the guide; ANY word the guide itself uses is protected and never changed;
  * "sounds like" = the same consonant skeleton, the same first letter and similar spelling (Papino ~ Papineau); similar spelling alone never
    counts ("Prince" is not changed to "Princess"); ties are left alone;
  * ordinary words (any word the guide uses in lower case, plus a small stoplist) are never touched, nor are words already in the vocabulary.
The learner always sees the corrected text in an editable box, and the original recognised text is kept separately.
"""
from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

from src.config import ROOT

# Names learners ask about. Not limited to the guide.
CURATED = [
    "Papineau", "Mackenzie", "Baldwin", "La Fontaine", "Brock", "Secord", "Tecumseh", "Riel", "Dumont", "Champlain", "Cartier", "Laurier", "Macdonald",
    "Trudeau", "Mulroney", "Diefenbaker", "Pearson", "Borden", "Harper", "Mackenzie King", "St. Laurent", "Douglas", "McClung", "Banting", "Naismith",
    "Frobisher", "Cabot", "Hudson", "Iqaluit", "Nunavut", "Saskatchewan", "Saskatoon", "Manitoba", "Winnipeg", "Charlottetown", "Fredericton", "Yellowknife",
    "Whitehorse", "Regina", "Edmonton", "Michif", "Inuktitut", "Iroquoian", "Algonquin", "Mohawk", "Haida", "Gwaii", "Mi'kmaq", "Haudenosaunee",
    "Acadians", "Loyalists", "Canadiens", "Confederation", "Quebec", "Québec", "Ottawa", "Nova Scotia", "Newfoundland", "Labrador",
]
_STOP_COMMON = {"canada", "canadian", "canadians", "english", "french", "ottawa", "quebec"}     # real words/places the learner may say normally
# everyday names and titles that must never be "corrected" into a Canadian name
_DONT_TOUCH = {"wales", "welsh", "england", "scotland", "ireland", "britain", "british", "france", "america", "american", "europe", "asia", "africa",
               "india", "china", "japan", "germany", "russia", "spain", "italy", "australia", "prince", "princess", "king", "queen", "governor",
               "parliament", "senate", "prime", "minister", "premier", "mayor", "judge", "chief", "sovereign", "crown"}
MIN_RATIO = 0.65
_WORD = re.compile(r"\b[A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]{3,}\b")


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if not unicodedata.combining(c))


def skeleton(s: str) -> str:
    """Consonant skeleton: accents folded, vowels (and the soft h/w/y) dropped, doubled letters collapsed ("Papineau", "Papino" and "Papinow" -> "pn")."""
    t = re.sub(r"[^a-z]", "", fold(s))
    t = re.sub(r"(?<=[aeiou])w", "", t)                 # "-ow" / "-au" endings sound like vowels; a leading w ("Wales") stays a consonant
    t = re.sub(r"[aeiouyh]", "", t)
    return re.sub(r"(.)\1+", r"\1", t)


MIN_WORD_LEN, MIN_TARGET_LEN = 5, 6
# "Louis-Joseph Papineau" is often heard as "Lee Joseph Papineau": only when the next word makes the name unmistakable
_ALIASES = [(re.compile(r"\bLee[ -]Joseph(?=\s+Papineau\b)"), "Louis-Joseph", "Lee Joseph"),
            (re.compile(r"\bLee[ -]Hippolyte(?=\s+La\s+Fontaine\b)"), "Louis-Hippolyte", "Lee Hippolyte")]


@lru_cache(maxsize=1)
def _vocabulary() -> tuple[tuple[str, ...], frozenset[str]]:
    """(correction targets, protected words). Protected = EVERY word in the guide's text (in any case) plus everyday names and titles: the guide's own
    wording is never a mistake. Targets = the curated names plus distinctive long names from the guide (>= 6 letters, never an ordinary lower-case word)."""
    targets = {fold(w): w for n in CURATED for w in re.split(r"[\s.]+", n) if len(w) >= 4}        # one entry per spelling, properly capitalised
    protected = {fold(w) for w in _STOP_COMMON | _DONT_TOUCH}
    lower: set[str] = set()
    capitalised: dict[str, str] = {}
    try:
        for chunk in json.loads((ROOT / "chunks_window.json").read_text()):
            text = chunk.get("text", "")
            for w in re.findall(r"[A-Za-zÀ-ÿ]+", text):
                protected.add(fold(w))
                if w[0].islower():
                    lower.add(fold(w))
                else:
                    capitalised.setdefault(fold(w), w)
    except (OSError, ValueError):
        pass
    for f, w in capitalised.items():
        if len(f) >= MIN_TARGET_LEN and f not in lower:
            targets.setdefault(f, w)
    return tuple(sorted(w for f, w in targets.items() if f not in _DONT_TOUCH)), frozenset(protected)


def correct_names(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Returns (corrected text, [(heard, fixed), …]). Unchanged text and [] when nothing needs fixing."""
    targets, protected = _vocabulary()
    changes: list[tuple[str, str]] = []

    def fix(m: re.Match) -> str:
        whole = m.group(0)
        word, tail = (whole[:-2], whole[-2:]) if re.search(r"['’][sS]$", whole) else (whole, "")           # "Papino's" -> fix "Papino", keep "'s"
        f = fold(word)
        if len(word) < MIN_WORD_LEN or f in protected:                         # short words and anything the guide itself says are never "mistakes"
            return whole
        sk = skeleton(word)
        scored = []
        for n in targets:
            if fold(n) == f:
                continue
            ratio = SequenceMatcher(None, f, fold(n)).ratio()
            if sk and skeleton(n) == sk and ratio >= MIN_RATIO and f[0] == fold(n)[0]:     # it must SOUND like the name; similar spelling alone never counts
                scored.append((ratio, n))
        if not scored:
            return whole
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.03:
            return whole                                                    # two equally good names: do not guess
        changes.append((word, scored[0][1]))
        return scored[0][1] + tail

    fixed = _WORD.sub(fix, text)
    for pattern, replacement, heard in _ALIASES:                           # whole-phrase slips that a single-word fix cannot see
        fixed, n = pattern.subn(replacement, fixed)
        changes += [(heard, replacement)] * n
    return fixed, changes
