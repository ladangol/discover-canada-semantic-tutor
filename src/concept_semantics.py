"""Semantic validation for concept hubs: entity types, relationship direction, and what the source text actually states.

`learning_validation` already checks mechanics (real source labels, verbatim quotes, years). That cannot catch a diagram that is
attractive but wrong, e.g. "Inuktitut" and "Michif" hung under "First Nation Languages" (they belong to the Inuit and the Métis, which
the guide names as separate peoples), or an arrow that reads backwards. This layer adds:

  1. ENTITY TYPES: person, people, language, language_family, word, place, event, institution, concept are different things and are
     never merged (a people is not a language; a word is not a language family; an institution is not an event).
  2. TYPED PREDICATES with a signature (subject type -> object type) and a direction. The displayed phrase is chosen from the predicate
     and the direction, so "<centre> <phrase> <node>" always reads correctly. A reversed arrow is corrected only when the types make the
     fix unambiguous; otherwise the item is rejected.
  3. SUPPORT: a typed predicate needs a cue in the evidence quote that actually states it (e.g. "speaks" needs speak/spoken/dialect).
     If the cue is missing but both ends appear in the same passage, the relationship is REVISED to the neutral "is mentioned with";
     if they do not, it is REJECTED. Nothing is inferred from outside knowledge.
  4. GROUP-SCOPE GUARD: First Nations, Métis and Inuit are three distinct peoples in the guide. An item whose own passage places it
     with one group cannot hang under a centre about another, and "Aboriginal"/"Indigenous" are never treated as synonyms for any of them.

Honest limit: these are mechanical checks over the quote and its passage. They catch wrong types, wrong direction, missing cues and
group mix-ups; they cannot prove that the model's one-sentence explanation adds nothing beyond the source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ENTITY_TYPES = ("person", "people", "language", "language_family", "word", "place", "event", "institution", "concept")
# The infographic theme knows five presentation categories; every entity type maps to one of them.
CATEGORY_FOR_TYPE = {"person": "person", "people": "person", "language": "concept", "language_family": "concept", "word": "concept",
                     "place": "place", "event": "event", "institution": "organization", "concept": "concept"}

MENTIONED_WITH = "mentioned_with"
OTHER = "other"


@dataclass(frozen=True)
class Predicate:
    name: str
    subject: frozenset
    object: frozenset
    forward: str            # how it reads when the CENTRE is the subject:  "<centre> <forward> <node>"
    reverse: str            # how it reads when the centre is the object:   "<centre> <reverse> <node>"
    cues: tuple             # the evidence quote must match at least one
    meaning: str            # shown to the model


def _p(name, subject, obj, forward, reverse, cues, meaning) -> Predicate:
    return Predicate(name, frozenset(subject.split()), frozenset(obj.split()), forward, reverse,
                     tuple(re.compile(c, re.I) for c in cues), meaning)


PREDICATES: dict[str, Predicate] = {p.name: p for p in [
    _p("speaks", "people person", "language", "speak", "is spoken by", (r"\bspeak", r"\bspoke", r"\bspoken\b", r"\bdialect\b", r"\bmother tongue\b", r"\blanguage of\b"),
       "a people/person speaks a language or dialect"),
    _p("word_in", "word", "language language_family people", "is a word in", "includes the word", (r"\bword\b", r"\bterm\b", r"\bmeans?\b", r"\bmeaning\b"),
       "a word belongs to / comes from a language or its speakers"),
    _p("member_of_family", "language", "language_family", "belongs to the family", "is a family that includes", (r"\bfamil", r"\bbranch\b"),
       "a language belongs to a language family"),
    _p("heard_by", "word concept", "person", "was heard by", "heard", (r"\bheard\b", r"\bhear\b"), "a word or sound was heard by a person"),
    _p("name_derived_from", "concept place word", "word person place people concept", "takes its name from", "is the source of the name of",
       (r"\bderiv", r"\bcomes? from\b", r"\bcame from\b", r"\borigin of\b", r"\bnamed (?:after|for)\b", r"\btakes? its name\b", r"\bfrom the \w+ word\b"),
       "a name is derived from a word/person/place; the text must say so"),
    _p("prohibited_in", "language concept", "institution event", "prohibited in", "prohibited", (r"\bprohibit", r"\bbann", r"\bforbid", r"\bforbade"),
       "a language or practice was banned within an institution or event"),
    _p("part_of", "person people language language_family word place event institution concept",
       "person people language language_family word place event institution concept", "is part of", "includes",
       (r"\bpart of\b", r"\bwithin\b", r"\bamong\b", r"\bone of\b", r"\binclud", r"\bconsist", r"\bmade up of\b", r"\bbelongs?\b"), "membership or inclusion"),
    _p("member_of", "person people", "institution people event", "is a member of", "has as a member", (r"\bmember", r"\bjoined\b", r"\bbelong", r"\bpart of\b"),
       "a person or people belongs to an institution or group"),
    _p("led_by", "event institution people place", "person", "was led by", "led", (r"\bled\b", r"\bleader", r"\bheaded\b", r"\bhead of\b", r"\bin charge\b"),
       "an event, body or people was led by a person"),
    _p("founded_by", "institution place event concept", "person people institution", "was founded by", "founded", (r"\bfound(?:ed|er|s)\b", r"\bestablish", r"\bcreated\b", r"\bbuilt\b"),
       "something was founded, created or built by someone"),
    _p("located_in", "place institution people event", "place", "is in", "contains", (r"\blocated\b", r"\blive[sd]? in\b", r"\bsituated\b", r"\b(?:is|are|was|were) in\b", r"\blies\b"),
       "something is located in a place"),
    _p("took_part_in", "person people", "event", "took part in", "involved", (r"\btook part\b", r"\bfought\b", r"\bparticipat", r"\bserved\b", r"\bdefend", r"\bled\b"),
       "a person or people took part in an event"),
]}

# First Nations, Métis and Inuit are three distinct peoples in the guide; "Aboriginal" is the umbrella and is NOT a scope of its own.
SCOPES = {"First Nations": re.compile(r"\bfirst nations?\b", re.I), "Métis": re.compile(r"\bm[ée]tis\b", re.I),
          "Inuit": re.compile(r"\binuit\b", re.I)}

# Used ONLY here (is the centre's topic near the quote?), never in retrieval or category mapping: a passage that says "dialect" is talking about a language.
_LANGUAGE_WORDS = {"langu", "diale", "tongu"}

_STOP = {"the", "and", "for", "with", "from", "that", "this", "its", "his", "her", "was", "were", "are", "of", "in", "by", "to", "a", "an", "on", "at"}
_PHRASE_STOP = _STOP | {"is", "be", "been", "as", "into", "include", "includes"}


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-zà-ÿ]+", " ", (text or "").casefold()).split())


def _stem(w: str) -> str:
    return w[:5] if len(w) > 5 else w


def _tokens(label: str) -> list[str]:
    return [t for t in norm(label).split() if len(t) >= 3 and t not in _STOP]


def mentions(label: str, text: str) -> bool:
    """Does `text` mention the thing named by `label`? At least half of its significant words (by 5-letter stem) must occur."""
    toks, body = _tokens(label), norm(text)
    if not toks:
        return False
    words = {_stem(w) for w in body.split()}
    return sum(_stem(t) in words for t in toks) >= max(1, (len(toks) + 1) // 2)


def scopes_in(text: str) -> set[str]:
    return {name for name, rx in SCOPES.items() if rx.search(text or "")}


def phrase_supported(phrase: str, quote: str) -> bool:
    """A free-text relationship ("invented by") is supported when each of its content words (by stem) occurs in the quote."""
    body = {_stem(w) for w in norm(quote).split()}
    content = [w for w in norm(phrase).split() if w not in _PHRASE_STOP]
    return bool(content) and all(_stem(w) in body for w in content)


def window(quote: str, chunk_text: str, before: int = 2) -> str:
    """The sentence holding the quote plus up to `before` sentences preceding it (a pronoun like "They" points back there)."""
    sentences = re.split(r"(?<=[.!?”\"])\s+", re.sub(r"\s+", " ", chunk_text))
    q = norm(quote)
    for i, s in enumerate(sentences):
        if q and (q in norm(s) or norm(s) in q and len(norm(s)) > 20):
            return " ".join(sentences[max(0, i - before): i + 1])
    return quote


def nearest_scopes(label: str, quote: str, chunk_text: str, back: int = 2) -> set[str]:
    """Which of First Nations / Métis / Inuit does the passage tie this item to?
    The quote's own sentence decides if it names a group. Otherwise look back up to `back` sentences (a pronoun like "They" points there),
    but a sentence that lists several groups at once ("65% First Nations, 30% Métis and 4% Inuit") ties the item to none of them."""
    sentences = re.split(r"(?<=[.!?”\"])\s+", re.sub(r"\s+", " ", chunk_text))
    q = norm(quote)
    i = next((k for k, s in enumerate(sentences) if q and (q in norm(s) or (norm(s) in q and len(norm(s)) > 20))), None)
    if i is None:
        return scopes_in(f"{label} {quote}")
    if own := scopes_in(f"{label} {sentences[i]}"):
        return own
    for j in range(i - 1, max(-1, i - 1 - back), -1):
        found = scopes_in(sentences[j])
        if len(found) >= 2:
            return set()
        if found:
            return found
    return set()


def scope_conflict(center: str, label: str, quote: str, chunk_text: str) -> str:
    """'' if fine, else why the item belongs to a different people than the centre."""
    want = scopes_in(center)
    if not want:
        return ""
    have = nearest_scopes(label, quote, chunk_text)
    if have and not (have & want):
        return f"its passage places it with the {', '.join(sorted(have))}, not the {', '.join(sorted(want))}"
    return ""


def centre_in_context(center: str, quote: str, chunk_text: str) -> bool:
    """For a weak "is mentioned with" link the centre's topic word must be near the quote, not just somewhere in the passage.
    The head word of the centre (its last significant word: "languages" in "First Nation Languages") has to occur in the quote's sentence
    or the two sentences before it (a pronoun like "They" points back there). A one-word centre needs only that word."""
    toks = _tokens(center)
    if not toks:
        return False
    head = _stem(toks[-1])
    words = {_stem(w) for w in norm(window(quote, chunk_text)).split()}
    return head in words or (head in _LANGUAGE_WORDS and bool(words & _LANGUAGE_WORDS))


def summary_scope_conflict(center: str, summary: str) -> bool:
    want = scopes_in(center)
    return bool(want) and bool(scopes_in(summary) - want)


@dataclass(frozen=True)
class Verdict:
    status: str                      # "ok" | "revised" | "rejected"
    predicate: str = ""
    subject_is_center: bool = True
    display: str = ""                # "<centre> <display> <node>" reads correctly
    support: str = ""                # "stated" (the quote states it) | "co-mentioned" (same passage only)
    reason: str = ""                 # why it was rejected or revised


def check_relationship(*, center: str, center_type: str, label: str, node_type: str, predicate: str, subject_is_center: bool,
                       relationship_text: str, quote: str, chunk_text: str) -> Verdict:
    """Validate ONE relationship between the centre and a node against types, direction and the source text."""
    predicate = (predicate or "").strip().lower() or (OTHER if relationship_text else MENTIONED_WITH)
    if conflict := scope_conflict(center, label, quote, chunk_text):
        return Verdict("rejected", reason=conflict)
    node_in_passage = mentions(label, chunk_text)
    if not node_in_passage:
        return Verdict("rejected", reason="the item is not named in the cited passage")
    co_mentioned = centre_in_context(center, quote, chunk_text)

    def weaker(why: str) -> Verdict:
        if co_mentioned:
            return Verdict("revised", MENTIONED_WITH, True, "is mentioned with", "co-mentioned", why + "; shown only as 'is mentioned with'")
        return Verdict("rejected", reason=why + f", and the centre's topic ('{(_tokens(center) or [center])[-1]}') is not near the quote")

    if predicate == MENTIONED_WITH:
        return (Verdict("ok", MENTIONED_WITH, True, "is mentioned with", "co-mentioned") if co_mentioned
                else Verdict("rejected", reason=f"the centre's topic ('{(_tokens(center) or [center])[-1]}') is not near the quote"))
    if predicate == OTHER or predicate not in PREDICATES:
        phrase = (relationship_text or "").strip()
        if not phrase:
            return Verdict("rejected", reason="no relationship stated")
        if phrase_supported(phrase, quote):
            return Verdict("ok", OTHER, subject_is_center, phrase, "stated")
        return weaker(f"'{phrase}' is not stated in the quote")

    rule = PREDICATES[predicate]
    if center_type not in ENTITY_TYPES or node_type not in ENTITY_TYPES:
        return weaker(f"cannot check '{predicate}': entity type missing or unknown")
    subj, obj = (center_type, node_type) if subject_is_center else (node_type, center_type)
    note = ""
    if not (subj in rule.subject and obj in rule.object):
        if obj in rule.subject and subj in rule.object:                      # the arrow is backwards and the types make the fix unambiguous
            subject_is_center, subj, obj, note = (not subject_is_center), obj, subj, "direction corrected"
        else:
            return weaker(f"types do not fit '{predicate}': {subj} -> {obj}, it needs {'/'.join(sorted(rule.subject))} -> {'/'.join(sorted(rule.object))}")
    if not any(c.search(quote) for c in rule.cues):
        return weaker(f"'{rule.forward}' is not stated in the quote")
    phrase = rule.forward if subject_is_center else rule.reverse
    return Verdict("revised" if note else "ok", predicate, subject_is_center, phrase, "stated", note)


def prompt_vocabulary() -> str:
    types = ("person (one named individual) | people (a nation, community or group: Inuit, Métis, First Nations, Loyalists) | language (a language or dialect) | "
             "language_family (a group of related languages, or the adjective naming one: Iroquoian) | word (a word or name quoted from a language: kanata) | "
             "place | event | institution (organisation, school system, government body, office) | concept (anything else)")
    preds = "\n".join(f'  - "{p.name}": {"/".join(sorted(p.subject))} -> {"/".join(sorted(p.object))}; {p.meaning}' for p in PREDICATES.values())
    return types, preds
