"""Renderer-independent specs for infographics.

`HubSpec` / `TimelineSpec` hold only validated content (labels, relationships, dates, pages,
quotes). They know nothing about SVG, colours or layout, so any theme or layout can draw them.
Conversion from the validated learning objects happens here; the LLM never supplies markup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.corpus import Corpus
from src.learning_models import CATEGORIES, ConceptHub, TimelineData
from src.retrieval import page_label


@dataclass(frozen=True)
class NodeSpec:
    id: str
    label: str
    relationship: str
    category: str
    explanation: str
    pages: list[str]
    section: str
    quote: str
    source_chunk_ids: list[str]


@dataclass(frozen=True)
class HubSpec:
    title: str
    subtitle: str
    center_category: str
    nodes: list[NodeSpec]
    pages: list[str] = field(default_factory=list)   # for the "Source" footer


@dataclass(frozen=True)
class EventSpec:
    id: str
    date_label: str
    title: str
    description: str
    importance: str
    band: str                       # the guide's own section heading
    pages: list[str]
    quote: str
    source_chunk_ids: list[str]
    start_year: int | None = None
    end_year: int | None = None


@dataclass(frozen=True)
class TimelineSpec:
    title: str
    subtitle: str
    events: list[EventSpec]         # chronological
    pages: list[str] = field(default_factory=list)


# --- helpers ----------------------------------------------------------------------------

_SMALL_WORDS = {"of", "the", "and", "in", "for", "to", "a", "an", "on", "de", "la"}


def tidy_title(text: str) -> str:
    out = []
    for i, w in enumerate(text.strip().split()):
        if i and w in _SMALL_WORDS:
            pass
        elif w[:1].isalpha() and not w[0].isupper():
            w = w.capitalize()
        elif w[:1].isalpha() and any(c.islower() for c in w[1:]) and any(c.isupper() for c in w[1:]) \
                and w[1:].lower() != w[1:] and not re.match(r"^(Mc|Mac|De|La|Le|O')", w):
            w = w.capitalize()
        out.append(w)
    return " ".join(out)


_PERSON = re.compile(r"^(sir|lord|lady|queen|king|prince|princess|dr\.?|mr\.?|mrs\.?|ms\.?|saint|st\.)\s", re.I)
_ORG = re.compile(r"\b(agency|council|parliament|cabinet|court|commission|government|assembly|senate|house of commons|"
                  r"army|navy|force|forces|corps|company|bank|police|rangers|department|ministry|party|"
                  r"aerospace|institute|university|society|association|monarchy|crown|legislature|governor general)\b", re.I)
_EVENT = re.compile(r"\b(war|battle|rebellion|rebellions|act|treaty|confederation|election|invasion|conference|"
                    r"expedition|revolution|siege|raid|crossing|landing|signing|founding|migration|apology|"
                    r"d-day|crash|depression|abolition|vote)\b|\b(1[0-9]{3}|20[0-9]{2})\b", re.I)
_PLACE = re.compile(r"\b(province|territory|territories|river|lake|ocean|bay|island|mountains|city|coast|region|"
                    r"nunavut|yukon|ontario|quebec|alberta|manitoba|saskatchewan|labrador|newfoundland|"
                    r"scotia|brunswick|columbia|halifax|ottawa|toronto|montreal|arctic|prairies?)\b", re.I)


def infer_category(label: str) -> str:
    """Deterministic fallback when the extractor gave no (valid) category."""
    if _PERSON.search(label):
        return "person"
    if _ORG.search(label):
        return "organization"
    if _EVENT.search(label):
        return "event"
    if _PLACE.search(label) and len(label.split()) <= 3:
        return "place"
    return "concept"


def _category(hint: str, label: str) -> str:
    return hint if hint in CATEGORIES else infer_category(label)


def _page_key(page: str) -> int:
    m = re.match(r"\d+", page)
    return int(m.group()) if m else 0


def source_pages(pages: list[str]) -> list[str]:
    return sorted(dict.fromkeys(pages), key=_page_key)


def pages_text(pages: list[str]) -> str:
    """"p. 27" / "pp. 26, 27" for the export footer."""
    ps = source_pages(pages)
    if not ps:
        return ""
    return page_label(ps[0]) if len(ps) == 1 else "pp. " + ", ".join(ps)


def lead_sentences(text: str, max_chars: int = 190) -> str:
    """Whole sentences from the start of `text` that fit `max_chars` (never cut mid-sentence)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = ""
    for s in sentences:
        if out and len(out) + 1 + len(s) > max_chars:
            break
        out = f"{out} {s}".strip()
    return out if len(out) <= max_chars * 1.35 else sentences[0]


# --- conversion from validated objects ---------------------------------------------------

def hub_spec(hub: ConceptHub, corpus: Corpus) -> HubSpec:
    nodes = []
    for n in hub.nodes:
        section = corpus.row(n.source_chunk_ids[0]).section if n.source_chunk_ids else ""
        nodes.append(NodeSpec(
            id=n.id, label=n.label, relationship=n.relationship,
            category=_category(n.category, n.label), explanation=n.explanation, pages=list(n.pages),
            section=section, quote=n.evidence_quote, source_chunk_ids=list(n.source_chunk_ids)))
    title = tidy_title(hub.central_concept)
    return HubSpec(
        title=title, subtitle=lead_sentences(hub.summary),
        center_category=_category(hub.central_category or ("person" if hub.kind == "person" else ""), title),
        nodes=nodes, pages=source_pages([p for n in nodes for p in n.pages]))


def timeline_spec(data: TimelineData, *, start: int = 0, stop: int | None = None) -> TimelineSpec:
    """`start`/`stop` select a slice of the chronological events (for fixed-size images)."""
    chosen = data.events[start:stop]
    events = [EventSpec(
        id=f"e{start + i}", date_label=e.date_label, title=e.title, description=e.description,
        importance=e.importance, band=e.section, pages=list(e.pages), quote=e.evidence_quote,
        source_chunk_ids=list(e.source_chunk_ids), start_year=e.start_year, end_year=e.end_year)
        for i, e in enumerate(chosen)]
    years = [y for e in chosen for y in (e.start_year, e.end_year) if y]
    subtitle = f"{min(years)}–{max(years)}" if years and min(years) != max(years) else (str(years[0]) if years else "")
    title = re.sub(r"^Timeline:\s*", "", data.title).strip()
    return TimelineSpec(title=f"{title}: timeline", subtitle=subtitle, events=events,
                        pages=source_pages([p for e in events for p in e.pages]))
