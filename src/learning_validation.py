"""Evidence validation for LLM-extracted learning objects.

The LLM proposes; this module disposes. An item survives only if
  * every cited source label resolves to a chunk that was actually supplied,
  * its `evidence_quote` appears (after punctuation/case normalisation) in a cited chunk,
  * every year it mentions appears in the cited text,
and duplicates are merged or dropped. Failures are omitted and recorded in
`omitted`, never repaired by guessing.

Honest limit: these are mechanical checks. They catch fabricated sources,
quotes and dates; they cannot prove that a free-text explanation adds no
paraphrase beyond the source.
"""
from __future__ import annotations

import re

from src.learning_models import ConceptNode, TimelineEvent
from src.retrieval import RetrievedChunk
from src.corpus import split_breadcrumb

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
MIN_QUOTE_WORDS = 3


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.casefold()).split())


def years_in(text: str) -> set[int]:
    return {int(y) for y in YEAR.findall(text)}


def _resolve(labels, by_label: dict[str, RetrievedChunk]) -> list[RetrievedChunk] | None:
    if not isinstance(labels, list) or not labels:
        return None
    out = []
    for label in labels:
        chunk = by_label.get(str(label).strip().upper())
        if chunk is None:
            return None           # a fabricated label invalidates the whole item
        if chunk not in out:
            out.append(chunk)
    return out


def _quote_ok(quote, chunks: list[RetrievedChunk]) -> bool:
    q = norm(str(quote or ""))
    return len(q.split()) >= MIN_QUOTE_WORDS and any(q in norm(c.text) for c in chunks)


_STOP = {"the", "and", "for", "with", "from", "that", "this", "its", "his", "her", "was", "were"}


def _tokens(label: str) -> set[str]:
    return {t for t in norm(label).split() if len(t) > 2 and t not in _STOP}


def _similar(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if norm(a) == norm(b):
        return True
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.7


def _pages(chunks: list[RetrievedChunk]) -> list[str]:
    return list(dict.fromkeys(c.page for c in chunks))


# --- concept hub --------------------------------------------------------------

def validate_hub_nodes(
    raw_nodes, central: str, by_label: dict[str, RetrievedChunk], max_nodes: int,
) -> tuple[list[ConceptNode], list[str]]:
    nodes: list[ConceptNode] = []
    omitted: list[str] = []
    for item in raw_nodes if isinstance(raw_nodes, list) else []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        name = label or "(unnamed)"
        chunks = _resolve(item.get("sources"), by_label)
        text = " ".join(c.text for c in chunks) if chunks else ""
        explanation = str(item.get("explanation", "")).strip()
        relationship = str(item.get("relationship", "")).strip()
        if not label or not relationship:
            omitted.append(f"{name}: missing label or relationship")
        elif chunks is None:
            omitted.append(f"{name}: cites a source that was not provided")
        elif not _quote_ok(item.get("evidence_quote"), chunks):
            omitted.append(f"{name}: evidence quote not found in the cited source")
        elif not years_in(explanation) <= years_in(text):
            omitted.append(f"{name}: explanation mentions a year not in the cited source")
        elif _similar(label, central) or any(_similar(label, n.label) for n in nodes):
            omitted.append(f"{name}: duplicate")
        else:
            nodes.append(ConceptNode(
                id=re.sub(r"[^a-z0-9]+", "_", norm(label)) or f"n{len(nodes)}",
                label=label, relationship=relationship, explanation=explanation,
                source_chunk_ids=[c.chunk_id for c in chunks], pages=_pages(chunks),
                evidence_quote=str(item["evidence_quote"]).strip(),
            ))
    if len(nodes) > max_nodes:
        omitted += [f"{n.label}: over the {max_nodes}-node limit" for n in nodes[max_nodes:]]
        nodes = nodes[:max_nodes]
    return nodes, omitted


def validate_summary(summary, labels, by_label: dict[str, RetrievedChunk]) -> tuple[str, list[str]]:
    """Keep the one-line definition only if its cited sources are real and any years match."""
    text, chunks = str(summary or "").strip(), _resolve(labels, by_label)
    if not text or chunks is None:
        return "", []
    if not years_in(text) <= years_in(" ".join(c.text for c in chunks)):
        return "", []
    return text, [c.chunk_id for c in chunks]


# --- timeline -----------------------------------------------------------------

def _int_or_none(v) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _dates_supported(label: str, start: int | None, end: int | None, text: str) -> bool:
    present = years_in(text)
    if not years_in(label) <= present:
        return False
    if start is None:
        return end is None
    if start not in present:
        return False
    if end is None or end == start or end in present:
        return True
    # abbreviated range such as "1837–38"
    return re.search(rf"{start}\s*[–—-]\s*{str(end)[-2:]}\b", text) is not None


def validate_events(
    raw_events, by_label: dict[str, RetrievedChunk],
) -> tuple[list[TimelineEvent], list[TimelineEvent], list[str]]:
    """Returns (dated events chronological, undated broad-period events, omitted reasons)."""
    kept: list[TimelineEvent] = []
    omitted: list[str] = []
    for item in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        label = str(item.get("date_label", "")).strip()
        name = f"{title or '(untitled)'} ({label or 'no date'})"
        chunks = _resolve(item.get("sources"), by_label)
        text = " ".join(c.text for c in chunks) if chunks else ""
        start, end = _int_or_none(item.get("start_year")), _int_or_none(item.get("end_year"))
        description = str(item.get("description", "")).strip()
        importance = str(item.get("importance", "")).strip()
        if not title or not label:
            omitted.append(f"{name}: missing title or date label")
        elif chunks is None:
            omitted.append(f"{name}: cites a source that was not provided")
        elif not _quote_ok(item.get("evidence_quote"), chunks):
            omitted.append(f"{name}: evidence quote not found in the cited source")
        elif not _dates_supported(label, start, end, text):
            omitted.append(f"{name}: date is not supported by the cited source")
        elif not years_in(f"{title} {description} {importance}") <= years_in(text):
            omitted.append(f"{name}: text mentions a year not in the cited source")
        else:
            first = chunks[0]
            row = split_breadcrumb(first.text)[0]
            kept.append(TimelineEvent(
                date_label=label, start_year=start, end_year=end if end is not None else start,
                title=title, description=description, importance=importance,
                source_chunk_ids=[c.chunk_id for c in chunks], pages=_pages(chunks),
                section=first.section, chapter=row[0] if len(row) >= 2 else first.section,
                evidence_quote=str(item["evidence_quote"]).strip(),
            ))
    merged, dupes = merge_duplicate_events(kept)
    omitted += dupes
    dated = sorted((e for e in merged if e.start_year is not None),
                   key=lambda e: (e.start_year, e.end_year or e.start_year))
    return dated, [e for e in merged if e.start_year is None], omitted


def merge_duplicate_events(events: list[TimelineEvent]) -> tuple[list[TimelineEvent], list[str]]:
    """Merge events with the same years and similar titles (sources are unioned)."""
    out: list[TimelineEvent] = []
    notes: list[str] = []
    for e in events:
        twin = next((o for o in out if (o.start_year, o.end_year) == (e.start_year, e.end_year)
                     and _similar(o.title, e.title)), None)
        if twin is None:
            out.append(e)
            continue
        i = out.index(twin)
        ids = list(dict.fromkeys(twin.source_chunk_ids + e.source_chunk_ids))
        pages = list(dict.fromkeys(twin.pages + e.pages))
        out[i] = TimelineEvent(**{**twin.__dict__, "source_chunk_ids": ids, "pages": pages})
        notes.append(f"{e.title} ({e.date_label}): merged into duplicate")
    return out, notes
