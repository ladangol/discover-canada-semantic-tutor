"""Typed containers for the learning visualizations (no logic beyond (de)serialisation)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ChapterSemanticStats:
    chapter: str
    n_chunks: int
    words: int                      # source-text volume
    breadth: float                  # mean cosine distance of chunks from the chapter centroid
    same_chapter_nn_rate: float     # share of chunks whose nearest neighbour is in the same chapter
    nearest: list[tuple[str, float]]  # other chapters by centroid cosine, best first


@dataclass(frozen=True)
class ChapterAnalysis:
    chapters: list[str]
    similarity: list[list[float]]   # centroid cosine matrix, rows/cols follow `chapters`
    stats: list[ChapterSemanticStats]


@dataclass(frozen=True)
class ConceptNode:
    id: str
    label: str
    relationship: str
    explanation: str
    source_chunk_ids: list[str]
    pages: list[str]
    evidence_quote: str             # verbatim (normalised) text from a cited chunk


@dataclass(frozen=True)
class ConceptHub:
    central_concept: str
    kind: str                       # "concept" | "person"
    summary: str
    nodes: list[ConceptNode]
    summary_source_chunk_ids: list[str] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)   # why items failed validation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptHub":
        return cls(**{**d, "nodes": [ConceptNode(**n) for n in d["nodes"]]})


@dataclass(frozen=True)
class TimelineEvent:
    date_label: str
    start_year: int | None
    end_year: int | None
    title: str
    description: str
    importance: str
    source_chunk_ids: list[str]
    pages: list[str]
    section: str                    # guide heading of the first cited chunk
    chapter: str
    evidence_quote: str


@dataclass(frozen=True)
class TimelineData:
    title: str
    events: list[TimelineEvent]     # dated events, chronological
    undated: list[TimelineEvent] = field(default_factory=list)  # broad periods without a year
    omitted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TimelineData":
        return cls(title=d["title"], omitted=d.get("omitted", []),
                   events=[TimelineEvent(**e) for e in d["events"]],
                   undated=[TimelineEvent(**e) for e in d.get("undated", [])])
