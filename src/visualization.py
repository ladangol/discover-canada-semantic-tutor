"""Atlas data + Plotly figures, and the query -> (retrieval, UMAP point) view.

UMAP coordinates are for display only. Similarity everywhere else (retrieval,
neighbours, diversity) uses the original 384-d cosine geometry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    DEFAULT_K,
    EMBEDDING_MAP_3D_PATH,
    EMBEDDING_MAP_PATH,
    UMAP_REDUCER_3D_PATH,
    UMAP_REDUCER_PATH,
)
from src.chapter_stats import mds_layout, strongest_edges
from src.embeddings import encode_query
from src.learning_models import ChapterAnalysis, ConceptHub, TimelineData
from src.retrieval import RetrievedChunk, page_label, retrieve

MISSING_ARTIFACT_HINT = "Run `python scripts/build_embedding_map.py` first."


# --- artifacts ----------------------------------------------------------------

@lru_cache(maxsize=1)
def load_map() -> pd.DataFrame:
    if not EMBEDDING_MAP_PATH.exists():
        raise FileNotFoundError(f"{EMBEDDING_MAP_PATH.name} not found. {MISSING_ARTIFACT_HINT}")
    payload = json.loads(EMBEDDING_MAP_PATH.read_text())
    return pd.DataFrame(payload["points"])


@lru_cache(maxsize=1)
def load_reducer():
    if not UMAP_REDUCER_PATH.exists():
        raise FileNotFoundError(f"{UMAP_REDUCER_PATH.name} not found. {MISSING_ARTIFACT_HINT}")
    return joblib.load(UMAP_REDUCER_PATH)


@lru_cache(maxsize=1)
def load_map_3d() -> pd.DataFrame | None:
    """3-D atlas points, or None if the 3-D artifacts have not been built."""
    if not (EMBEDDING_MAP_3D_PATH.exists() and UMAP_REDUCER_3D_PATH.exists()):
        return None
    return pd.DataFrame(json.loads(EMBEDDING_MAP_3D_PATH.read_text())["points"])


@lru_cache(maxsize=1)
def _reducer_3d():
    return joblib.load(UMAP_REDUCER_3D_PATH)


def project_query_3d(vector: np.ndarray) -> tuple[float, float, float] | None:
    if load_map_3d() is None:
        return None
    xyz = _reducer_3d().transform(vector.reshape(1, -1))[0]
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def project_query(vector: np.ndarray) -> tuple[float, float]:
    """Place a query embedding in the atlas via the fitted reducer's transform()."""
    xy = load_reducer().transform(vector.reshape(1, -1))[0]
    return float(xy[0]), float(xy[1])


# --- query view ----------------------------------------------------------------

@dataclass(frozen=True)
class QueryView:
    """One query embedding, used two ways: cosine retrieval (original vector)
    and UMAP projection (display only)."""

    question: str
    chunks: list[RetrievedChunk]
    xy: tuple[float, float]
    xyz: tuple[float, float, float] | None = None


def analyze_query(question: str, k: int = DEFAULT_K) -> QueryView:
    vector = encode_query(question)
    chunks = retrieve(question, k=k, query_vector=vector)
    return QueryView(question=question, chunks=chunks, xy=project_query(vector),
                     xyz=project_query_3d(vector))


# --- figures -------------------------------------------------------------------

def chapter_colors(chapters: list[str]) -> dict[str, str]:
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    return {c: palette[i % len(palette)] for i, c in enumerate(chapters)}


def _hover(df: pd.DataFrame) -> list[str]:
    return [
        f"<b>{r.section}</b><br>{page_label(r.page)} · {r.block_type}<br>"
        f"<i>{r.chapter}</i><br>{_wrap(r.preview)}"
        for r in df.itertuples()
    ]


def _wrap(text: str, width: int = 60, max_lines: int = 4) -> str:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return "<br>".join(lines[:max_lines])


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.02, font=dict(size=11)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode="pan",
        clickmode="event+select",
        hoverlabel=dict(align="left"),
    )
    return fig


def atlas_figure(
    df: pd.DataFrame,
    colors: dict[str, str],
    *,
    visible_ids: set[str],
    selected_id: str | None = None,
    neighbour_ids: list[str] | None = None,
    height: int = 620,
) -> go.Figure:
    """Chunks coloured by chapter. Filtered-out chunks stay as faint grey dots
    so the global layout is never lost."""
    fig = go.Figure()

    hidden = df[~df["chunk_id"].isin(visible_ids)]
    if len(hidden):
        fig.add_trace(go.Scattergl(
            x=hidden["x"], y=hidden["y"], mode="markers", name="filtered out",
            marker=dict(size=6, color="#d0d0d0", opacity=0.5),
            hoverinfo="skip", showlegend=False,
        ))

    shown = df[df["chunk_id"].isin(visible_ids)]
    for chapter, group in shown.groupby("chapter", sort=False):
        fig.add_trace(go.Scatter(
            x=group["x"], y=group["y"], mode="markers", name=chapter,
            customdata=group["chunk_id"], text=_hover(group), hoverinfo="text",
            marker=dict(size=9, color=colors[chapter], line=dict(width=0.5, color="white")),
        ))

    by_id = df.set_index("chunk_id")
    if neighbour_ids:
        nb = by_id.loc[neighbour_ids].reset_index()
        fig.add_trace(go.Scatter(
            x=nb["x"], y=nb["y"], mode="markers+text", name="semantic neighbours",
            text=[str(i + 1) for i in range(len(nb))], textposition="top center",
            customdata=nb["chunk_id"], hovertext=_hover(nb), hoverinfo="text",
            marker=dict(size=15, symbol="circle-open", color="#111", line=dict(width=2)),
        ))
    if selected_id and selected_id in by_id.index:
        s = by_id.loc[[selected_id]].reset_index()
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], mode="markers", name="selected",
            customdata=s["chunk_id"], hovertext=_hover(s), hoverinfo="text",
            marker=dict(size=20, symbol="star", color="#c8102e", line=dict(width=1.5, color="white")),
        ))
    return _base_layout(fig, height)


def query_figure(
    df: pd.DataFrame,
    colors: dict[str, str],
    view: QueryView,
    *,
    emphasize: bool,
    height: int = 560,
) -> go.Figure:
    """The map as the retriever's question sees it: query star + retrieved top-k.

    Retrieved chunks are drawn wherever UMAP put them, even if that is far from
    the query star — 2-D distance is not retrieval similarity.
    """
    fig = go.Figure()
    retrieved_ids = [c.chunk_id for c in view.chunks]
    rest = df[~df["chunk_id"].isin(retrieved_ids)]
    opacity = 0.18 if emphasize else 0.8

    for chapter, group in rest.groupby("chapter", sort=False):
        fig.add_trace(go.Scatter(
            x=group["x"], y=group["y"], mode="markers", name=chapter,
            text=_hover(group), hoverinfo="text",
            marker=dict(size=8, color=colors[chapter], opacity=opacity),
        ))

    by_id = df.set_index("chunk_id")
    qx, qy = view.xy
    for c in view.chunks:
        p = by_id.loc[c.chunk_id]
        fig.add_trace(go.Scatter(
            x=[qx, p["x"]], y=[qy, p["y"]], mode="lines", showlegend=False,
            line=dict(color="rgba(200,16,46,0.35)", width=1, dash="dot"), hoverinfo="skip",
        ))
    hits = by_id.loc[retrieved_ids].reset_index()
    fig.add_trace(go.Scatter(
        x=hits["x"], y=hits["y"], mode="markers+text", name="retrieved top-k",
        text=[str(i + 1) for i in range(len(hits))], textposition="middle center",
        textfont=dict(color="white", size=11),
        hovertext=[
            f"<b>#{i + 1} · cos {c.similarity:.3f}</b><br>{h}"
            for i, (c, h) in enumerate(zip(view.chunks, _hover(hits)))
        ],
        hoverinfo="text",
        marker=dict(size=22, color="#c8102e", line=dict(width=2, color="white")),
    ))
    fig.add_trace(go.Scatter(
        x=[qx], y=[qy], mode="markers", name="your question",
        hovertext=[f"<b>Query</b><br>{view.question}"], hoverinfo="text",
        marker=dict(size=24, symbol="star", color="#111", line=dict(width=2, color="gold")),
    ))
    return _base_layout(fig, height)



# --- chapter semantic map (original-embedding statistics; not UMAP) -------------

def chapter_heatmap(analysis: ChapterAnalysis, height: int = 560) -> go.Figure:
    """Centroid cosine similarity. The diagonal (always 1) is left blank so the
    colour range shows the differences between chapters."""
    z = np.array(analysis.similarity)
    np.fill_diagonal(z, np.nan)
    fig = go.Figure(go.Heatmap(
        z=z, x=analysis.chapters, y=analysis.chapters, colorscale="YlOrRd",
        colorbar=dict(title="cosine"), texttemplate="%{z:.2f}", textfont=dict(size=10),
        hovertemplate="%{y}<br>vs %{x}<br>centroid cosine %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-40))
    return fig


def chapter_network(
    analysis: ChapterAnalysis, colors: dict[str, str], per_chapter: int = 2, height: int = 560,
) -> go.Figure:
    """Chapters as nodes (size = chunk count); each chapter keeps its N strongest
    centroid-cosine links. Node positions are an MDS of centroid distances, so they
    are approximate; the edge labels carry the actual similarity."""
    sim = np.array(analysis.similarity)
    pos = mds_layout(sim)
    edges = strongest_edges(sim, per_chapter)
    lo, hi = min(edges.values()), max(edges.values())
    fig = go.Figure()
    for (i, j), w in edges.items():
        width = 1.5 + 7 * ((w - lo) / (hi - lo) if hi > lo else 0.5)
        fig.add_trace(go.Scatter(
            x=[pos[i, 0], pos[j, 0]], y=[pos[i, 1], pos[j, 1]], mode="lines", showlegend=False,
            line=dict(width=width, color="rgba(120,120,120,0.55)"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[(pos[i, 0] + pos[j, 0]) / 2], y=[(pos[i, 1] + pos[j, 1]) / 2], mode="text",
            text=[f"{w:.2f}"], textfont=dict(size=10, color="#555"), showlegend=False,
            hovertext=f"{analysis.chapters[i]} ↔ {analysis.chapters[j]}<br>centroid cosine {w:.3f}",
            hoverinfo="text"))
    st_ = analysis.stats
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1], mode="markers+text", showlegend=False,
        text=[x.chapter for x in st_], textposition="top center", textfont=dict(size=11),
        marker=dict(size=[16 + 5 * np.sqrt(x.n_chunks) for x in st_],
                    color=[colors[x.chapter] for x in st_], line=dict(width=1.5, color="white")),
        hovertext=[f"<b>{x.chapter}</b><br>{x.n_chunks} chunks · {x.words} words<br>breadth {x.breadth:.3f}"
                   for x in st_], hoverinfo="text"))
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), dragmode="pan")
    return fig


# --- concept / person hub --------------------------------------------------------

def hub_figure(hub: ConceptHub, height: int = 560) -> go.Figure:
    """Radial hub-and-spoke. Hover a node for explanation, page and the supporting quote."""
    palette = px.colors.qualitative.Set2
    n = len(hub.nodes)
    angles = [np.pi / 2 - 2 * np.pi * i / max(n, 1) for i in range(n)]
    pts = [(np.cos(a), np.sin(a)) for a in angles]
    fig = go.Figure()
    for node, (x, y) in zip(hub.nodes, pts):
        fig.add_trace(go.Scatter(x=[0, x], y=[0, y], mode="lines", showlegend=False,
                                 line=dict(color="#bbb", width=2), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[x * 0.66], y=[y * 0.66], mode="text", text=[node.relationship], showlegend=False,
            textfont=dict(size=10, color="#666"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers+text", showlegend=False,
        text=[nd.label for nd in hub.nodes],
        textposition=["top center" if p[1] > 0.3 else "bottom center" if p[1] < -0.3
                      else "middle right" if p[0] > 0 else "middle left" for p in pts],
        textfont=dict(size=12),
        marker=dict(size=34, color=[palette[i % len(palette)] for i in range(n)],
                    line=dict(width=2, color="white")),
        hovertext=[f"<b>{nd.label}</b> — {nd.relationship}<br>{_wrap(nd.explanation, 55, 5)}<br>"
                   f"<i>p. {', '.join(nd.pages)}</i><br>“{_wrap(nd.evidence_quote, 55, 3)}”"
                   for nd in hub.nodes], hoverinfo="text"))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text", showlegend=False, text=[f"<b>{hub.central_concept}</b>"],
        textposition="middle center", textfont=dict(size=13),
        marker=dict(size=92, color="#ffd54f", line=dict(width=3, color="white")),
        hovertext=_wrap(hub.summary, 55, 6) or hub.central_concept, hoverinfo="text"))
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False, range=[-1.9, 1.9]),
                      yaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="x"),
                      dragmode="pan")
    return fig


# --- timelines --------------------------------------------------------------------

def _event_hover(e) -> str:
    imp = f"<br><i>{_wrap(e.importance, 55, 2)}</i>" if e.importance else ""
    return (f"<b>{e.date_label} — {e.title}</b><br>{_wrap(e.description, 55, 4)}{imp}"
            f"<br>p. {', '.join(e.pages)} · {e.section}")


def timeline_simple(data: TimelineData) -> go.Figure:
    """Vertical study timeline in chronological order (order matters here, not year
    gaps). Marker colour = the guide section the event comes from."""
    ev = data.events
    sections = list(dict.fromkeys(e.section for e in ev))
    palette = px.colors.qualitative.Dark24
    color = {sec: palette[i % len(palette)] for i, sec in enumerate(sections)}
    ys = list(range(len(ev)))
    fig = go.Figure(go.Scatter(x=[0, 0], y=[-0.6, len(ev) - 0.4], mode="lines", showlegend=False,
                               line=dict(color="#bbb", width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[0] * len(ev), y=ys, mode="markers+text", showlegend=False,
        text=[f"<b>{e.date_label}</b>  {e.title}" for e in ev], textposition="middle right",
        textfont=dict(size=12),
        marker=dict(size=13, color=[color[e.section] for e in ev], line=dict(width=2, color="white")),
        hovertext=[_event_hover(e) for e in ev], hoverinfo="text"))
    fig.update_layout(template="plotly_white", height=max(300, 34 * len(ev) + 40),
                      margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False, range=[-0.1, 5]),
                      yaxis=dict(visible=False, autorange="reversed"))
    return fig


def timeline_layered(data: TimelineData) -> go.Figure:
    """Rows are the guide's own section headings (earliest first); x is the real year.
    Ranges are thick bars. Row labels give the year span and event count; details on hover."""
    ev = data.events
    first: dict[str, int] = {}
    for e in ev:
        first.setdefault(e.section, e.start_year)
    rows = sorted(first, key=lambda sec: first[sec])
    palette = px.colors.qualitative.Dark24
    label_of = {}
    fig = go.Figure()
    for i, sec in enumerate(rows):
        mine = [e for e in ev if e.section == sec]
        lo = min(e.start_year for e in mine)
        hi = max(e.end_year or e.start_year for e in mine)
        label_of[sec] = f"{sec} · {lo}" + (f"–{hi}" if hi != lo else "") + f" · {len(mine)}"
        color = palette[i % len(palette)]
        for e in mine:
            if e.end_year and e.end_year > e.start_year:
                fig.add_trace(go.Scatter(x=[e.start_year, e.end_year], y=[label_of[sec]] * 2, mode="lines",
                                         showlegend=False, line=dict(color=color, width=10),
                                         hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[e.start_year for e in mine], y=[label_of[sec]] * len(mine), mode="markers", showlegend=False,
            marker=dict(size=12, color=color, line=dict(width=1.5, color="white")),
            hovertext=[_event_hover(e) for e in mine], hoverinfo="text"))
    fig.update_layout(template="plotly_white", height=140 + 48 * len(rows),
                      margin=dict(l=0, r=10, t=10, b=0), xaxis=dict(title="year", gridcolor="#eee"),
                      yaxis=dict(autorange="reversed", categoryorder="array",
                                 categoryarray=[label_of[r] for r in rows]),
                      dragmode="pan")
    return fig
