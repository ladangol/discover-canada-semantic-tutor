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

from src.config import DEFAULT_K, EMBEDDING_MAP_PATH, UMAP_REDUCER_PATH
from src.embeddings import encode_query
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


def analyze_query(question: str, k: int = DEFAULT_K) -> QueryView:
    vector = encode_query(question)
    chunks = retrieve(question, k=k, query_vector=vector)
    return QueryView(question=question, chunks=chunks, xy=project_query(vector))


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


def _wrap(text: str, width: int = 60) -> str:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return "<br>".join(lines[:4])


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
