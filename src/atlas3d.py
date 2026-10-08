"""Thin Streamlit wrapper around the three.js 3-D atlas (src/atlas3d.js).

The component draws a 3-D UMAP projection of the stored embeddings and sends
the clicked chunk id back to Python. Like the 2-D map it is for looking
around only; similarity comes from the original embeddings.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_HTML = """
<div class="a3d">
  <div class="tip"></div>
  <div class="hint">drag to rotate · scroll to zoom · click a point</div>
</div>
"""

_CSS = """
.a3d { position: relative; width: 100%; height: 600px; border: 1px solid #e3e3e3;
       border-radius: 10px; overflow: hidden; background: radial-gradient(#ffffff, #f1f3f6);
       font: 13px/1.4 system-ui, sans-serif; color: #222; }
.a3d canvas { display: block; width: 100%; height: 100%; cursor: grab; }
.tip { position: absolute; display: none; max-width: 260px; padding: 6px 9px; pointer-events: none;
       background: rgba(255,255,255,.96); border: 1px solid #ccc; border-radius: 6px; font-size: 12px;
       white-space: pre-line; }
.hint { position: absolute; left: 10px; bottom: 8px; font-size: 11px; color: #777; pointer-events: none; }
"""

_component = st.components.v2.component(
    "discover_canada_atlas_3d",
    html=_HTML,
    css=_CSS,
    js=(Path(__file__).parent / "atlas3d.js").read_text(),
)


def _tooltip(r) -> str:
    return f"{r.section}\np. {r.page} · {r.block_type}\n{r.chapter}\n{r.preview[:110]}"


def atlas_3d(
    df3: pd.DataFrame,
    colors: dict[str, str],
    *,
    key: str,
    visible_ids: set[str] | None = None,
    selected_id: str | None = None,
    neighbour_ids: list[str] | None = None,
    retrieved_ids: list[str] | None = None,
    query_xyz: tuple[float, float, float] | None = None,
    emphasize: bool = False,
) -> str | None:
    """Render the 3-D atlas; return the chunk id clicked on this run, if any."""
    points = [
        {"id": r.chunk_id, "x": r.x, "y": r.y, "z": r.z, "c": colors[r.chapter],
         "v": visible_ids is None or r.chunk_id in visible_ids, "t": _tooltip(r)}
        for r in df3.itertuples()
    ]
    result = _component(
        data={
            "points": points,
            "selected": selected_id,
            "neighbours": neighbour_ids or [],
            "retrieved": retrieved_ids or [],
            "query": list(query_xyz) if query_xyz else None,
            "emphasize": emphasize,
        },
        key=key,
        on_selected_change=lambda: None,
    )
    return result.selected
