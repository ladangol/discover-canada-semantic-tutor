"""Shared Streamlit UI for the infographics (used by app.py and study_app.py).

`render_hub_infographic` / `render_timeline_infographic` take validated learning objects and
show: an interactive view (hover/click), an evidence panel, and blog exports (SVG / PNG).
All drawing happens in src/infographic_*; this module only wires controls and the component.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.corpus import Corpus
from src.infographic_layout import fit_count, layout_timeline, max_events
from src.infographic_models import (
    HubSpec, TimelineSpec, hub_spec, pages_text, timeline_spec,
)
from src.infographic_renderers import (
    export_name, hub_svg, svg_to_png, timeline_svg,
)
from src.infographic_themes import CANVASES, THEMES
from src.learning_models import ConceptHub, TimelineData
from src.retrieval import page_label

_CSS = """
.ig { position: relative; width: 100%; margin: 0 auto; }
.stage svg { width: 100%; height: auto; display: block; border-radius: 14px;
             box-shadow: 0 1px 6px rgba(0,0,0,.10); }
.stage g.card:not(.center), .stage g.edge, .stage g.pill { transition: opacity .15s; }
.stage g.card.node, .stage g.card.event { cursor: pointer; }
.ig.hover .stage g.card:not(.center):not(.hl), .ig.hover .stage g.edge:not(.hl),
.ig.hover .stage g.pill:not(.hl) { opacity: .28; }
.stage g.hl rect { stroke-width: 2.4; }
.stage g.sel > rect { stroke-width: 3.2; }
.tip { position: absolute; display: none; max-width: 260px; padding: 9px 11px; z-index: 5;
       pointer-events: none; background: #fff; color: #1f2430; border: 1px solid #d8dbe2; border-radius: 10px;
       box-shadow: 0 6px 18px rgba(0,0,0,.18); font: 13px/1.4 system-ui, sans-serif; }
.tip .tt { font-weight: 700; margin-bottom: 3px; }
.tip .tw { margin-top: 5px; font-size: 11.5px; color: #6b7280; }
"""

_component = st.components.v2.component(
    "discover_canada_infographic",
    html='<div class="ig"><div class="stage"></div><div class="tip"></div></div>',
    css=_CSS,
    js=(Path(__file__).parent / "infographic_viewer.js").read_text(),
)


def _view(svg_wide: str, svg_narrow: str | None, details: dict, *, key: str, selected: str | None,
          max_width: int | None = None) -> str | None:
    res = _component(
        data={"svg_wide": svg_wide, "svg_narrow": svg_narrow, "details": details,
              "selected": selected, "max_width": max_width},
        key=key, on_selected_change=lambda: None)
    return res.selected


def _download_buttons(key: str, title: str, canvas: str, build_svg) -> None:
    c1, c2 = st.columns(2)
    c1.download_button("SVG (scalable)", data=lambda: build_svg().encode("utf-8"),
                       file_name=export_name(title, canvas, "svg"), mime="image/svg+xml",
                       on_click="ignore", key=f"{key}_svg", icon=":material/download:", width="stretch")
    c2.download_button("PNG (high resolution)", data=lambda: svg_to_png(build_svg(), 2.0),
                       file_name=export_name(title, canvas, "png"), mime="image/png",
                       on_click="ignore", key=f"{key}_png", icon=":material/image:", width="stretch")


# --- concept / person hub -----------------------------------------------------------------

def _hub_details(spec: HubSpec) -> dict:
    return {n.id: {"title": n.label, "text": n.explanation,
                   "where": f"Discover Canada, {pages_text(n.pages)}" + (f" · {n.section}" if n.section else "")}
            for n in spec.nodes}


def render_hub_infographic(hub: ConceptHub, corpus: Corpus, *, key: str, learner: bool = False) -> None:
    spec = hub_spec(hub, corpus)
    theme = st.segmented_control("Style", list(THEMES), default="Editorial", key=f"{key}_theme",
                                 label_visibility="collapsed") or "Editorial"
    if not spec.nodes:
        st.warning("The guide does not state enough connected facts about this topic to draw a map.")
    sel_key = f"{key}_sel"
    clicked = _view(hub_svg(spec, theme, "landscape", interactive=True, responsive=True),
                    hub_svg(spec, theme, "portrait", interactive=True, responsive=True),
                    _hub_details(spec), key=f"{key}_view", selected=st.session_state.get(sel_key))
    if clicked and clicked != st.session_state.get(sel_key):
        st.session_state[sel_key] = clicked
        st.rerun()
    st.caption("Hover a card to highlight its link; click it to see where it comes from in the guide.")

    node = next((n for n in spec.nodes if n.id == st.session_state.get(sel_key)), None)
    if node:
        with st.container(border=True):
            st.markdown(f"**{node.label}** — _{node.relationship}_")
            st.write(node.explanation)
            st.caption(f"Discover Canada, {pages_text(node.pages)}" + (f" · {node.section}" if node.section else ""))
            with st.expander("Show evidence from the guide"):
                st.markdown(f"> {node.quote}")
                if not learner:
                    for cid in node.source_chunk_ids:
                        row = corpus.row(cid)
                        st.caption(f"{page_label(row.page)} · {row.section}")
                        st.write(row.body)
    if learner:                       # the learner app has no blog export
        return
    with st.popover("Export for blog", icon=":material/ios_share:"):
        canvas = st.selectbox("Layout", list(CANVASES), format_func=lambda c: CANVASES[c].label, key=f"{key}_xc")
        theme_x = st.selectbox("Style", list(THEMES), index=list(THEMES).index(theme), key=f"{key}_xt")
        _download_buttons(key, spec.title, canvas, lambda: hub_svg(spec, theme_x, canvas))
        st.caption("Exports contain the title, labels, relationships and a source footer only.")


# --- timeline -------------------------------------------------------------------------------

def _event_details(spec: TimelineSpec) -> dict:
    return {e.id: {"title": f"{e.date_label} — {e.title}", "text": e.description,
                   "where": f"Discover Canada, {pages_text(e.pages)} · {e.band}"} for e in spec.events}


def render_timeline_infographic(data: TimelineData, corpus: Corpus, *, key: str, learner: bool = False) -> None:
    c1, c2 = st.columns([3, 2])
    style_label = c1.segmented_control("Layout", ["Chronological", "Layered by period"], default="Chronological",
                                       key=f"{key}_style", label_visibility="collapsed") or "Chronological"
    theme = c2.segmented_control("Style", list(THEMES), default="Editorial", key=f"{key}_theme",
                                 label_visibility="collapsed") or "Editorial"
    style = "layered" if style_label.startswith("Layered") else "chronological"
    spec = timeline_spec(data)
    if not spec.events:
        st.warning("No dated events in the guide for this request.")
    sel_key = f"{key}_sel"
    clicked = _view(timeline_svg(spec, theme, "portrait", style, interactive=True, responsive=True), None,
                    _event_details(spec), key=f"{key}_view_{style}", selected=st.session_state.get(sel_key),
                    max_width=760)
    if clicked and clicked != st.session_state.get(sel_key):
        st.session_state[sel_key] = clicked
        st.rerun()
    st.caption("Hover an event to see its description and page; click it for the evidence.")

    ev = next((e for e in spec.events if e.id == st.session_state.get(sel_key)), None)
    if ev:
        with st.container(border=True):
            st.markdown(f"**{ev.date_label} — {ev.title}**")
            st.write(ev.description)
            if ev.importance:
                st.caption(ev.importance)
            st.caption(f"Discover Canada, {pages_text(ev.pages)} · {ev.band}")
            with st.expander("Show evidence from the guide"):
                st.markdown(f"> {ev.quote}")
    if data.undated:
        with st.expander(f"Broad periods without a precise year ({len(data.undated)})"):
            for e in data.undated:
                st.markdown(f"**{e.date_label}** — {e.title}. {e.description} _({page_label(e.pages[0])}, {e.section})_")

    if learner:
        return
    with st.popover("Export for blog", icon=":material/ios_share:"):
        canvas = st.selectbox("Layout", list(CANVASES), format_func=lambda c: CANVASES[c].label,
                              index=len(CANVASES) - 1, key=f"{key}_xc")
        theme_x = st.selectbox("Style", list(THEMES), index=list(THEMES).index(theme), key=f"{key}_xt")
        total = len(spec.events)
        start, stop = 0, None
        if CANVASES[canvas].height is not None and total:
            start = st.number_input("First event", 1, total, 1, key=f"{key}_xs") - 1
            rest = timeline_spec(data, start=start)
            n = fit_count(rest, THEMES[theme_x], canvas, style)
            count = st.number_input("Events in image", 1, max(total - start, 1), min(max(n, 1), total - start),
                                    key=f"{key}_xn_{canvas}_{style}_{start}")
            stop = start + int(count)
            cap = max_events(canvas, style)
            sub = timeline_spec(data, start=start, stop=stop)
            if layout_timeline(sub, THEMES[theme_x], canvas, style).overflow:
                st.caption(f"{n} events fit this shape cleanly; more will make the image taller than "
                           f"{CANVASES[canvas].label.split()[0].lower()}.")
            elif cap:
                st.caption(f"Up to {cap} events fit side by side; {n} fit with their descriptions.")
        else:
            st.caption("Portrait includes every event, however many.")
        sub = timeline_spec(data, start=start, stop=stop)
        _download_buttons(key, sub.title, canvas, lambda: timeline_svg(sub, theme_x, canvas, style))
