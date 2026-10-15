"""Analyze tab: chapter semantic map, concept/person hub, timeline.

UI only. Statistics come from src/chapter_stats.py (original embeddings, no LLM);
hubs and timelines from src/learning_extractors.py (retrieve -> LLM JSON -> validate).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import chapter_stats, learning_extractors as ex, tutor, visualization as viz
from src.corpus import load_corpus
from src.generation import GenerationNotConfigured
from src.learning_models import ConceptHub, TimelineData
from ui.infographic_components import render_hub_infographic, render_timeline_infographic


@st.cache_data(show_spinner=False)
def _analysis():
    return chapter_stats.analyze_chapters(load_corpus())


def _generate(fn, *args, **kwargs):
    """Run an extractor, turning expected failures into friendly messages."""
    try:
        with st.spinner("Reading the guide and building the structure… (first build can take a minute or two)"):
            return fn(*args, **kwargs)
    except GenerationNotConfigured as exc:
        st.warning(f"{exc} Add it to a local `.env` to build hubs and timelines. "
                   "The chapter semantic map works without it.")
    except (tutor.QuizGenerationError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
    return None


def analyze_tab() -> None:
    corpus = load_corpus()
    view = st.radio("Analysis", ["Chapter semantic map", "Concept hub", "Timeline"], horizontal=True,
                    key="analyze_view")
    if view == "Chapter semantic map":
        chapter_map(corpus)
    elif view == "Concept hub":
        concept_hub(corpus)
    else:
        timeline(corpus)


# --- 1. chapter semantic map ----------------------------------------------------------

def chapter_map(corpus) -> None:
    analysis = _analysis()
    colors = viz.chapter_colors(corpus.chapters)
    st.caption("This map describes similarity in the **embedding space**, computed from the original "
               "384-d MiniLM vectors (chapter centroids and cosine). It does not mean that one "
               "chapter historically causes or depends on another, and it does not use UMAP.")
    mode = st.radio("View", ["Similarity matrix", "Semantic network", "Breadth table"], horizontal=True)
    if mode == "Similarity matrix":
        st.plotly_chart(viz.chapter_heatmap(analysis), width="stretch", key="chapter_heatmap")
        st.caption("Cell = cosine similarity between the two chapters' centroids (mean embedding). "
                   "Diagonal left blank.")
    elif mode == "Semantic network":
        n = st.slider("Strongest links kept per chapter", 1, 4, 2)
        st.plotly_chart(viz.chapter_network(analysis, colors, n), width="stretch", key="chapter_net")
        st.caption("Node size = number of chunks. Edge width and label = centroid cosine. Rule: every "
                   "chapter keeps its N strongest links. Node positions are an approximate MDS of "
                   "centroid distances.")
    else:
        rows = [{"chapter": s.chapter, "chunks": s.n_chunks, "words": s.words,
                 "semantic breadth": round(s.breadth, 3),
                 "same-chapter nearest-neighbour rate": round(s.same_chapter_nn_rate, 2),
                 "nearest chapters": "; ".join(f"{c} ({v:.2f})" for c, v in s.nearest[:2])}
                for s in sorted(analysis.stats, key=lambda s: -s.breadth)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("**Semantic breadth** = mean cosine distance of a chapter's chunks from its centroid: "
                   "higher means the chapter covers a wider variety of content. A one-chunk chapter has "
                   "breadth 0 by construction. **Nearest-neighbour rate** is inflated by overlapping "
                   "chunk windows within a section.")


# --- 2. concept / person hub ------------------------------------------------------------

def concept_hub(corpus) -> None:
    st.caption("Embeddings choose which passages to read; the source text states each relationship; "
               "a validator drops anything without a verbatim supporting quote.")
    with st.form("hub_form"):
        concept = st.text_input("Concept or person", placeholder="Prime Minister")
        c1, c2, c3 = st.columns([2, 2, 2])
        kind_label = c1.radio("Type", ["Concept", "Person"], horizontal=True)
        max_nodes = c2.slider("Max nodes", 4, 8, 6)
        fresh = c3.checkbox("Rebuild (ignore cache)")
        go_ = st.form_submit_button("Build concept hub", type="primary")
    if go_ and concept.strip():
        res = _generate(ex.build_concept_hub, concept, corpus,
                        kind=kind_label.lower(), max_nodes=max_nodes, use_cache=not fresh)
        st.session_state.hub = res
    res = st.session_state.get("hub")
    if not res:
        return
    hub, cached = res
    show_hub(hub, cached, corpus)


def show_hub(hub: ConceptHub, cached: bool, corpus) -> None:
    render_hub_infographic(hub, corpus, key="hub")
    st.caption(("Loaded from cache · " if cached else "") + f"{len(hub.nodes)} source-supported nodes.")
    if hub.omitted:
        with st.expander(f"{len(hub.omitted)} item(s) omitted by validation"):
            st.write("\n".join(f"- {o}" for o in hub.omitted))


# --- 3. timeline ---------------------------------------------------------------------------

def timeline(corpus) -> None:
    st.caption("Embeddings are used only to find passages; every date comes from the source text and "
               "is checked against it. Chronology is never inferred from the map.")
    scope = st.radio("Build from", ["Entire chapter", "Section", "Topic query"], horizontal=True)
    chapter = section = topic = None
    if scope == "Entire chapter":
        chapter = st.selectbox("Chapter", corpus.chapters,
                               index=corpus.chapters.index("Canada’s History")
                               if "Canada’s History" in corpus.chapters else 0)
    elif scope == "Section":
        chapter = st.selectbox("Chapter", corpus.chapters, key="tl_chapter")
        section = st.selectbox("Section", corpus.sections(chapter))
    else:
        topic = st.text_input("Topic", placeholder="voting rights")
    fresh = st.checkbox("Rebuild (ignore cache)", key="tl_fresh")
    if st.button("Build timeline", type="primary") and (chapter or topic):
        res = _generate(ex.build_timeline, corpus, chapter=chapter, section=section,
                        topic=topic, use_cache=not fresh)
        st.session_state.timeline = res
    res = st.session_state.get("timeline")
    if res:
        show_timeline(res[0], res[1], corpus)


def show_timeline(data: TimelineData, cached: bool, corpus) -> None:
    st.subheader(data.title)
    if not data.events and not data.undated:
        st.warning("No source-supported dated events were found for this request.")
        return
    st.caption(("Loaded from cache · " if cached else "") +
               f"{len(data.events)} dated events, {len(data.undated)} broad-period items.")
    render_timeline_infographic(data, corpus, key="tl")
    if data.omitted:
        with st.expander(f"{len(data.omitted)} item(s) omitted or merged by validation"):
            st.write("\n".join(f"- {o}" for o in data.omitted))
