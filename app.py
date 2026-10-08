"""Discover Canada Atlas — thin Streamlit UI. Logic lives in src/."""
from __future__ import annotations

import random

import pandas as pd
import streamlit as st

from src import tutor, visualization as viz
from src.config import DEFAULT_K, EMBEDDING_MODEL_NAME, generation_configured, generation_model
from src.corpus import load_corpus
from src.generation import GenerationNotConfigured
from src.retrieval import page_label

st.set_page_config(page_title="Discover Canada Atlas", page_icon="🍁", layout="wide")


# --- cached resources ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading corpus…")
def corpus():
    return load_corpus()


@st.cache_data(show_spinner="Loading atlas…")
def atlas_df() -> pd.DataFrame:
    return viz.load_map()


@st.cache_resource(show_spinner="Warming up the encoder and map projection…")
def warm_up() -> bool:
    """Pay the one-time model load / UMAP JIT cost at startup, not on the first question."""
    viz.analyze_query("warm up")
    return True


def init_state() -> None:
    st.session_state.setdefault("selected_chunk", None)
    st.session_state.setdefault("last_click", None)
    st.session_state.setdefault("query_view", None)
    st.session_state.setdefault("ask_result", None)
    st.session_state.setdefault("history", [])           # list[tutor.AnswerRecord]
    st.session_state.setdefault("quiz", None)            # current question + outcome
    st.session_state.setdefault("asked_ids", set())
    st.session_state.setdefault("review_expl", None)
    st.session_state.setdefault("rng", random.Random())


def run_generation(fn, *args, **kwargs):
    """Call a tutor function, turning expected failures into friendly messages."""
    try:
        with st.spinner("Thinking…"):
            return fn(*args, **kwargs)
    except GenerationNotConfigured as exc:
        st.warning(f"{exc} Add it to a local `.env` (see `.env.example`) to enable the tutor. "
                   "Retrieval and the atlas work without it.")
    except (tutor.QuizGenerationError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
    return None


def show_answer(result: tutor.TutorAnswer) -> None:
    st.markdown(result.answer)
    if result.truncated:
        st.caption("⚠️ The answer hit the output limit and may be cut off.")
    if result.cited:
        with st.expander(f"Sources cited ({len(result.cited)})"):
            for c in result.cited:
                st.markdown(f"**{tutor.citation_label(c)}**")
                st.caption(tutor.split_text(c.text))


def chunk_table(rows: list[dict]) -> None:
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# --- sidebar -------------------------------------------------------------------

def sidebar() -> None:
    with st.sidebar:
        st.title("🍁 Discover Canada Atlas")
        st.caption("Explore, question, and study the citizenship guide through its embedding space.")
        st.metric("Chunks indexed", len(corpus()))
        st.caption(f"Retrieval (frozen): `{EMBEDDING_MODEL_NAME.split('/')[-1]}` · cosine · dense · k={DEFAULT_K}")
        if generation_configured():
            st.success(f"Tutor ready · `{generation_model()}`")
        else:
            st.warning("VENICE_API_KEY is not configured. Atlas and retrieval work; Ask/Study answers are off.")
        st.divider()
        st.caption("Map positions come from UMAP and are for **visualization only**. "
                   "Retrieval and neighbours use cosine similarity on the original 384-d embeddings.")


# --- Atlas tab ------------------------------------------------------------------

def atlas_tab() -> None:
    c, df = corpus(), atlas_df()
    colors = viz.chapter_colors(c.chapters)

    f1, f2, f3 = st.columns(3)
    chapters = f1.multiselect("Chapter", c.chapters)
    section_options = [s for ch in (chapters or c.chapters) for s in c.sections(ch)]
    sections = f2.multiselect("Section", list(dict.fromkeys(section_options)))
    types = f3.multiselect("Block type", sorted(df["block_type"].unique()))

    mask = pd.Series(True, index=df.index)
    if chapters:
        mask &= df["chapter"].isin(chapters)
    if sections:
        mask &= df["section"].isin(sections)
    if types:
        mask &= df["block_type"].isin(types)
    visible = df[mask]
    visible_ids = set(visible["chunk_id"])

    selected = st.session_state.selected_chunk
    neighbours = c.neighbours(selected, 5) if selected else []
    fig = viz.atlas_figure(df, colors, visible_ids=visible_ids, selected_id=selected,
                           neighbour_ids=[c.chunks.chunk_id.iat[j] for j, _ in neighbours])

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.caption(f"{len(visible)} of {len(df)} chunks shown · click a point to inspect it")
        event = st.plotly_chart(fig, key="atlas_plot", on_select="rerun",
                                selection_mode="points", width="stretch")
        pts = event.selection.points if event and event.selection else []
        if pts:
            clicked = pts[0].get("customdata")
            clicked = clicked[0] if isinstance(clicked, list) else clicked
            if clicked and clicked != st.session_state.last_click:
                st.session_state.last_click = clicked
                if clicked != selected:
                    st.session_state.selected_chunk = clicked
                    st.rerun()

    with right:
        ids = list(visible["chunk_id"]) or list(df["chunk_id"])
        labels = {r.chunk_id: f"{page_label(r.page)} · {r.section} · {r.chunk_id.split(':')[-1]}"
                  for r in c.chunks.itertuples()}
        index = ids.index(selected) if selected in ids else None
        pick = st.selectbox("…or pick a chunk", ids, index=index, format_func=labels.get,
                            placeholder="Select a chunk")
        if pick and pick != selected:
            st.session_state.selected_chunk = pick
            st.rerun()
        if not selected:
            st.info("Select a chunk on the map to see its full text and nearest semantic neighbours.")
            return
        inspector(c, selected, neighbours)


def inspector(c, chunk_id: str, neighbours: list[tuple[int, float]]) -> None:
    row = c.row(chunk_id)
    st.subheader(row.section)
    st.caption(f"{row.chapter} · {page_label(row.page)} · {row.block_type} · `{row.chunk_id}`")
    st.write(row.body)

    st.markdown("**Nearest semantic neighbours** — cosine similarity on the original embeddings")
    chunk_table([
        {"rank": n, "cosine": round(sim, 3), "page": c.chunks.page.iat[j],
         "section": c.chunks.section.iat[j], "type": c.chunks.block_type.iat[j],
         "preview": c.chunks.preview.iat[j]}
        for n, (j, sim) in enumerate(neighbours, start=1)
    ])
    st.caption("Ringed points on the map are these neighbours. They may not look close in 2-D — UMAP distorts distance.")

    if st.button("Explain this chunk", key="explain_chunk"):
        result = run_generation(tutor.explain_chunk, c, chunk_id)
        if result:
            show_answer(result)


# --- Ask tab --------------------------------------------------------------------

def ask_tab() -> None:
    c, df = corpus(), atlas_df()
    colors = viz.chapter_colors(c.chapters)

    with st.form("ask_form"):
        question = st.text_input("Ask about Discover Canada", placeholder="What happened during the First World War?")
        b1, b2, b3, _ = st.columns([1, 1, 1.3, 4])
        ask_btn = b1.form_submit_button("Ask", type="primary")
        explain_btn = b2.form_submit_button("Explain")
        retrieve_btn = b3.form_submit_button("Retrieval only")

    pending = None
    if (ask_btn or explain_btn or retrieve_btn) and question.strip():
        question = question.strip()
        with st.spinner("Retrieving…"):
            st.session_state.query_view = viz.analyze_query(question)
        st.session_state.ask_result = None
        pending = "ask" if ask_btn else "explain" if explain_btn else None

    view = st.session_state.query_view
    if view is None:
        st.info("Ask a question to see what the retriever finds — and where it sits on the map.")
        return

    st.markdown(f"#### “{view.question}”")
    mode = st.radio("Map view", ["All chunks", "Retrieved chunks emphasized"], index=1, horizontal=True)
    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = viz.query_figure(df, colors, view, emphasize=mode != "All chunks")
        st.plotly_chart(fig, width="stretch", key="query_plot")
        st.caption("★ your question · numbered red dots are the retrieved top-5. 2-D distance is not "
                   "retrieval similarity — a retrieved chunk can look far from the star.")
    with right:
        st.markdown(f"**What the retriever sees** (top {DEFAULT_K}, cosine)")
        for n, ch in enumerate(view.chunks, start=1):
            with st.container(border=True):
                st.markdown(f"**#{n}** · cos **{ch.similarity:.3f}** · {page_label(ch.page)} · "
                            f"{ch.section} · _{ch.block_type}_")
                st.caption(c.row(ch.chunk_id).preview)

    # Generation runs after the retrieval view is on screen.
    if pending:
        fn = tutor.ask if pending == "ask" else tutor.explain_question
        result = run_generation(fn, view.question, view.chunks)
        label = "Answer" if pending == "ask" else "Explanation"
        st.session_state.ask_result = (label, result) if result else None

    saved = st.session_state.ask_result
    if saved:
        heading, result = saved
        st.markdown(f"### {heading}")
        show_answer(result)
        with st.expander("Retrieved source text"):
            for n, ch in enumerate(view.chunks, start=1):
                st.markdown(f"**#{n} · {tutor.citation_label(ch)}** · cos {ch.similarity:.3f}")
                st.caption(tutor.split_text(ch.text))


# --- Study tab ------------------------------------------------------------------

def scope_picker(c, key: str, *, allow_query: bool = True) -> dict:
    options = ["Whole guide", "Chapter", "Section"]
    if allow_query and st.session_state.query_view:
        options.append("Current query")
    scope = st.radio("Study scope", options, horizontal=True, key=f"{key}_scope")
    chapter = section = None
    if scope == "Chapter":
        chapter = st.selectbox("Chapter", c.chapters, key=f"{key}_chapter")
    elif scope == "Section":
        chapter = st.selectbox("Chapter", c.chapters, key=f"{key}_chapter2")
        section = st.selectbox("Section", c.sections(chapter), key=f"{key}_section")
    return {"scope": scope, "chapter": chapter, "section": section}


def study_tab() -> None:
    c = corpus()
    mode = st.radio("Mode", ["Quiz me", "Flashcards", "Explain a section"], horizontal=True)
    if mode == "Quiz me":
        quiz_ui(c)
    elif mode == "Flashcards":
        flashcards_ui(c)
    else:
        explain_ui(c)


def explain_ui(c) -> None:
    sc = scope_picker(c, "explain", allow_query=False)
    if sc["scope"] == "Whole guide":
        st.info("Choose a chapter or a section to explain.")
        return
    if st.button("Explain", type="primary"):
        result = run_generation(tutor.explain_scope, c, sc["chapter"], sc["section"])
        if result:
            show_answer(result)


def flashcards_ui(c) -> None:
    sc = scope_picker(c, "cards")
    n = st.slider("Number of cards", 3, 8, 5)
    if st.button("Make flashcards", type="primary"):
        view = st.session_state.query_view
        kwargs = ({"question": view.question, "query_chunks": view.chunks}
                  if sc["scope"] == "Current query" else
                  {"chapter": sc["chapter"], "section": sc["section"]})
        cards = run_generation(tutor.generate_flashcards, c, n=n, **kwargs)
        st.session_state.cards = cards
    for card in st.session_state.get("cards") or []:
        with st.container(border=True):
            st.markdown(f"**{card.front}**")
            with st.expander("Show answer"):
                st.write(card.back)
                st.caption(f"Discover Canada, {page_label(card.page)}, {card.section}")


def record(q: tutor.QuizQuestion, correct: bool) -> None:
    st.session_state.history.append(tutor.AnswerRecord(q.chunk_id, q.source.section, q.question, correct))


def new_question(c, sc: dict, kind: str, *, focus: str | None = None, review_of: str | None = None) -> None:
    view = st.session_state.query_view
    kwargs = dict(kind=kind, exclude_ids=st.session_state.asked_ids, rng=st.session_state.rng)
    if focus:
        kwargs["focus_chunk_id"] = focus
    elif sc["scope"] == "Current query" and view:
        kwargs["candidate_ids"] = [ch.chunk_id for ch in view.chunks]
    else:
        kwargs.update(chapter=sc["chapter"], section=sc["section"])
    q = run_generation(tutor.generate_quiz, c, **kwargs)
    st.session_state.quiz = {"q": q, "result": None, "id": random.random()} if q else None
    st.session_state.review_expl = None
    if q:
        st.session_state.asked_ids.add(q.chunk_id)


def quiz_ui(c) -> None:
    sc = scope_picker(c, "quiz")
    kind_label = st.radio("Question type", ["Multiple choice", "Free response"], horizontal=True)
    kind = "mcq" if kind_label == "Multiple choice" else "free"

    if st.button("New question", type="primary"):
        new_question(c, sc, kind)

    state = st.session_state.quiz
    if state and state["q"]:
        q: tutor.QuizQuestion = state["q"]
        st.markdown(f"#### {q.question}")
        key = f"quiz_{state['id']}"
        if state["result"] is None:
            if q.kind == "mcq":
                choice = st.radio("Your answer", range(len(q.options)), index=None,
                                  format_func=lambda i: f"{'ABCD'[i]}. {q.options[i]}", key=key)
                if st.button("Check answer", disabled=choice is None):
                    state["result"] = tutor.grade_mcq(q, choice)
                    record(q, state["result"].correct)
                    st.rerun()
            else:
                answer = st.text_area("Your answer", key=key)
                if st.button("Check answer", disabled=not answer.strip()):
                    res = run_generation(tutor.grade_free, q, answer)
                    if res:
                        state["result"] = res
                        record(q, res.correct)
                        st.rerun()
        else:
            res: tutor.QuizResult = state["result"]
            (st.success if res.correct else st.error)(res.feedback)
            if res.explanation:
                st.write(res.explanation)
            st.caption(f"Source: {tutor.citation_label(res.source)}")
            review_panel(c, sc, kind, q, res)

    score_panel(c)


def review_panel(c, sc: dict, kind: str, q: tutor.QuizQuestion, res: tutor.QuizResult) -> None:
    """After a miss: semantic neighbours of the missed concept (adaptive review)."""
    cols = st.columns([1, 1, 3])
    if cols[0].button("Next question"):
        new_question(c, sc, kind)
        st.rerun()
    if res.correct:
        return
    near = tutor.review_candidates(c, q.chunk_id, st.session_state.history)
    st.markdown("##### Review the neighbourhood of this concept")
    st.caption("Nearby ideas by cosine similarity of the stored embeddings:")
    for j in near:
        st.markdown(f"- **{c.chunks.section.iat[j]}** · {page_label(c.chunks.page.iat[j])}")
    if cols[1].button("Quiz nearby", disabled=not near):
        review_kind = kind
        cand = c.chunks.chunk_id.iat[st.session_state.rng.choice(near)]
        new_question(c, sc, review_kind, focus=cand)
        st.rerun()
    if st.button("Explain this neighbourhood"):
        st.session_state.review_expl = run_generation(tutor.explain_neighbourhood, c, q.chunk_id)
    if st.session_state.review_expl:
        show_answer(st.session_state.review_expl)


def score_panel(c) -> None:
    history = st.session_state.history
    if not history:
        return
    right = sum(r.correct for r in history)
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Correct", right)
    m2.metric("Missed", len(history) - right)
    weak = tutor.weak_sections(history)
    if weak:
        st.markdown("**Worth reviewing:** " + ", ".join(f"{s}" for s, _ in weak[:5]))
    if st.button("Reset session"):
        st.session_state.history = []
        st.session_state.asked_ids = set()
        st.session_state.quiz = None
        st.rerun()


# --- main -----------------------------------------------------------------------

def main() -> None:
    init_state()
    try:
        corpus()
        atlas_df()
        warm_up()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    sidebar()
    atlas, ask, study = st.tabs(["🗺️ Atlas", "💬 Ask", "📚 Study"])
    with atlas:
        atlas_tab()
    with ask:
        ask_tab()
    with study:
        study_tab()


main()
