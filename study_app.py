"""Discover Canada Study: a simple, friendly citizenship-test study app.

Same engine as app.py (frozen retrieval, tutor, validated hubs/timelines, infographic renderer),
without the technical views: no embeddings, UMAP, similarity scores, rankings or chunk ids.
Run:  streamlit run study_app.py
"""
from __future__ import annotations

import html
import logging
import os
import random

import streamlit as st

from src import budget, facts, learning_extractors as ex, tutor
from src.config import generation_configured, voice_enabled
from src.corpus import load_corpus
from src.embeddings import encode_query
from src.retrieval import page_label, retrieve
from ui import study_theme as theme
from ui.common import generate_or_explain
from ui.chapter_carousel import chapter_carousel
from ui.flashcards import flashcard_deck
from ui.infographic_components import render_hub_infographic, render_timeline_infographic
from ui import voice_tutor
from ui.theme_sync import sync as sync_theme
from ui.session_limits import LIMIT_MESSAGE, charge, clean_topic, refund

logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
st.set_page_config(page_title="Discover Canada Study", page_icon="static/favicon.png", layout="wide",
                   initial_sidebar_state="collapsed")

EVERYWHERE = "Anywhere in the guide"
GOAL = 5                                                  # "daily goal" shown in the dashboard


@st.cache_resource(show_spinner="Getting things ready…")
def corpus():
    c = load_corpus()
    encode_query("warm up")            # load the search model once, not on the first question
    return c


def init_state() -> None:
    for k, v in {"page": "Learn", "history": [], "asked": set(), "quiz": None, "cards": None, "learn": None,
                 "rng": random.Random(), "study_hub": None, "study_tl": None, "voice_quiz": None}.items():
        st.session_state.setdefault(k, v)
    if "fact" not in st.session_state:
        st.session_state.fact = facts.pick(st.session_state.rng)
        st.session_state.mascot = st.session_state.rng.choice(theme.MASCOTS)


def cite(chunk) -> str:
    return f"{page_label(chunk.page)} · {chunk.section}"


def mascot() -> str:
    return st.session_state.get("mascot", theme.MASCOTS[0])


def another_fact() -> None:
    st.session_state.fact = facts.pick(st.session_state.rng, avoid=st.session_state.fact)
    st.session_state.mascot = st.session_state.rng.choice(theme.MASCOTS)


def accent(page: str) -> str:
    return theme.PAGES[page][2]


def go_to(page: str) -> None:
    st.session_state.page = page


# --- shell: hero, navigation, footer ----------------------------------------------------------------

def nav() -> None:
    with st.container(key="nav"):
        pages = {k: v for k, v in theme.PAGES.items() if k != "Talk" or voice_enabled()}   # Talk is opt-in (STUDY_VOICE=1)
        cols = st.columns(len(pages))
        for col, (name, (label, icon, _)) in zip(cols, pages.items()):
            k = name.lower().replace(" ", "_")
            on = st.session_state.page == name
            col.button(label, icon=icon, key=f"nav_{k}{'_on' if on else ''}", width="stretch",
                       on_click=go_to, args=(name,))


def header(page: str, text: str) -> None:
    label, _, color = theme.PAGES[page]
    icon = {"Learn": "school", "Practice": "bolt", "Flashcards": "style", "Concept map": "hub", "Timeline": "timeline",
            "Talk": "record_voice_over"}[page]
    st.html(theme.section_header(icon, label, text, color))


def suggestions(items: list[str], prefix: str, target_key: str) -> str | None:
    """A row of one-tap example chips; returns the chosen text."""
    picked = None
    with st.container(key=f"chips_{prefix}", horizontal=True):
        for n, item in enumerate(items):
            if st.button(item, key=f"{prefix}_{n}", icon=":material/bolt:"):
                picked = item
    return picked


# --- Learn ----------------------------------------------------------------------------------------------

def show_answer(result: tutor.TutorAnswer) -> None:
    with st.container(key="answer"):
        st.html(f'<div class="ans-head"><img src="{theme.mascot_url(mascot())}" alt=""><span>Here’s what the guide says</span></div>')
        st.markdown(result.answer)
    if result.cited:
        st.html(theme.pills([cite(c) for c in result.cited], accent("Learn")))


def learn_page() -> None:
    c = corpus()
    fact = facts.FACTS[st.session_state.fact]
    st.html(theme.hero(mascot(), fact.text, page_label(fact.page)))
    with st.container(key="chips_fact", horizontal=True):
        st.button("Another fact", icon=":material/casino:", on_click=another_fact, key="fact_btn")

    header("Learn", "Ask anything about Canada, or open a chapter for a quick summary.")
    with st.form("learn_form", border=False):
        st.html('<div class="ask-title"><span class="mi">forum</span> What do you want to know?</div>')
        c1, c2 = st.columns([5, 1.2], vertical_alignment="bottom")
        text = c1.text_input("Your question", placeholder="e.g. Who can vote? · What is responsible government?",
                             max_chars=120, icon=":material/search:", label_visibility="collapsed")
        go = c2.form_submit_button("Ask", type="primary", icon=":material/arrow_forward:", width="stretch")
    answer_slot = st.container()            # the answer lands here, right under the question box

    st.markdown("##### Or explore a chapter")
    items = [{"title": ch, "icon": theme.CHAPTER_STYLE.get(ch, theme.DEFAULT_STYLE)[0],
              "a": theme.CHAPTER_STYLE.get(ch, theme.DEFAULT_STYLE)[1], "b": theme.CHAPTER_STYLE.get(ch, theme.DEFAULT_STYLE)[2]}
             for ch in c.chapters]
    clicked = chapter_carousel(items, key="chapter_carousel")

    topic = clean_topic(text)
    if (go and topic) or clicked:
        with answer_slot:
            if not charge():
                st.warning(LIMIT_MESSAGE)
            elif clicked:
                st.session_state.learn = generate_or_explain(tutor.explain_scope, c, clicked, None)
            else:
                chunks = retrieve(tutor.search_text(topic))   # same frozen retrieval; boilerplate like "in Discover Canada" removed from the search string only
                st.session_state.learn = generate_or_explain(tutor.ask, topic, chunks)
    if st.session_state.learn:
        with answer_slot:
            show_answer(st.session_state.learn)


# --- Practice -------------------------------------------------------------------------------------------

def record(q: tutor.QuizQuestion, res: tutor.QuizResult) -> None:
    st.session_state.history.append(tutor.AnswerRecord(q.chunk_id, q.source.section, q.question, res.correct))


def pick_option(i: int) -> None:
    """Tap-to-answer: grade immediately so the options can be coloured in the same run."""
    state = st.session_state.quiz
    if state and state["res"] is None:
        state["pick"] = i
        state["res"] = tutor.grade_mcq(state["q"], i)
        record(state["q"], state["res"])


def streak(history: list) -> int:
    n = 0
    for r in reversed(history):
        if not r.correct:
            break
        n += 1
    return n


def new_quiz_question(c, scope: str, *, kind: str = "mcq", focus: str | None = None, state_key: str = "quiz") -> bool:
    """Generate the next quiz question into session state `state_key` (Practice: "quiz"; Talk: "voice_quiz").
    Answer history and the asked-set are shared, so both pages count toward the same goal."""
    if not charge(2):
        st.warning(LIMIT_MESSAGE)
        return False
    kwargs = dict(kind=kind, difficulty=tutor.DEFAULT_DIFFICULTY, exclude_ids=st.session_state.asked,
                  rng=st.session_state.rng)
    if focus:
        kwargs["focus_chunk_id"] = focus
    else:
        kwargs["chapter"] = None if scope == EVERYWHERE else scope
    q = generate_or_explain(tutor.generate_quiz, c, **kwargs)
    if not q:
        refund(2)
    st.session_state[state_key] = {"q": q, "res": None, "pick": None, "id": random.random()} if q else None
    if q:
        st.session_state.asked.add(q.chunk_id)
    return bool(q)


def practice_page() -> None:
    c = corpus()
    header("Practice", "Tap an answer. You'll see right away how you did, and why.")
    h = st.session_state.history
    right = sum(r.correct for r in h)
    st.html(theme.stats_strip(right, len(h), streak(h)))
    if len(h) < GOAL:
        st.progress(len(h) / GOAL, text=f":material/flag: Goal: {GOAL} questions · {len(h)} done")
    else:
        st.progress(1.0, text=f":material/emoji_events: Goal reached: {len(h)} questions answered!")

    scope = st.selectbox("Topic", [EVERYWHERE] + c.chapters, key="pq_scope")

    def new_question(focus: str | None = None) -> None:
        new_quiz_question(c, scope, focus=focus)

    state = st.session_state.quiz
    if not state or not state["q"]:
        st.html(theme.empty_state(mascot(), "Ready when you are! Press the button for your first question."))
    if st.button("New question" if state else "Start practising", icon=":material/auto_awesome:", type="primary", key="new_q"):
        new_question()
        st.rerun()

    state = st.session_state.quiz
    if not (state and state["q"]):
        return
    q: tutor.QuizQuestion = state["q"]
    res = state["res"]
    st.html(f'<div class="question">{html.escape(q.question)}</div>')

    if q.kind == "mcq":
        with st.container(key="options"):
            for row in (0, 2):
                for col, i in zip(st.columns(2), (row, row + 1)):
                    if i >= len(q.options):
                        continue
                    st_ = "plain"
                    if res is not None:
                        st_ = "right" if i == q.correct_index else "wrong" if i == state["pick"] else "plain"
                    col.button(f"**{'ABCD'[i]}**   {q.options[i]}", key=f"opt_{i}_{st_}", width="stretch",
                               disabled=res is not None, on_click=pick_option, args=(i,))

    if res is not None:
        good = res.correct
        kind, icon, msg = (("good", "check_circle", "Correct! Nicely done.") if good
                           else ("bad", "lightbulb", "Not quite, but now you know!"))
        st.html(f'<div class="banner {kind}"><span class="mi">{icon}</span> {msg}</div>')
        if res.explanation:
            st.markdown(f"**Why:** {res.explanation}")
        st.html(theme.pills([f"From the guide: {cite(res.source)}"], accent("Practice")))
        b1, b2, _ = st.columns([1, 1.4, 2])
        if b1.button("Next question", icon=":material/arrow_forward:", type="primary", key="next_q"):
            new_question()
            st.rerun()
        if not good:
            related = tutor.review_candidates(c, q.chunk_id, st.session_state.history)
            if related and b2.button("Practise related ideas", key="rel_q"):
                new_question(focus=c.chunks.chunk_id.iat[st.session_state.rng.choice(related)])
                st.rerun()
    weak = tutor.weak_sections(h)
    if weak:
        st.html(theme.pills(["Worth another look: " + s for s, _ in weak[:3]], "#E89A0C"))


# --- Flashcards -----------------------------------------------------------------------------------------

def flashcards_page() -> None:
    c = corpus()
    header("Flashcards", "Flip, test yourself, and mark what you've got.")
    c1, c2 = st.columns([3, 1])
    scope = c1.selectbox("Topic", [EVERYWHERE] + c.chapters, key="fc_scope")
    n = c2.number_input("Cards", 3, 8, 5, key="fc_n")
    typed = clean_topic(st.text_input("…or type your own topic", max_chars=120, key="fc_typed",
                                      placeholder="e.g. voting, the Charter, hockey"))
    if st.button("Make my flashcards", icon=":material/style:", type="primary"):
        if not charge():
            st.warning(LIMIT_MESSAGE)
        else:
            kwargs = {"question": typed} if typed else {"chapter": None if scope == EVERYWHERE else scope}
            st.session_state.cards = generate_or_explain(tutor.generate_flashcards, c, n=int(n), **kwargs)
    cards = st.session_state.cards
    if cards:
        flashcard_deck([{"front": k.front, "back": k.back, "meta": f"{page_label(k.page)} · {k.section}"}
                        for k in cards], key=f"deck_{id(cards)}", accent=accent("Flashcards"))
    else:
        st.html(theme.empty_state(mascot(), "Make a deck to start flipping."))


# --- Concept map & timeline -------------------------------------------------------------------------------

def concept_page() -> None:
    c = corpus()
    header("Concept map", "See how an idea or a person connects to others, using only what the guide says.")
    chip = suggestions(["Prime Minister", "Canadarm", "Confederation", "Sir John A. Macdonald"], "cm", "hub_name")
    with st.form("hub_form", border=False):
        c1, c2, c3 = st.columns([4, 2, 1.3], vertical_alignment="bottom")
        name = c1.text_input("Idea or person", placeholder="e.g. Federal elections, Terry Fox", max_chars=120,
                             icon=":material/hub:")
        kind = c2.segmented_control("This is a", ["Idea", "Person"], default="Idea", key="cm_kind") or "Idea"
        go = c3.form_submit_button("Build", type="primary", width="stretch")
    topic = chip or (clean_topic(name) if go else "")
    if topic:
        is_person = chip == "Sir John A. Macdonald" or kind == "Person"
        if not charge():
            st.warning(LIMIT_MESSAGE)
        else:
            res = generate_or_explain(ex.build_concept_hub, topic, c, kind="person" if is_person else "concept")
            if res and res[1] or not res:
                refund()                                   # cached results cost nothing
            st.session_state.study_hub = res
    if st.session_state.get("study_hub"):
        render_hub_infographic(st.session_state.study_hub[0], c, key="study_hub", learner=True)
    else:
        st.html(theme.empty_state(mascot(), "Pick an example above or type your own to draw a map."))


def timeline_page() -> None:
    c = corpus()
    header("Timeline", "Dates and events exactly as stated in Discover Canada.")
    chip = suggestions(["Canada’s History", "Modern Canada", "Voting rights", "Confederation"], "tl", "tl_pick")
    by = st.segmented_control("Build from", ["A chapter", "A topic"], default="A chapter", key="tl_by") or "A chapter"
    if by == "A chapter":
        chapter = st.selectbox("Chapter", c.chapters,
                               index=c.chapters.index("Canada’s History") if "Canada’s History" in c.chapters else 0)
        topic = None
    else:
        chapter, topic = None, clean_topic(st.text_input("Topic", placeholder="e.g. voting rights", max_chars=120))
    go = st.button("Build timeline", icon=":material/timeline:", type="primary")
    if chip:
        chapter, topic, go = (chip, None, True) if chip in c.chapters else (None, chip, True)
    if go and (chapter or topic):
        if not charge(2):
            st.warning(LIMIT_MESSAGE)
        else:
            res = generate_or_explain(ex.build_timeline, c, chapter=chapter, topic=topic,
                                      cache_only=os.environ.get("STUDY_LIVE_TIMELINES", "1") == "0")
            if res and res[1] or not res:
                refund(2)
            st.session_state.study_tl = res
    if st.session_state.get("study_tl"):
        render_timeline_infographic(st.session_state.study_tl[0], c, key="study_tl", learner=True)
    else:
        st.html(theme.empty_state(mascot(), "Choose a chapter or topic to see its story in order."))


# --- Talk (voice tutor; Milestone 1) -------------------------------------------------------------------

def talk_page() -> None:
    c = corpus()
    header("Talk", "Record your voice, check what I heard, then I'll answer, grade, and read it back to you.")
    voice_tutor.render(c, new_question=lambda kind: new_quiz_question(c, EVERYWHERE, kind=kind, state_key="voice_quiz"), record=record,
                       cite=cite, accent=accent("Talk"))


# --- main -------------------------------------------------------------------------------------------------

PAGE_FUNCS = {"Learn": learn_page, "Practice": practice_page, "Flashcards": flashcards_page,
              "Concept map": concept_page, "Timeline": timeline_page, "Talk": talk_page}


def main() -> None:
    init_state()
    if st.session_state.page == "Talk" and not voice_enabled():
        st.session_state.page = "Learn"                        # the voice page is opt-in (STUDY_VOICE=1)
    theme.inject()
    sync_theme()
    corpus()
    nav()
    if not generation_configured():
        st.info("The study tools are temporarily unavailable. Please try again later.")
    elif budget.status().exhausted:
        st.info("The study helper has used up today’s AI allowance. You can still browse everything already "
                "created, and new questions and summaries are back tomorrow.")
    PAGE_FUNCS[st.session_state.page]()
    st.html('<div class="footer">Discover Canada · powered by AI embeddings · every answer comes from the official study guide</div>')


main()
