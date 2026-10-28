"""Talk page (Milestone 1): record → transcribe → learner edits and confirms → shared tutor → text + spoken reply.

Thin by design: all rules live in src/voice/session.py and src/tutor.py; this file only draws widgets and wires them.
The page never grades or answers anything itself, and nothing reaches the tutor until the learner confirms the text."""
from __future__ import annotations

import html
import importlib.util

import streamlit as st

from src import config, tutor
from src.voice import asr, session as vsession, tts
from src.voice.models import SpeechOutput, VoiceError, VoiceTimings
from ui import study_theme as theme
from ui.common import generate_or_explain
from ui.session_limits import LIMIT_MESSAGE, charge, refund

ASK, QUIZ = "Ask a question", "Practice quiz"
MCQ, OPEN = "Multiple choice", "Open answer"
STATE = "voice"                 # st.session_state key holding this browser session's VoiceState
QUIZ_KEY = "voice_quiz"         # the Talk page's own open quiz question (history/asked are shared with Practice)


def _state() -> vsession.VoiceState:
    if STATE not in st.session_state:
        st.session_state[STATE] = vsession.VoiceState()
    return st.session_state[STATE]


def _speak(text: str) -> SpeechOutput:
    return tts.get_synthesizer().synthesize(text)


def _readiness() -> tuple[bool, bool]:
    """(speech-to-text installed, the configured voice engine usable) — checked without loading any model."""
    return importlib.util.find_spec("faster_whisper") is not None, tts.engine_ready()


def _question_audio(q: tutor.QuizQuestion, qid) -> None:
    """Read the open question (and its options) aloud; synthesized once per question, then kept in the session."""
    cache = st.session_state.setdefault("voice_q_audio", {})
    if cache.get("id") != qid:
        spoken = q.question
        if q.kind == "mcq":
            spoken += " " + " ".join(f"{'ABCD'[i]}. {o}." for i, o in enumerate(q.options))
        try:
            with st.spinner("Getting the question ready…"):
                cache.clear()
                cache.update(id=qid, audio=_speak(spoken), note=None)
        except VoiceError as exc:
            cache.update(id=qid, audio=None, note=str(exc))
    if cache.get("audio"):
        st.audio(cache["audio"].audio_bytes, format=cache["audio"].mime_type)
    elif cache.get("note"):
        st.caption(cache["note"])


# The learner app has custom CSS; the technical Atlas does not, so every styled element has a plain-widget twin.
def _title(styled: bool, text: str) -> None:
    if styled:
        st.html(f'<div class="ask-title"><span class="mi">edit_note</span> {html.escape(text)}</div>')
    else:
        st.markdown(f"**{text}**")


def _banner(styled: bool, good: bool, msg: str) -> None:
    if styled:
        kind, icon = ("good", "check_circle") if good else ("bad", "lightbulb")
        st.html(f'<div class="banner {kind}"><span class="mi">{icon}</span> {html.escape(msg)}</div>')
    else:
        (st.success if good else st.warning)(msg)


def _question(styled: bool, text: str) -> None:
    st.html(f'<div class="question">{html.escape(text)}</div>') if styled else st.markdown(f"#### {text}")


def _pills(styled: bool, items: list[str], accent: str) -> None:
    st.html(theme.pills(items, accent)) if styled else st.caption("Sources: " + " · ".join(dict.fromkeys(items)))


def _empty(styled: bool, text: str) -> None:
    if styled:
        st.html(theme.empty_state(st.session_state.get("mascot", theme.MASCOTS[0]), text))
    else:
        st.info(text)


def _pick_suggestion(text: str) -> None:
    st.session_state["voice_pending_text"] = text          # handled at the top of the next run, once


def _show_turn(turn: vsession.VoiceTurn, *, accent: str, cite, latest: bool, styled: bool) -> None:
    r = turn.result
    with st.chat_message("user", avatar=":material/mic:"):
        st.markdown(turn.said)
        if turn.edited:
            st.caption("You corrected what I heard.")
    with st.chat_message("assistant", avatar=":material/school:"):
        if r.kind == "quiz_feedback":
            good = r.quiz_result.correct
            _banner(styled, good, "Correct! Nicely done." if good else "Not quite, but now you know!")
        st.markdown(r.text)
        if r.cited:
            _pills(styled, [cite(c) for c in r.cited], accent)
        if latest:
            for i, sug in enumerate(r.suggestions):
                st.button(sug, key=f"voice_sug_{turn.id}_{i}", icon=":material/help:", on_click=_pick_suggestion, args=(sug,))
        if turn.audio:
            st.audio(turn.audio.audio_bytes, format=turn.audio.mime_type, autoplay=False)
        elif turn.audio_note and latest:
            st.caption(turn.audio_note)


def _diagnostics(t: VoiceTimings) -> None:
    with st.expander("Developer diagnostics (measured processing time)"):
        st.table({k: [v] for k, v in t.rows()})
        st.caption("Processing time only: it excludes recording, transcript correction and pressing Play. "
                   f"Total processing: {t.processing_total():.2f} s.")


def render(c, *, new_question, record, cite, accent: str = "#E11D74", styled: bool = True, diagnostics: bool = False) -> None:
    """styled=True in the learner app; the technical Atlas passes styled=False (plain widgets) and diagnostics=True."""
    vs = _state()
    stt_ok, voice_ok = _readiness()
    lim = config.voice_limits()

    mode = st.segmented_control("What would you like to do?", [ASK, QUIZ], default=ASK, key="voice_mode") or ASK
    quiz_state = st.session_state.get(QUIZ_KEY)
    q: tutor.QuizQuestion | None = quiz_state["q"] if quiz_state and quiz_state["q"] else None
    open_q = q is not None and quiz_state["res"] is None          # a question waiting for the learner's answer

    if mode == QUIZ:
        kind_label = st.segmented_control("Question type", [MCQ, OPEN], default=MCQ, key="voice_qkind") or MCQ
        if q is not None:
            _question(styled, q.question)
            if q.kind == "mcq":
                st.markdown("\n".join(f"**{'ABCD'[i]}.** {o}" for i, o in enumerate(q.options)))
            _question_audio(q, quiz_state["id"])
        if not open_q:
            label = "New question" if q is not None else "Give me a question"
            if st.button(label, icon=":material/auto_awesome:", type="primary", key="voice_new_q"):
                new_question("mcq" if kind_label == MCQ else "free")
                st.rerun()
            if q is None:
                st.info("Press the button for a question, then answer it out loud.")

    can_record = mode == ASK or open_q
    if mode == QUIZ and open_q:
        st.caption("Answer out loud: say a letter (“B”), a position (“the second one”), or the answer itself.")

    def send_text(text: str) -> None:
        """Send confirmed text to the tutor exactly once (form submit or a tapped suggestion)."""
        answering = mode == QUIZ and open_q
        needs_llm = (not answering) or q.kind == "free"
        if needs_llm and not charge():
            st.warning(LIMIT_MESSAGE)
            return
        active_q = q if answering else None
        turn = generate_or_explain(
            vsession.submit_turn, vs, text, mode="quiz" if answering else "ask",
            turn_fn=lambda t: tutor.handle_turn(t, quiz=active_q, corpus=c),
            synth_fn=_speak if voice_ok else None)
        if turn is None:
            if needs_llm:
                refund()
            return
        if turn.result.kind == "clarify" and needs_llm:
            refund()
        if turn.result.abstained and turn.result.suggestions:
            charge()                                         # the suggestion guess was one extra AI call: count it (best effort)
        if turn.result.kind == "quiz_feedback":          # grade once, record once
            quiz_state["res"], quiz_state["pick"] = turn.result.quiz_result, turn.result.chosen_index
            record(q, turn.result.quiz_result)
        if not voice_ok:
            turn.audio_note = turn.audio_note or "Spoken replies aren't set up on this server. The written answer is below."
        st.rerun()

    pending = st.session_state.pop("voice_pending_text", None)      # a "did you mean…" suggestion that was tapped
    if pending and mode == ASK:
        vsession.begin_text(vs, pending)
        send_text(pending)

    # --- record ------------------------------------------------------------------------------------------
    if can_record:
        rec = st.audio_input("Record your voice", key=f"voice_rec_{vs.rec_n}", sample_rate=16000,
                             help="Press the microphone, speak, then press stop.")
        st.caption("Your recording is processed on this server and not saved. Only the text you confirm is used.")
        if rec is not None:
            if rec.size > lim["max_bytes"]:
                vs.notice, vs.has_draft, vs.heard = "That recording is too large. Please keep it short and try again.", True, ""
            else:
                with st.spinner("Listening…"):
                    vsession.ingest_recording(vs, rec.getvalue(), asr.get_transcriber)

    # --- confirm / correct --------------------------------------------------------------------------------
    if can_record and vs.has_draft:
        if vs.notice:
            st.warning(vs.notice)
        with st.form("voice_confirm", border=False):
            _title(styled, "You said — fix anything I got wrong")
            text = st.text_input("Your message", value=vs.heard, key=f"voice_draft_{vs.draft_version}", max_chars=300,
                                 label_visibility="collapsed")
            b1, b2, _ = st.columns([1.3, 1.1, 2.6])
            send = b1.form_submit_button("Send to tutor", type="primary", icon=":material/send:", width="stretch")
            again = b2.form_submit_button("Record again", icon=":material/replay:", width="stretch")
        if again:
            vsession.discard_draft(vs)
            st.rerun()
        if send and text.strip():
            send_text(text)
        elif send:
            st.warning("Please say or type something first.")
    elif can_record and not vs.turns:
        _empty(styled, "Press the microphone, ask your question, and I'll listen.")

    if not stt_ok or not voice_ok:
        missing = [n for n, ok in (("speech recognition", stt_ok), ("spoken replies", voice_ok)) if not ok]
        st.info(f"Setup needed for {' and '.join(missing)} — see the README (“Voice tutor”). "
                "You can still type into the box after pressing record, and read the written answers.")

    # --- latest turn + history ----------------------------------------------------------------------------
    if vs.turns:
        _show_turn(vs.turns[-1], accent=accent, cite=cite, latest=True, styled=styled)
        if diagnostics or config.voice_diagnostics():
            _diagnostics(vs.turns[-1].timings)
        if len(vs.turns) > 1:
            with st.expander(f"Conversation history ({len(vs.turns) - 1} earlier)"):
                for turn in vs.turns[:-1]:
                    _show_turn(turn, accent=accent, cite=cite, latest=False, styled=styled)
