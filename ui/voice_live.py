"""Live page (Milestone 2): hands-free conversation. No Record or Stop: it listens, notices when you have finished, answers, speaks, and
listens again.

    listening → hearing you → (you stop; a pause of `hangover` passes) → thinking → speaking → listening …

How it works: the browser streams the microphone over WebRTC (streamlit-webrtc) to this server; Silero VAD finds the start and end of each turn
(src/voice/endpointing.py); the finished turn is transcribed (names fixed), sent through the SAME `voice_tutor.send_turn` as the Talk page, and the
spoken reply is played. While the tutor is thinking or speaking the microphone is gated shut, so it cannot hear itself (use headphones if your
speakers are loud). Nothing is recorded to disk; audio is kept in memory only until a turn has been transcribed.

Local test page: shown in the technical Atlas (not the deployed learner app) when STUDY_VOICE=1."""
from __future__ import annotations

import importlib.util
import time

import streamlit as st

from src import config
from src.voice import asr, audio, live as vlive, session as vsession
from ui import voice_tutor as vt

STATE = "voice_live"                 # this page's VoiceState (separate from the Talk page's)
QUIZ_KEY = "voice_live_quiz"
RTC_KEY = "voice_live_rtc"
REPLY_LENGTH = {"Short": 350, "Medium": 700, "Full": None}       # characters spoken (≈ 20 s / 40 s / everything); the full text is always on screen
STATUS = {vlive.LISTENING: ("Listening", ":material/hearing:", "Say something. I'll answer when you pause."),
          vlive.HEARING: ("I hear you", ":material/graphic_eq:", "Keep going; I wait for a pause."),
          vlive.THINKING: ("Thinking", ":material/psychology:", "Transcribing and working out the answer. The microphone is paused."),
          vlive.SPEAKING: ("Speaking", ":material/volume_up:", "The microphone is paused while I talk, so I don't hear myself.")}


def installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("streamlit_webrtc", "av", "silero_vad"))


def _vs() -> vsession.VoiceState:
    if STATE not in st.session_state:
        st.session_state[STATE] = vsession.VoiceState()
    return st.session_state[STATE]


def _status_banner(s: vlive.LiveStatus) -> None:
    label, icon, hint = STATUS[s.state]
    show = {vlive.LISTENING: st.success, vlive.HEARING: st.warning, vlive.THINKING: st.info, vlive.SPEAKING: st.info}[s.state]
    show(f"**{label}** — {hint}", icon=icon)
    st.progress(min(1.0, s.level * 12), text=f"Microphone level · detector: {s.vad} · speech probability {s.probability:.2f}")


def _latest(vs: vsession.VoiceState, cite, fresh_id) -> None:
    """Show the newest turn; only a brand-new reply autoplays (the same element on every refresh, so the sound is not cut off)."""
    if not vs.turns:
        return
    turn = vs.turns[-1]
    vt._show_turn(turn, accent="#E11D74", cite=cite, latest=True, styled=False, autoplay=turn.id == fresh_id)
    vt._diagnostics(turn.timings)
    lat = st.session_state.get("live_latency", [])
    if lat:
        st.caption(f"Reply latency after I decided you had finished (transcribe + answer + speech): {lat[-1]:.1f} s. "
                   f"The pause I wait for ({st.session_state.get('live_pause_s', 0.8):.1f} s) comes on top of that.")


def _answer(proc, c, vs, text, mode, q, quiz_state, open_q, record, voice_ok):
    chars = REPLY_LENGTH[st.session_state.get("live_reply_len", "Short")]
    """Send `text` to the tutor with the microphone gated shut for the whole thinking AND speaking time. Returns the turn (or None)."""
    proc.engine.hold()
    speak, turn = 0.0, None
    try:
        turn = vt.send_turn(c, vs, text, mode=mode, q=q, quiz_state=quiz_state, open_q=open_q, record=record, voice_ok=voice_ok, speech_chars=chars)
        if turn is not None and turn.audio:
            speak = audio.wav_seconds(turn.audio.audio_bytes)
    finally:
        proc.engine.release(speak)                              # reopens after the reply has finished playing (+ a margin)
    if turn is not None:
        st.session_state["live_fresh"] = turn.id
    return turn


@st.fragment(run_every=0.5)
def _live(ctx, c, record, cite, mode, auto_send, voice_ok) -> None:
    """Runs every half second on its own: updates the status, and when a finished turn is waiting, answers it."""
    vs = _vs()
    quiz_state = st.session_state.get(QUIZ_KEY)
    q = quiz_state["q"] if quiz_state and quiz_state["q"] else None
    open_q = q is not None and quiz_state["res"] is None
    proc = ctx.audio_processor if (ctx is not None and ctx.state.playing) else None
    fresh = st.session_state.get("live_fresh")

    if proc is None:
        if st.session_state.get("live_idle_stopped"):
            st.warning("I stopped listening after a long silence. Press START to talk again.", icon=":material/mic_off:")
        else:
            st.info("Press START above (and allow the microphone). I only listen while it says Listening.", icon=":material/mic:")
        _latest(vs, cite, fresh)
        return
    st.session_state["live_idle_stopped"] = False
    proc.engine.set_config(vlive.endpoint_config(threshold=st.session_state.get("live_threshold", 0.5),
                                                 hangover_ms=int(st.session_state.get("live_pause_s", 0.8) * 1000),
                                                 idle_timeout_s=st.session_state.get("live_idle_min", 3) * 60.0))
    status = proc.engine.status()
    if status.idle_expired:
        st.session_state["live_stop"], st.session_state["live_idle_stopped"] = True, True
        proc.engine.reset_idle()
        st.rerun()                                              # full rerun: the page asks the stream to stop, so the microphone indicator goes off
    if status.state == vlive.LISTENING and fresh is not None:
        st.session_state["live_fresh"] = fresh = None           # the reply has finished playing and the microphone is open again
    slot = st.empty()
    with slot.container():
        _status_banner(status)

    if mode == vt.QUIZ and not open_q:
        while proc.engine.pop_utterance():                      # nothing to answer yet: ignore speech rather than treat it as a question
            pass
    elif (utt := proc.engine.pop_utterance()) is not None:
        slot.info("**Thinking** — transcribing and working out the answer.", icon=":material/psychology:")
        proc.engine.hold()                                      # closed while transcribing (the tutor may then keep it closed while speaking)
        try:
            heard = vlive.ingest_utterance(vs, utt, asr.get_transcriber)
        except Exception:
            proc.engine.release(0.0)
            raise
        proc.engine.release(0.0)
        if not heard:
            st.warning(vs.notice or "I didn't catch that. Try again.")
            vsession.discard_draft(vs)
        elif auto_send:
            turn = _answer(proc, c, vs, vs.heard, mode, q, quiz_state, open_q, record, voice_ok)
            if turn is not None:
                st.session_state["live_latency"] = st.session_state.get("live_latency", []) + [time.monotonic() - utt.heard_at]
                fresh = turn.id

    if vs.has_draft and vs.heard.strip() and not auto_send:      # review mode: the turn waits until the learner confirms it
        _review(proc, c, vs, mode, q, quiz_state, open_q, record, voice_ok)
    _latest(vs, cite, st.session_state.get("live_fresh"))
    if len(vs.turns) > 1:
        with st.expander(f"Conversation history ({len(vs.turns) - 1} earlier)"):
            for t in vs.turns[:-1]:
                vt._show_turn(t, accent="#E11D74", cite=cite, latest=False, styled=False)


def _review(proc, c, vs, mode, q, quiz_state, open_q, record, voice_ok) -> None:
    if vs.fixes:
        st.info("I fixed a name: " + ", ".join(f"“{a}” → “{b}”" for a, b in vs.fixes), icon=":material/spellcheck:")
    text = st.text_input("I heard (fix anything, then send)", value=vs.heard, key=f"live_draft_{vs.draft_version}", max_chars=300)
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Send", type="primary", key=f"live_send_{vs.draft_version}", width="stretch") and text.strip():
        _answer(proc, c, vs, text, mode, q, quiz_state, open_q, record, voice_ok)
    if b2.button("Discard", key=f"live_discard_{vs.draft_version}", width="stretch"):
        vsession.discard_draft(vs)


def render(c, *, new_question, record, cite) -> None:
    if not installed():
        st.info("The live page needs `streamlit-webrtc` and `silero-vad`: `pip install -r requirements-voice.txt` (see the README, “Live voice”).")
        return
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    from ui.live_audio import LiveAudioProcessor

    stt_ok, voice_ok = vt._readiness()
    live_cfg = config.live_settings()
    st.caption("Hands-free: no Record or Stop. Press START once, then talk. Local test page (it is not part of the deployed learner app). "
               "Use headphones if your speakers are loud.")
    mode = st.segmented_control("What would you like to do?", [vt.ASK, vt.QUIZ], default=vt.ASK, key="live_mode") or vt.ASK
    with st.expander("Settings: how quickly it decides you have finished", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.slider("Pause that ends your turn (seconds)", 0.4, 2.0, live_cfg["hangover_ms"] / 1000, 0.1, key="live_pause_s",
                  help="Shorter = replies sooner but may cut you off mid-thought. Longer = patient, but slower.")
        c2.slider("Speech sensitivity", 0.3, 0.8, live_cfg["threshold"], 0.05, key="live_threshold",
                  help="Lower picks up quieter speech but may react to noise. Higher ignores more noise.")
        c3.slider("Stop listening after silence (min)", 1, 15, 3, key="live_idle_min")
        c4.selectbox("Spoken reply", list(REPLY_LENGTH), key="live_reply_len",
                     help="The microphone stays closed while I talk, so long replies mean long waits. The full text is always on screen.")
        auto_send = st.toggle("Send automatically when I pause (otherwise let me review the text first)", value=True, key="live_auto")
    auto_send = st.session_state.get("live_auto", True)

    vs = _vs()
    quiz_state = st.session_state.get(QUIZ_KEY)
    q = quiz_state["q"] if quiz_state and quiz_state["q"] else None
    open_q = q is not None and quiz_state["res"] is None
    if mode == vt.QUIZ:
        kind_label = st.segmented_control("Question type", [vt.MCQ, vt.OPEN], default=vt.MCQ, key="live_qkind") or vt.MCQ
        if q is not None:
            vt._question(False, q.question)
            if q.kind == "mcq":
                st.markdown("\n".join(f"**{'ABCD'[i]}.** {o}" for i, o in enumerate(q.options)))
            vt._question_audio(q, quiz_state["id"], cache_key="live_q_audio")
        if not open_q and st.button("New question" if q is not None else "Give me a question", icon=":material/auto_awesome:",
                                    type="primary", key="live_new_q"):
            new_question("mcq" if kind_label == vt.MCQ else "free")
            st.rerun()
        if open_q:
            st.caption("Answer out loud: say a letter (“B”), a position (“the second one”), or the answer itself.")

    stop = st.session_state.pop("live_stop", False)
    ctx = webrtc_streamer(
        key=RTC_KEY, mode=WebRtcMode.SENDONLY, audio_processor_factory=LiveAudioProcessor, async_processing=True,
        desired_playing_state=False if stop else None,
        media_stream_constraints={"audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True}, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] if live_cfg["stun"] else []},
        translations={"start": "Start listening", "stop": "Stop", "select_device": "Select microphone", "device_ask_permission": "Allow the microphone",
                      "device_not_available": "No microphone available", "media_api_not_available": "This browser can't use the microphone here (HTTPS or localhost needed)"})
    if not stt_ok or not voice_ok:
        missing = [n for n, ok in (("speech recognition", stt_ok), ("spoken replies", voice_ok)) if not ok]
        st.info(f"Setup needed for {' and '.join(missing)} (see the README, “Voice tutor”). Without it I can listen but not transcribe or speak.")
    _live(ctx, c, record, cite, mode, auto_send, voice_ok)
