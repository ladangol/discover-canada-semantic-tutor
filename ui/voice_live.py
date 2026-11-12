"""Live page (Milestone 2): hands-free conversation. No Record or Stop: it listens, notices when you have finished, answers, speaks, and
listens again.

    listening → hearing you → (you stop; a pause of `hangover` passes) → thinking → speaking → listening …

How it works: the browser streams the microphone over WebRTC (streamlit-webrtc) to this server; Silero VAD finds the start and end of each turn
(src/voice/endpointing.py); the finished turn is transcribed (names fixed), sent through the SAME `voice_tutor.send_turn` as the Talk page, and the
spoken reply is played. While the tutor is thinking or speaking the microphone is gated shut, so it cannot hear itself (use headphones if your
speakers are loud). Nothing is recorded to disk; audio is kept in memory only until a turn has been transcribed.

Local test page: shown in the technical Atlas (not the deployed learner app) when STUDY_VOICE=1."""
from __future__ import annotations

import contextlib
import importlib.util
import threading
import time

import streamlit as st

from src import config, tutor
from src.voice import asr, audio, intents, live as vlive, session as vsession, speak, tts
from src.voice.models import VoiceError
from ui import voice_tutor as vt

STATE = "voice_live"                 # this page's VoiceState (separate from the Talk page's)
PIPE = "voice_live_pipe"             # the audio pipe of the current WebRTC session (microphone in, speech out)
_ACKS = speak.Acks()                 # "One moment." / "I'm still looking." phrases, synthesised once per process
QUIZ_KEY = "voice_live_quiz"
RTC_KEY = "voice_live_rtc"
REPLY_LENGTH = {"Everything": None, "Medium": 500, "Short": 240}   # characters spoken (all / ≈ 30 s / ≈ 15 s); the full text is always on screen
VOICE_HELP = ("Say a question about the guide. Or say “ask me a multiple choice question” / “ask me an open question” / “quiz me”; answer out loud "
              "(“B”, “the second one”, or the answer itself); say “another question” for the next one, or “stop the quiz” to leave it.")
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
    streamed = st.session_state.get("live_streamed", False)               # already played live: the player below is only for replaying it
    vt._show_turn(turn, accent="#E11D74", cite=cite, latest=True, styled=False, autoplay=turn.id == fresh_id and not streamed)
    vt._diagnostics(turn.timings)
    lat = st.session_state.get("live_timing")
    if lat:
        ack = f" I said “one moment” after {lat['ack']:.1f} s." if lat.get("ack") is not None else ""
        first = f"My answer began {lat['first']:.1f} s after I decided you had finished." if lat.get("first") is not None else \
            f"The whole reply was ready {lat['total']:.1f} s after I decided you had finished."
        st.caption(f"{first}{ack} The pause I wait for ({st.session_state.get('live_pause_s', 0.8):.1f} s) comes on top of that.")


def _stop_talking(pipe) -> None:
    """Button callback: cut the tutor's speech now and reopen the microphone."""
    pipe.interrupt()
    pipe.engine.stop_speaking()


def _streamer(pipe, voice_ok: bool, max_chars: int | None):
    """A SpeechStreamer for one reply (speech goes to the browser as it is produced), or None when there is no streaming voice."""
    if not voice_ok or pipe.speaker is None:
        return None
    try:
        synth = tts.get_synthesizer()
        if not tts.can_stream(synth):
            return None
        pipe.active = speak.SpeechStreamer(synth, pipe.speaker, max_chars=max_chars, verdict=tutor.speak_verdict)
        return pipe.active                                            # remembered so that talking over it can stop it
    except VoiceError:
        return None                                               # no voice engine: the answer is text only (send_turn explains)


def _say(pipe, voice_ok: bool, text: str) -> None:
    """Speak `text` (not a tutor answer: a question to ask, or a short reply to a command). Best effort."""
    streamer = _streamer(pipe, voice_ok, None)
    if streamer is not None:
        try:
            streamer.finish(text)
        except VoiceError:
            pass


def _patience(pipe, spoke, heard_at):
    """Says "I'm still looking" if nothing real has been spoken for a few seconds. A no-op context when there is no voice."""
    return speak.Patience(_ACKS, pipe.speaker, spoke, started_at=heard_at) if pipe.speaker is not None else contextlib.nullcontext()


def _answer(pipe, c, vs, text, mode, q, quiz_state, open_q, record, voice_ok, heard_at=None):
    """Send `text` to the tutor with the microphone gated shut for the whole thinking AND speaking time. Returns the turn (or None).
    The reply is spoken sentence by sentence while it is still being written (the microphone gate follows the audio that was queued)."""
    chars = REPLY_LENGTH[st.session_state.get("live_reply_len", "Everything")]
    pipe.engine.hold()
    streamer, turn = _streamer(pipe, voice_ok, chars), None
    speak_s = 0.0
    try:
        with _patience(pipe, lambda: streamer is not None and streamer.first_audio_at is not None, heard_at):
            turn = vt.send_turn(c, vs, text, mode=mode, q=q, quiz_state=quiz_state, open_q=open_q, record=record, voice_ok=voice_ok,
                                speech_chars=chars, streamer=streamer)
        if turn is not None and turn.audio and streamer is None:
            speak_s = audio.wav_seconds(turn.audio.audio_bytes)   # whole-reply fallback: the page's player speaks it, so wait that long
    finally:
        pipe.engine.release(speak_s)                              # audio already queued keeps the microphone shut; this never shortens it
    if turn is not None:
        st.session_state["live_fresh"], st.session_state["live_streamed"] = turn.id, streamer is not None
        if heard_at is not None:
            first = streamer.first_audio_at if streamer is not None else None
            ack = getattr(pipe, "last_ack_at", None)
            st.session_state["live_timing"] = {"first": first - heard_at if first else None, "total": time.monotonic() - heard_at,
                                               "ack": ack - heard_at if ack and ack >= heard_at else None}
    return turn


def _ask_for_question(pipe, new_question, kind: str, voice_ok: bool, heard_at=None) -> None:
    """The learner asked, out loud, to be quizzed: make a question and read it (with its options) aloud."""
    pipe.engine.hold()
    try:
        with _patience(pipe, lambda: False, heard_at):
            new_question(kind)
        st.session_state["live_last_kind"] = kind
        qs = st.session_state.get(QUIZ_KEY)
        q = qs["q"] if qs and qs["q"] else None
        if q is None:
            _say(pipe, voice_ok, "Sorry, I couldn't make a question just now. Ask me something about the guide, or ask for a question again.")
        else:
            spoken = q.question
            if q.kind == "mcq":
                spoken += " " + " ".join(f"{'ABCD'[i]}. {o}." for i, o in enumerate(q.options))
            _say(pipe, voice_ok, spoken)
    finally:
        pipe.engine.release(0.0)


def _handle_text(pipe, c, vs, text, new_question, record, voice_ok, heard_at=None, ack=None):
    """Decide what the learner's confirmed words are, then do it. Hesitations are ignored; "ask me a question" starts a quiz question; "stop the quiz"
    leaves one; while a question is open the words are the ANSWER; otherwise they are a question for the guide. Returns the turn (or None)."""
    quiz_state = st.session_state.get(QUIZ_KEY)
    q = quiz_state["q"] if quiz_state and quiz_state["q"] else None
    open_q = q is not None and quiz_state["res"] is None
    last = vs.turns[-1].result if vs.turns else None
    expecting = open_q or bool(last is not None and last.suggestions)                  # a question or a "did you mean…?" is waiting for a reply
    if open_q and intents.quiz_exit(text):
        st.session_state[QUIZ_KEY] = None
        vsession.discard_draft(vs)
        pipe.engine.hold()
        try:
            _say(pipe, voice_ok, "Okay. Ask me anything about the guide.")
        finally:
            pipe.engine.release(0.0)
        return None
    kind = intents.quiz_request(text, st.session_state.get("live_last_kind"))
    if kind is None and intents.is_noise(text, expecting_reply=expecting):
        vsession.discard_draft(vs)                                                      # a breath, "um", "cool", or a word Whisper invented: stay quiet
        return None
    if ack is not None:
        ack()                                                                           # now we know it is a real request: "One moment."
    if kind:
        vsession.discard_draft(vs)
        _ask_for_question(pipe, new_question, kind, voice_ok, heard_at)
        return None
    return _answer(pipe, c, vs, text, vt.QUIZ if open_q else vt.ASK, q, quiz_state, open_q, record, voice_ok, heard_at)


def _show_question(q, vs, open_q: bool) -> None:
    """The open question (or the one just answered) on screen. The page does not need a click: it is asked and answered by voice."""
    answered_now = bool(vs.turns) and vs.turns[-1].result.kind == "quiz_feedback"
    if not (open_q or answered_now):
        return
    vt._question(False, q.question)
    if q.kind == "mcq":
        st.markdown("\n".join(f"**{'ABCD'[i]}.** {o}" for i, o in enumerate(q.options)))
    if open_q:
        st.caption("Answer out loud: a letter (“B”), a position (“the second one”), or the answer itself. Say “another question” or “stop the quiz” if you like.")


def _warm(stt_ok: bool, voice_ok: bool) -> None:
    """Load the speech models in the background when the page opens, so the first turn does not pay for it."""
    def work() -> None:
        try:
            if voice_ok:
                _ACKS.warm(tts.get_synthesizer())
            if stt_ok:
                asr.get_transcriber()
        except Exception:
            pass
    threading.Thread(target=work, name="voice-warmup", daemon=True).start()


@st.fragment(run_every=0.25)
def _live(ctx, pipe, c, record, cite, new_question, auto_send, voice_ok) -> None:
    """Runs every half second on its own: updates the status, and when a finished turn is waiting, handles it."""
    vs = _vs()
    quiz_state = st.session_state.get(QUIZ_KEY)
    q = quiz_state["q"] if quiz_state and quiz_state["q"] else None
    open_q = q is not None and quiz_state["res"] is None
    proc = pipe if (ctx is not None and ctx.state.playing and not pipe.closed) else None
    fresh = st.session_state.get("live_fresh")

    if proc is None:
        if st.session_state.get("live_idle_stopped"):
            st.warning("I stopped listening after a long silence. Press START to talk again.", icon=":material/mic_off:")
        else:
            st.info("Press START above (and allow the microphone). I only listen while it says Listening.", icon=":material/mic:")
        if q is not None:
            _show_question(q, vs, open_q)
        _latest(vs, cite, fresh)
        return
    st.session_state["live_idle_stopped"] = False
    proc.ack_enabled = bool(st.session_state.get("live_ack", True)) and not open_q       # an answer to a question needs no "one moment"
    proc.engine.set_barge_in(bool(st.session_state.get("live_barge", True)), min_ms=int(st.session_state.get("live_barge_ms", 600)))
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
        if status.state == vlive.SPEAKING:
            st.button("Stop talking", icon=":material/stop_circle:", key="live_stop_talking", on_click=_stop_talking, args=(proc,),
                      help="Cut me off and listen again. You can also just start talking (with headphones).")

    if (utt := proc.engine.pop_utterance()) is not None:
        slot.info("**Thinking** — transcribing and working out the answer.", icon=":material/psychology:")
        proc.engine.hold()                                      # closed while transcribing (the tutor may then keep it closed while speaking)
        try:
            heard = vlive.ingest_utterance(vs, utt, asr.get_transcriber)
        except Exception:
            proc.engine.release(0.0)
            raise
        proc.engine.release(0.0)
        if not heard:
            (st.caption if not vs.raw_heard.strip() else st.warning)(vs.notice or "I didn't catch that. Try again.")
            vsession.discard_draft(vs)
        elif auto_send:
            turn = _handle_text(proc, c, vs, vs.heard, new_question, record, voice_ok, heard_at=utt.heard_at, ack=lambda: proc.acknowledge(utt))
            if turn is not None:
                fresh = turn.id
            quiz_state = st.session_state.get(QUIZ_KEY)         # a spoken command may have started, changed or ended a question
            q = quiz_state["q"] if quiz_state and quiz_state["q"] else None
            open_q = q is not None and quiz_state["res"] is None

    if vs.has_draft and vs.heard.strip() and not auto_send:      # review mode: the turn waits until the learner confirms it
        _review(proc, c, vs, new_question, record, voice_ok)
    if q is not None:
        _show_question(q, vs, open_q)
    _latest(vs, cite, st.session_state.get("live_fresh"))
    if len(vs.turns) > 1:
        with st.expander(f"Conversation history ({len(vs.turns) - 1} earlier)"):
            for t in vs.turns[:-1]:
                vt._show_turn(t, accent="#E11D74", cite=cite, latest=False, styled=False)


def _review(proc, c, vs, new_question, record, voice_ok) -> None:
    if vs.fixes:
        st.info("I fixed a name: " + ", ".join(f"“{a}” → “{b}”" for a, b in vs.fixes), icon=":material/spellcheck:")
    text = st.text_input("I heard (fix anything, then send)", value=vs.heard, key=f"live_draft_{vs.draft_version}", max_chars=300)
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Send", type="primary", key=f"live_send_{vs.draft_version}", width="stretch") and text.strip():
        _handle_text(proc, c, vs, text, new_question, record, voice_ok)
    if b2.button("Discard", key=f"live_discard_{vs.draft_version}", width="stretch"):
        vsession.discard_draft(vs)


def render(c, *, new_question, record, cite) -> None:
    if not installed():
        st.info("The live page needs `streamlit-webrtc` and `silero-vad`: `pip install -r requirements-voice.txt` (see the README, “Live voice”).")
        return
    from streamlit_webrtc import WebRtcMode, create_audio_sink_track, create_pcm_audio_source_track, webrtc_streamer
    from ui.live_audio import LiveAudioPipe

    stt_ok, voice_ok = vt._readiness()
    live_cfg = config.live_settings()
    st.caption("Hands-free: no Record or Stop, and nothing to click once you press START. Local test page (it is not part of the deployed learner app). "
               "Use headphones if your speakers are loud.")
    st.caption(VOICE_HELP)
    with st.expander("Settings: how quickly it decides you have finished", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.slider("Pause that ends your turn (seconds)", 0.4, 2.0, live_cfg["hangover_ms"] / 1000, 0.1, key="live_pause_s",
                  help="Shorter = replies sooner but may cut you off mid-thought. Longer = patient, but slower.")
        c2.slider("Speech sensitivity", 0.3, 0.8, live_cfg["threshold"], 0.05, key="live_threshold",
                  help="Lower picks up quieter speech but may react to noise. Higher ignores more noise (breathing, a fan).")
        c3.slider("Stop listening after silence (min)", 1, 15, 3, key="live_idle_min")
        c4.selectbox("Spoken reply", list(REPLY_LENGTH), key="live_reply_len",
                     help="Everything reads the whole answer. The microphone stays closed while I talk (press Stop talking to cut me off), so shorter replies mean shorter waits. The full text is always on screen.")
        st.toggle("Send automatically when I pause (otherwise let me review the text first)", value=True, key="live_auto")
        st.toggle("Say “one moment” once I've understood a request (not while answering a question)", value=True, key="live_ack",
                  help="A short spoken acknowledgement while the answer is being worked out, and “I'm still looking” if it takes longer. It is only said after I have recognised real words, so a breath or an “um” gets nothing. It does not make the answer faster; it makes the wait feel shorter.")
        b1, b2 = st.columns(2)
        b1.toggle("Let me talk over it (use headphones)", value=True, key="live_barge",
                  help="While I'm speaking the microphone stays on, and speaking clearly for about half a second cuts me off and I listen to you. With loudspeakers I could hear myself: switch this off.")
        b2.slider("How long I must speak to cut it off (seconds)", 0.3, 1.5, 0.6, 0.1, key="live_barge_ms",
                  help="Longer = a cough or “mm-hm” will not interrupt it. Shorter = quicker to cut it off.")
    auto_send = st.session_state.get("live_auto", True)

    pipe = st.session_state.get(PIPE)
    if pipe is None or pipe.closed:                              # a new WebRTC session gets a fresh pipe (engine state, gate, queue)
        pipe = st.session_state[PIPE] = LiveAudioPipe(acks=_ACKS)
        _warm(stt_ok, voice_ok)
    out = create_pcm_audio_source_track(key=RTC_KEY + "_speech", sample_rate=speak.SAMPLE_RATE)      # the tutor's voice, sent to the browser
    if getattr(pipe, "speaker_source", None) is not out:          # one Speaker per output track: its interruption state must survive page reruns
        pipe.speaker, pipe.speaker_source = speak.Speaker(out.push, out.clear, pipe.engine.speak), out
    sink = create_audio_sink_track(pipe.on_frame, key=RTC_KEY + "_mic", on_ended=pipe.close)           # the microphone, received here

    stop = st.session_state.pop("live_stop", False)
    ctx = webrtc_streamer(
        key=RTC_KEY, mode=WebRtcMode.SENDRECV, sink_audio_track=sink, source_audio_track=out.track, sendback_video=False,
        desired_playing_state=False if stop else None,
        media_stream_constraints={"audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True}, "video": False},
        audio_html_attrs={"autoPlay": True, "controls": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] if live_cfg["stun"] else []},
        translations={"start": "Start listening", "stop": "Stop", "select_device": "Select microphone", "device_ask_permission": "Allow the microphone",
                      "device_not_available": "No microphone available", "media_api_not_available": "This browser can't use the microphone here (HTTPS or localhost needed)"})
    if not stt_ok or not voice_ok:
        missing = [n for n, ok in (("speech recognition", stt_ok), ("spoken replies", voice_ok)) if not ok]
        st.info(f"Setup needed for {' and '.join(missing)} (see the README, “Voice tutor”). Without it I can listen but not transcribe or speak.")
    _live(ctx, pipe, c, record, cite, new_question, auto_send, voice_ok)
