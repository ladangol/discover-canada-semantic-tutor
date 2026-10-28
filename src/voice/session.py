"""Per-browser-session voice state and the small, testable rules around it (no Streamlit imports).

Guarantees (each covered by tests/test_voice_session.py):
  * a recording is transcribed once, however many times Streamlit reruns the script;
  * nothing reaches the tutor until the learner confirms text, and a confirmed draft is used exactly once;
  * the spoken reply is synthesized once, when the turn is created, and stored with that turn.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.voice import audio
from src.voice.models import SpeechOutput, VoiceError, VoiceTimings

KEEP_AUDIO_TURNS = 6          # older turns keep their text but drop their audio bytes


@dataclass
class VoiceTurn:
    id: int
    mode: str                          # "ask" | "quiz"
    heard: str                         # original ASR text (kept separately from the learner's correction)
    said: str                          # the confirmed text that was sent to the tutor
    result: Any                        # tutor.TurnResult
    audio: SpeechOutput | None = None
    audio_note: str | None = None      # why there is no audio (safe to show)
    timings: VoiceTimings = field(default_factory=VoiceTimings)

    @property
    def edited(self) -> bool:
        return self.heard.strip() != self.said.strip()


@dataclass
class VoiceState:
    rec_n: int = 0                     # bump to give the recorder widget a fresh, empty state
    last_audio_hash: str | None = None
    has_draft: bool = False
    heard: str = ""
    draft_version: int = 0             # bump so the editable text box is re-seeded with the new transcript
    notice: str | None = None          # a friendly ASR/audio message for the learner
    draft_timings: VoiceTimings = field(default_factory=VoiceTimings)
    turns: list[VoiceTurn] = field(default_factory=list)
    next_id: int = 1


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_recording(state: VoiceState, data: bytes, get_transcriber: Callable[[], Any]) -> bool:
    """Transcribe `data` unless this exact recording was already handled. Returns True if it did work.

    On any voice error the learner gets a notice and an empty, typeable draft instead of a traceback."""
    fp = fingerprint(data)
    if fp == state.last_audio_hash:
        return False
    state.last_audio_hash = fp
    state.notice, state.heard, state.has_draft = None, "", True
    state.draft_version += 1
    state.draft_timings = VoiceTimings()
    try:
        samples, info = audio.read_wav(data)
        state.draft_timings.audio_seconds = info.duration_seconds
        res = get_transcriber().transcribe(samples, info.duration_seconds)
        state.heard = res.text
        state.draft_timings.asr = res.inference_seconds
        if not res.text.strip():
            state.notice = "I couldn't make out any words. Please record again, or type your message."
    except VoiceError as exc:
        state.notice = str(exc)
    return True


def begin_text(state: VoiceState, text: str) -> None:
    """Start a draft from text that was typed or picked from a suggestion (nothing was 'heard', so nothing is marked corrected)."""
    state.has_draft, state.heard, state.notice = True, text, None
    state.draft_version += 1
    state.draft_timings = VoiceTimings()


def discard_draft(state: VoiceState) -> None:
    """'Record again' / after submit: clear the draft and give the recorder widget a fresh state."""
    state.has_draft, state.heard, state.notice = False, "", None
    state.last_audio_hash = None
    state.rec_n += 1
    state.draft_timings = VoiceTimings()


def submit_turn(state: VoiceState, text: str, *, mode: str, turn_fn: Callable[[str], Any],
                synth_fn: Callable[[str], SpeechOutput] | None) -> VoiceTurn | None:
    """Send the learner-confirmed `text` to the tutor once. Returns None (and calls nothing) if there is no pending
    draft or the text is blank — which is what a Streamlit rerun after submitting looks like."""
    text = (text or "").strip()
    if not state.has_draft or not text:
        return None
    t = state.draft_timings
    heard = state.heard
    t0 = time.perf_counter()
    result = turn_fn(text)
    t.retrieval, t.generation = result.timings.get("retrieval"), result.timings.get("generation")
    t.tutor_total = result.timings.get("total", time.perf_counter() - t0)
    turn = VoiceTurn(state.next_id, mode, heard, text, result, timings=t)
    if synth_fn is not None and result.speech_text.strip():
        try:
            turn.audio = synth_fn(result.speech_text)
            t.tts = turn.audio.synthesis_seconds
        except VoiceError as exc:
            turn.audio_note = str(exc)
    state.next_id += 1
    state.turns.append(turn)
    for old in state.turns[:-KEEP_AUDIO_TURNS]:
        old.audio = None
    discard_draft(state)
    return turn
