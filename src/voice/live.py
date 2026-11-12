"""Live (hands-free) voice engine, Milestone 2: audio in -> utterances out, with a half-duplex gate. No Streamlit, no WebRTC imports.

`LiveEngine` is what the audio thread feeds (16 kHz mono float32) and what the page polls. It owns an `Endpointer` and decides what the learner
is told: listening / hearing you / thinking / speaking. While the tutor is thinking the microphone is gated shut and the endpointer forgets everything.
While it speaks a reply the gate stays shut (half-duplex) unless barge-in is on, in which case the microphone stays open in a stricter mode and the learner
can talk over it. Thread-safe: the audio thread and the Streamlit thread use one lock."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from src import config
from src.voice import audio, session as vsession
from src.voice.endpointing import EndpointConfig, Endpointer
from src.voice.vad import Vad, get_vad

LISTENING, HEARING, THINKING, SPEAKING = "listening", "hearing", "thinking", "speaking"


@dataclass(frozen=True)
class Utterance:
    samples: np.ndarray
    seconds: float
    reason: str                       # "utterance" (the learner stopped) | "max_length" (cut off at the limit)
    heard_at: float


@dataclass(frozen=True)
class LiveStatus:
    state: str
    level: float                      # loudness of the latest frame, 0..1-ish, for a meter
    probability: float                # the detector's speech probability for the latest frame
    vad: str
    queued: int
    idle_expired: bool


def endpoint_config(**overrides) -> EndpointConfig:
    s = config.live_settings() | overrides
    return EndpointConfig(threshold=s["threshold"], min_speech_ms=s["min_speech_ms"], hangover_ms=s["hangover_ms"],
                          max_utterance_s=float(config.voice_limits()["max_seconds"]), idle_timeout_s=s["idle_timeout_s"])


class LiveEngine:
    """The microphone has three modes, chosen from what the tutor is doing:

      open    listening for the learner (normal sensitivity)
      closed  the tutor is thinking, or is playing a short phrase / has just finished: the gate is shut and everything heard is forgotten
      barge   the tutor is speaking a reply and `barge_in` is on: the microphone stays open but STRICT (sustained, clearer speech is needed), so a
              cough, a breath or the tutor's own voice leaking through does not count; real speech cuts the tutor off (`on_interrupt`) and is
              kept, from its first word, as the learner's next utterance. Use headphones; with speakers leave barge-in off."""

    def __init__(self, vad: Vad | None = None, cfg: EndpointConfig | None = None, *, margin_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic, on_utterance: Callable[[Utterance], None] | None = None,
                 on_interrupt: Callable[[], None] | None = None, barge_in: bool = False, barge_min_ms: int = 600, barge_threshold: float = 0.65):
        self._vad = vad or get_vad(config.live_settings()["vad"])
        self._normal = cfg or endpoint_config()
        self._ep = Endpointer(self._vad, self._normal)
        self._clock = clock
        self._margin = config.live_settings()["margin_s"] if margin_s is None else margin_s
        self._lock = threading.Lock()
        self._queue: deque[Utterance] = deque()
        self._held = False                # the tutor is busy (transcribing / answering)
        self._gate_until = 0.0            # the tutor is speaking until this moment (+ margin)
        self._speech_end = 0.0            # when the audio queued so far finishes playing
        self._barge_in, self._barge_min_ms, self._barge_threshold = barge_in, barge_min_ms, barge_threshold
        self._barge_start = self._barge_end = 0.0      # the window in which interruptible speech (a reply) is playing
        self._applied = "open"
        self._on_utterance, self._on_interrupt = on_utterance, on_interrupt
        self.idle_expired = False

    # --- the mode ------------------------------------------------------------------------------------------------------------------
    def _wanted(self) -> str:
        now = self._clock()
        if self._barge_in and self._barge_start <= now < self._barge_end:
            return "barge"
        if self._held or now < self._gate_until:
            return "closed"
        return "open"

    def _barge_cfg(self) -> EndpointConfig:
        n = self._normal
        return replace(n, min_speech_ms=max(self._barge_min_ms, n.min_speech_ms), threshold=max(n.threshold, self._barge_threshold), idle_timeout_s=None)

    def _sync(self) -> None:
        want = self._wanted()
        if want == self._applied:
            return
        self._applied = want
        if want == "closed":
            self._ep.close_gate()
        else:
            self._ep.cfg = self._barge_cfg() if want == "barge" else self._normal
            self._ep.open_gate()

    def _interrupt_locked(self) -> None:
        """The learner spoke over the tutor: stop treating the room as busy and keep what is being heard (no reset)."""
        now = self._clock()
        self._speech_end = self._gate_until = now
        self._barge_start = self._barge_end = 0.0
        self._held = False
        self._applied = "open"
        self._ep.cfg = self._normal

    # --- audio thread ---------------------------------------------------------------------------------------------------------------
    def feed(self, samples: np.ndarray) -> None:
        fresh: list[Utterance] = []
        interrupted = False
        with self._lock:
            self._sync()
            for e in self._ep.feed(samples):
                if e.kind == "speech_start" and self._applied == "barge":
                    self._interrupt_locked()
                    interrupted = True
                elif e.kind in ("utterance", "max_length"):
                    u = Utterance(e.samples, e.seconds, e.kind, self._clock())
                    self._queue.append(u)
                    fresh.append(u)
                elif e.kind == "idle_timeout" and self._applied == "open":
                    self.idle_expired = True
        if interrupted and self._on_interrupt is not None:
            self._on_interrupt()
        if self._on_utterance is not None:
            for u in fresh:
                self._on_utterance(u)

    # --- page thread ------------------------------------------------------------------------------------------------------------------
    def pop_utterance(self) -> Utterance | None:
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def hold(self) -> None:
        """The tutor starts working on a turn: close the gate (and forget a half-heard turn). A reply that is already being spoken can still be interrupted."""
        with self._lock:
            self._held = True
            self._sync()

    def release(self, speak_seconds: float = 0.0) -> None:
        """The tutor is done thinking. It speaks for `speak_seconds` (0 if silent); the microphone reopens after that plus a margin.
        Audio already queued with `speak()` keeps the gate shut: a release never shortens it."""
        with self._lock:
            self._held = False
            self._gate_until = max(self._gate_until, self._clock() + max(0.0, speak_seconds) + (self._margin if speak_seconds > 0 else 0.0))
            self._sync()

    def speak(self, seconds: float, interruptible: bool = True) -> None:
        """`seconds` of speech were just queued for playback (streaming): close the gate at once and keep it shut until all queued speech has
        played plus a margin. `interruptible` = a reply (the learner may talk over it, if barge-in is on); a short phrase ("One moment") is not.
        Safe to call from any thread, as often as sentences arrive."""
        if seconds <= 0:
            return
        with self._lock:
            now = self._clock()
            before = max(self._speech_end, now)
            self._speech_end = before + seconds
            self._gate_until = max(self._gate_until, self._speech_end + self._margin)
            if interruptible:
                if self._barge_end <= now:
                    self._barge_start = before
                self._barge_end = self._speech_end
            self._sync()

    def stop_speaking(self) -> None:
        """"Stop talking" (button): forget queued speech and reopen the microphone after a short margin (the page also clears the audio)."""
        with self._lock:
            now = self._clock()
            self._speech_end = now
            self._gate_until = now + min(self._margin, 0.3)
            self._barge_start = self._barge_end = 0.0
            self._sync()

    def set_config(self, cfg: EndpointConfig) -> None:
        with self._lock:
            if cfg != self._normal:
                self._normal = cfg
                if self._applied == "open":
                    self._ep.cfg = cfg
                elif self._applied == "barge":
                    self._ep.cfg = self._barge_cfg()

    def set_barge_in(self, enabled: bool, *, min_ms: int | None = None, threshold: float | None = None) -> None:
        with self._lock:
            self._barge_in = enabled
            if min_ms is not None:
                self._barge_min_ms = min_ms
            if threshold is not None:
                self._barge_threshold = threshold

    def reset_idle(self) -> None:
        with self._lock:
            self.idle_expired = False

    def status(self) -> LiveStatus:
        with self._lock:
            self._sync()
            if self._applied == "closed":
                state = THINKING if self._held else SPEAKING
            elif self._applied == "barge":
                state = SPEAKING
            elif self._ep.state == "speech":
                state = HEARING
            else:
                state = LISTENING
            return LiveStatus(state, self._ep.level, self._ep.probability, self._vad.name, len(self._queue), self.idle_expired)


# --- one utterance -> the learner's draft text ----------------------------------------------------------------------------------------------

def ingest_utterance(vs: vsession.VoiceState, utt: Utterance, get_transcriber) -> bool:
    """Transcribe an utterance into the draft (names fixed, original kept) using the same session logic as Milestone 1.
    Returns True if there is text to send; False if the recogniser heard nothing usable (`vs.notice` says why)."""
    vsession.ingest_recording(vs, audio.encode_wav(utt.samples), get_transcriber)
    return bool(vs.heard.strip()) and vs.notice is None
