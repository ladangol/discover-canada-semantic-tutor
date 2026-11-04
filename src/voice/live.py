"""Live (hands-free) voice engine, Milestone 2: audio in -> utterances out, with a half-duplex gate. No Streamlit, no WebRTC imports.

`LiveEngine` is what the audio thread feeds (16 kHz mono float32) and what the page polls. It owns an `Endpointer` and decides what the learner
is told: listening / hearing you / thinking / speaking. While the tutor is thinking or speaking the microphone is gated shut and the endpointer
forgets everything, so the tutor's own voice can never start a turn. Thread-safe: the audio thread and the Streamlit thread use one lock."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
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
    def __init__(self, vad: Vad | None = None, cfg: EndpointConfig | None = None, *, margin_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._vad = vad or get_vad(config.live_settings()["vad"])
        self._ep = Endpointer(self._vad, cfg or endpoint_config())
        self._clock = clock
        self._margin = config.live_settings()["margin_s"] if margin_s is None else margin_s
        self._lock = threading.Lock()
        self._queue: deque[Utterance] = deque()
        self._held = False                # the tutor is busy (transcribing / answering)
        self._gate_until = 0.0            # the tutor is speaking until this moment (+ margin)
        self.idle_expired = False

    # --- audio thread ---------------------------------------------------------------------------------------------------------
    def feed(self, samples: np.ndarray) -> None:
        with self._lock:
            self._reopen_if_due()
            for e in self._ep.feed(samples):
                if e.kind in ("utterance", "max_length"):
                    self._queue.append(Utterance(e.samples, e.seconds, e.kind, self._clock()))
                elif e.kind == "idle_timeout":
                    self.idle_expired = True

    # --- page thread ------------------------------------------------------------------------------------------------------------
    def pop_utterance(self) -> Utterance | None:
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def hold(self) -> None:
        """The tutor starts working on a turn: close the gate (and forget a half-heard turn)."""
        with self._lock:
            self._held = True
            self._ep.close_gate()

    def release(self, speak_seconds: float = 0.0) -> None:
        """The tutor is done thinking. It speaks for `speak_seconds` (0 if silent); the microphone reopens after that plus a margin."""
        with self._lock:
            self._held = False
            self._gate_until = self._clock() + max(0.0, speak_seconds) + (self._margin if speak_seconds > 0 else 0.0)
            self._reopen_if_due()

    def set_config(self, cfg: EndpointConfig) -> None:
        with self._lock:
            if cfg != self._ep.cfg:
                self._ep.cfg = cfg

    def reset_idle(self) -> None:
        with self._lock:
            self.idle_expired = False

    def status(self) -> LiveStatus:
        with self._lock:
            self._reopen_if_due()
            if self._held:
                state = THINKING
            elif not self._ep.gate_open:
                state = SPEAKING
            elif self._ep.state == "speech":
                state = HEARING
            else:
                state = LISTENING
            return LiveStatus(state, self._ep.level, self._ep.probability, self._vad.name, len(self._queue), self.idle_expired)

    def _reopen_if_due(self) -> None:
        if not self._held and not self._ep.gate_open and self._clock() >= self._gate_until:
            self._ep.open_gate()


# --- one utterance -> the learner's draft text ----------------------------------------------------------------------------------------------

def ingest_utterance(vs: vsession.VoiceState, utt: Utterance, get_transcriber) -> bool:
    """Transcribe an utterance into the draft (names fixed, original kept) using the same session logic as Milestone 1.
    Returns True if there is text to send; False if the recogniser heard nothing usable (`vs.notice` says why)."""
    vsession.ingest_recording(vs, audio.encode_wav(utt.samples), get_transcriber)
    return bool(vs.heard.strip()) and vs.notice is None
