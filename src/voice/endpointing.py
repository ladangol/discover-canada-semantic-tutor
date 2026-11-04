"""Turn-taking without Record/Stop: turn a stream of audio into utterances (Milestone 2). Pure Python; no Streamlit, no models.

Feed 16 kHz mono float32 audio in any chunk size. Events come out:
  speech_start  speech has begun (at least `min_speech_ms` of consecutive speech frames: a cough or a click never starts a turn);
  utterance     speech ended after `hangover_ms` of silence: the audio, with a little lead-in and tail;
  max_length    the speaker did not stop: the utterance was cut at `max_utterance_s`;
  idle_timeout  nothing was said for `idle_timeout_s` (the caller can stop the microphone).

Hysteresis: speech starts at `threshold` and only ends when the probability stays below `end_threshold`, so a short dip inside a word does not
split an utterance. While the gate is closed (the tutor is thinking or speaking) audio is thrown away and nothing is remembered, so the tutor's
own voice can never be mistaken for the learner (half-duplex)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.voice.vad import FRAME, FRAME_MS, SAMPLE_RATE, Vad

Kind = Literal["speech_start", "utterance", "max_length", "idle_timeout"]


@dataclass(frozen=True)
class EndpointConfig:
    threshold: float = 0.5
    end_threshold: float | None = None          # default: threshold - 0.15
    min_speech_ms: int = 250
    hangover_ms: int = 800                      # silence that ends a turn
    preroll_ms: int = 320                       # audio kept from before the speech was confirmed
    tail_ms: int = 200                          # silence kept after the last word
    max_utterance_s: float = 30.0
    idle_timeout_s: float | None = None

    @property
    def end(self) -> float:
        return self.end_threshold if self.end_threshold is not None else max(0.1, self.threshold - 0.15)


@dataclass(frozen=True)
class Event:
    kind: Kind
    samples: np.ndarray | None = None
    seconds: float = 0.0


def _frames(ms: float) -> int:
    return max(1, int(round(ms / FRAME_MS)))


class Endpointer:
    def __init__(self, vad: Vad, config: EndpointConfig | None = None):
        self.vad, self.cfg = vad, config or EndpointConfig()
        self._carry = np.zeros(0, np.float32)
        self.gate_open = True
        self.level = 0.0                         # loudness of the latest frame (for a meter)
        self.probability = 0.0                   # speech probability of the latest frame
        self._reset_state()

    # --- state ---------------------------------------------------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.state = "idle"                      # "idle" | "speech"
        self._pre: deque[np.ndarray] = deque(maxlen=_frames(self.cfg.preroll_ms) + _frames(self.cfg.min_speech_ms) + 2)
        self._run = 0                            # consecutive speech frames while idle
        self._frames: list[np.ndarray] = []
        self._speech = 0                         # speech frames in the current utterance
        self._silence = 0
        self._idle_frames = 0

    def reset(self) -> None:
        self._reset_state()
        self._carry = np.zeros(0, np.float32)
        self.vad.reset()

    def close_gate(self) -> None:
        """The tutor is thinking or speaking: ignore the microphone and forget everything heard so far."""
        self.gate_open = False
        self.reset()

    def open_gate(self) -> None:
        self.gate_open = True
        self.reset()

    # --- audio in --------------------------------------------------------------------------------------------------------------
    def feed(self, samples: np.ndarray) -> list[Event]:
        if not self.gate_open or samples is None or not len(samples):
            return []
        x = np.concatenate([self._carry, np.asarray(samples, dtype=np.float32).reshape(-1)])
        n = (len(x) // FRAME) * FRAME
        self._carry = x[n:]
        events: list[Event] = []
        for i in range(0, n, FRAME):
            events += self._frame(x[i:i + FRAME])
        return events

    def _frame(self, f: np.ndarray) -> list[Event]:
        cfg = self.cfg
        p = self.probability = float(self.vad.probability(f))
        self.level = float(np.sqrt(np.mean(np.square(f, dtype=np.float64))))
        if self.state == "idle":
            self._pre.append(f)
            self._run = self._run + 1 if p >= cfg.threshold else 0
            if self._run >= _frames(cfg.min_speech_ms):
                self.state, self._frames = "speech", list(self._pre)
                self._speech, self._silence, self._idle_frames = self._run, 0, 0
                return [Event("speech_start")]
            self._idle_frames += 1
            if cfg.idle_timeout_s and self._idle_frames * FRAME_MS >= cfg.idle_timeout_s * 1000:
                self._idle_frames = 0
                return [Event("idle_timeout")]
            return []
        self._frames.append(f)
        if p >= cfg.end:
            self._speech += 1
            self._silence = 0
        else:
            self._silence += 1
        if len(self._frames) * FRAME_MS >= cfg.max_utterance_s * 1000:
            return [self._finish("max_length", trim=False)]
        if self._silence * FRAME_MS >= cfg.hangover_ms:
            return [self._finish("utterance", trim=True)]
        return []

    def _finish(self, kind: Kind, *, trim: bool) -> Event:
        frames = self._frames
        if trim:                                                  # drop the silence beyond a short tail
            drop = max(0, self._silence - _frames(self.cfg.tail_ms))
            frames = frames[: len(frames) - drop] if drop else frames
        samples = np.concatenate(frames) if frames else np.zeros(0, np.float32)
        self._after_turn()
        return Event(kind, samples, len(samples) / SAMPLE_RATE)

    def _after_turn(self) -> None:
        self._reset_state()
        self.vad.reset()
