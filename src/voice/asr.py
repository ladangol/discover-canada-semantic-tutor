"""Speech-to-text with faster-whisper (local; CTranslate2). The model loads once per process, lazily."""
from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Protocol

import numpy as np

from src import config
from src.voice.models import TranscriptionResult, VoiceUnavailable

SLOT_WAIT_SECONDS = 10
_slots: threading.BoundedSemaphore | None = None
_slots_lock = threading.Lock()


def acquire_slot() -> threading.BoundedSemaphore:
    """One inference at a time by default; a second visitor waits briefly, then gets a friendly 'busy'."""
    global _slots
    with _slots_lock:
        if _slots is None:
            _slots = threading.BoundedSemaphore(config.voice_limits()["concurrency"])
    if not _slots.acquire(timeout=SLOT_WAIT_SECONDS):
        raise VoiceUnavailable("The voice tutor is busy with another learner. Please try again in a moment.")
    return _slots


class Transcriber(Protocol):
    def transcribe(self, samples: np.ndarray, duration_seconds: float) -> TranscriptionResult: ...


class FasterWhisperTranscriber:
    def __init__(self, model: str, device: str, compute_type: str, language: str | None, initial_prompt: str | None):
        self.language, self.initial_prompt = language, initial_prompt
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise VoiceUnavailable("Speech recognition isn't installed on this server. You can still type your question.") from None
        try:
            self._model = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception:                                   # download/offline/model errors: never leak internals
            raise VoiceUnavailable("The speech-recognition model could not be loaded. You can still type your question.") from None

    def transcribe(self, samples: np.ndarray, duration_seconds: float) -> TranscriptionResult:
        slot = acquire_slot()
        try:
            t0 = time.perf_counter()
            segments, info = self._model.transcribe(samples, language=self.language, initial_prompt=self.initial_prompt,
                                                    beam_size=5, vad_filter=True, condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in list(segments))     # the generator is lazy: consume it fully
            return TranscriptionResult(" ".join(text.split()), getattr(info, "language", self.language), duration_seconds,
                                       time.perf_counter() - t0)
        finally:
            slot.release()


@lru_cache(maxsize=1)
def get_transcriber() -> Transcriber:
    s = config.asr_settings()
    return FasterWhisperTranscriber(s["model"], s["device"], s["compute_type"], s["language"], s["initial_prompt"])
