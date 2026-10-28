"""Validate and convert a browser recording (WAV) into what the speech model needs.

Pure Python + numpy (`wave`): nothing is written to disk and nothing is logged."""
from __future__ import annotations

import io
import wave

import numpy as np

from src.config import voice_limits
from src.voice.models import AudioError, AudioInfo

TARGET_RATE = 16_000
MIN_SECONDS = 0.3


def read_wav(data: bytes, *, max_seconds: float | None = None, max_bytes: int | None = None) -> tuple[np.ndarray, AudioInfo]:
    """Return (mono float32 samples in [-1, 1] at 16 kHz, info). Raises AudioError with a learner-safe message."""
    lim = voice_limits()
    max_seconds = max_seconds or lim["max_seconds"]
    max_bytes = max_bytes or lim["max_bytes"]
    if not data:
        raise AudioError("The recording is empty. Please record again.")
    if len(data) > max_bytes:
        raise AudioError("That recording is too large. Please keep it short and try again.")
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            rate, ch, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            if rate <= 0 or ch <= 0 or frames <= 0:
                raise AudioError("The recording is empty. Please record again.")
            duration = frames / rate
            if duration > max_seconds:                      # reject before decoding the samples
                raise AudioError(f"That recording is longer than {int(max_seconds)} seconds. Please keep it shorter.")
            raw = w.readframes(frames)
    except AudioError:
        raise
    except (wave.Error, EOFError):
        raise AudioError("I couldn't read that recording. Please record again.") from None
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioError("That audio format isn't supported. Please record again.")
    x = x[: (len(x) // ch) * ch].reshape(-1, ch).mean(axis=1)    # to mono
    if duration < MIN_SECONDS or not np.any(x):
        raise AudioError("I didn't hear anything. Please check your microphone and record again.")
    if rate != TARGET_RATE:                                       # linear resample; fine for speech recognition
        n = max(1, int(round(len(x) * TARGET_RATE / rate)))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x.astype(np.float32), AudioInfo(duration, rate, ch)
