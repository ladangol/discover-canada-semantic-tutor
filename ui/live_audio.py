"""WebRTC audio adapter for the Live page (Milestone 2): browser microphone frames -> 16 kHz mono float32 -> `LiveEngine`.

streamlit-webrtc hands `recv` one `av.AudioFrame` at a time on its own thread (typically 20 ms, 48 kHz, stereo or mono). Nothing is stored or
sent anywhere: frames are resampled, fed to the voice-activity detector, and only a finished utterance is kept in memory until it is transcribed."""
from __future__ import annotations

import av
import numpy as np
from streamlit_webrtc import AudioProcessorBase

from src.voice import live


def new_resampler() -> av.AudioResampler:
    return av.AudioResampler(format="s16", layout="mono", rate=16_000)


def frame_to_pcm16k(frame: av.AudioFrame, resampler: av.AudioResampler) -> np.ndarray:
    """Any sample rate / layout / format -> float32 in [-1, 1] at 16 kHz mono. The resampler keeps state between frames, so reuse one per stream."""
    out = resampler.resample(frame)
    frames = out if isinstance(out, list) else [out]
    chunks = [f.to_ndarray().reshape(-1) for f in frames if f is not None]
    if not chunks:
        return np.zeros(0, np.float32)
    return (np.concatenate(chunks).astype(np.float32)) / 32768.0


class LiveAudioProcessor(AudioProcessorBase):
    """One per WebRTC stream. `engine` is read by the Streamlit thread (status, utterances, gate)."""

    def __init__(self, engine: live.LiveEngine | None = None):
        super().__init__()
        self.engine = engine or live.LiveEngine()
        self._resampler = new_resampler()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.engine.feed(frame_to_pcm16k(frame, self._resampler))
        return frame
