"""WebRTC audio adapter for the Live page (Milestone 2): browser microphone frames -> 16 kHz mono float32 -> `LiveEngine`.

streamlit-webrtc delivers one `av.AudioFrame` at a time to a sink callback (typically 20 ms, 48 kHz, stereo or mono). Nothing is stored or
sent anywhere: frames are resampled, fed to the voice-activity detector, and only a finished utterance is kept in memory until it is transcribed."""
from __future__ import annotations

import queue
import threading
import time

import av
import numpy as np

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


class LiveAudioPipe:
    """One per WebRTC session. Microphone frames arrive on the WebRTC event loop (`on_frame`, which only queues them); a worker thread resamples
    them and feeds the engine, so voice detection can never stall the audio that is being sent back. `engine` is read by the Streamlit thread
    (status, utterances, gate). `speaker` (set once the output track exists) is where the tutor's voice goes."""

    MAX_QUEUED = 500                                   # ≈ 10 s of 20 ms frames; beyond that the oldest are dropped rather than lagging behind

    def __init__(self, engine: live.LiveEngine | None = None, *, acks=None):
        self.engine = engine or live.LiveEngine(on_interrupt=self.interrupt)
        self.speaker = None                            # src.voice.speak.Speaker, set by the page once the output track exists
        self.acks = acks                               # src.voice.speak.Acks (pre-synthesised "One moment.")
        self.closed = False
        self.ack_enabled = True
        self.active = None                             # the SpeechStreamer of the reply being spoken (so an interruption can stop it)
        self.last_ack_at: float | None = None          # monotonic time the last acknowledgement was pushed (latency display)
        self._resampler = new_resampler()
        self._frames: queue.Queue = queue.Queue(self.MAX_QUEUED)
        self._thread: threading.Thread | None = None

    def interrupt(self) -> None:
        """The learner spoke over the tutor (or pressed Stop talking): drop what is queued and stop producing the rest of the reply."""
        if self.speaker is not None:
            self.speaker.clear()
        if self.active is not None:
            self.active.abort()

    def acknowledge(self, utt) -> None:
        """Say "One moment." The page calls this only AFTER the words were recognised and are a real request, so a breath or an "um" never gets one."""
        if self.ack_enabled and self.speaker is not None and self.acks is not None:
            x = self.acks.pick()
            if x is not None:
                self.last_ack_at = time.monotonic()
                self.speaker.push(x, epoch=self.speaker.epoch, interruptible=False)

    def feed_frame(self, frame: av.AudioFrame) -> None:
        """Resample one frame and give it to the engine (synchronous; the worker thread calls this)."""
        self.engine.feed(frame_to_pcm16k(frame, self._resampler))

    def on_frame(self, frame: av.AudioFrame) -> None:
        """WebRTC callback: hand the frame to the worker and return at once."""
        if self.closed:
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="live-audio", daemon=True)
            self._thread.start()
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass

    def _run(self) -> None:
        while not self.closed:
            try:
                frame = self._frames.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.feed_frame(frame)
            except Exception:                           # one bad frame must not end the conversation
                pass

    def close(self) -> None:
        self.closed = True
