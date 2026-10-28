"""Small typed records and errors shared by the voice adapters. No Streamlit, no tutor imports."""
from __future__ import annotations

from dataclasses import dataclass, field


class VoiceError(RuntimeError):
    """Base class. The message is safe to show to a learner (no paths, keys or audio content)."""


class AudioError(VoiceError):
    """The recording is empty, unreadable, too long or too large."""


class VoiceUnavailable(VoiceError):
    """ASR or TTS cannot run: package/model not installed, or the host is busy."""


@dataclass(frozen=True)
class AudioInfo:
    duration_seconds: float
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration_seconds: float          # length of the recording
    inference_seconds: float         # time spent in the model (measured, not estimated)


@dataclass(frozen=True)
class SpeechOutput:
    audio_bytes: bytes
    mime_type: str
    synthesis_seconds: float


@dataclass
class VoiceTimings:
    """Measured processing time per stage, in seconds; None = stage did not run. Never includes the time the
    learner spends recording, correcting the transcript or pressing Play."""
    audio_seconds: float | None = None
    asr: float | None = None
    retrieval: float | None = None
    generation: float | None = None
    tts: float | None = None
    tutor_total: float | None = None

    def processing_total(self) -> float:
        return sum(v for v in (self.asr, self.tutor_total, self.tts) if v)

    def rows(self) -> list[tuple[str, str]]:
        names = [("Audio length", self.audio_seconds), ("Speech to text", self.asr), ("Retrieval", self.retrieval),
                 ("Generation / grading", self.generation), ("Tutor total", self.tutor_total), ("Text to speech", self.tts)]
        return [(n, "–" if v is None else f"{v:.2f} s") for n, v in names]
