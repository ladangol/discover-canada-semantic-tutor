"""Shared fakes for the voice tests: synthetic WAVs, a fake tutor source/quiz, and call-counting adapters."""
from __future__ import annotations

import io
import wave

import numpy as np

from src.generation import GenerationResult
from src.retrieval import RetrievedChunk
from src.tutor import QuizQuestion
from src.voice.models import SpeechOutput, TranscriptionResult


def make_wav(seconds=1.0, rate=16000, channels=1, width=2, amp=0.3, freq=440.0) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    x = (amp * np.sin(2 * np.pi * freq * t)) if amp else np.zeros_like(t)
    if width == 2:
        pcm = (x * 32767).astype("<i2")
    elif width == 1:
        pcm = (x * 127 + 128).astype(np.uint8)
    else:
        pcm = (x * 2147483647).astype("<i4")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels), w.setsampwidth(width), w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def chunk(section="Confederation", page="18") -> RetrievedChunk:
    return RetrievedChunk("S1", "c1", "Confederation took place in 1867 when four provinces joined.", page, section, "p")


def mcq(correct_index=0) -> QuizQuestion:
    return QuizQuestion("mcq", "Which year did Confederation take place?", chunk(), "It was 1867.",
                        options=["1867", "1812", "Ontario and Quebec", "Nova Scotia"], correct_index=correct_index)


def free_q() -> QuizQuestion:
    return QuizQuestion("free", "Which year did Confederation take place?", chunk(), "It was 1867.", reference_answer="1867")


class Gen:
    """A fake `generate` that counts calls and returns canned text."""
    def __init__(self, text="Confederation was in 1867 [S1]."):
        self.text, self.calls = text, 0

    def __call__(self, messages, *a, **k):
        self.calls += 1
        return GenerationResult(self.text, "stop", 10, 10)


class NoCalls:
    def __call__(self, *a, **k):
        raise AssertionError("must not be called")


class FakeTranscriber:
    def __init__(self, text="Can you explain responsible government?"):
        self.text, self.calls = text, 0

    def transcribe(self, samples, duration):
        self.calls += 1
        return TranscriptionResult(self.text, "en", duration, 0.25)


class FakeSynth:
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return SpeechOutput(b"RIFF" + str(len(self.calls)).encode(), "audio/wav", 0.1)


class SeqGen:
    """Returns the given texts in order (one per call) and counts calls; raises if a later call is made unexpectedly."""
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), 0

    def __call__(self, messages, *a, **k):
        self.calls += 1
        if not self.texts:
            raise AssertionError("unexpected extra LLM call")
        return GenerationResult(self.texts.pop(0), "stop", 10, 10)


class FakeCorpus:
    chapters = ["Rights and Responsibilities of Citizenship", "Canada's History"]

    def sections(self, chapter=None):
        return ["The Charter", "Confederation"] if chapter else ["The Charter", "Confederation"]

