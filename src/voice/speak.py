"""Speaking while the answer is still being written (Live page). No Streamlit, no WebRTC, no tutor/retrieval imports.

    answer text arrives in pieces ──► SpeechStreamer ──► whole sentences ──► worker thread: Kokoro, one sentence at a time ──► Speaker ──► browser

* `SpeechStreamer` cleans the text exactly as the one-shot path does (`tts.prepare_speech_text`: no citations, markup or JSON read aloud; "Queen Elizabeth
  II" -> "the Second"), cuts it into sentences, and hands each finished sentence to a worker thread, so the first sentence plays while the second is
  still being written. It never speaks the "I don't know" reply (the caller tells it how to recognise one) and it stops after a character budget, then
  says that there is more on the screen. When the turn is over, `finish()` returns the whole spoken reply as one WAV for the replay widget.
* `Speaker` is the output end: it pushes samples to whatever plays them (the browser, through WebRTC) and tells the live engine how long that speech will
  last, so the microphone stays shut while it plays.
* `Acks` are short phrases ("One moment.") synthesised once, in the background, so one can be played the instant the learner stops talking."""
from __future__ import annotations

import io
import queue
import random
import threading
import time
import wave
from typing import Callable

import numpy as np

from src.voice import tts
from src.voice.models import SpeechOutput, VoiceUnavailable

SAMPLE_RATE = 24_000
MORE_ON_SCREEN = "There is more on the screen."
ACK_PHRASES = ("Let me check.", "One moment.", "Hmm, let me look.")
STILL_PHRASES = ("I'm still looking.", "Just a little longer.", "Still working on it.", "Almost there.")
MAX_SPOKEN = 4000                      # characters of one reply that may be read when no budget is set (the reply itself is capped by the model)


class Speaker:
    """Pushes int16 samples to the output (`sink`) and reports how long they will play (`on_speak(seconds, interruptible)`).

    `clear()` drops everything queued AND starts a new `epoch`: whoever was still producing speech for the old epoch (a reply being synthesised, a "still
    looking" timer) has its later pushes ignored, so nothing from an interrupted reply can leak out after the learner cut it off."""

    def __init__(self, sink: Callable[[np.ndarray], None], clear: Callable[[], None] | None = None,
                 on_speak: Callable[..., None] | None = None, rate: int = SAMPLE_RATE):
        self._sink, self._clear, self._on_speak, self.rate = sink, clear, on_speak, rate
        self.epoch = 0

    def push(self, pcm: np.ndarray, *, epoch: int | None = None, interruptible: bool = True) -> None:
        if len(pcm) == 0 or (epoch is not None and epoch != self.epoch):
            return
        if self._on_speak is not None:
            self._on_speak(len(pcm) / self.rate, interruptible)      # close the microphone BEFORE the sound can leave
        self._sink(pcm)

    def clear(self) -> None:
        self.epoch += 1
        if self._clear is not None:
            self._clear()


class Acks:
    """Pre-synthesised short phrases: "One moment." (`pick()`, played when the learner stops talking) and "I'm still looking." (`pick_still()`, played when a
    reply is slow). `warm()` fills them in the background; a pick is None until a phrase is ready, and never repeats the previous one."""

    def __init__(self, phrases=ACK_PHRASES, still=STILL_PHRASES, rng: random.Random | None = None):
        self._groups = {"ack": phrases, "still": still}
        self._rng = rng or random.Random()
        self._ready: dict[str, list[tuple[int, np.ndarray]]] = {"ack": [], "still": []}
        self._last: dict[str, int | None] = {"ack": None, "still": None}
        self._lock = threading.Lock()
        self._started = False

    def warm(self, synth) -> None:
        if self._started or not tts.can_stream(synth):
            return
        self._started = True

        def work() -> None:
            for group, phrases in self._groups.items():
                for n, ph in enumerate(phrases):
                    try:
                        x = synth.pcm(ph)
                    except Exception:                       # a courtesy phrase is never an error
                        continue
                    with self._lock:
                        self._ready[group].append((n, x))

        threading.Thread(target=work, name="ack-warmup", daemon=True).start()

    def _pick(self, group: str) -> np.ndarray | None:
        with self._lock:
            options = [(n, x) for n, x in self._ready[group] if n != self._last[group]] or self._ready[group]
            if not options:
                return None
            n, x = self._rng.choice(options)
            self._last[group] = n
            return x

    def pick(self) -> np.ndarray | None:
        return self._pick("ack")

    def pick_still(self) -> np.ndarray | None:
        return self._pick("still")


class Patience:
    """While a reply is being prepared, says a short "still looking" phrase when nothing real has been spoken for a while: first after `first_after`
    seconds from `started_at`, then every `every` seconds, at most `limit` times. Use as a context manager around the slow work; it stops at once on exit
    and never speaks once `spoke()` is true (the reply has begun)."""

    def __init__(self, acks: Acks, speaker: Speaker, spoke: Callable[[], bool], *, started_at: float | None = None, first_after: float = 5.0,
                 every: float = 6.0, limit: int = 3, clock: Callable[[], float] = time.monotonic):
        self._acks, self._speaker, self._spoke, self._clock = acks, speaker, spoke, clock
        self._epoch = speaker.epoch                                  # an interruption (clear) ends this timer's phrases
        self._start = clock() if started_at is None else started_at
        self._first, self._every, self._limit = first_after, every, limit
        self._stop = threading.Event()
        self.said = 0
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while self.said < self._limit and not self._stop.is_set():
            due = self._start + self._first + self.said * self._every
            wait = due - self._clock()
            if wait > 0:
                if self._stop.wait(min(wait, 0.25)):
                    return
                continue
            if self._spoke() or self._stop.is_set() or self._speaker.epoch != self._epoch:
                return
            x = self._acks.pick_still()
            self.said += 1
            if x is not None:
                self._speaker.push(x, epoch=self._epoch, interruptible=False)

    def __enter__(self) -> "Patience":
        self._thread = threading.Thread(target=self._run, name="patience", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class SpeechStreamer:
    """One reply. Feed it the answer's text as it is written, then call `finish()` once the turn is complete."""

    def __init__(self, synth, speaker: Speaker, *, max_chars: int | None = 240,
                 verdict: Callable[[str], bool | None] = lambda text: True, clock: Callable[[], float] = time.monotonic):
        self._synth, self._speaker, self._cap, self._verdict, self._clock = synth, speaker, max_chars, verdict, clock
        self._gap = np.zeros(int(speaker.rate * (getattr(synth, "pause_ms", 150)) / 1000), np.int16)
        self._epoch = speaker.epoch                    # speech for an older epoch (the learner interrupted) is dropped by the speaker
        self._aborted = False
        self._raw = ""
        self._muted = False                  # the reply is the "I don't know" one: it is spoken from the turn's own text instead
        self._streamed = False               # at least one sentence came from the stream
        self._done = 0                       # sentences already handed to the worker
        self._spoken_chars = 0
        self._truncated = False
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pcm: list[np.ndarray] = []
        self._synth_seconds = 0.0
        self._error: Exception | None = None
        self.first_audio_at: float | None = None      # clock time the first sentence of the reply was pushed (for latency measurement)

    # --- input ------------------------------------------------------------------------------------------------------------------------
    def feed(self, piece: str) -> None:
        """Text so far grows by `piece`. Speaks every sentence that is now complete."""
        self._raw += piece
        if self._muted or self._aborted:
            return
        v = self._verdict(self._raw)
        if v is None:
            return                            # could still turn out to be the "I don't know" reply
        if v is False:
            self._muted = True
            return
        self._streamed = True
        self._queue_sentences(final=False)

    def _sentences(self) -> list[str]:
        raw = self._raw
        k = raw.rfind("[")
        if k > raw.rfind("]"):                # a citation that is still being written ("[Discover Canada, p.") is not text to speak or to split at
            raw = raw[:k]
        return tts.split_sentences(tts.prepare_speech_text(raw, limit=MAX_SPOKEN if self._cap is None else tts.MAX_SPOKEN_CHARS))

    def _queue_sentences(self, final: bool) -> None:
        pieces = self._sentences()
        for sentence in (pieces if final else pieces[:-1])[self._done:]:     # the last piece may still be growing
            self._done += 1
            self._offer(sentence)

    def _offer(self, sentence: str) -> None:
        """Hand one sentence to the voice unless the spoken-length budget is used up."""
        if self._truncated or not sentence.strip():
            return
        n = len(sentence)
        if self._cap is not None:
            if self._spoken_chars and self._spoken_chars + 1 + n > self._cap:
                self._truncated = True
                return
            if not self._spoken_chars and n > self._cap * 1.5:               # one very long first sentence: cut at the last space
                sentence = sentence[: int(self._cap)].rsplit(" ", 1)[0].rstrip(",;:") + "."
                n = len(sentence)
                self._truncated = True
        self._spoken_chars += n + 1
        self._enqueue(sentence)

    def _enqueue(self, sentence: str) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._work, name="speech-stream", daemon=True)
            self._thread.start()
        self._queue.put(sentence)

    # --- the voice ----------------------------------------------------------------------------------------------------------------------
    def _work(self) -> None:
        first = True
        while True:
            sentence = self._queue.get()
            if sentence is None:
                return
            if self._error is not None or self._aborted:
                continue
            try:
                t0 = time.perf_counter()
                x = self._synth.pcm(sentence)
                self._synth_seconds += time.perf_counter() - t0
                if not len(x):
                    continue
                if not first and len(self._gap):
                    x = np.concatenate([self._gap, x])
                if self._aborted:
                    continue
                if first:
                    self.first_audio_at, first = self._clock(), False
                self._pcm.append(x)
                self._speaker.push(x, epoch=self._epoch)
            except Exception as exc:                  # a voice problem must not break the answer; finish() reports it
                self._error = exc

    def abort(self) -> None:
        """The learner interrupted: speak nothing more of this reply (what was already heard stays in the replay)."""
        self._aborted = True
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    @property
    def aborted(self) -> bool:
        return self._aborted

    # --- the end of the turn --------------------------------------------------------------------------------------------------------------
    def finish(self, speech_text: str) -> SpeechOutput:
        """The turn is complete. Speaks whatever is left (the whole `speech_text` if nothing was streamed: the "I don't know" reply, a quiz
        result, a clarification), waits for the voice, and returns the reply that was spoken as one WAV."""
        if self._aborted:
            if self._thread is not None:
                self._queue.put(None)
                self._thread.join()
            if not self._pcm:
                raise VoiceUnavailable("You interrupted me.")
        elif self._streamed and not self._muted:
            self._queue_sentences(final=True)
            if self._truncated:
                self._enqueue(MORE_ON_SCREEN)
        else:
            for sentence in tts.split_sentences(tts.brief_speech(speech_text, self._cap, limit=MAX_SPOKEN if self._cap is None else tts.MAX_SPOKEN_CHARS)):
                self._enqueue(sentence)
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
        if not self._pcm:
            if isinstance(self._error, VoiceUnavailable):
                raise self._error
            raise VoiceUnavailable("There is nothing to read aloud.")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(self._speaker.rate)
            w.writeframes(np.concatenate(self._pcm).tobytes())
        return SpeechOutput(buf.getvalue(), "audio/wav", self._synth_seconds)

    @property
    def streamed(self) -> bool:
        return self._streamed and not self._muted
