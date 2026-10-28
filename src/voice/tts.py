"""Text-to-speech, local. Default engine: Kokoro-82M (TTS_ENGINE=kokoro; kokoro-onnx = same model via ONNX Runtime). Alternative: Piper (TTS_ENGINE=piper, voice file via
TTS_VOICE_PATH; not bundled, see README). Both sit behind `Synthesizer`, and the tutor never knows which one is used."""
from __future__ import annotations

import importlib.util
import io
import re
import threading
import time
import wave
from functools import lru_cache
from typing import Protocol

import numpy as np

from src import config
from src.voice.asr import acquire_slot
from src.voice.models import SpeechOutput, VoiceUnavailable

MAX_SPOKEN_CHARS = 1200


# --- spoken forms: how the voice should SAY things that are written compactly (speech only; the screen text is untouched) ---
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
_ORDINAL = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth", 6: "Sixth", 7: "Seventh", 8: "Eighth", 9: "Ninth", 10: "Tenth"}
# Only given names that carry regnal/papal numbers, so "Section I", "Part II" or the pronoun "I" are never touched.
_REGNAL = ("Elizabeth|Victoria|George|Charles|Edward|Henry|William|James|John|Francis|Louis|Philip|Richard|Anne|Mary|"
           "Pius|Paul|Benedict|Leo|Alexander|Nicholas")
_REGNAL_RE = re.compile(rf"\b({_REGNAL})\s+(I{{1,3}}|IV|VI{{0,3}}|IX|X)\b(?![.\w])")
_WORLD_WAR_RE = re.compile(r"\bWorld War (I{1,2})\b(?![.\w])")
_YEAR_RANGE_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\s?[–-]\s?(\d{4}|\d{2})\b")


def _year_range(m: re.Match) -> str:
    a, b = m.group(1), m.group(2)
    b = a[:2] + b if len(b) == 2 else b
    return f"{a} to {b}" if int(b) > int(a) else m.group(0)


def spoken_forms(text: str) -> str:
    """Deterministic, rule-based (no LLM): 'Queen Elizabeth II' -> 'Queen Elizabeth the Second', 'World War II' -> 'World War
    Two', 'St. Lawrence' -> 'Saint Lawrence', 'Dr. Banting' -> 'Doctor Banting', '1837–38' -> '1837 to 1838'."""
    t = _REGNAL_RE.sub(lambda m: f"{m.group(1)} the {_ORDINAL[_ROMAN[m.group(2)]]}", text)
    t = _WORLD_WAR_RE.sub(lambda m: "World War " + ("One" if m.group(1) == "I" else "Two"), t)
    t = re.sub(r"\bSt\.\s+(?=[A-Z])", "Saint ", t)
    t = re.sub(r"\bDr\.\s+(?=[A-Z])", "Doctor ", t)
    return _YEAR_RANGE_RE.sub(_year_range, t)


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_DOTTED_ABBREVIATION = re.compile(r"(?:\b[A-Z]\.){2,}$")           # P.E.I.  B.C.  U.S.A.  N.W.T.


def split_sentences(text: str) -> list[str]:
    """Sentence pieces for the speech engines. A dotted abbreviation such as 'P.E.I.' followed by a lower-case word is not a sentence
    end ("P.E.I. is the smallest province"), so it is kept with what follows instead of being spoken alone with a pause."""
    out: list[str] = []
    for piece in _SENTENCE_BREAK.split(text.strip()):
        if out and piece[:1].islower() and _DOTTED_ABBREVIATION.search(out[-1]):
            out[-1] = f"{out[-1]} {piece}"
        elif piece.strip():
            out.append(piece)
    return out


def prepare_speech_text(text: str) -> str:
    """Deterministic clean-up so the voice never reads citation syntax, markup or JSON. Meaning is not changed.

    '[Discover Canada, p. 18, Confederation]', '[S2]', '[p.18]'   -> removed (citations stay visible on screen)
    markdown (#, *, _, `, > and bullets), links, raw URLs         -> plain words
    'Queen Elizabeth II', 'World War II', 'St.', 'Dr.', '1837–38' -> spoken forms (see spoken_forms)
    """
    t = text or ""
    t = re.sub(r"\[[^\]]*\]\([^)]*\)", lambda m: re.match(r"\[([^\]]*)\]", m.group(0)).group(1), t)   # [label](url) -> label
    t = re.sub(r"\[[^\]]*\]", "", t)                                                                  # citations / labels
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"`+", "", t)
    t = re.sub(r"(\*\*|__|\*|_)(\S.*?\S|\S)\1", r"\2", t)                                             # bold / italics
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)                                               # headings
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)
    t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", t, flags=re.M)                                        # list markers
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    lines = [ln if re.search(r"[.!?:]$", ln) else ln + "." for ln in lines]                           # pause between bullets
    t = " ".join(lines)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = spoken_forms(t)
    if len(t) > MAX_SPOKEN_CHARS:                                                                      # keep it a short lesson
        cut = t.rfind(". ", 0, MAX_SPOKEN_CHARS)
        t = t[: cut + 1] if cut > 200 else t[:MAX_SPOKEN_CHARS]
    return t


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> SpeechOutput: ...


class PiperSynthesizer:
    def __init__(self, voice_path):
        try:
            from piper import PiperVoice
        except ImportError:
            raise VoiceUnavailable("Spoken replies aren't installed on this server. The written answer is below.") from None
        if not voice_path or not voice_path.exists():
            raise VoiceUnavailable("No voice model is set up, so I can't speak. The written answer is below.")
        try:
            self._voice = PiperVoice.load(str(voice_path))
        except Exception:
            raise VoiceUnavailable("The voice model could not be loaded. The written answer is below.") from None
        from piper.config import SynthesisConfig
        knobs = config.tts_settings()
        self._pause_ms = knobs.pop("pause_ms")
        self._pause_ms = 280 if self._pause_ms is None else self._pause_ms
        self._cfg = SynthesisConfig(**{k: v for k, v in knobs.items() if v is not None})

    def synthesize(self, text: str) -> SpeechOutput:
        spoken = prepare_speech_text(text)
        if not spoken:
            raise VoiceUnavailable("There is nothing to read aloud.")
        slot = acquire_slot()
        try:
            t0 = time.perf_counter()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                first = True
                for chunk in self._voice.synthesize(spoken, syn_config=self._cfg):      # one chunk per sentence
                    if first:
                        w.setframerate(chunk.sample_rate), w.setsampwidth(chunk.sample_width), w.setnchannels(chunk.sample_channels)
                        first = False
                    elif self._pause_ms:                                                 # a short, natural pause between sentences
                        n = int(chunk.sample_rate * self._pause_ms / 1000) * chunk.sample_width * chunk.sample_channels
                        w.writeframes(b"\x00" * n)
                    w.writeframes(chunk.audio_int16_bytes)
            return SpeechOutput(buf.getvalue(), "audio/wav", time.perf_counter() - t0)
        finally:
            slot.release()


# Best-effort pronunciations for names in the guide that Kokoro's G2P gets wrong (checked by printing its phonemes, not by ear
# for every word). Kokoro's inline markup is [word](/phonemes/); A=ay, I=eye, O=oh, ɹ=r. Applied only for the Kokoro engine.
KOKORO_PRONUNCIATIONS = {
    "Iqaluit": "ikˈɑluɪt",              # ee-KAH-loo-it
    "Champlain": "ʃæmplˈA",             # sham-PLAY(n)
    "Cartier": "kɑɹtiˈA",               # kar-tee-AY
    "Laurier": "lˈɔɹiA",                # LOR-ee-ay
    "Métis": "mAtˈi",                   # may-TEE
    "Metis": "mAtˈi",
    "Papineau": "pæpinˈO",              # pap-ee-NOH
    "Pierre": "piˈɛɹ",                  # pee-AIR
    "Hippolyte": "ipəlˈit",             # ee-puh-LEET
    "Fontaine": "fɔntˈɛn",              # fon-TEN
    "Nunavut": "nˈunəvˌut",             # NOO-nuh-voot
    "Michif": "miʃˈif",                 # mi-SHEEF (Kokoro says "MICH-if")
}
_PRON_RE = re.compile(r"\b(" + "|".join(sorted(KOKORO_PRONUNCIATIONS, key=len, reverse=True)) + r")\b(?!\]|\))")


def apply_pronunciations(text: str) -> str:
    return _PRON_RE.sub(lambda m: f"[{m.group(1)}](/{KOKORO_PRONUNCIATIONS[m.group(1)]}/)", text)


class KokoroSynthesizer:
    """Kokoro-82M (Apache-2.0): 24 kHz, 54 built-in voices, runs on CPU. The model downloads from Hugging Face on first use."""
    SAMPLE_RATE = 24_000

    def __init__(self, voice: str, speed: float, pause_ms: int | None = None):
        if importlib.util.find_spec("kokoro") is None:
            raise VoiceUnavailable("Kokoro isn't installed on this server. The written answer is below.")
        self._voice, self._speed = voice, speed
        self._pause_ms = 150 if pause_ms is None else pause_ms
        try:
            from kokoro import KPipeline
            self._pipe = KPipeline(lang_code="b" if voice[:2] in ("bf", "bm") else "a", repo_id="hexgrad/Kokoro-82M")
        except Exception:
            raise VoiceUnavailable("The Kokoro voice could not be loaded. The written answer is below.") from None

    def synthesize(self, text: str) -> SpeechOutput:
        spoken = prepare_speech_text(text)
        if not spoken:
            raise VoiceUnavailable("There is nothing to read aloud.")
        spoken = apply_pronunciations(spoken)
        slot = acquire_slot()
        try:
            t0 = time.perf_counter()
            pieces: list[np.ndarray] = []
            gap = np.zeros(int(self.SAMPLE_RATE * self._pause_ms / 1000), np.int16)
            for sentence in split_sentences(spoken):
                for _gs, _ps, audio in self._pipe(sentence, voice=self._voice, speed=self._speed, split_pattern=r"\n+"):
                    if audio is None:
                        continue
                    if pieces and len(gap):
                        pieces.append(gap)
                    x = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
                    pieces.append((x * 32767).astype("<i2"))
            if not pieces:
                raise VoiceUnavailable("There is nothing to read aloud.")
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(self.SAMPLE_RATE)
                w.writeframes(np.concatenate(pieces).tobytes())
            return SpeechOutput(buf.getvalue(), "audio/wav", time.perf_counter() - t0)
        finally:
            slot.release()


class KokoroOnnxSynthesizer:
    """Kokoro-82M through ONNX Runtime (kokoro-onnx): int8 model ≈ 92 MB, no torch. Pronunciation uses the same misaki front-end (and
    the same name fixes) as the PyTorch engine, because kokoro-onnx's own espeak front-end mispronounces e.g. Tecumseh and Champlain."""
    SAMPLE_RATE = 24_000

    def __init__(self, model_path, voices_path, voice: str, speed: float, pause_ms: int | None = None):
        if importlib.util.find_spec("kokoro_onnx") is None or importlib.util.find_spec("misaki") is None:
            raise VoiceUnavailable("Kokoro (ONNX) isn't installed on this server. The written answer is below.")
        if not (model_path and model_path.exists() and voices_path and voices_path.exists()):
            raise VoiceUnavailable("The Kokoro (ONNX) model files are missing. The written answer is below.")
        self._voice, self._speed = voice, speed
        self._pause_ms = 150 if pause_ms is None else pause_ms
        try:
            from kokoro_onnx import Kokoro
            from misaki import en, espeak
            british = voice[:2] in ("bf", "bm")
            self._ko = Kokoro(str(model_path), str(voices_path))
            self._g2p = en.G2P(trf=False, british=british, fallback=espeak.EspeakFallback(british=british))
        except Exception:
            raise VoiceUnavailable("The Kokoro (ONNX) voice could not be loaded. The written answer is below.") from None

    def synthesize(self, text: str) -> SpeechOutput:
        spoken = prepare_speech_text(text)
        if not spoken:
            raise VoiceUnavailable("There is nothing to read aloud.")
        spoken = apply_pronunciations(spoken)
        slot = acquire_slot()
        try:
            t0 = time.perf_counter()
            pieces: list[np.ndarray] = []
            gap = np.zeros(int(self.SAMPLE_RATE * self._pause_ms / 1000), np.int16)
            for sentence in split_sentences(spoken):                              # one model call per sentence (token limit per call)
                phonemes, _ = self._g2p(sentence)
                if not phonemes:
                    continue
                audio, _sr = self._ko.create(phonemes, voice=self._voice, speed=self._speed, is_phonemes=True)
                if pieces and len(gap):
                    pieces.append(gap)
                pieces.append((np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2"))
            if not pieces:
                raise VoiceUnavailable("There is nothing to read aloud.")
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(self.SAMPLE_RATE)
                w.writeframes(np.concatenate(pieces).tobytes())
            return SpeechOutput(buf.getvalue(), "audio/wav", time.perf_counter() - t0)
        finally:
            slot.release()


def engine_ready() -> bool:
    """Is the configured engine usable? Checked without loading any model (Kokoro's weights download on first use)."""
    engine = config.tts_engine()
    if engine == "kokoro":
        return importlib.util.find_spec("kokoro") is not None
    if engine == "kokoro-onnx":
        m, v = config.kokoro_onnx_paths()
        return all(importlib.util.find_spec(x) is not None for x in ("kokoro_onnx", "misaki")) and m.exists() and v.exists()
    p = config.tts_voice_path()
    return importlib.util.find_spec("piper") is not None and bool(p and p.exists())


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load() -> Synthesizer:
    engine = config.tts_engine()
    if engine in ("kokoro", "kokoro-onnx"):
        k = config.kokoro_settings()
        pause = config.tts_settings()["pause_ms"]
        if engine == "kokoro-onnx":
            return KokoroOnnxSynthesizer(*config.kokoro_onnx_paths(), k["voice"], k["speed"], pause)
        return KokoroSynthesizer(k["voice"], k["speed"], pause)
    return PiperSynthesizer(config.tts_voice_path())


def get_synthesizer() -> Synthesizer:
    with _lock:
        return _load()
