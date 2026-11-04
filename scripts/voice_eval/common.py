"""Shared helpers for the voice evaluation harness (Milestone 4). Evaluation only: nothing here is imported by the app.

Honesty rules baked in: every result file records WHAT audio was used (synthetic voices, not people), the machine, and the sample counts."""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts" / "voice_eval"
OUT.mkdir(parents=True, exist_ok=True)
SR = 16_000

# Synthetic "speakers". macOS `say` voices need no download and differ in accent; af_heart is the app's own TTS voice (so it is the easiest case).
SAY_VOICES = {"samantha": ("Samantha", "US"), "daniel": ("Daniel", "UK"), "karen": ("Karen", "AU"), "rishi": ("Rishi", "IN")}


def questions() -> list[dict]:
    return json.loads((Path(__file__).parent / "questions.json").read_text())["questions"]


def save(name: str, payload: dict) -> Path:
    p = OUT / f"{name}.json"
    p.write_text(json.dumps({"environment": environment(), **payload}, indent=1, ensure_ascii=False, default=float))
    return p


def load(name: str) -> dict | None:
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pkg(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment() -> dict:
    from src import config
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()) / 2**30
    except Exception:
        cpu, mem = platform.processor(), 0.0
    asr = config.asr_settings()
    return {"machine": cpu, "memory_gb": round(mem), "os": platform.platform(), "python": platform.python_version(),
            "packages": {p: _pkg(p) for p in ("faster-whisper", "torch", "kokoro", "silero-vad", "onnxruntime", "sentence-transformers", "chromadb",
                                                "streamlit", "streamlit-webrtc", "jiwer")},
            "asr": {k: asr[k] for k in ("model", "device", "compute_type", "language")}, "tts_engine": config.tts_engine(),
            "tts_voice": config.kokoro_settings()["voice"], "llm_model": config.generation_model(),
            "vad": config.live_settings()["vad"], "hangover_ms": config.live_settings()["hangover_ms"],
            "browser": "not used: the harness drives the Python pipeline directly; browser microphone/WebRTC path is NOT measured"}


def pct(values: list[float]) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def norm(text: str) -> str:
    """For scoring: lower-case, accents folded, punctuation and hyphens -> spaces."""
    t = fold(text.lower().replace("’", "'").replace("'", ""))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def compact(text: str) -> str:
    return norm(text).replace(" ", "")


def entity_found(entity: str, heard: str) -> bool:
    """Is the name in the transcript? Space/hyphen/accent/case-insensitive, so 'La Fontaine' = 'Lafontaine' = 'la-fontaine'."""
    return compact(entity) in compact(heard)


# --- synthetic speech --------------------------------------------------------------------------------------------------------------------

def say_samples(voice_key: str, text: str) -> np.ndarray:
    """16 kHz mono float32 speech from a macOS `say` voice (offline, no download)."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "s.wav"
        subprocess.run(["say", "-v", SAY_VOICES[voice_key][0], "-o", str(f), "--data-format=LEI16@16000", text], check=True, capture_output=True)
        from src.voice import audio
        x, _ = audio.read_wav(f.read_bytes())
    return x


def kokoro_samples(text: str) -> np.ndarray:
    from src.voice import audio, tts
    x, _ = audio.read_wav(tts.get_synthesizer().synthesize(text).audio_bytes)
    return x


def speech_samples(voice_key: str, text: str) -> np.ndarray:
    return kokoro_samples(text) if voice_key == "kokoro" else say_samples(voice_key, text)


def trim(x: np.ndarray, floor: float = 0.01) -> tuple[np.ndarray, int]:
    """Drop leading/trailing near-silence. Returns (speech, number of samples kept from the start)."""
    env = np.abs(x)
    idx = np.where(env > floor)[0]
    return (x[idx[0]: idx[-1] + 1], idx[0]) if len(idx) else (x, 0)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.maximum(np.arange(len(spec)), 1)
    out = np.fft.irfft(spec / np.sqrt(f), n)
    return (out / (np.std(out) + 1e-9)).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mix pink noise at a given signal-to-noise ratio (speech power measured over the speech only)."""
    speech_rms = float(np.sqrt(np.mean(trim(x)[0] ** 2))) + 1e-9
    noise = pink_noise(len(x), rng) * (speech_rms / (10 ** (snr_db / 20)))
    return np.clip(x + noise, -1.0, 1.0).astype(np.float32)


def stream(speech: np.ndarray, lead_s: float = 0.6, tail_s: float = 2.5, noise_snr_db: float | None = None,
           rng: np.random.Generator | None = None) -> tuple[np.ndarray, int]:
    """Silence + speech + silence, as a microphone would deliver it. Returns (signal, index of the last speech sample)."""
    speech, _ = trim(speech)
    lead, tail = np.zeros(int(lead_s * SR), np.float32), np.zeros(int(tail_s * SR), np.float32)
    sig = np.concatenate([lead, speech, tail])
    end = len(lead) + len(speech)
    if noise_snr_db is not None:
        sig = add_noise(sig, noise_snr_db, rng or np.random.default_rng(0))
    return sig, end


def capture(engine, signal: np.ndarray, frame: int = 512):
    """Feed `signal` to a LiveEngine frame by frame like the audio thread does. Returns (utterance or None, sample index when it was emitted, wall seconds)."""
    t0 = time.perf_counter()
    for i in range(0, len(signal) - frame + 1, frame):
        engine.feed(signal[i: i + frame])
        utt = engine.pop_utterance()
        if utt is not None:
            return utt, i + frame, time.perf_counter() - t0
    return None, len(signal), time.perf_counter() - t0
