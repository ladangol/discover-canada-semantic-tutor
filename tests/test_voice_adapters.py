import io
import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from src.voice import asr, tts
from src.voice.models import VoiceUnavailable


# --- ASR ------------------------------------------------------------------------------------------------------

class LazyModel:
    """Like faster-whisper: transcribe() returns a LAZY generator; text only exists once it is consumed."""
    def __init__(self):
        self.consumed = False
        self.kwargs = None

    def transcribe(self, samples, **kwargs):
        self.kwargs = kwargs

        def gen():
            self.consumed = True
            yield SimpleNamespace(text=" Louis-Hippolyte ")
            yield SimpleNamespace(text="La Fontaine. ")
        return gen(), SimpleNamespace(language="en")


def make_transcriber(model):
    t = object.__new__(asr.FasterWhisperTranscriber)           # skip loading a real model
    t._model, t.language, t.initial_prompt = model, "en", "hint"
    return t


def test_transcribe_consumes_lazy_segments_and_reports_measured_time():
    m = LazyModel()
    res = make_transcriber(m).transcribe(np.ones(16000, np.float32), 1.0)
    assert m.consumed and res.text == "Louis-Hippolyte La Fontaine."
    assert res.language == "en" and res.duration_seconds == 1.0 and res.inference_seconds >= 0
    assert m.kwargs["language"] == "en" and m.kwargs["initial_prompt"] == "hint"


def test_busy_host_gives_friendly_error_not_a_hang(monkeypatch):
    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr(asr, "_slots", sem)
    monkeypatch.setattr(asr, "SLOT_WAIT_SECONDS", 0.05)
    with pytest.raises(VoiceUnavailable, match="busy"):
        make_transcriber(LazyModel()).transcribe(np.ones(16000, np.float32), 1.0)


def test_slot_is_released_after_a_failure(monkeypatch):
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(asr, "_slots", sem)

    class Boom:
        def transcribe(self, *a, **k):
            raise RuntimeError("model crashed")
    with pytest.raises(RuntimeError):
        make_transcriber(Boom()).transcribe(np.ones(16000, np.float32), 1.0)
    assert sem.acquire(timeout=0.1)                              # not leaked


def test_missing_packages_are_friendly(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_voice_libs(name, *a, **k):
        if name in ("faster_whisper", "piper"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_voice_libs)
    with pytest.raises(VoiceUnavailable, match="isn't installed"):
        asr.FasterWhisperTranscriber("small", "cpu", "int8", "en", None)
    with pytest.raises(VoiceUnavailable, match="aren't installed"):
        tts.PiperSynthesizer(None)


def test_missing_model_is_a_friendly_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", SimpleNamespace(WhisperModel=lambda *a, **k: 1 / 0))
    with pytest.raises(VoiceUnavailable, match="could not be loaded"):
        asr.FasterWhisperTranscriber("small", "cpu", "int8", "en", None)


# --- TTS ------------------------------------------------------------------------------------------------------

def test_prepare_speech_text_removes_citations_and_markup():
    src = "**Confederation** took place in 1867 [Discover Canada, p. 18, Confederation].\n\n- Four provinces [S2][S3]\n- A new country"
    out = tts.prepare_speech_text(src)
    assert out == "Confederation took place in 1867. Four provinces. A new country."
    assert "[" not in out and "*" not in out and "p. 18" not in out


@pytest.mark.parametrize("raw,spoken", [
    ("The answer is Confederation [p.18].", "The answer is Confederation."),
    ("See [the guide](https://example.com/x) now", "See the guide now."),
    ("# Heading\n> quote with `code`", "Heading. quote with code."),
    ("", ""),
    ("[S1]", ""),
])
def test_prepare_speech_text_cases(raw, spoken):
    assert tts.prepare_speech_text(raw) == spoken


def test_prepare_speech_text_keeps_meaning_of_plain_text():
    s = "Quebec joined in 1867. Canada has ten provinces."
    assert tts.prepare_speech_text(s) == s


def test_long_text_is_trimmed_at_a_sentence():
    out = tts.prepare_speech_text("A sentence here. " * 200)
    assert len(out) <= tts.MAX_SPOKEN_CHARS and out.endswith(".")


class FakeVoice:
    """Like Piper 1.8: synthesize() yields one chunk per sentence."""
    def __init__(self):
        self.said, self.cfg = [], None

    def synthesize(self, text, syn_config=None):
        self.said.append(text)
        self.cfg = syn_config
        for _ in text.split(". "):
            yield SimpleNamespace(sample_rate=22050, sample_width=2, sample_channels=1,
                                  audio_int16_bytes=np.ones(2205, "<i2").tobytes())


def piper_with(voice, pause_ms=280):
    s = object.__new__(tts.PiperSynthesizer)
    s._voice, s._cfg, s._pause_ms = voice, "cfg", pause_ms
    return s


def test_piper_returns_a_playable_wav_of_the_cleaned_text():
    s = piper_with(FakeVoice(), pause_ms=0)
    out = s.synthesize("Correct! It was 1867 [Discover Canada, p. 18, Confederation].")
    assert s._voice.said == ["Correct! It was 1867."] and s._voice.cfg == "cfg"
    assert out.mime_type == "audio/wav" and out.synthesis_seconds >= 0
    with wave.open(io.BytesIO(out.audio_bytes)) as w:
        assert w.getframerate() == 22050 and w.getnframes() == 2205


def test_sentences_are_separated_by_a_short_pause():
    s = piper_with(FakeVoice(), pause_ms=200)
    out = s.synthesize("First sentence. Second sentence. Third sentence.")
    with wave.open(io.BytesIO(out.audio_bytes)) as w:
        assert w.getnframes() == 3 * 2205 + 2 * int(22050 * 0.2)       # three sentences, two pauses (none at the ends)


def test_nothing_to_speak_is_an_error_not_silence():
    s = piper_with(FakeVoice())
    with pytest.raises(VoiceUnavailable):
        s.synthesize("[S1]")


def test_missing_voice_file_is_friendly(tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "piper", SimpleNamespace(PiperVoice=object))   # CI has no piper-tts
    with pytest.raises(VoiceUnavailable, match="No voice model"):
        tts.PiperSynthesizer(tmp_path / "nope.onnx")
    with pytest.raises(VoiceUnavailable):
        tts.PiperSynthesizer(None)


# --- spoken forms (roman numerals after royal names etc.) --------------------------------------------------------------

@pytest.mark.parametrize("written,spoken", [
    ("Queen Elizabeth II", "Queen Elizabeth the Second"),                      # the cases that occur in the guide
    ("Queen Elizabeth I", "Queen Elizabeth the First"),
    ("King George III", "King George the Third"),
    ("King George V", "King George the Fifth"),
    ("King Charles II", "King Charles the Second"),
    ("King Francis I", "King Francis the First"),
    ("Elizabeth II's reign", "Elizabeth the Second's reign"),                 # possessive, no title
    ("Pope Pius IX", "Pope Pius the Ninth"),
    ("the Second World War", "the Second World War"),                           # already words: untouched
    ("World War II", "World War Two"), ("World War I.", "World War I."),       # "I." could be a sentence end: left alone
    ("St. Lawrence River", "Saint Lawrence River"), ("Dr. Banting", "Doctor Banting"),
    ("1837–38", "1837 to 1838"), ("1899–1902", "1899 to 1902"), ("1935-40", "1935 to 1940"),
])
def test_spoken_forms(written, spoken):
    assert tts.spoken_forms(written) == spoken


@pytest.mark.parametrize("text", [
    "I think Section I and Part II apply.", "Sir John A. Macdonald", "Victoria Cross (V.C.)", "Main St. is busy.",
    "In 1867 Canada had four provinces.", "John II was a king.".replace("John II", "Ontario"), "Chapter V", "Vitamin X",
])
def test_spoken_forms_leave_everything_else_alone(text):
    assert tts.spoken_forms(text) == text


def test_voice_says_queen_elizabeth_the_second_but_the_screen_text_is_unchanged():
    written = "Queen Elizabeth II signed the proclamation in 1982 [Discover Canada, p. 8, Rights]."
    assert tts.prepare_speech_text(written) == "Queen Elizabeth the Second signed the proclamation in 1982."
    s = piper_with(FakeVoice(), pause_ms=0)
    s.synthesize(written)
    assert s._voice.said == ["Queen Elizabeth the Second signed the proclamation in 1982."]
    assert "Elizabeth II" in written                                              # the displayed answer still reads "II"


# --- Kokoro engine -----------------------------------------------------------------------------------------------------

class FakeKokoro:
    """Like KPipeline: yields (graphemes, phonemes, audio) per sentence for the split pattern."""
    def __init__(self):
        self.calls = []

    def __call__(self, text, voice, speed, split_pattern):
        self.calls.append((text, voice, speed, split_pattern))
        yield "g", "p", np.full(2400, 0.25, np.float32)                 # one chunk (0.1 s at 24 kHz) per call


def kokoro_with(pipe, pause_ms=150):
    k = object.__new__(tts.KokoroSynthesizer)
    k._pipe, k._voice, k._speed, k._pause_ms = pipe, "af_heart", 1.0, pause_ms
    return k


def test_kokoro_returns_a_24khz_wav_with_pauses_between_sentences():
    pipe = FakeKokoro()
    out = kokoro_with(pipe, pause_ms=200).synthesize("First one. Second one. Third one.")
    with wave.open(io.BytesIO(out.audio_bytes)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getnframes() == 3 * 2400 + 2 * 4800                    # three sentences, two 200 ms pauses
    assert out.mime_type == "audio/wav" and out.synthesis_seconds >= 0
    assert pipe.calls[0][1:3] == ("af_heart", 1.0) and [c[0] for c in pipe.calls] == ["First one.", "Second one.", "Third one."]


def test_kokoro_gets_clean_spoken_text_with_names_respelled_only_for_kokoro():
    pipe = FakeKokoro()
    kokoro_with(pipe).synthesize("**Samuel de Champlain** [Discover Canada, p. 12, Explorers] reached Iqaluit. Queen Elizabeth II visited.")
    said = " ".join(c[0] for c in pipe.calls)
    assert "[Champlain](/ʃæmplˈA/)" in said and "[Iqaluit](/ikˈɑluɪt/)" in said
    assert "Queen Elizabeth the Second" in said and "Discover Canada, p." not in said and "**" not in said
    assert "[" not in tts.prepare_speech_text("Champlain and Iqaluit")      # the engine-neutral text has no phoneme markup (Piper path)


def test_pronunciation_overrides_do_not_double_wrap_or_touch_other_words():
    once = tts.apply_pronunciations("Cartier met Laurier. Cartiers are jewellers.")
    assert once.count("[Cartier](/") == 1 and "Cartiers" in once                # whole words only
    assert tts.apply_pronunciations(once) == once                              # idempotent


def test_kokoro_silence_is_an_error_and_the_slot_is_released(monkeypatch):
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(asr, "_slots", sem)

    class Empty:
        def __call__(self, *a, **k):
            return iter(())
    with pytest.raises(VoiceUnavailable):
        kokoro_with(Empty()).synthesize("Hello there.")
    assert sem.acquire(timeout=0.1)


def test_kokoro_not_installed_is_friendly(monkeypatch):
    monkeypatch.setattr(tts.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(VoiceUnavailable, match="isn't installed"):
        tts.KokoroSynthesizer("af_heart", 1.0)


def test_engine_selection_and_readiness(monkeypatch):
    from src import config
    monkeypatch.delenv("TTS_ENGINE", raising=False)
    assert config.tts_engine() == "kokoro"
    monkeypatch.setenv("TTS_ENGINE", "PIPER")
    assert config.tts_engine() == "piper"
    monkeypatch.delenv("TTS_VOICE_PATH", raising=False)
    assert tts.engine_ready() is False                                         # Piper needs a voice file
    monkeypatch.setenv("TTS_ENGINE", "kokoro")
    monkeypatch.setattr(tts.importlib.util, "find_spec", lambda name: object() if name == "kokoro" else None)
    assert tts.engine_ready() is True


def test_kokoro_settings_are_clamped(monkeypatch):
    from src import config
    monkeypatch.setenv("KOKORO_SPEED", "9")
    monkeypatch.setenv("KOKORO_VOICE", "bf_emma")
    assert config.kokoro_settings() == {"voice": "bf_emma", "speed": 1.4}
    monkeypatch.setenv("KOKORO_SPEED", "fast")
    assert config.kokoro_settings()["speed"] == 1.0


# --- Kokoro via ONNX Runtime ---------------------------------------------------------------------------------------------

class FakeOnnx:
    def __init__(self):
        self.calls = []

    def create(self, phonemes, voice, speed, is_phonemes):
        self.calls.append((phonemes, voice, speed, is_phonemes))
        return np.full(2400, 0.25, np.float32), 24000                      # 0.1 s


def onnx_with(ko, g2p=None, pause_ms=100):
    k = object.__new__(tts.KokoroOnnxSynthesizer)
    k._ko, k._voice, k._speed, k._pause_ms = ko, "af_heart", 1.0, pause_ms
    k._g2p = g2p or (lambda s: (f"PH<{s}>", None))
    return k


def test_kokoro_onnx_speaks_sentence_by_sentence_from_phonemes_with_name_fixes():
    ko, seen = FakeOnnx(), []

    def g2p(s):
        seen.append(s)
        return f"PH<{s}>", None
    out = onnx_with(ko, g2p, pause_ms=100).synthesize("Champlain arrived. Queen Elizabeth II visited! Done?")
    assert len(ko.calls) == 3 and all(c[3] is True and c[1] == "af_heart" for c in ko.calls)       # phonemes in, not raw text
    assert "[Champlain](/ʃæmplˈA/)" in seen[0] and "Elizabeth the Second" in seen[1]
    with wave.open(io.BytesIO(out.audio_bytes)) as w:
        assert w.getframerate() == 24000 and w.getnframes() == 3 * 2400 + 2 * 2400            # three sentences, two 100 ms pauses


def test_kokoro_onnx_empty_and_missing_files_are_friendly(tmp_path, monkeypatch):
    with pytest.raises(VoiceUnavailable):
        onnx_with(FakeOnnx()).synthesize("[S1]")
    monkeypatch.setattr(tts.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(VoiceUnavailable, match="model files are missing"):
        tts.KokoroOnnxSynthesizer(tmp_path / "no.onnx", tmp_path / "no.bin", "af_heart", 1.0)
    monkeypatch.setattr(tts.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(VoiceUnavailable, match="isn't installed"):
        tts.KokoroOnnxSynthesizer(tmp_path / "no.onnx", tmp_path / "no.bin", "af_heart", 1.0)


def test_engine_names_and_onnx_readiness(monkeypatch, tmp_path):
    from src import config
    monkeypatch.setenv("TTS_ENGINE", "Kokoro-ONNX")
    assert config.tts_engine() == "kokoro-onnx"
    monkeypatch.setenv("TTS_ENGINE", "nonsense")
    assert config.tts_engine() == "kokoro"
    monkeypatch.setenv("TTS_ENGINE", "kokoro-onnx")
    m, v = tmp_path / "m.onnx", tmp_path / "v.bin"
    monkeypatch.setenv("KOKORO_ONNX_MODEL", str(m)), monkeypatch.setenv("KOKORO_ONNX_VOICES", str(v))
    monkeypatch.setattr(tts.importlib.util, "find_spec", lambda name: object())
    assert tts.engine_ready() is False                                          # files not there yet
    m.write_bytes(b"x"), v.write_bytes(b"x")
    assert tts.engine_ready() is True


@pytest.mark.parametrize("text,pieces", [
    ("P.E.I. is the smallest province. It is in the east.", ["P.E.I. is the smallest province.", "It is in the east."]),
    ("She lives in B.C. and he lives in P.E.I. Both are provinces.", ["She lives in B.C. and he lives in P.E.I.", "Both are provinces."]),
    ("It began in 1873. Then it grew!", ["It began in 1873.", "Then it grew!"]),
    ("Who was he? A king.", ["Who was he?", "A king."]),
    ("   ", []),
])
def test_split_sentences_keeps_dotted_abbreviations_with_their_sentence(text, pieces):
    assert tts.split_sentences(text) == pieces


def test_michif_is_pronounced_and_known_to_the_speech_recognizer():
    from src import config
    assert tts.apply_pronunciations("The Métis speak Michif.") == "The [Métis](/mAtˈi/) speak [Michif](/miʃˈif/)."
    assert "Michif" in config.ASR_INITIAL_PROMPT_DEFAULT and "Inuktitut" in config.ASR_INITIAL_PROMPT_DEFAULT
