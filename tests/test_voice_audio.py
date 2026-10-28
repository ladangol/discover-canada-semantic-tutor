import numpy as np
import pytest

from src.voice import audio
from src.voice.models import AudioError
from tests.voice_helpers import make_wav


def test_valid_recording_is_mono_16k_float32():
    x, info = audio.read_wav(make_wav(1.0, rate=16000))
    assert x.dtype == np.float32 and x.ndim == 1 and len(x) == 16000
    assert info.sample_rate == 16000 and info.channels == 1 and info.duration_seconds == pytest.approx(1.0)
    assert np.max(np.abs(x)) <= 1.0


def test_resamples_and_downmixes():
    x, info = audio.read_wav(make_wav(1.0, rate=48000, channels=2))
    assert len(x) == 16000 and info.sample_rate == 48000 and info.channels == 2


@pytest.mark.parametrize("width", [1, 2, 4])
def test_sample_widths(width):
    x, _ = audio.read_wav(make_wav(0.5, width=width))
    assert len(x) == 8000 and np.max(np.abs(x)) > 0.1


def test_empty_bytes_rejected():
    with pytest.raises(AudioError, match="empty"):
        audio.read_wav(b"")


def test_corrupted_audio_rejected_without_leaking_details():
    with pytest.raises(AudioError, match="couldn't read") as e:
        audio.read_wav(b"this is not a wav file at all" * 10)
    assert "RIFF" not in str(e.value)


def test_truncated_wav_rejected():
    with pytest.raises(AudioError):
        audio.read_wav(make_wav(1.0)[:30])


def test_silence_rejected():
    with pytest.raises(AudioError, match="didn't hear"):
        audio.read_wav(make_wav(1.0, amp=0))


def test_too_short_rejected():
    with pytest.raises(AudioError, match="didn't hear"):
        audio.read_wav(make_wav(0.1))


def test_long_recording_rejected_by_duration():
    with pytest.raises(AudioError, match="longer than 5 seconds"):
        audio.read_wav(make_wav(6.0), max_seconds=5)


def test_oversized_upload_rejected_by_bytes():
    with pytest.raises(AudioError, match="too large"):
        audio.read_wav(make_wav(2.0), max_bytes=1000)
