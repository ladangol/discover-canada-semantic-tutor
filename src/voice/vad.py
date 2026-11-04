"""Voice-activity detection: how likely is this 32 ms frame of 16 kHz audio to be speech? (Milestone 2, local, no network.)

`SileroVad` (Silero VAD v5 through ONNX Runtime, ~2 MB, ships inside the `silero-vad` package) is the default. `EnergyVad` is a
dependency-free fallback that tracks the room's noise floor; it is cruder (it cannot tell speech from other loud sounds)."""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np

try:                                       # remember PyTorch's thread count BEFORE silero_vad is imported (see SileroVad)
    import torch as _torch
    _TORCH_THREADS: int | None = _torch.get_num_threads()
except ImportError:                        # the energy detector needs no torch
    _torch, _TORCH_THREADS = None, None

SAMPLE_RATE = 16_000
FRAME = 512                      # samples per VAD frame at 16 kHz = 32 ms (what Silero v5 expects)
FRAME_MS = FRAME * 1000 / SAMPLE_RATE


class Vad(Protocol):
    name: str

    def probability(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class SileroVad:
    name = "silero"

    def __init__(self):
        import torch
        from silero_vad import load_silero_vad
        self._torch = torch
        self._model = load_silero_vad(onnx=True)
        if _TORCH_THREADS:
            # Importing silero_vad calls torch.set_num_threads(1) for the WHOLE process, which made Kokoro and the search model ~5x slower.
            # The detector runs on ONNX Runtime and does not need that, so give PyTorch its threads back.
            torch.set_num_threads(_TORCH_THREADS)

    def probability(self, frame: np.ndarray) -> float:
        return float(self._model(self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)), SAMPLE_RATE).reshape(-1)[0])

    def reset(self) -> None:
        self._model.reset_states()


class EnergyVad:
    """Loudness against an adaptive noise floor. Probability rises from 0 at 2x the floor to 1 at 8x."""
    name = "energy"

    def __init__(self, floor: float = 0.003):
        self.floor = floor

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else 0.0
        ratio = rms / max(self.floor, 1e-6)
        if ratio < 2.0:                                           # quiet frames teach the noise floor (slowly up, quickly down)
            self.floor = 0.98 * self.floor + 0.02 * max(rms, 1e-4) if rms > self.floor else 0.8 * self.floor + 0.2 * max(rms, 1e-4)
        return min(1.0, max(0.0, (math.log2(max(ratio, 1e-6)) - 1.0) / 2.0))

    def reset(self) -> None:
        pass                                                      # the noise floor is a property of the room: keep it


def get_vad(engine: str = "silero") -> Vad:
    """The requested engine, falling back to the energy detector if Silero cannot load (the caller can read `.name`)."""
    if engine == "silero":
        try:
            return SileroVad()
        except Exception:
            pass
    return EnergyVad()
