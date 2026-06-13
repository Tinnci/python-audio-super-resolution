from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class EnhancementResult:
    """Metadata returned after writing an enhanced audio file."""

    input_path: Path
    output_path: Path
    sample_rate: int
    duration_seconds: float
    backend: str


class EnhancementBackend(Protocol):
    """Interface implemented by audio super-resolution backends."""

    name: str

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        """Return audio enhanced to target_sample_rate."""


class SincResampleBackend:
    """Deterministic baseline backend using polyphase sinc resampling."""

    name = "sinc-resample"

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        if sample_rate == target_sample_rate:
            return np.asarray(audio)

        gcd = np.gcd(sample_rate, target_sample_rate)
        up = target_sample_rate // gcd
        down = sample_rate // gcd
        return resample_poly(audio, up, down, axis=0)


class AudioSuperResolver:
    """High-level API for audio super-resolution workflows."""

    def __init__(self, target_sr: int = 48000, backend: EnhancementBackend | None = None) -> None:
        if target_sr <= 0:
            raise ValueError("target_sr must be greater than zero")

        self.target_sr = target_sr
        self.backend = backend or SincResampleBackend()

    def enhance(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_sr: int | None = None,
    ) -> EnhancementResult:
        input_path = Path(input_path)
        output_path = Path(output_path)
        requested_sr = target_sr or self.target_sr

        if requested_sr <= 0:
            raise ValueError("target_sr must be greater than zero")

        audio, sample_rate = sf.read(input_path, always_2d=True)
        enhanced = self.backend.enhance(audio, sample_rate, requested_sr)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, enhanced, requested_sr)

        frames = enhanced.shape[0]
        return EnhancementResult(
            input_path=input_path,
            output_path=output_path,
            sample_rate=requested_sr,
            duration_seconds=frames / requested_sr,
            backend=self.backend.name,
        )
