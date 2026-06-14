from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from ..config import InferenceConfig
from ..specs import ModelSpec
from .base import DEFAULT_ARRAY_BACKEND_CAPABILITY


class SincResampleBackend:
    """Deterministic baseline backend using polyphase sinc resampling."""

    name = "sinc-resample"
    description = "Deterministic polyphase sinc resampling baseline."
    optional_dependency = None
    package_extra = None

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()

    @classmethod
    def model_specs(cls) -> tuple[ModelSpec, ...]:
        return (
            ModelSpec(
                id="sinc-resample",
                backend=cls.name,
                name="Sinc Resample Baseline",
                description=cls.description,
                implementation="baseline",
                domain=("general",),
                architecture="polyphase-sinc-resampling",
                target_sample_rates=None,
                weights_license=None,
                weights_source=None,
                maturity="stable",
                capability=DEFAULT_ARRAY_BACKEND_CAPABILITY,
            ),
        )

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        if sample_rate == target_sample_rate:
            return np.asarray(audio)

        gcd = np.gcd(sample_rate, target_sample_rate)
        up = target_sample_rate // gcd
        down = sample_rate // gcd
        return resample_poly(audio, up, down, axis=0)
