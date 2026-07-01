from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from ..config import InferenceConfig
from ..devices import resolve_device
from ..runtime import resolve_runtime_provider
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
                tasks=("resampling",),
                architecture="polyphase-sinc-resampling",
                target_sample_rates=None,
                input_sample_rates=None,
                code_license="MIT",
                weights_license=None,
                weights_source=None,
                upstream_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html",
                recommended_for=("baseline", "offline", "deterministic", "no-weights"),
                known_limitations=("not-generative", "does-not-reconstruct-missing-bandwidth"),
                validation=("default-unit-tests", "cli-smoke"),
                maturity="stable",
                capability=DEFAULT_ARRAY_BACKEND_CAPABILITY,
            ),
        )

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        resolve_runtime_provider(self.config.runtime_provider, DEFAULT_ARRAY_BACKEND_CAPABILITY.runtime_providers)
        resolve_device(self.config.device, supported_devices=DEFAULT_ARRAY_BACKEND_CAPABILITY.accelerators)

        if sample_rate == target_sample_rate:
            return np.asarray(audio)

        gcd = np.gcd(sample_rate, target_sample_rate)
        up = target_sample_rate // gcd
        down = sample_rate // gcd
        return resample_poly(audio, up, down, axis=0)
