from __future__ import annotations

import importlib.util

import numpy as np

from ..config import InferenceConfig
from ..specs import BackendCapability, ModelSpec, WeightFileSpec

LAVASR_SAMPLE_RATE = 48000
LAVASR_MODEL_ID = "lavasr-v2-bwe"
LAVASR_REVISION = "b98dc8be472da45ab7b6346ad7997e1dfeb5911d"


class LavaSRCompatBackend:
    """Self-contained LavaSR-compatible backend placeholder for verified BWE weights."""

    name = "lavasr-compat"
    description = "Self-contained LavaSR-compatible speech bandwidth extension backend."
    optional_dependency = "torch"
    package_extra = "lavasr"

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("yaml") is not None

    @classmethod
    def model_specs(cls) -> tuple[ModelSpec, ...]:
        return (
            ModelSpec(
                id=LAVASR_MODEL_ID,
                backend=cls.name,
                model_name=LAVASR_MODEL_ID,
                name="LavaSR v2 BWE",
                description="LavaSR v2 bandwidth extension weights for 48 kHz output.",
                implementation="self_torch",
                domain=("speech",),
                architecture="vocos-istft-bwe",
                target_sample_rates=(LAVASR_SAMPLE_RATE,),
                weights_license="Apache-2.0",
                weights_source="YatharthS/LavaSR",
                weight_provider="huggingface",
                default_weight_revision=LAVASR_REVISION,
                requires_weights=True,
                maturity="experimental",
                weight_files=(
                    WeightFileSpec(
                        path="enhancer_v2/config.yaml",
                        sha256="0d970f5cdc1913730c417b49e476bb09bb8b874583d113f71a7f10c8bb3a4b7d",
                        size=526,
                    ),
                    WeightFileSpec(
                        path="enhancer_v2/pytorch_model.bin",
                        sha256="d100db961b2c125d77a52a12215c689e44cd9926a72f117513395b7e25e6de12",
                        size=56316591,
                    ),
                ),
                capability=BackendCapability(
                    supports_array_io=True,
                    supports_file_io=False,
                    supports_chunking=True,
                    deterministic=True,
                    supports_cpu=True,
                    supports_cuda=True,
                    supports_mps=True,
                    precision_modes=("float32", "auto"),
                ),
            ),
        )

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        if target_sample_rate != LAVASR_SAMPLE_RATE:
            raise ValueError("The lavasr-compat backend outputs 48000 Hz audio; set target_sr=48000.")
        if self.config.denoise:
            raise ValueError("denoise is reserved but not supported by lavasr-compat yet")

        from ..weight_store import resolve_weights_for_spec
        from .lavasr_validation import validate_lavasr_v2_weight_bundle

        resolved_weights = resolve_weights_for_spec(
            self.model_specs()[0],
            self.config,
            allow_download=self.config.download_weights,
        )
        validate_lavasr_v2_weight_bundle(resolved_weights)
        raise RuntimeError("lavasr-compat weight management is available, but inference is not implemented yet.")
