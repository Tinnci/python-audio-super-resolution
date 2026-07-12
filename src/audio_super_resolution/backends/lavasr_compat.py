from __future__ import annotations

import importlib.util

import numpy as np
from scipy.signal import resample_poly

from ..config import InferenceConfig
from ..devices import resolve_device
from ..runtime import resolve_runtime_provider
from ..specs import BackendCapability, ModelSpec, WeightFileSpec
from .lavasr_validation import LAVASR_WEIGHTS_PATH, LavaSRWeightBundleInfo

LAVASR_SAMPLE_RATE = 48000
LAVASR_MODEL_ID = "lavasr-v2-bwe"
LAVASR_REVISION = "b98dc8be472da45ab7b6346ad7997e1dfeb5911d"
LAVASR_CAPABILITY = BackendCapability(
    supports_array_io=True,
    supports_file_io=False,
    supports_chunking=True,
    deterministic=True,
    supports_cpu=True,
    supports_cuda=True,
    supports_mps=True,
    precision_modes=("float32", "auto"),
    accelerators=("cpu", "cuda", "mps"),
    runtime_providers=("torch-eager",),
)


class LavaSRCompatBackend:
    """Self-contained LavaSR-compatible backend placeholder for verified BWE weights."""

    name = "lavasr-compat"
    description = "Self-contained LavaSR-compatible speech bandwidth extension backend."
    optional_dependency = "torch"
    package_extra = "lavasr"

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._model: object | None = None
        self._model_cache_key: tuple[str, str, str] | None = None

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("torch") is not None

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
                tasks=("speech-bandwidth-extension", "audio-super-resolution"),
                architecture="vocos-istft-bwe",
                input_sample_rate_range=(8000, LAVASR_SAMPLE_RATE),
                target_sample_rates=(LAVASR_SAMPLE_RATE,),
                code_license="MIT/Apache-2.0",
                weights_license="Apache-2.0",
                weights_source="YatharthS/LavaSR",
                weight_provider="huggingface",
                default_weight_revision=LAVASR_REVISION,
                requires_weights=True,
                upstream_url="https://github.com/ysharma3501/LavaSR",
                recommended_for=("speech", "fast-bwe", "self-contained-runtime"),
                known_limitations=(
                    "experimental",
                    "denoise-reserved-not-supported",
                    "requires-local-verified-weights",
                ),
                validation=(
                    "default-unit-tests",
                    "real-weight-download",
                    "torch-smoke",
                    "upstream-parity",
                ),
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
                capability=LAVASR_CAPABILITY,
            ),
        )

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        if target_sample_rate != LAVASR_SAMPLE_RATE:
            raise ValueError("The lavasr-compat backend outputs 48000 Hz audio; set target_sr=48000.")
        if self.config.denoise:
            raise ValueError("denoise is reserved but not supported by lavasr-compat yet")
        if self.config.precision not in {"float32", "auto"}:
            raise ValueError("lavasr-compat supports precision modes: float32, auto")

        from ..weight_store import resolve_weights_for_spec
        from .lavasr_validation import validate_lavasr_v2_weight_bundle

        resolved_weights = resolve_weights_for_spec(
            self.model_specs()[0],
            self.config,
            allow_download=self.config.download_weights,
        )
        bundle_info = validate_lavasr_v2_weight_bundle(resolved_weights)
        resolve_runtime_provider(self.config.runtime_provider, LAVASR_CAPABILITY.runtime_providers)
        device = resolve_device(self.config.device, supported_devices=LAVASR_CAPABILITY.accelerators)
        _require_torch_runtime()

        prepared_audio, was_mono = _prepare_lavasr_input(audio, sample_rate)
        if not np.any(prepared_audio):
            return _restore_lavasr_output(np.zeros_like(prepared_audio).T, was_mono=was_mono)

        import torch

        model = self._load_model(resolved_weights, bundle_info, device=device)
        waveform = torch.from_numpy(prepared_audio.T.copy()).to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            enhanced = _run_lavasr_model(
                model,
                waveform,
                cutoff_hz=_merge_cutoff_hz(sample_rate),
            )

        return _restore_lavasr_output(enhanced.cpu().numpy(), was_mono=was_mono)

    def _load_model(self, resolved_weights, bundle_info: LavaSRWeightBundleInfo, *, device: str):
        weights_path = resolved_weights.path_for(LAVASR_WEIGHTS_PATH)
        cache_key = (
            str(resolved_weights.manifest_path.resolve(strict=False)),
            str(weights_path.resolve(strict=False)),
            device,
        )
        if self._model is not None and self._model_cache_key == cache_key:
            return self._model

        from .lavasr_torch import build_lavasr_v2_model, load_lavasr_v2_state_dict

        model = build_lavasr_v2_model(bundle_info.config)
        load_lavasr_v2_state_dict(model, weights_path)
        model.eval().to(device)
        self._model = model
        self._model_cache_key = cache_key
        return model


def _require_torch_runtime() -> None:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("lavasr-compat inference requires `uv pip install audio-super-resolution[lavasr]`.")


def _prepare_lavasr_input(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, bool]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    audio_array = np.asarray(audio, dtype=np.float32)
    was_mono = audio_array.ndim == 1
    if was_mono:
        audio_array = audio_array[:, None]
    if audio_array.ndim != 2:
        raise ValueError("lavasr-compat expects mono or channel-last 2D audio arrays")
    if sample_rate == LAVASR_SAMPLE_RATE:
        return audio_array, was_mono

    divisor = np.gcd(sample_rate, LAVASR_SAMPLE_RATE)
    up = LAVASR_SAMPLE_RATE // divisor
    down = sample_rate // divisor
    return resample_poly(audio_array, up, down, axis=0).astype(np.float32, copy=False), was_mono


def _run_lavasr_model(model, waveform, *, cutoff_hz: float):
    from .lavasr_torch import FastLRMerge

    predicted = model(waveform)
    predicted = predicted[:, : waveform.shape[1]].float()
    source = waveform[:, : predicted.shape[1]].float()
    return FastLRMerge(cutoff=cutoff_hz, transition_bins=1024)(predicted, source)


def _merge_cutoff_hz(input_sample_rate: int) -> float:
    return min(float(input_sample_rate) / 2, float(LAVASR_SAMPLE_RATE) / 2)


def _restore_lavasr_output(enhanced_channels_first: np.ndarray, *, was_mono: bool) -> np.ndarray:
    enhanced = enhanced_channels_first.T
    if was_mono:
        return enhanced[:, 0]
    return enhanced
