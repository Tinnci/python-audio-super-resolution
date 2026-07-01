from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .devices import VALID_DEVICES
from .runtime import VALID_RUNTIME_PROVIDERS

VALID_PRECISIONS = ("float32", "float16", "bfloat16", "auto")
VALID_PREPROCESSING_MODES = ("none", "lowpass")


def default_model_cache_dir() -> Path:
    """Return the default directory used for model weights."""

    configured = os.environ.get("AUDIO_SUPER_RESOLUTION_CACHE")
    if configured:
        return Path(configured).expanduser()

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "audio-super-resolution" / "models"

    return Path.home() / ".cache" / "audio-super-resolution" / "models"


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime options shared by enhancement backends."""

    device: str = "cpu"
    runtime_provider: str = "auto"
    precision: str = "float32"
    chunked: bool = False
    chunk_seconds: float = 30.0
    overlap_seconds: float = 1.0
    seed: int = 0
    model_cache_dir: Path = field(default_factory=default_model_cache_dir)
    model_name: str = "basic"
    ddim_steps: int = 50
    guidance_scale: float = 3.5
    preprocess: str = "none"
    lowpass_cutoff_hz: float | None = None
    lowpass_order: int = 8
    weights_manifest: Path | None = None
    download_weights: bool = False
    force_download: bool = False
    weight_revision: str | None = None
    denoise: bool = False

    def __post_init__(self) -> None:
        device = self.device.lower()
        runtime_provider = self.runtime_provider.lower()
        precision = self.precision.lower()
        preprocess = self.preprocess.lower()
        model_cache_dir = Path(self.model_cache_dir).expanduser()
        weights_manifest = Path(self.weights_manifest).expanduser() if self.weights_manifest is not None else None

        _validate_choice("device", device, VALID_DEVICES)
        _validate_choice("runtime_provider", runtime_provider, VALID_RUNTIME_PROVIDERS)
        _validate_choice("precision", precision, VALID_PRECISIONS)
        _validate_choice("preprocess", preprocess, VALID_PREPROCESSING_MODES)
        _validate_chunk_options(self.chunk_seconds, self.overlap_seconds)
        _validate_positive("ddim_steps", self.ddim_steps)
        _validate_positive("guidance_scale", self.guidance_scale)
        _validate_optional_positive("lowpass_cutoff_hz", self.lowpass_cutoff_hz)
        _validate_positive("lowpass_order", self.lowpass_order)

        object.__setattr__(self, "device", device)
        object.__setattr__(self, "runtime_provider", runtime_provider)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "preprocess", preprocess)
        object.__setattr__(self, "model_cache_dir", model_cache_dir)
        object.__setattr__(self, "weights_manifest", weights_manifest)
        object.__setattr__(self, "model_name", self.model_name.lower())

    def ensure_model_cache_dir(self) -> Path:
        """Create and return the configured model cache directory."""

        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        return self.model_cache_dir

    def as_dict(self) -> dict[str, str | float | int | bool | None]:
        """Return a JSON-friendly representation."""

        return {
            "device": self.device,
            "runtime_provider": self.runtime_provider,
            "precision": self.precision,
            "chunked": self.chunked,
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "seed": self.seed,
            "model_cache_dir": str(self.model_cache_dir),
            "model_name": self.model_name,
            "ddim_steps": self.ddim_steps,
            "guidance_scale": self.guidance_scale,
            "preprocess": self.preprocess,
            "lowpass_cutoff_hz": self.lowpass_cutoff_hz,
            "lowpass_order": self.lowpass_order,
            "weights_manifest": None if self.weights_manifest is None else str(self.weights_manifest),
            "download_weights": self.download_weights,
            "force_download": self.force_download,
            "weight_revision": self.weight_revision,
            "denoise": self.denoise,
        }


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(choices)}")


def _validate_chunk_options(chunk_seconds: float, overlap_seconds: float) -> None:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than zero")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be greater than or equal to zero")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be less than chunk_seconds")


def _validate_positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_optional_positive(name: str, value: float | None) -> None:
    if value is not None:
        _validate_positive(name, value)
