from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

VALID_DEVICES = ("cpu", "cuda", "mps", "directml", "auto")
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
    precision: str = "float32"
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

    def __post_init__(self) -> None:
        device = self.device.lower()
        precision = self.precision.lower()
        preprocess = self.preprocess.lower()
        model_cache_dir = Path(self.model_cache_dir).expanduser()

        if device not in VALID_DEVICES:
            raise ValueError(f"device must be one of: {', '.join(VALID_DEVICES)}")
        if precision not in VALID_PRECISIONS:
            raise ValueError(f"precision must be one of: {', '.join(VALID_PRECISIONS)}")
        if preprocess not in VALID_PREPROCESSING_MODES:
            raise ValueError(f"preprocess must be one of: {', '.join(VALID_PREPROCESSING_MODES)}")
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be greater than zero")
        if self.overlap_seconds < 0:
            raise ValueError("overlap_seconds must be greater than or equal to zero")
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("overlap_seconds must be less than chunk_seconds")
        if self.ddim_steps <= 0:
            raise ValueError("ddim_steps must be greater than zero")
        if self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be greater than zero")
        if self.lowpass_cutoff_hz is not None and self.lowpass_cutoff_hz <= 0:
            raise ValueError("lowpass_cutoff_hz must be greater than zero")
        if self.lowpass_order <= 0:
            raise ValueError("lowpass_order must be greater than zero")

        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "preprocess", preprocess)
        object.__setattr__(self, "model_cache_dir", model_cache_dir)
        object.__setattr__(self, "model_name", self.model_name.lower())

    def ensure_model_cache_dir(self) -> Path:
        """Create and return the configured model cache directory."""

        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        return self.model_cache_dir

    def as_dict(self) -> dict[str, str | float | int | None]:
        """Return a JSON-friendly representation."""

        return {
            "device": self.device,
            "precision": self.precision,
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
        }
