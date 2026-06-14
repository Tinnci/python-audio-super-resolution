from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import InferenceConfig
from ..specs import BackendCapability, ModelSpec


@dataclass(frozen=True)
class BackendInfo:
    """User-facing metadata for an enhancement backend."""

    name: str
    description: str
    installed: bool
    optional_dependency: str | None = None
    package_extra: str | None = None


class EnhancementBackend(Protocol):
    """Interface implemented by audio super-resolution backends."""

    name: str
    description: str
    config: InferenceConfig

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        """Return audio enhanced to target_sample_rate."""


class FileEnhancementBackend(EnhancementBackend, Protocol):
    """Optional file-native backend interface."""

    def enhance_file(self, input_path: str | Path, output_path: str | Path, target_sample_rate: int) -> None:
        """Write enhanced audio to output_path."""


DEFAULT_ARRAY_BACKEND_CAPABILITY = BackendCapability(
    supports_array_io=True,
    supports_file_io=False,
    supports_chunking=True,
    deterministic=True,
    supports_cpu=True,
)


DEFAULT_FILE_BACKEND_CAPABILITY = BackendCapability(
    supports_array_io=False,
    supports_file_io=True,
    supports_chunking=True,
    deterministic=False,
    supports_cpu=True,
    supports_cuda=True,
    supports_mps=True,
)


def backend_model_specs(backend_type: type[EnhancementBackend]) -> tuple[ModelSpec, ...]:
    """Return model specs exposed by a backend type."""

    model_specs = getattr(backend_type, "model_specs", None)
    if callable(model_specs):
        return tuple(model_specs())
    return ()
