from __future__ import annotations

from .audiosr_external import AUDIOSR_MODEL_NAMES, AUDIOSR_SAMPLE_RATE, AudiosrBackend
from .base import BackendInfo, EnhancementBackend, FileEnhancementBackend
from .lavasr_compat import LAVASR_MODEL_ID, LAVASR_REVISION, LAVASR_SAMPLE_RATE, LavaSRCompatBackend
from .registry import (
    available_backends,
    get_backend,
    list_backend_model_specs,
    register_backend,
    registered_backend_types,
)
from .sinc import SincResampleBackend

__all__ = [
    "AUDIOSR_MODEL_NAMES",
    "AUDIOSR_SAMPLE_RATE",
    "LAVASR_MODEL_ID",
    "LAVASR_REVISION",
    "LAVASR_SAMPLE_RATE",
    "AudiosrBackend",
    "BackendInfo",
    "EnhancementBackend",
    "FileEnhancementBackend",
    "LavaSRCompatBackend",
    "SincResampleBackend",
    "available_backends",
    "get_backend",
    "list_backend_model_specs",
    "register_backend",
    "registered_backend_types",
]
