"""Audio super-resolution tools for Python."""

from .resolver import (
    DEFAULT_AUDIO_EXTENSIONS,
    AudioSuperResolver,
    BackendInfo,
    EnhancementResult,
    PlannedEnhancement,
    available_backends,
    discover_audio_files,
    get_backend,
    plan_enhancements,
)

__all__ = [
    "DEFAULT_AUDIO_EXTENSIONS",
    "AudioSuperResolver",
    "BackendInfo",
    "EnhancementResult",
    "PlannedEnhancement",
    "available_backends",
    "discover_audio_files",
    "get_backend",
    "plan_enhancements",
]

__version__ = "0.1.0"
