"""Audio super-resolution tools for Python."""

from .config import InferenceConfig, default_model_cache_dir
from .quality import AudioQualityReport, format_quality_report, inspect_audio_quality
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
    "InferenceConfig",
    "PlannedEnhancement",
    "AudioQualityReport",
    "available_backends",
    "discover_audio_files",
    "default_model_cache_dir",
    "format_quality_report",
    "get_backend",
    "inspect_audio_quality",
    "plan_enhancements",
]

__version__ = "0.1.0"
