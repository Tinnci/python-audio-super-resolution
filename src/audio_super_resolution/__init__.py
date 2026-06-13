"""Audio super-resolution tools for Python."""

from .audiosr_backend import AUDIOSR_MODEL_NAMES, AUDIOSR_SAMPLE_RATE, AudiosrBackend
from .config import InferenceConfig, default_model_cache_dir
from .manifest import (
    ManifestComparison,
    ManifestDifference,
    build_manifest,
    compare_manifests,
    format_manifest_comparison,
    load_manifest,
    manifest_comparison_to_dict,
    write_manifest,
)
from .models import ModelInfo, list_models
from .preprocess import DEFAULT_LOWPASS_CUTOFF_HZ, apply_preprocessing, lowpass_filter
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
    "AudiosrBackend",
    "BackendInfo",
    "EnhancementResult",
    "InferenceConfig",
    "ManifestComparison",
    "ManifestDifference",
    "ModelInfo",
    "PlannedEnhancement",
    "AudioQualityReport",
    "AUDIOSR_MODEL_NAMES",
    "AUDIOSR_SAMPLE_RATE",
    "DEFAULT_LOWPASS_CUTOFF_HZ",
    "apply_preprocessing",
    "available_backends",
    "build_manifest",
    "compare_manifests",
    "discover_audio_files",
    "default_model_cache_dir",
    "format_manifest_comparison",
    "format_quality_report",
    "get_backend",
    "inspect_audio_quality",
    "list_models",
    "lowpass_filter",
    "load_manifest",
    "manifest_comparison_to_dict",
    "plan_enhancements",
    "write_manifest",
]

__version__ = "0.1.0"
