"""Audio super-resolution tools for Python."""

from .audiosr_backend import AUDIOSR_MODEL_NAMES, AUDIOSR_SAMPLE_RATE, AudiosrBackend
from .backends import register_backend, registered_backend_types
from .config import InferenceConfig, default_model_cache_dir
from .devices import DeviceInfo, available_devices, resolve_device
from .downloads import (
    download_model_weights as download_model_weights_for_spec,
)
from .downloads import (
    download_weights_for_spec,
    register_weight_provider,
)
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
from .models import ModelInfo, find_model_spec, get_model_spec, list_models
from .preprocess import DEFAULT_LOWPASS_CUTOFF_HZ, apply_preprocessing, lowpass_filter
from .quality import (
    AudioQualityReport,
    build_quality_report_bundle,
    format_quality_report,
    inspect_audio_quality,
    quality_report_to_dict,
    write_quality_report_bundle,
)
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
from .specs import BackendCapability, ModelSpec, WeightFileSpec
from .weight_store import (
    ResolvedWeights,
    download_model_weights,
    resolve_backend_model_weights,
    resolve_model_weights,
    validate_weight_manifest_matches_spec,
    verify_model_weights,
)
from .weights import (
    WeightFile,
    WeightManifest,
    load_safetensors,
    read_weight_manifest,
    resolve_manifest_file_paths,
    resolve_weight_file_path,
    resolve_weight_path,
    sha256_file,
    validate_weight_file_path,
    verify_weight_file,
    verify_weight_manifest,
    write_weight_manifest,
)

__all__ = [
    "DEFAULT_AUDIO_EXTENSIONS",
    "AudioSuperResolver",
    "AudiosrBackend",
    "BackendInfo",
    "BackendCapability",
    "DeviceInfo",
    "EnhancementResult",
    "InferenceConfig",
    "ManifestComparison",
    "ManifestDifference",
    "ModelInfo",
    "ModelSpec",
    "PlannedEnhancement",
    "ResolvedWeights",
    "AudioQualityReport",
    "AUDIOSR_MODEL_NAMES",
    "AUDIOSR_SAMPLE_RATE",
    "DEFAULT_LOWPASS_CUTOFF_HZ",
    "WeightFile",
    "WeightFileSpec",
    "WeightManifest",
    "apply_preprocessing",
    "available_backends",
    "available_devices",
    "build_manifest",
    "build_quality_report_bundle",
    "compare_manifests",
    "discover_audio_files",
    "download_model_weights",
    "download_model_weights_for_spec",
    "download_weights_for_spec",
    "default_model_cache_dir",
    "find_model_spec",
    "format_manifest_comparison",
    "format_quality_report",
    "get_backend",
    "get_model_spec",
    "inspect_audio_quality",
    "list_models",
    "lowpass_filter",
    "load_manifest",
    "manifest_comparison_to_dict",
    "plan_enhancements",
    "quality_report_to_dict",
    "read_weight_manifest",
    "register_backend",
    "registered_backend_types",
    "register_weight_provider",
    "resolve_backend_model_weights",
    "resolve_device",
    "resolve_manifest_file_paths",
    "resolve_model_weights",
    "resolve_weight_file_path",
    "resolve_weight_path",
    "sha256_file",
    "validate_weight_file_path",
    "validate_weight_manifest_matches_spec",
    "verify_model_weights",
    "verify_weight_manifest",
    "verify_weight_file",
    "write_manifest",
    "write_quality_report_bundle",
    "write_weight_manifest",
    "load_safetensors",
]

__version__ = "0.1.0"
