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
from .golden import (
    GoldenAudioStats,
    GoldenComparisonReport,
    GoldenThresholds,
    compare_golden_fixture,
    compare_golden_outputs,
    golden_report_to_dict,
    inspect_golden_audio,
    load_golden_fixture,
    write_golden_report,
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
from .model_weights import (
    download_model_weights,
    resolve_backend_model_weights,
    resolve_model_weights,
    verify_model_weights,
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
    resolve_weights_for_spec,
    validate_weight_manifest_matches_spec,
    verify_weights_for_spec,
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
    "AUDIOSR_MODEL_NAMES",
    "AUDIOSR_SAMPLE_RATE",
    "DEFAULT_AUDIO_EXTENSIONS",
    "DEFAULT_LOWPASS_CUTOFF_HZ",
    "AudioQualityReport",
    "AudioSuperResolver",
    "AudiosrBackend",
    "BackendCapability",
    "BackendInfo",
    "DeviceInfo",
    "EnhancementResult",
    "GoldenAudioStats",
    "GoldenComparisonReport",
    "GoldenThresholds",
    "InferenceConfig",
    "ManifestComparison",
    "ManifestDifference",
    "ModelInfo",
    "ModelSpec",
    "PlannedEnhancement",
    "ResolvedWeights",
    "WeightFile",
    "WeightFileSpec",
    "WeightManifest",
    "apply_preprocessing",
    "available_backends",
    "available_devices",
    "build_manifest",
    "build_quality_report_bundle",
    "compare_golden_fixture",
    "compare_manifests",
    "compare_golden_outputs",
    "default_model_cache_dir",
    "discover_audio_files",
    "download_model_weights",
    "download_model_weights_for_spec",
    "download_weights_for_spec",
    "find_model_spec",
    "format_manifest_comparison",
    "format_quality_report",
    "get_backend",
    "get_model_spec",
    "inspect_audio_quality",
    "inspect_golden_audio",
    "list_models",
    "load_manifest",
    "load_golden_fixture",
    "load_safetensors",
    "lowpass_filter",
    "manifest_comparison_to_dict",
    "plan_enhancements",
    "quality_report_to_dict",
    "golden_report_to_dict",
    "read_weight_manifest",
    "register_backend",
    "register_weight_provider",
    "registered_backend_types",
    "resolve_backend_model_weights",
    "resolve_device",
    "resolve_manifest_file_paths",
    "resolve_model_weights",
    "resolve_weights_for_spec",
    "resolve_weight_file_path",
    "resolve_weight_path",
    "sha256_file",
    "validate_weight_file_path",
    "validate_weight_manifest_matches_spec",
    "verify_model_weights",
    "verify_weights_for_spec",
    "verify_weight_file",
    "verify_weight_manifest",
    "write_manifest",
    "write_golden_report",
    "write_quality_report_bundle",
    "write_weight_manifest",
]

__version__ = "0.1.1"
