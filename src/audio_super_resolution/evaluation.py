from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .backends import available_backends
from .config import InferenceConfig
from .models import find_model_spec
from .preprocess import lowpass_filter
from .quality import inspect_audio_quality, quality_report_to_dict
from .resolver import AudioSuperResolver, discover_audio_files
from .runtime_stats import peak_rss_snapshot
from .specs import BackendCapability, ModelSpec

SUPPORTED_DEGRADERS = (
    "lowpass_4k",
    "narrowband_8k",
    "wideband_16k",
    "opus_16k_24kbps",
    "mp3_32kbps",
    "noisy_16k",
)
FULL_REFERENCE_METRICS = (
    "si_sdr_db",
    "sdr_db",
    "lsd_db",
    "spectral_convergence",
    "highband_lsd_4_8k",
    "highband_lsd_8_16k",
)
OPTIONAL_FULL_REFERENCE_METRICS = ("pesq", "stoi", "estoi", "mcd")
DOWNSTREAM_METRICS = (
    "wer",
    "cer",
    "speaker_similarity",
    "vad_accuracy",
    "endpoint_accuracy",
    "keyword_accuracy",
)
ENGINEERING_METRICS = (
    "load_time_seconds",
    "backend_init_seconds",
    "elapsed_seconds",
    "total_elapsed_seconds",
    "rtf",
    "peak_rss_mb",
    "peak_rss_delta_mb",
)
STABILITY_METRICS = ("duration_drift_seconds", "clipped_fraction")
SIGNAL_STATS_NO_REFERENCE_METRICS = (
    "rms_dbfs",
    "peak_level",
    "clipped_fraction",
    "silence_fraction",
    "dc_offset",
)
SUPPORTED_NO_REFERENCE_EVALUATORS = ("signal-stats", "dnsmos", "nisqa", "utmos", "visqol")
SUPPORTED_DOWNSTREAM_EVALUATORS = (
    "transcript-error-rate",
    "speaker-similarity",
    "vad-endpoint",
    "keyword-spotting",
)
SUPPORTED_LISTENING_PROTOCOLS = ("mushra", "ab", "abx")
LISTENING_DIMENSIONS = (
    "clarity",
    "naturalness",
    "high_frequency_harshness",
    "metallic_artifacts",
    "noise",
    "intelligibility",
    "speaker_fidelity",
    "music_environment_artifacts",
    "latency",
    "stability",
)
PLANNED_NO_REFERENCE_ADAPTERS = (
    {
        "name": "dnsmos",
        "status": "planned_optional",
        "expected_scores": ["ovrl_mos", "sig_mos", "bak_mos", "p808_mos"],
        "model_source": "Microsoft DNSMOS ONNX models",
        "license_status": "requires review before redistribution",
        "runtime_requirements": ["onnxruntime", "model weights"],
        "install_guidance": "Install a future optional no-reference extra and provide explicit DNSMOS model paths.",
    },
    {
        "name": "nisqa",
        "status": "planned_optional",
        "expected_scores": ["mos", "noisiness", "coloration", "discontinuity", "loudness"],
        "model_source": "NISQA upstream checkpoints",
        "license_status": "requires review before redistribution",
        "runtime_requirements": ["torch", "model weights"],
        "install_guidance": "Install a future optional no-reference extra and provide explicit NISQA checkpoint paths.",
    },
    {
        "name": "utmos",
        "status": "planned_optional",
        "expected_scores": ["mos"],
        "model_source": "UTMOS upstream checkpoints",
        "license_status": "requires review before redistribution",
        "runtime_requirements": ["torch", "model weights"],
        "install_guidance": "Install a future optional no-reference extra and provide explicit UTMOS checkpoint paths.",
    },
    {
        "name": "visqol",
        "status": "planned_optional",
        "expected_scores": ["mos_lqo"],
        "model_source": "ViSQOL upstream binary/models",
        "license_status": "requires review before redistribution",
        "runtime_requirements": ["visqol binary or Python binding"],
        "install_guidance": "Install ViSQOL separately and configure a future adapter with the binary/model path.",
    },
)
PLANNED_FULL_REFERENCE_ADAPTERS = (
    {
        "name": "pesq",
        "status": "planned_optional",
        "expected_scores": ["pesq"],
        "runtime_requirements": ["pesq optional dependency"],
        "sample_rate_constraints": ["8000 Hz narrowband", "16000 Hz wideband"],
        "install_guidance": (
            "Install a future optional full-reference extra; record PESQ mode and effective sample rate."
        ),
    },
    {
        "name": "stoi",
        "status": "planned_optional",
        "expected_scores": ["stoi", "estoi"],
        "runtime_requirements": ["pystoi optional dependency"],
        "sample_rate_constraints": ["explicitly resampled evaluator input"],
        "install_guidance": "Install a future optional full-reference extra; record resampling settings.",
    },
    {
        "name": "mcd",
        "status": "planned_optional",
        "expected_scores": ["mcd"],
        "runtime_requirements": ["cepstral feature implementation"],
        "sample_rate_constraints": ["record mel/cepstral feature settings"],
        "install_guidance": "Use a maintained feature extractor or a tested local implementation before enabling MCD.",
    },
)
PLANNED_DOWNSTREAM_ADAPTERS = (
    {
        "name": "speaker-similarity",
        "status": "planned_optional",
        "task": "speaker",
        "expected_scores": ["speaker_similarity"],
        "runtime_requirements": ["speaker embedding model", "model weights"],
        "install_guidance": "Install a future optional downstream extra and provide explicit speaker model paths.",
    },
    {
        "name": "vad-endpoint",
        "status": "planned_optional",
        "task": "vad",
        "expected_scores": ["vad_accuracy", "endpoint_accuracy"],
        "runtime_requirements": ["VAD model or labeled endpoint fixtures"],
        "install_guidance": "Provide labeled VAD/endpoint fixtures and configure a future optional VAD adapter.",
    },
    {
        "name": "keyword-spotting",
        "status": "planned_optional",
        "task": "kws",
        "expected_scores": ["keyword_accuracy"],
        "runtime_requirements": ["keyword labels", "KWS model"],
        "install_guidance": "Provide keyword labels and configure a future optional KWS adapter.",
    },
)
LOWER_IS_BETTER_METRICS = {
    "lsd_db",
    "mcd",
    "spectral_convergence",
    "highband_lsd_4_8k",
    "highband_lsd_8_16k",
    "wer",
    "cer",
    "endpoint_error",
    "backend_init_seconds",
    "elapsed_seconds",
    "rtf",
    "peak_rss_mb",
    "peak_rss_delta_mb",
    "duration_drift_seconds",
    "clipped_fraction",
}
SILENCE_RMS_THRESHOLD = 1e-5
HALLUCINATION_RMS_THRESHOLD = 1e-4
LOW_VOLUME_RMS_THRESHOLD = 0.01
OVERAMPLIFICATION_GAIN_THRESHOLD = 12.0
OVERAMPLIFICATION_OUTPUT_RMS_THRESHOLD = 0.05
SPEECH_BWE_EVALSET_ID = "speech_bwe_v1_tiny"
SPEECH_BWE_EVALSET_PROFILES = (
    ("zh_female_slow_clean", "zh", "female", "slow", "clean", 185.0),
    ("zh_male_normal_clean", "zh", "male", "normal", "clean", 125.0),
    ("zh_female_fast_noisy", "zh", "female", "fast", "noisy", 210.0),
    ("zh_male_slow_reverb", "zh", "male", "slow", "reverb", 110.0),
    ("en_female_normal_clean", "en", "female", "normal", "clean", 195.0),
    ("en_male_fast_noisy", "en", "male", "fast", "noisy", 135.0),
    ("en_female_slow_reverb", "en", "female", "slow", "reverb", 175.0),
    ("en_male_normal_clean", "en", "male", "normal", "clean", 120.0),
)


@dataclass(frozen=True)
class DegradedAudio:
    """Audio produced by a controlled degradation recipe."""

    audio: np.ndarray
    sample_rate: int
    recipe: dict[str, object]


def run_eval_dataset(
    *,
    dataset_dir: str | Path,
    backend: str,
    output_path: str | Path,
    work_dir: str | Path,
    target_sample_rate: int = 48000,
    degrader: str = "wideband_16k",
    config: InferenceConfig | None = None,
    limit: int | None = None,
    optional_metrics: tuple[str, ...] = (),
) -> dict[str, object]:
    """Run a lightweight full-reference eval over clean reference WAV files."""

    dataset_path = Path(dataset_dir)
    output = Path(output_path)
    work = Path(work_dir)
    references = _reference_files(dataset_path, limit=limit)
    if not references:
        raise ValueError(f"No .wav reference files found in {dataset_path}")

    resolved_config = config or InferenceConfig()
    degraded_dir = work / "degraded" / degrader
    enhanced_dir = work / "enhanced" / backend / degrader
    degraded_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _run_eval_item(
            reference_path=reference,
            dataset_dir=dataset_path,
            backend=backend,
            target_sample_rate=target_sample_rate,
            degrader=degrader,
            degraded_dir=degraded_dir,
            enhanced_dir=enhanced_dir,
            config=resolved_config,
            optional_metrics=optional_metrics,
        )
        for reference in references
    ]
    manifest = build_eval_manifest(
        dataset_dir=dataset_path,
        backend=backend,
        target_sample_rate=target_sample_rate,
        degrader=degrader,
        config=resolved_config,
        results=results,
        optional_metrics=optional_metrics,
    )
    write_eval_manifest(output, manifest)
    return manifest


def init_speech_bwe_evalset(
    *,
    output_dir: str | Path,
    count: int = 20,
    sample_rate: int = 48000,
    duration_seconds: float = 0.35,
    force: bool = False,
) -> dict[str, object]:
    """Create a deterministic tiny speech-BWE evalset fixture.

    The generated audio is synthetic and intended for smoke/regression workflows, not for
    publishable perceptual claims.
    """

    if count <= 0:
        raise ValueError("count must be greater than zero")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")

    output_path = Path(output_dir)
    clean_dir = output_path / "speech_clean_48k"
    manifest_path = output_path / "manifest.json"
    if output_path.exists() and any(output_path.iterdir()) and not force:
        raise FileExistsError(f"{output_path} is not empty; pass force=True or --force to replace generated files")
    clean_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for index in range(count):
        profile = SPEECH_BWE_EVALSET_PROFILES[index % len(SPEECH_BWE_EVALSET_PROFILES)]
        profile_id, language, speaker_gender, speaking_rate, condition, base_frequency = profile
        item_id = f"sample_{index + 1:03d}_{profile_id}"
        path = clean_dir / f"{item_id}.wav"
        audio = _synthetic_speech_like_audio(
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            base_frequency=base_frequency,
            seed=index,
            condition=condition,
        )
        sf.write(path, audio, sample_rate)
        records.append(
            {
                "id": item_id,
                "path": str(path.relative_to(output_path)),
                "language": language,
                "speaker_gender": speaker_gender,
                "speaking_rate": speaking_rate,
                "condition": condition,
                "sample_rate": sample_rate,
                "duration_seconds": duration_seconds,
                "synthetic": True,
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset_id": SPEECH_BWE_EVALSET_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(output_path),
        "reference_dir": str(clean_dir.relative_to(output_path)),
        "sample_rate": sample_rate,
        "record_count": len(records),
        "degraders": list(SUPPORTED_DEGRADERS),
        "recommended_metrics": {
            "full_reference": list(FULL_REFERENCE_METRICS),
            "optional_full_reference": list(OPTIONAL_FULL_REFERENCE_METRICS),
            "downstream": ["wer", "cer"],
            "engineering": ["rtf", "peak_rss_mb", "load_time_seconds", "duration_drift_seconds"],
            "stability": ["clipped_fraction", "sample_rate_correct", "failure_cases"],
        },
        "notes": [
            "Synthetic fixture for smoke and regression tests only.",
            "Use real licensed clean speech before making backend quality claims.",
        ],
        "records": records,
    }
    write_eval_manifest(manifest_path, manifest)
    return manifest


def run_no_reference_eval(
    *,
    input_path: str | Path,
    output_path: str | Path,
    recursive: bool = False,
    evaluator: str = "signal-stats",
    limit: int | None = None,
) -> dict[str, object]:
    """Run no-reference objective screening over one file or a directory."""

    evaluator = evaluator.lower()
    if evaluator != "signal-stats":
        raise ValueError(_unsupported_no_reference_evaluator_message(evaluator))

    audio_paths = discover_audio_files(input_path, recursive=recursive)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        audio_paths = audio_paths[:limit]
    if not audio_paths:
        raise ValueError(f"No supported audio files found in {Path(input_path)}")

    records = [_signal_stats_no_reference_record(path, root=Path(input_path)) for path in audio_paths]
    manifest = build_no_reference_manifest(
        input_path=Path(input_path),
        evaluator=evaluator,
        records=records,
    )
    write_eval_manifest(output_path, manifest)
    return manifest


def run_downstream_eval(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    evaluator: str = "transcript-error-rate",
    dataset_id: str | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    """Run lightweight downstream evaluation from precomputed task outputs."""

    evaluator = evaluator.lower()
    if evaluator != "transcript-error-rate":
        raise ValueError(_unsupported_downstream_evaluator_message(evaluator))

    dataset = _load_downstream_dataset(dataset_path)
    loaded_records = dataset["records"]
    if not isinstance(loaded_records, list):
        raise ValueError("downstream dataset must include a records list")
    records: list[object] = loaded_records
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        records = records[:limit]
    if not records:
        raise ValueError(f"No downstream records found in {Path(dataset_path)}")

    resolved_dataset_id = dataset_id or str(dataset.get("dataset_id") or Path(dataset_path))
    eval_records = [
        _transcript_error_rate_record(record, dataset_id=resolved_dataset_id)
        for record in records
        if isinstance(record, dict)
    ]
    manifest = build_downstream_manifest(
        dataset_path=Path(dataset_path),
        dataset_id=resolved_dataset_id,
        evaluator=evaluator,
        records=eval_records,
    )
    write_eval_manifest(output_path, manifest)
    return manifest


def run_listening_export(
    *,
    manifest_paths: list[str | Path],
    output_dir: str | Path,
    protocol: str = "mushra",
    seed: int = 0,
) -> dict[str, object]:
    """Export a browser/runtime-neutral listening-test bundle from eval manifests."""

    protocol = protocol.lower()
    if protocol not in SUPPORTED_LISTENING_PROTOCOLS:
        choices = ", ".join(SUPPORTED_LISTENING_PROTOCOLS)
        raise ValueError(f"Unsupported listening protocol {protocol!r}. Choices: {choices}")
    if not manifest_paths:
        raise ValueError("at least one eval manifest is required for listening export")

    output_path = Path(output_dir)
    stimuli_dir = output_path / "stimuli"
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    loaded_manifests = [(Path(path), load_eval_manifest(path)) for path in manifest_paths]
    trials, answer_key_trials = _listening_trials(
        loaded_manifests,
        stimuli_dir=stimuli_dir,
        protocol=protocol,
        seed=seed,
    )
    public_manifest = build_listening_manifest(
        protocol=protocol,
        output_dir=output_path,
        manifest_paths=[path for path, _manifest in loaded_manifests],
        trials=trials,
        seed=seed,
    )
    answer_key = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "seed": seed,
        "trials": answer_key_trials,
    }
    write_eval_manifest(output_path / "listening_manifest.json", public_manifest)
    write_eval_manifest(output_path / "answer_key.json", answer_key)
    return {
        "manifest_path": str(output_path / "listening_manifest.json"),
        "answer_key_path": str(output_path / "answer_key.json"),
        "manifest": public_manifest,
        "answer_key": answer_key,
    }


def build_no_reference_manifest(
    *,
    input_path: Path,
    evaluator: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Build a JSON-serializable no-reference eval manifest."""

    status_counts = _status_counts(records)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "no_reference",
        "passed": all(record.get("status") == "passed" for record in records),
        "input": str(input_path),
        "evaluator": _no_reference_evaluator_info(evaluator),
        "metric_groups": {
            "no_reference": list(SIGNAL_STATS_NO_REFERENCE_METRICS),
            "planned_optional_no_reference": [adapter["name"] for adapter in PLANNED_NO_REFERENCE_ADAPTERS],
        },
        "result_count": len(records),
        "status_counts": status_counts,
        "planned_adapters": list(PLANNED_NO_REFERENCE_ADAPTERS),
        "records": records,
        "results": records,
    }


def no_reference_signal_stats(path: str | Path) -> dict[str, float]:
    """Compute lightweight no-reference signal screening metrics for one audio file."""

    audio, _sample_rate = sf.read(path, always_2d=True)
    absolute = np.abs(audio)
    peak_level = float(absolute.max()) if absolute.size else 0.0
    clipped_samples = int(np.count_nonzero(absolute >= 0.999))
    clipped_fraction = clipped_samples / absolute.size if absolute.size else 0.0
    silence_fraction = int(np.count_nonzero(absolute <= 1e-4)) / absolute.size if absolute.size else 0.0
    dc_offset = float(np.mean(audio)) if audio.size else 0.0
    rms = _rms(audio)
    return {
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-12))),
        "peak_level": peak_level,
        "clipped_fraction": clipped_fraction,
        "silence_fraction": silence_fraction,
        "dc_offset": dc_offset,
    }


def build_downstream_manifest(
    *,
    dataset_path: Path,
    dataset_id: str,
    evaluator: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Build a JSON-serializable downstream eval manifest."""

    status_counts = _status_counts(records)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "downstream",
        "passed": all(record.get("status") == "passed" for record in records),
        "dataset": str(dataset_path),
        "dataset_id": dataset_id,
        "evaluator": _downstream_evaluator_info(evaluator),
        "metric_groups": {
            "downstream": ["wer", "cer"],
            "planned_optional_downstream": [adapter["name"] for adapter in PLANNED_DOWNSTREAM_ADAPTERS],
        },
        "result_count": len(records),
        "status_counts": status_counts,
        "planned_adapters": list(PLANNED_DOWNSTREAM_ADAPTERS),
        "records": records,
        "results": records,
    }


def build_listening_manifest(
    *,
    protocol: str,
    output_dir: Path,
    manifest_paths: list[Path],
    trials: list[dict[str, object]],
    seed: int,
) -> dict[str, object]:
    """Build the public, blind listening-test manifest."""

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "listening_export",
        "protocol": protocol,
        "seed": seed,
        "source_manifests": [str(path) for path in manifest_paths],
        "stimuli_dir": str(output_dir / "stimuli"),
        "answer_key_external": True,
        "rating_dimensions": list(LISTENING_DIMENSIONS),
        "guidance": {
            "do_not_ask_only_which_is_better": True,
            "combine_with": [
                "full_reference_metrics",
                "no_reference_screening",
                "downstream_eval",
                "engineering_performance",
                "stability",
                "governance",
            ],
        },
        "trial_count": len(trials),
        "trials": trials,
    }


def transcript_error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    """Compute WER/CER for a reference and hypothesis transcript."""

    reference_words = _word_tokens(reference)
    hypothesis_words = _word_tokens(hypothesis)
    reference_chars = _character_tokens(reference)
    hypothesis_chars = _character_tokens(hypothesis)
    return {
        "wer": _error_rate(reference_words, hypothesis_words),
        "cer": _error_rate(reference_chars, hypothesis_chars),
    }


def build_eval_manifest(
    *,
    dataset_dir: Path,
    backend: str,
    target_sample_rate: int,
    degrader: str,
    config: InferenceConfig,
    results: list[dict[str, object]],
    optional_metrics: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build a JSON-serializable eval manifest."""

    status_counts = _status_counts(results)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(result.get("status") == "passed" for result in results),
        "dataset": str(dataset_dir),
        "backend": backend,
        "backend_profile": build_backend_profile(backend, config=config),
        "target_sample_rate": target_sample_rate,
        "degrader": {"name": degrader},
        "config": config.as_dict(),
        "optional_metrics_requested": list(optional_metrics),
        "metric_groups": {
            "full_reference": list(FULL_REFERENCE_METRICS),
            "optional_full_reference": list(OPTIONAL_FULL_REFERENCE_METRICS),
            "engineering": [
                "backend_init_seconds",
                "load_time_seconds",
                "elapsed_seconds",
                "total_elapsed_seconds",
                "rtf",
                "peak_rss_mb",
                "peak_rss_delta_mb",
            ],
            "stability": [
                "sample_rate_correct",
                "duration_drift_seconds",
                "duration_drift_exceeded",
                "clipped_fraction",
                "clipped",
                "failure_status",
                "passed",
            ],
            "governance": [
                "offline",
                "reproducible",
                "license_usable",
                "explicit_weights",
                "dependency_footprint",
            ],
        },
        "result_count": len(results),
        "status_counts": status_counts,
        "failure_count": sum(count for status, count in status_counts.items() if status != "passed"),
        "planned_adapters": {
            "full_reference": list(PLANNED_FULL_REFERENCE_ADAPTERS),
            "no_reference": list(PLANNED_NO_REFERENCE_ADAPTERS),
            "downstream": list(PLANNED_DOWNSTREAM_ADAPTERS),
        },
        "results": results,
    }


def build_backend_profile(backend: str, *, config: InferenceConfig) -> dict[str, object]:
    """Return backend capability and governance facts for eval manifests."""

    try:
        spec = find_model_spec(backend, _model_name_for_backend(backend, config))
    except ValueError:
        return {
            "backend": backend,
            "model_id": None,
            "capabilities": {
                "batch": "unknown",
                "stream": False,
                "chunking": False,
                "supports_cpu": False,
                "supports_cuda": False,
                "supports_mps": False,
                "cpu_only": False,
                "offline": False,
                "reproducible": False,
            },
            "governance": {
                "code_license": None,
                "weights_license": None,
                "license_usable": False,
                "explicit_weights": False,
                "weights_source": None,
                "weights_hash": None,
                "weight_provider": None,
                "weight_file_count": 0,
                "weight_size_bytes": None,
                "requires_weights": None,
            },
            "dependency_footprint": {
                "package_extra": None,
                "optional_dependency": None,
                "strategy": "unknown-backend",
                "dependency_tier": "unknown",
            },
        }

    capability = spec.capability
    weight_size_bytes = _total_weight_size_bytes(spec)
    explicit_weights = _has_explicit_weight_governance(spec)
    license_usable = _license_usable(spec.weights_license, requires_weights=spec.requires_weights)
    reproducible = bool(
        capability is not None
        and capability.deterministic
        and explicit_weights
        and (not spec.requires_weights or _all_weight_hashes_present(spec))
    )

    return {
        "backend": backend,
        "model_id": spec.id,
        "model_name": spec.model_name,
        "implementation": spec.implementation,
        "maturity": spec.maturity,
        "domain": list(spec.domain),
        "tasks": list(spec.tasks),
        "capabilities": {
            "batch": "file-loop",
            "stream": False,
            "chunking": capability.supports_chunking if capability is not None else False,
            "supports_array_io": capability.supports_array_io if capability is not None else False,
            "supports_file_io": capability.supports_file_io if capability is not None else False,
            "supports_cpu": capability.supports_cpu if capability is not None else False,
            "supports_cuda": capability.supports_cuda if capability is not None else False,
            "supports_mps": capability.supports_mps if capability is not None else False,
            "requires_gpu": capability.requires_gpu if capability is not None else False,
            "cpu_only": _cpu_only(capability),
            "offline": _offline_capable(spec),
            "reproducible": reproducible,
            "deterministic": capability.deterministic if capability is not None else False,
            "precision_modes": list(capability.precision_modes) if capability is not None else [],
            "accelerators": list(capability.accelerators) if capability is not None else [],
            "runtime_providers": list(capability.runtime_providers) if capability is not None else [],
        },
        "governance": {
            "code_license": spec.code_license,
            "weights_license": spec.weights_license,
            "license_usable": license_usable,
            "explicit_weights": explicit_weights,
            "weights_source": spec.weights_source,
            "weights_hash": spec.weights_hash,
            "weight_provider": spec.weight_provider,
            "weight_file_count": len(spec.weight_files),
            "weight_size_bytes": weight_size_bytes,
            "weight_manifest_url": spec.weight_manifest_url,
            "default_weight_revision": spec.default_weight_revision,
            "requires_weights": spec.requires_weights,
            "attribution_required": spec.attribution_required,
        },
        "dependency_footprint": {
            "package_extra": _package_extra_for_backend(backend),
            "optional_dependency": _optional_dependency_for_backend(backend),
            "strategy": "declared-optional-extra-and-weight-metadata",
            "dependency_tier": _dependency_tier(spec.implementation, requires_weights=spec.requires_weights),
            "weight_size_bytes": weight_size_bytes,
        },
        "known_limitations": list(spec.known_limitations),
        "validation": list(spec.validation),
    }


def _signal_stats_no_reference_record(path: Path, *, root: Path) -> dict[str, object]:
    audio_info = sf.info(path)
    root_path = root if root.is_dir() else root.parent
    try:
        item_id = path.relative_to(root_path).with_suffix("").as_posix()
    except ValueError:
        item_id = path.with_suffix("").name
    return {
        "id": item_id,
        "input_path": str(path),
        "evaluator": _no_reference_evaluator_info("signal-stats"),
        "status": "passed",
        "scores": no_reference_signal_stats(path),
        "metadata": {
            "sample_rate": audio_info.samplerate,
            "duration_seconds": audio_info.frames / audio_info.samplerate,
            "channels": audio_info.channels,
            "frames": audio_info.frames,
        },
        "error": None,
        "install_guidance": None,
    }


def _no_reference_evaluator_info(evaluator: str) -> dict[str, object]:
    if evaluator == "signal-stats":
        return {
            "name": "signal-stats",
            "version": "builtin",
            "status": "implemented",
            "score_fields": list(SIGNAL_STATS_NO_REFERENCE_METRICS),
            "model_source": None,
            "license_status": "project-license",
            "runtime_requirements": ["numpy", "soundfile"],
            "screening_signal": True,
            "absolute_truth": False,
        }
    for adapter in PLANNED_NO_REFERENCE_ADAPTERS:
        if adapter["name"] == evaluator:
            return dict(adapter)
    return {
        "name": evaluator,
        "status": "unsupported",
        "score_fields": [],
        "model_source": None,
        "license_status": None,
        "runtime_requirements": [],
        "screening_signal": True,
        "absolute_truth": False,
    }


def _unsupported_no_reference_evaluator_message(evaluator: str) -> str:
    evaluator_info = _no_reference_evaluator_info(evaluator)
    if evaluator_info.get("status") == "planned_optional":
        return (
            f"No-reference evaluator {evaluator!r} is documented but not enabled in the default install. "
            f"{evaluator_info['install_guidance']} Heavy no-reference evaluators must remain opt-in."
        )
    choices = ", ".join(SUPPORTED_NO_REFERENCE_EVALUATORS)
    return f"Unsupported no-reference evaluator {evaluator!r}. Choices: {choices}"


def _load_downstream_dataset(dataset_path: str | Path) -> dict[str, object]:
    loaded = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        return {"records": loaded}
    if not isinstance(loaded, dict):
        raise ValueError("downstream dataset must be a JSON object or list")
    records = loaded.get("records")
    if not isinstance(records, list):
        raise ValueError("downstream dataset must include a records list")
    return loaded


def _transcript_error_rate_record(record: dict[str, object], *, dataset_id: str) -> dict[str, object]:
    item_id = str(record.get("id", ""))
    reference = _required_text(record, "reference_transcript", item_id=item_id)
    baseline = _required_text(record, "baseline_transcript", item_id=item_id)
    enhanced = _required_text(record, "enhanced_transcript", item_id=item_id)
    baseline_scores = transcript_error_rates(reference, baseline)
    enhanced_scores = transcript_error_rates(reference, enhanced)
    delta = {
        "wer": enhanced_scores["wer"] - baseline_scores["wer"],
        "cer": enhanced_scores["cer"] - baseline_scores["cer"],
    }
    return {
        "id": item_id or str(record.get("input_path", "record")),
        "dataset_id": dataset_id,
        "task": "asr",
        "evaluator": _downstream_evaluator_info("transcript-error-rate"),
        "evaluator_version": "builtin",
        "status": "passed",
        "baseline_input_score": baseline_scores,
        "enhanced_score": enhanced_scores,
        "delta": delta,
        "scores": {
            "wer": enhanced_scores["wer"],
            "cer": enhanced_scores["cer"],
            "baseline_wer": baseline_scores["wer"],
            "baseline_cer": baseline_scores["cer"],
            "wer_delta": delta["wer"],
            "cer_delta": delta["cer"],
        },
        "metadata": {
            "reference_transcript": reference,
            "baseline_transcript": baseline,
            "enhanced_transcript": enhanced,
            "input_path": record.get("input_path"),
            "enhanced_path": record.get("enhanced_path"),
        },
        "error": None,
        "install_guidance": None,
    }


def _required_text(record: dict[str, object], field: str, *, item_id: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        label = f" for record {item_id!r}" if item_id else ""
        raise ValueError(f"downstream transcript dataset missing string field {field!r}{label}")
    return value


def _downstream_evaluator_info(evaluator: str) -> dict[str, object]:
    if evaluator == "transcript-error-rate":
        return {
            "name": "transcript-error-rate",
            "version": "builtin",
            "status": "implemented",
            "task": "asr",
            "score_fields": ["wer", "cer"],
            "runtime_requirements": ["precomputed reference/baseline/enhanced transcripts"],
            "lower_is_better": True,
            "model_source": None,
            "license_status": "project-license",
        }
    for adapter in PLANNED_DOWNSTREAM_ADAPTERS:
        if adapter["name"] == evaluator:
            return dict(adapter)
    return {
        "name": evaluator,
        "status": "unsupported",
        "task": "unknown",
        "score_fields": [],
        "runtime_requirements": [],
    }


def _unsupported_downstream_evaluator_message(evaluator: str) -> str:
    evaluator_info = _downstream_evaluator_info(evaluator)
    if evaluator_info.get("status") == "planned_optional":
        return (
            f"Downstream evaluator {evaluator!r} is documented but not enabled in the default install. "
            f"{evaluator_info['install_guidance']} Downstream models must remain opt-in."
        )
    choices = ", ".join(SUPPORTED_DOWNSTREAM_EVALUATORS)
    return f"Unsupported downstream evaluator {evaluator!r}. Choices: {choices}"


def _listening_trials(
    loaded_manifests: list[tuple[Path, dict[str, object]]],
    *,
    stimuli_dir: Path,
    protocol: str,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped_results: dict[str, list[tuple[Path, dict[str, object]]]] = {}
    for manifest_path, manifest in loaded_manifests:
        for result in _manifest_results(manifest):
            grouped_results.setdefault(str(result.get("id")), []).append((manifest_path, result))

    trials: list[dict[str, object]] = []
    answer_key_trials: list[dict[str, object]] = []
    rng = random.Random(seed)
    for trial_index, item_id in enumerate(sorted(grouped_results), start=1):
        sources = _listening_sources(grouped_results[item_id])
        rng.shuffle(sources)
        public_stimuli: list[dict[str, object]] = []
        answer_stimuli: list[dict[str, object]] = []
        for stimulus_index, source in enumerate(sources, start=1):
            blind_id = f"t{trial_index:03d}_s{stimulus_index:02d}"
            source_path = Path(str(source["path"]))
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            stimulus_path = stimuli_dir / f"{blind_id}{source_path.suffix.lower() or '.wav'}"
            shutil.copy2(source_path, stimulus_path)
            public_stimuli.append(
                {
                    "blind_id": blind_id,
                    "path": str(stimulus_path),
                }
            )
            answer_stimuli.append(
                {
                    "blind_id": blind_id,
                    "path": str(stimulus_path),
                    "source_path": str(source_path),
                    "source_manifest": source["source_manifest"],
                    "item_id": item_id,
                    "role": source["role"],
                    "backend": source["backend"],
                }
            )
        trials.append(
            {
                "id": item_id,
                "protocol": protocol,
                "stimuli": public_stimuli,
                "rating_dimensions": list(LISTENING_DIMENSIONS),
            }
        )
        answer_key_trials.append({"id": item_id, "stimuli": answer_stimuli})
    return trials, answer_key_trials


def _manifest_results(manifest: dict[str, object]) -> list[dict[str, object]]:
    results = manifest.get("results")
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, dict)]


def _listening_sources(results: list[tuple[Path, dict[str, object]]]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    first_manifest_path, first_result = results[0]
    reference_path = first_result.get("reference_path")
    if isinstance(reference_path, str):
        sources.append(
            {
                "path": reference_path,
                "source_manifest": str(first_manifest_path),
                "role": "reference",
                "backend": None,
            }
        )
    degraded_path = first_result.get("degraded_path") or first_result.get("input_path")
    if isinstance(degraded_path, str):
        sources.append(
            {
                "path": degraded_path,
                "source_manifest": str(first_manifest_path),
                "role": "anchor",
                "backend": "degraded_input",
            }
        )
    for manifest_path, result in results:
        enhanced_path = result.get("enhanced_path")
        if not isinstance(enhanced_path, str):
            continue
        sources.append(
            {
                "path": enhanced_path,
                "source_manifest": str(manifest_path),
                "role": "system",
                "backend": result.get("backend"),
            }
        )
    if len(sources) < 2:
        raise ValueError("listening export requires at least two stimuli per trial")
    return sources


def _word_tokens(text: str) -> list[str]:
    return text.casefold().split()


def _character_tokens(text: str) -> list[str]:
    return [character for character in text.casefold() if not character.isspace()]


def _error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(reference, hypothesis) / len(reference)


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if reference_token == hypothesis_token else 1
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def write_eval_manifest(path: str | Path, manifest: dict[str, object]) -> Path:
    """Write an eval manifest."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def load_eval_manifest(path: str | Path) -> dict[str, object]:
    """Load and minimally validate an eval manifest."""

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("eval manifest root must be a JSON object")
    if loaded.get("schema_version") != 1:
        raise ValueError("eval manifest schema_version must be 1")
    if not isinstance(loaded.get("results"), list):
        raise ValueError("eval manifest results must be a list")
    return loaded


def compare_eval_manifests(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Compare two eval manifests without collapsing metrics into one score."""

    baseline_results = _indexed_results(baseline)
    candidate_results = _indexed_results(candidate)
    metric_summary = _metric_summary(baseline_results, candidate_results)
    regressions = _eval_regressions(
        baseline_results,
        candidate_results,
        thresholds=thresholds or {},
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not regressions,
        "baseline_backend": baseline.get("backend"),
        "candidate_backend": candidate.get("backend"),
        "item_count": len(set(baseline_results) & set(candidate_results)),
        "missing_from_candidate": sorted(set(baseline_results) - set(candidate_results)),
        "unexpected_candidate_items": sorted(set(candidate_results) - set(baseline_results)),
        "metric_summary": metric_summary,
        "tables": _comparison_tables(baseline, candidate, metric_summary, candidate_results),
        "thresholds": thresholds or {},
        "regressions": regressions,
    }


def degrade_audio(audio: np.ndarray, sample_rate: int, degrader: str) -> DegradedAudio:
    """Apply a controlled degradation recipe."""

    if degrader == "lowpass_4k":
        return DegradedAudio(
            audio=lowpass_filter(audio, sample_rate=sample_rate, cutoff_hz=4000.0),
            sample_rate=sample_rate,
            recipe={"name": degrader, "lowpass_cutoff_hz": 4000.0},
        )
    if degrader == "narrowband_8k":
        degraded = _resample(audio, sample_rate, 8000)
        return DegradedAudio(audio=degraded, sample_rate=8000, recipe={"name": degrader, "sample_rate": 8000})
    if degrader == "wideband_16k":
        degraded = _resample(audio, sample_rate, 16000)
        return DegradedAudio(audio=degraded, sample_rate=16000, recipe={"name": degrader, "sample_rate": 16000})
    if degrader == "opus_16k_24kbps":
        degraded = _resample(audio, sample_rate, 16000)
        degraded = _codec_like_quantize(degraded, levels=512)
        return DegradedAudio(
            audio=degraded,
            sample_rate=16000,
            recipe={"name": degrader, "sample_rate": 16000, "codec": "opus-like", "bitrate": "24kbps"},
        )
    if degrader == "mp3_32kbps":
        degraded = lowpass_filter(audio, sample_rate=sample_rate, cutoff_hz=min(11000.0, sample_rate * 0.45))
        degraded = _codec_like_quantize(degraded, levels=384)
        return DegradedAudio(
            audio=degraded.astype(np.float32, copy=False),
            sample_rate=sample_rate,
            recipe={"name": degrader, "codec": "mp3-like", "bitrate": "32kbps", "lowpass_cutoff_hz": 11000.0},
        )
    if degrader == "noisy_16k":
        degraded = _resample(audio, sample_rate, 16000)
        noise = _deterministic_noise(degraded.shape, scale=0.005)
        return DegradedAudio(
            audio=np.asarray(degraded + noise, dtype=np.float32),
            sample_rate=16000,
            recipe={"name": degrader, "sample_rate": 16000, "noise_rms": 0.005},
        )
    choices = ", ".join(SUPPORTED_DEGRADERS)
    raise ValueError(f"Unsupported degrader {degrader!r}. Choices: {choices}")


def full_reference_metrics(
    enhanced_audio: np.ndarray,
    reference_audio: np.ndarray,
    *,
    sample_rate: int,
) -> dict[str, float | None]:
    """Compute lightweight full-reference metrics for one enhanced/reference pair."""

    enhanced = _mixdown(enhanced_audio)
    reference = _mixdown(reference_audio)
    enhanced, reference = _align(enhanced, reference)
    enhanced_mag = _stft_magnitude(enhanced)
    reference_mag = _stft_magnitude(reference)
    return {
        "si_sdr_db": _si_sdr_db(enhanced, reference),
        "sdr_db": _sdr_db(enhanced, reference),
        "lsd_db": _lsd_db(enhanced_mag, reference_mag),
        "spectral_convergence": _spectral_convergence(enhanced_mag, reference_mag),
        "highband_lsd_4_8k": _band_lsd_db(
            enhanced_mag, reference_mag, sample_rate=sample_rate, low_hz=4000, high_hz=8000
        ),
        "highband_lsd_8_16k": _band_lsd_db(
            enhanced_mag,
            reference_mag,
            sample_rate=sample_rate,
            low_hz=8000,
            high_hz=16000,
        ),
    }


def optional_full_reference_metrics(
    enhanced_audio: np.ndarray,
    reference_audio: np.ndarray,
    *,
    sample_rate: int,
    metrics: tuple[str, ...],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run explicitly requested optional full-reference metrics when their dependencies exist."""

    scores: dict[str, float] = {}
    records: list[dict[str, object]] = []
    for metric in metrics:
        metric_name = metric.lower()
        if metric_name not in OPTIONAL_FULL_REFERENCE_METRICS:
            choices = ", ".join(OPTIONAL_FULL_REFERENCE_METRICS)
            raise ValueError(f"Unsupported optional full-reference metric {metric!r}. Choices: {choices}")
        try:
            score = _run_optional_full_reference_metric(
                metric_name,
                enhanced_audio,
                reference_audio,
                sample_rate=sample_rate,
            )
        except ImportError as exc:
            records.append(_optional_metric_record(metric_name, status="skipped", error=str(exc)))
            continue
        except (RuntimeError, ValueError) as exc:
            records.append(_optional_metric_record(metric_name, status="failed", error=str(exc)))
            continue
        if score is None:
            records.append(
                _optional_metric_record(
                    metric_name,
                    status="skipped",
                    error=f"{metric_name} does not have a default lightweight implementation",
                )
            )
            continue
        scores[metric_name] = score
        records.append(
            {
                "name": metric_name,
                "status": "passed",
                "score": score,
                "install_guidance": None,
                "error": None,
            }
        )
    return scores, records


def _run_optional_full_reference_metric(
    metric: str,
    enhanced_audio: np.ndarray,
    reference_audio: np.ndarray,
    *,
    sample_rate: int,
) -> float | None:
    enhanced = _mixdown(enhanced_audio)
    reference = _mixdown(reference_audio)
    enhanced, reference = _align(enhanced, reference)
    if metric == "pesq":
        pesq_module = import_module("pesq")
        pesq_func = cast(Any, pesq_module).pesq
        eval_sample_rate = 16000 if sample_rate >= 16000 else 8000
        mode = "wb" if eval_sample_rate == 16000 else "nb"
        reference_eval = _resample(reference, sample_rate, eval_sample_rate)
        enhanced_eval = _resample(enhanced, sample_rate, eval_sample_rate)
        return float(pesq_func(eval_sample_rate, reference_eval, enhanced_eval, mode))
    if metric in {"stoi", "estoi"}:
        pystoi_module = import_module("pystoi.stoi")
        stoi_func = cast(Any, pystoi_module).stoi
        eval_sample_rate = min(sample_rate, 16000)
        reference_eval = _resample(reference, sample_rate, eval_sample_rate)
        enhanced_eval = _resample(enhanced, sample_rate, eval_sample_rate)
        return float(stoi_func(reference_eval, enhanced_eval, eval_sample_rate, extended=metric == "estoi"))
    return None


def _optional_metric_record(metric: str, *, status: str, error: str) -> dict[str, object]:
    adapter_info = next((adapter for adapter in PLANNED_FULL_REFERENCE_ADAPTERS if adapter["name"] == metric), {})
    return {
        "name": metric,
        "status": status,
        "score": None,
        "install_guidance": adapter_info.get("install_guidance"),
        "error": error,
    }


def _failed_eval_result(
    *,
    item_id: str,
    reference_path: Path,
    degraded_path: Path,
    enhanced_path: Path,
    backend: str,
    degraded_recipe: dict[str, object],
    failure_stage: str,
    exc: Exception,
    performance: dict[str, object],
) -> dict[str, object]:
    failure = {
        "stage": failure_stage,
        "type": exc.__class__.__name__,
        "message": str(exc),
    }
    stability = _failed_stability_report(failure=failure, enhanced_path=enhanced_path)
    return {
        "id": item_id,
        "status": "failed",
        "failure": failure,
        "input_path": str(degraded_path),
        "reference_path": str(reference_path),
        "degraded_path": str(degraded_path),
        "enhanced_path": str(enhanced_path),
        "backend": backend,
        "degrader": degraded_recipe,
        "metrics": {},
        "optional_metric_records": [],
        "quality": None,
        "stability": stability,
        "failure_cases": stability["failure_cases"],
        "performance": performance,
    }


def _performance_report(
    *,
    backend_init_seconds: float,
    elapsed_seconds: float,
    audio_duration_seconds: float,
    memory_before: dict[str, object],
    memory_after: dict[str, object],
) -> dict[str, object]:
    peak_rss_mb = memory_after.get("peak_rss_mb")
    start_peak_rss_mb = memory_before.get("peak_rss_mb")
    peak_rss_delta_mb = None
    if isinstance(peak_rss_mb, int | float) and isinstance(start_peak_rss_mb, int | float):
        peak_rss_delta_mb = max(float(peak_rss_mb) - float(start_peak_rss_mb), 0.0)

    return {
        "load_time_seconds": backend_init_seconds,
        "backend_init_seconds": backend_init_seconds,
        "elapsed_seconds": elapsed_seconds,
        "total_elapsed_seconds": backend_init_seconds + elapsed_seconds,
        "audio_duration_seconds": audio_duration_seconds,
        "rtf": elapsed_seconds / audio_duration_seconds if audio_duration_seconds > 0 else None,
        "memory": {
            "strategy": memory_after["strategy"],
            "platform": memory_after["platform"],
            "available": memory_after["available"],
            "start_peak_rss_mb": start_peak_rss_mb,
            "peak_rss_mb": peak_rss_mb,
            "peak_rss_delta_mb": peak_rss_delta_mb,
            "unit_note": memory_after["unit_note"],
            "fallback": memory_after["fallback"],
        },
        "peak_rss_mb": peak_rss_mb,
        "peak_rss_delta_mb": peak_rss_delta_mb,
    }


def _memory_snapshot() -> dict[str, object]:
    return peak_rss_snapshot()


def _stability_report(
    *,
    quality: dict[str, str | int | float | bool | list[str] | None],
    reference_audio: np.ndarray,
    degraded_audio: np.ndarray,
    enhanced_audio: np.ndarray,
) -> dict[str, object]:
    sample_rate = quality.get("sample_rate")
    expected_sample_rate = quality.get("expected_sample_rate")
    duration_drift_seconds = quality.get("duration_drift_seconds")
    issues = quality.get("issues")
    issue_list = [str(issue) for issue in issues] if isinstance(issues, list) else []

    sample_rate_correct = expected_sample_rate is None or sample_rate == expected_sample_rate
    duration_drift_exceeded = any(issue.startswith("duration drift") for issue in issue_list)
    clipped_samples = quality.get("clipped_samples")
    clipped = isinstance(clipped_samples, int) and clipped_samples > 0

    failure_case_classification = _classify_failure_cases(
        reference_audio=reference_audio,
        degraded_audio=degraded_audio,
        enhanced_audio=enhanced_audio,
        sample_rate_correct=sample_rate_correct,
        duration_drift_exceeded=duration_drift_exceeded,
        clipped=clipped,
    )
    failed_cases = sorted(name for name, failed in failure_case_classification.items() if failed)
    return {
        "passed": bool(quality.get("passed")) and not failed_cases,
        "failure_status": "none" if not failed_cases else "stability_failed",
        "sample_rate_correct": sample_rate_correct,
        "duration_drift_seconds": duration_drift_seconds,
        "duration_drift_exceeded": duration_drift_exceeded,
        "clipped_fraction": quality.get("clipped_fraction"),
        "clipped": clipped,
        "issues": issue_list,
        "failure_cases": failed_cases,
        "failure_case_classification": failure_case_classification,
    }


def _failed_stability_report(*, failure: dict[str, str], enhanced_path: Path) -> dict[str, object]:
    output_missing = not enhanced_path.exists()
    failure_case_classification = {
        "backend_failure": True,
        "output_missing": output_missing,
        "sample_rate_incorrect": False,
        "duration_drift": False,
        "clipping": False,
        "silence_hallucination": False,
        "low_volume_overamplification": False,
        "channel_count_changed": False,
    }
    failed_cases = sorted(name for name, failed in failure_case_classification.items() if failed)
    return {
        "passed": False,
        "failure_status": f"{failure['stage']}_failed",
        "failure": failure,
        "sample_rate_correct": False,
        "duration_drift_seconds": None,
        "duration_drift_exceeded": False,
        "clipped_fraction": None,
        "clipped": False,
        "issues": [failure["message"]],
        "failure_cases": failed_cases,
        "failure_case_classification": failure_case_classification,
    }


def _classify_failure_cases(
    *,
    reference_audio: np.ndarray,
    degraded_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    sample_rate_correct: bool,
    duration_drift_exceeded: bool,
    clipped: bool,
) -> dict[str, bool]:
    degraded_rms = _rms(degraded_audio)
    enhanced_rms = _rms(enhanced_audio)
    reference_channels = _channel_count(reference_audio)
    enhanced_channels = _channel_count(enhanced_audio)
    return {
        "sample_rate_incorrect": not sample_rate_correct,
        "duration_drift": duration_drift_exceeded,
        "clipping": clipped,
        "silence_hallucination": degraded_rms <= SILENCE_RMS_THRESHOLD and enhanced_rms > HALLUCINATION_RMS_THRESHOLD,
        "low_volume_overamplification": 0 < degraded_rms <= LOW_VOLUME_RMS_THRESHOLD
        and enhanced_rms >= OVERAMPLIFICATION_OUTPUT_RMS_THRESHOLD
        and enhanced_rms / degraded_rms >= OVERAMPLIFICATION_GAIN_THRESHOLD,
        "channel_count_changed": reference_channels != enhanced_channels,
    }


def _rms(audio: np.ndarray) -> float:
    audio_array = np.asarray(audio, dtype=np.float64)
    if audio_array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_array))))


def _channel_count(audio: np.ndarray) -> int:
    audio_array = np.asarray(audio)
    if audio_array.ndim < 2:
        return 1
    return int(audio_array.shape[1])


def _run_eval_item(
    *,
    reference_path: Path,
    dataset_dir: Path,
    backend: str,
    target_sample_rate: int,
    degrader: str,
    degraded_dir: Path,
    enhanced_dir: Path,
    config: InferenceConfig,
    optional_metrics: tuple[str, ...],
) -> dict[str, object]:
    reference_audio, reference_sample_rate = sf.read(reference_path, always_2d=True)
    degraded = degrade_audio(reference_audio, reference_sample_rate, degrader)
    item_id = reference_path.relative_to(dataset_dir).with_suffix("").as_posix()
    safe_name = item_id.replace("/", "__")
    degraded_path = degraded_dir / f"{safe_name}.wav"
    enhanced_path = enhanced_dir / f"{safe_name}.wav"
    sf.write(degraded_path, degraded.audio, degraded.sample_rate)

    init_started_at = time.perf_counter()
    memory_before = _memory_snapshot()
    try:
        resolver = AudioSuperResolver(target_sr=target_sample_rate, backend=backend, config=config)
    except (OSError, RuntimeError, ValueError) as exc:
        backend_init_seconds = time.perf_counter() - init_started_at
        return _failed_eval_result(
            item_id=item_id,
            reference_path=reference_path,
            degraded_path=degraded_path,
            enhanced_path=enhanced_path,
            backend=backend,
            degraded_recipe=degraded.recipe,
            failure_stage="init",
            exc=exc,
            performance=_performance_report(
                backend_init_seconds=backend_init_seconds,
                elapsed_seconds=0.0,
                audio_duration_seconds=degraded.audio.shape[0] / degraded.sample_rate,
                memory_before=memory_before,
                memory_after=_memory_snapshot(),
            ),
        )
    backend_init_seconds = time.perf_counter() - init_started_at
    started_at = time.perf_counter()
    try:
        result = resolver.enhance(degraded_path, enhanced_path, target_sr=target_sample_rate)
    except (OSError, RuntimeError, ValueError) as exc:
        elapsed_seconds = time.perf_counter() - started_at
        return _failed_eval_result(
            item_id=item_id,
            reference_path=reference_path,
            degraded_path=degraded_path,
            enhanced_path=enhanced_path,
            backend=backend,
            degraded_recipe=degraded.recipe,
            failure_stage="enhance",
            exc=exc,
            performance=_performance_report(
                backend_init_seconds=backend_init_seconds,
                elapsed_seconds=elapsed_seconds,
                audio_duration_seconds=degraded.audio.shape[0] / degraded.sample_rate,
                memory_before=memory_before,
                memory_after=_memory_snapshot(),
            ),
        )
    elapsed_seconds = time.perf_counter() - started_at

    try:
        enhanced_audio, enhanced_sample_rate = sf.read(enhanced_path, always_2d=True)
        comparison_reference = reference_audio
        if reference_sample_rate != enhanced_sample_rate:
            comparison_reference = _resample(reference_audio, reference_sample_rate, enhanced_sample_rate)

        metrics = full_reference_metrics(enhanced_audio, comparison_reference, sample_rate=enhanced_sample_rate)
        optional_scores, optional_records = optional_full_reference_metrics(
            enhanced_audio,
            comparison_reference,
            sample_rate=enhanced_sample_rate,
            metrics=optional_metrics,
        )
        metrics.update(optional_scores)
        quality = inspect_audio_quality(
            enhanced_path,
            expected_sample_rate=target_sample_rate,
            expected_duration_seconds=reference_audio.shape[0] / reference_sample_rate,
        )
        quality_dict = quality_report_to_dict(quality)
        stability = _stability_report(
            quality=quality_dict,
            reference_audio=reference_audio,
            degraded_audio=degraded.audio,
            enhanced_audio=enhanced_audio,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return _failed_eval_result(
            item_id=item_id,
            reference_path=reference_path,
            degraded_path=degraded_path,
            enhanced_path=enhanced_path,
            backend=backend,
            degraded_recipe=degraded.recipe,
            failure_stage="inspect",
            exc=exc,
            performance=_performance_report(
                backend_init_seconds=backend_init_seconds,
                elapsed_seconds=elapsed_seconds,
                audio_duration_seconds=result.duration_seconds,
                memory_before=memory_before,
                memory_after=_memory_snapshot(),
            ),
        )
    status = "passed" if stability["passed"] else "stability_failed"
    return {
        "id": item_id,
        "status": status,
        "failure": None,
        "input_path": str(degraded_path),
        "reference_path": str(reference_path),
        "degraded_path": str(degraded_path),
        "enhanced_path": str(enhanced_path),
        "backend": backend,
        "degrader": degraded.recipe,
        "metrics": metrics,
        "optional_metric_records": optional_records,
        "quality": quality_dict,
        "stability": stability,
        "failure_cases": stability["failure_cases"],
        "performance": _performance_report(
            backend_init_seconds=backend_init_seconds,
            elapsed_seconds=elapsed_seconds,
            audio_duration_seconds=result.duration_seconds,
            memory_before=memory_before,
            memory_after=_memory_snapshot(),
        ),
    }


def _reference_files(dataset_path: Path, *, limit: int | None) -> list[Path]:
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    references = sorted(path for path in dataset_path.rglob("*.wav") if path.is_file())
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        return references[:limit]
    return references


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(audio)
    divisor = np.gcd(source_sr, target_sr)
    up = target_sr // divisor
    down = source_sr // divisor
    return resample_poly(audio, up, down, axis=0).astype(np.float32, copy=False)


def _deterministic_noise(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, scale, size=shape).astype(np.float32)


def _codec_like_quantize(audio: np.ndarray, *, levels: int) -> np.ndarray:
    clipped = np.clip(audio, -1.0, 1.0)
    return (np.round(clipped * levels) / levels).astype(np.float32, copy=False)


def _synthetic_speech_like_audio(
    *,
    sample_rate: int,
    duration_seconds: float,
    base_frequency: float,
    seed: int,
    condition: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = int(sample_rate * duration_seconds)
    time_axis = np.arange(frames, dtype=np.float64) / sample_rate
    envelope = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0.0, 1.0, frames))
    modulation = 0.65 + 0.35 * np.sin(2 * np.pi * 5.0 * time_axis + seed)
    audio = np.zeros(frames, dtype=np.float64)
    for harmonic, gain in enumerate((1.0, 0.45, 0.24, 0.12, 0.06), start=1):
        audio += gain * np.sin(2 * np.pi * base_frequency * harmonic * time_axis + harmonic * 0.13 * seed)
    audio *= envelope * modulation * 0.11
    audio += 0.015 * np.sin(2 * np.pi * min(9000.0 + 137.0 * seed, sample_rate * 0.45) * time_axis)
    if condition == "noisy":
        audio += rng.normal(0.0, 0.003, size=frames)
    elif condition == "reverb":
        delay = max(int(sample_rate * 0.018), 1)
        echo = np.zeros_like(audio)
        echo[delay:] = audio[:-delay] * 0.28
        audio += echo
    return np.clip(audio, -0.95, 0.95).astype(np.float32)


def _mixdown(audio: np.ndarray) -> np.ndarray:
    audio_array = np.asarray(audio, dtype=np.float64)
    if audio_array.ndim == 1:
        return audio_array
    return np.mean(audio_array, axis=1)


def _align(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = min(first.shape[0], second.shape[0])
    return first[:frames], second[:frames]


def _stft_magnitude(audio: np.ndarray, *, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float64)
    if audio.shape[0] < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.shape[0]))
    starts = range(0, max(audio.shape[0] - n_fft + 1, 1), hop_length)
    window = np.hanning(n_fft)
    spectra = [np.abs(np.fft.rfft(audio[start : start + n_fft] * window)) for start in starts]
    return np.stack(spectra, axis=1)


def _si_sdr_db(enhanced: np.ndarray, reference: np.ndarray) -> float:
    enhanced = enhanced - np.mean(enhanced)
    reference = reference - np.mean(reference)
    reference_energy = float(np.sum(reference**2))
    if reference_energy <= 1e-12:
        return 0.0
    projection = np.sum(enhanced * reference) * reference / reference_energy
    noise = enhanced - projection
    return _power_ratio_db(np.sum(projection**2), np.sum(noise**2))


def _sdr_db(enhanced: np.ndarray, reference: np.ndarray) -> float:
    error = enhanced - reference
    return _power_ratio_db(np.sum(reference**2), np.sum(error**2))


def _power_ratio_db(numerator: float | np.floating[Any], denominator: float | np.floating[Any]) -> float:
    return float(10 * np.log10((float(numerator) + 1e-12) / (float(denominator) + 1e-12)))


def _lsd_db(enhanced_mag: np.ndarray, reference_mag: np.ndarray) -> float:
    return _lsd_from_bins(enhanced_mag, reference_mag, np.ones(enhanced_mag.shape[0], dtype=bool))


def _band_lsd_db(
    enhanced_mag: np.ndarray,
    reference_mag: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> float | None:
    frequencies = np.fft.rfftfreq((enhanced_mag.shape[0] - 1) * 2, d=1 / sample_rate)
    mask = (frequencies >= low_hz) & (frequencies < min(high_hz, sample_rate / 2))
    if not np.any(mask):
        return None
    return _lsd_from_bins(enhanced_mag, reference_mag, mask)


def _lsd_from_bins(enhanced_mag: np.ndarray, reference_mag: np.ndarray, mask: np.ndarray) -> float:
    frames = min(enhanced_mag.shape[1], reference_mag.shape[1])
    enhanced_db = 20 * np.log10(np.clip(enhanced_mag[mask, :frames], 1e-8, None))
    reference_db = 20 * np.log10(np.clip(reference_mag[mask, :frames], 1e-8, None))
    return float(np.sqrt(np.mean((enhanced_db - reference_db) ** 2)))


def _spectral_convergence(enhanced_mag: np.ndarray, reference_mag: np.ndarray) -> float:
    frames = min(enhanced_mag.shape[1], reference_mag.shape[1])
    diff = enhanced_mag[:, :frames] - reference_mag[:, :frames]
    return float(np.linalg.norm(diff) / max(np.linalg.norm(reference_mag[:, :frames]), 1e-12))


def _indexed_results(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    records = manifest.get("results")
    if not isinstance(records, list):
        return {}
    return {str(record.get("id", index)): record for index, record in enumerate(records) if isinstance(record, dict)}


def _metric_summary(
    baseline_results: dict[str, dict[str, object]],
    candidate_results: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    metric_names = sorted(
        {
            metric
            for record in [*baseline_results.values(), *candidate_results.values()]
            for metric in _numeric_eval_values(record)
        }
    )
    common_ids = sorted(set(baseline_results) & set(candidate_results))
    for metric in metric_names:
        baseline_values = [_numeric_eval_values(baseline_results[item_id]).get(metric) for item_id in common_ids]
        candidate_values = [_numeric_eval_values(candidate_results[item_id]).get(metric) for item_id in common_ids]
        paired = [
            (baseline, candidate)
            for baseline, candidate in zip(baseline_values, candidate_values, strict=False)
            if baseline is not None and candidate is not None
        ]
        if not paired:
            continue
        baseline_mean = float(np.mean([value[0] for value in paired]))
        candidate_mean = float(np.mean([value[1] for value in paired]))
        summary[metric] = {
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "delta": candidate_mean - baseline_mean,
            "direction": _metric_direction(metric),
        }
    return summary


def _comparison_tables(
    baseline: dict[str, object],
    candidate: dict[str, object],
    metric_summary: dict[str, dict[str, object]],
    candidate_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "audio_quality": _summary_subset(
            metric_summary,
            set(FULL_REFERENCE_METRICS) | set(OPTIONAL_FULL_REFERENCE_METRICS),
        ),
        "no_reference": _summary_subset(metric_summary, set(SIGNAL_STATS_NO_REFERENCE_METRICS)),
        "downstream": _summary_subset(metric_summary, set(DOWNSTREAM_METRICS)),
        "engineering": _summary_subset(metric_summary, set(ENGINEERING_METRICS)),
        "stability": {
            "metrics": _summary_subset(metric_summary, set(STABILITY_METRICS)),
            "candidate_status_counts": _status_counts(list(candidate_results.values())),
            "candidate_failure_cases": _failure_case_counts(candidate_results),
        },
        "governance": {
            "baseline": _governance_table(baseline.get("backend_profile")),
            "candidate": _governance_table(candidate.get("backend_profile")),
        },
    }


def _eval_regressions(
    baseline_results: dict[str, dict[str, object]],
    candidate_results: dict[str, dict[str, object]],
    *,
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    regressions: list[dict[str, object]] = []
    for item_id in sorted(set(baseline_results) - set(candidate_results)):
        regressions.append({"id": item_id, "field": "results", "message": "missing candidate result"})
    for item_id in sorted(set(baseline_results) & set(candidate_results)):
        candidate_status = candidate_results[item_id].get("status")
        if candidate_status is not None and candidate_status != "passed":
            regressions.append(
                {
                    "id": item_id,
                    "field": "status",
                    "message": f"candidate eval result status is {candidate_status}",
                }
            )
        candidate_stability = candidate_results[item_id].get("stability")
        if isinstance(candidate_stability, dict) and candidate_stability.get("passed") is False:
            regressions.append({"id": item_id, "field": "stability", "message": "candidate stability checks failed"})
        candidate_quality = candidate_results[item_id].get("quality")
        if isinstance(candidate_quality, dict) and candidate_quality.get("passed") is False:
            regressions.append({"id": item_id, "field": "quality", "message": "candidate quality checks failed"})
        baseline_values = _numeric_eval_values(baseline_results[item_id])
        candidate_values = _numeric_eval_values(candidate_results[item_id])
        for metric, allowed_regression in thresholds.items():
            if metric in baseline_values and metric in candidate_values:
                delta = candidate_values[metric] - baseline_values[metric]
                if _threshold_regressed(metric, delta, allowed_regression):
                    regressions.append(
                        {
                            "id": item_id,
                            "field": metric,
                            "delta": delta,
                            "allowed_regression": allowed_regression,
                            "direction": _metric_direction(metric),
                            "message": f"{metric} regressed by more than {allowed_regression}",
                        }
                    )
    return regressions


def _numeric_eval_values(record: dict[str, object]) -> dict[str, float]:
    values = _numeric_metrics(record)
    values.update(_numeric_scores(record))
    values.update(_numeric_nested_values(record.get("performance"), ENGINEERING_METRICS))
    quality_values = _numeric_nested_values(record.get("quality"), STABILITY_METRICS)
    stability_values = _numeric_nested_values(record.get("stability"), STABILITY_METRICS)
    values.update(quality_values)
    values.update(stability_values)
    return values


def _numeric_metrics(record: dict[str, object]) -> dict[str, float]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        str(name): float(value)
        for name, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _numeric_scores(record: dict[str, object]) -> dict[str, float]:
    scores = record.get("scores")
    if not isinstance(scores, dict):
        return {}
    return {
        str(name): float(value)
        for name, value in scores.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _numeric_nested_values(container: object, names: tuple[str, ...]) -> dict[str, float]:
    if not isinstance(container, dict):
        return {}
    return {
        name: float(value)
        for name in names
        if isinstance((value := container.get(name)), int | float) and not isinstance(value, bool)
    }


def _summary_subset(
    metric_summary: dict[str, dict[str, object]],
    names: set[str],
) -> dict[str, dict[str, object]]:
    return {name: metric_summary[name] for name in sorted(names & set(metric_summary))}


def _failure_case_counts(candidate_results: dict[str, dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in candidate_results.values():
        failure_cases = result.get("failure_cases")
        if not isinstance(failure_cases, list):
            continue
        for failure_case in failure_cases:
            name = str(failure_case)
            counts[name] = counts.get(name, 0) + 1
    return counts


def _governance_table(profile: object) -> dict[str, object]:
    if not isinstance(profile, dict):
        return {}
    capabilities = profile.get("capabilities")
    governance = profile.get("governance")
    dependency_footprint = profile.get("dependency_footprint")
    return {
        "backend": profile.get("backend"),
        "model_id": profile.get("model_id"),
        "offline": capabilities.get("offline") if isinstance(capabilities, dict) else None,
        "reproducible": capabilities.get("reproducible") if isinstance(capabilities, dict) else None,
        "license_usable": governance.get("license_usable") if isinstance(governance, dict) else None,
        "explicit_weights": governance.get("explicit_weights") if isinstance(governance, dict) else None,
        "dependency_footprint": dependency_footprint if isinstance(dependency_footprint, dict) else None,
    }


def _threshold_regressed(metric: str, delta: float, allowed_regression: float) -> bool:
    if allowed_regression < 0:
        raise ValueError("threshold values must be non-negative")
    if _metric_direction(metric) == "lower_is_better":
        return delta > allowed_regression
    return delta < -allowed_regression


def _metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER_METRICS:
        return "lower_is_better"
    return "higher_is_better"


def _status_counts(results: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _model_name_for_backend(backend: str, config: InferenceConfig) -> str | None:
    if backend == "audiosr":
        return config.model_name
    return None


def _package_extra_for_backend(backend: str) -> str | None:
    for info in available_backends():
        if info.name == backend:
            return info.package_extra
    return None


def _optional_dependency_for_backend(backend: str) -> str | None:
    for info in available_backends():
        if info.name == backend:
            return info.optional_dependency
    return None


def _total_weight_size_bytes(spec: ModelSpec) -> int | None:
    if not spec.weight_files:
        return None
    sizes = [weight_file.size for weight_file in spec.weight_files]
    if any(size is None for size in sizes):
        return None
    return sum(size for size in sizes if size is not None)


def _has_explicit_weight_governance(spec: ModelSpec) -> bool:
    if not spec.requires_weights:
        return True
    if spec.weights_hash:
        return True
    return bool(spec.weight_files) and _all_weight_hashes_present(spec)


def _all_weight_hashes_present(spec: ModelSpec) -> bool:
    return bool(spec.weight_files) and all(weight_file.sha256 for weight_file in spec.weight_files)


def _license_usable(weights_license: str | None, *, requires_weights: bool) -> bool:
    if not requires_weights:
        return True
    if weights_license is None:
        return False
    return weights_license.strip().lower() not in {"", "unknown", "unspecified"}


def _offline_capable(spec: ModelSpec) -> bool:
    if not spec.requires_weights:
        return True
    if spec.implementation == "external_package":
        return False
    return _has_explicit_weight_governance(spec)


def _cpu_only(capability: BackendCapability | None) -> bool:
    if capability is None:
        return False
    return (
        capability.supports_cpu
        and not capability.supports_cuda
        and not capability.supports_mps
        and not capability.requires_gpu
    )


def _dependency_tier(implementation: str, *, requires_weights: bool) -> str:
    if implementation == "baseline" and not requires_weights:
        return "baseline-no-weights"
    if implementation == "external_package":
        return "external-heavy"
    if requires_weights:
        return "optional-model-weights"
    return "optional-light"
