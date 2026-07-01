from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .backends import available_backends
from .config import InferenceConfig
from .models import find_model_spec
from .preprocess import lowpass_filter
from .quality import inspect_audio_quality, quality_report_to_dict
from .resolver import AudioSuperResolver
from .specs import BackendCapability, ModelSpec

_resource_module: Any
try:
    import resource as _resource_module
except ImportError:  # pragma: no cover - exercised on platforms without resource.
    _resource_module = None

_resource: Any | None = _resource_module

SUPPORTED_DEGRADERS = (
    "lowpass_4k",
    "narrowband_8k",
    "wideband_16k",
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
SILENCE_RMS_THRESHOLD = 1e-5
HALLUCINATION_RMS_THRESHOLD = 1e-4
LOW_VOLUME_RMS_THRESHOLD = 0.01
OVERAMPLIFICATION_GAIN_THRESHOLD = 12.0
OVERAMPLIFICATION_OUTPUT_RMS_THRESHOLD = 0.05


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
    )
    write_eval_manifest(output, manifest)
    return manifest


def build_eval_manifest(
    *,
    dataset_dir: Path,
    backend: str,
    target_sample_rate: int,
    degrader: str,
    config: InferenceConfig,
    results: list[dict[str, object]],
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
        "metric_groups": {
            "full_reference": list(FULL_REFERENCE_METRICS),
            "optional_full_reference": list(OPTIONAL_FULL_REFERENCE_METRICS),
            "engineering": [
                "backend_init_seconds",
                "elapsed_seconds",
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
        "backend_init_seconds": backend_init_seconds,
        "elapsed_seconds": elapsed_seconds,
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
    system = platform.system()
    if _resource is None:
        return {
            "strategy": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "platform": system,
            "available": False,
            "peak_rss_mb": None,
            "unit_note": None,
            "fallback": "peak RSS unavailable because the platform does not provide the resource module",
        }

    usage = _resource.getrusage(_resource.RUSAGE_SELF)
    raw_peak_rss = float(usage.ru_maxrss)
    if system == "Darwin":
        peak_rss_mb = raw_peak_rss / (1024 * 1024)
        unit_note = "ru_maxrss reports bytes on Darwin"
    else:
        peak_rss_mb = raw_peak_rss / 1024
        unit_note = "ru_maxrss reports kilobytes on Linux and most Unix platforms"

    return {
        "strategy": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "platform": system,
        "available": True,
        "peak_rss_mb": peak_rss_mb,
        "unit_note": unit_note,
        "fallback": None,
    }


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
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    metric_names = sorted(
        {
            metric
            for record in [*baseline_results.values(), *candidate_results.values()]
            for metric in _numeric_metrics(record)
        }
    )
    common_ids = sorted(set(baseline_results) & set(candidate_results))
    for metric in metric_names:
        baseline_values = [_numeric_metrics(baseline_results[item_id]).get(metric) for item_id in common_ids]
        candidate_values = [_numeric_metrics(candidate_results[item_id]).get(metric) for item_id in common_ids]
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
        }
    return summary


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
        baseline_metrics = _numeric_metrics(baseline_results[item_id])
        candidate_metrics = _numeric_metrics(candidate_results[item_id])
        for metric, allowed_drop in thresholds.items():
            if metric in baseline_metrics and metric in candidate_metrics:
                delta = candidate_metrics[metric] - baseline_metrics[metric]
                if delta < -allowed_drop:
                    regressions.append(
                        {
                            "id": item_id,
                            "field": metric,
                            "delta": delta,
                            "message": f"{metric} regressed by more than {allowed_drop}",
                        }
                    )
    return regressions


def _numeric_metrics(record: dict[str, object]) -> dict[str, float]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        str(name): float(value)
        for name, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


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
