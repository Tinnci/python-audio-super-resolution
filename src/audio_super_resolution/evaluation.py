from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import InferenceConfig
from .preprocess import lowpass_filter
from .quality import inspect_audio_quality, quality_report_to_dict
from .resolver import AudioSuperResolver

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

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_dir),
        "backend": backend,
        "target_sample_rate": target_sample_rate,
        "degrader": {"name": degrader},
        "config": config.as_dict(),
        "metric_groups": {
            "full_reference": list(FULL_REFERENCE_METRICS),
            "optional_full_reference": list(OPTIONAL_FULL_REFERENCE_METRICS),
            "engineering": ["elapsed_seconds", "rtf"],
            "stability": ["sample_rate", "duration_drift_seconds", "clipped_fraction", "passed"],
        },
        "results": results,
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

    resolver = AudioSuperResolver(target_sr=target_sample_rate, backend=backend, config=config)
    started_at = time.perf_counter()
    result = resolver.enhance(degraded_path, enhanced_path, target_sr=target_sample_rate)
    elapsed_seconds = time.perf_counter() - started_at

    enhanced_audio, enhanced_sample_rate = sf.read(enhanced_path, always_2d=True)
    comparison_reference = reference_audio
    if reference_sample_rate != enhanced_sample_rate:
        comparison_reference = _resample(reference_audio, reference_sample_rate, enhanced_sample_rate)

    quality = inspect_audio_quality(
        enhanced_path,
        expected_sample_rate=target_sample_rate,
        expected_duration_seconds=reference_audio.shape[0] / reference_sample_rate,
    )
    return {
        "id": item_id,
        "input_path": str(degraded_path),
        "reference_path": str(reference_path),
        "degraded_path": str(degraded_path),
        "enhanced_path": str(enhanced_path),
        "backend": backend,
        "degrader": degraded.recipe,
        "metrics": full_reference_metrics(enhanced_audio, comparison_reference, sample_rate=enhanced_sample_rate),
        "quality": quality_report_to_dict(quality),
        "performance": {
            "elapsed_seconds": elapsed_seconds,
            "audio_duration_seconds": result.duration_seconds,
            "rtf": elapsed_seconds / result.duration_seconds if result.duration_seconds > 0 else None,
        },
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
