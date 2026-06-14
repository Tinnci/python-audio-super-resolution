from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class GoldenThresholds:
    """Tolerances used when comparing backend output against a golden reference."""

    max_duration_drift_seconds: float = 0.05
    max_peak_delta: float = 0.02
    max_rms_delta: float = 0.02
    max_log_mel_l1: float = 0.5
    max_hf_energy_ratio_delta: float = 0.05
    high_frequency_start_hz: float = 8000.0


@dataclass(frozen=True)
class GoldenAudioStats:
    """Stable audio statistics used for golden comparisons."""

    path: Path
    sample_rate: int
    duration_seconds: float
    channels: int
    peak_level: float
    rms_level: float
    high_frequency_energy_ratio: float


@dataclass(frozen=True)
class GoldenComparisonReport:
    """Comparison result for one actual output and one golden reference."""

    actual: GoldenAudioStats
    reference: GoldenAudioStats
    thresholds: GoldenThresholds
    duration_drift_seconds: float
    peak_delta: float
    rms_delta: float
    log_mel_l1: float | None
    hf_energy_ratio_delta: float
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues


def compare_golden_outputs(
    actual_path: str | Path,
    reference_path: str | Path,
    *,
    thresholds: GoldenThresholds | None = None,
) -> GoldenComparisonReport:
    """Compare actual audio output against a golden reference file."""

    resolved_thresholds = thresholds or GoldenThresholds()
    actual = inspect_golden_audio(actual_path, thresholds=resolved_thresholds)
    reference = inspect_golden_audio(reference_path, thresholds=resolved_thresholds)

    duration_drift = abs(actual.duration_seconds - reference.duration_seconds)
    peak_delta = abs(actual.peak_level - reference.peak_level)
    rms_delta = abs(actual.rms_level - reference.rms_level)
    hf_delta = abs(actual.high_frequency_energy_ratio - reference.high_frequency_energy_ratio)
    log_mel_l1 = _log_mel_l1(actual.path, reference.path) if actual.sample_rate == reference.sample_rate else None

    issues = _golden_issues(
        actual=actual,
        reference=reference,
        thresholds=resolved_thresholds,
        duration_drift=duration_drift,
        peak_delta=peak_delta,
        rms_delta=rms_delta,
        log_mel_l1=log_mel_l1,
        hf_delta=hf_delta,
    )
    return GoldenComparisonReport(
        actual=actual,
        reference=reference,
        thresholds=resolved_thresholds,
        duration_drift_seconds=duration_drift,
        peak_delta=peak_delta,
        rms_delta=rms_delta,
        log_mel_l1=log_mel_l1,
        hf_energy_ratio_delta=hf_delta,
        issues=tuple(issues),
    )


def compare_golden_fixture(
    fixture_path: str | Path,
    actual_path: str | Path,
) -> GoldenComparisonReport:
    """Compare actual output against the reference declared by a golden fixture."""

    fixture = load_golden_fixture(fixture_path)
    reference = _resolve_fixture_audio_path(fixture_path, _fixture_section(fixture, "reference")["path"])
    return compare_golden_outputs(
        actual_path,
        reference,
        thresholds=_thresholds_from_fixture(fixture),
    )


def inspect_golden_audio(path: str | Path, *, thresholds: GoldenThresholds | None = None) -> GoldenAudioStats:
    """Inspect audio statistics used by golden-sample comparison."""

    audio_path = Path(path)
    audio, sample_rate = sf.read(audio_path, always_2d=True)
    duration_seconds = audio.shape[0] / sample_rate
    absolute = np.abs(audio)
    peak_level = float(absolute.max()) if absolute.size else 0.0
    rms_level = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    hf_ratio = _high_frequency_energy_ratio(
        _mixdown(audio),
        sample_rate=sample_rate,
        high_frequency_start_hz=(thresholds or GoldenThresholds()).high_frequency_start_hz,
    )
    return GoldenAudioStats(
        path=audio_path,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        channels=audio.shape[1],
        peak_level=peak_level,
        rms_level=rms_level,
        high_frequency_energy_ratio=hf_ratio,
    )


def load_golden_fixture(path: str | Path) -> dict[str, Any]:
    """Load a golden fixture JSON file."""

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("golden fixture root must be a JSON object")
    if loaded.get("schema_version") != 1:
        raise ValueError("golden fixture schema_version must be 1")
    for key in ("id", "backend"):
        if not isinstance(loaded.get(key), str):
            raise ValueError(f"golden fixture field {key!r} must be a string")
    for key in ("input", "reference"):
        _fixture_section(loaded, key)
    return loaded


def golden_report_to_dict(report: GoldenComparisonReport) -> dict[str, Any]:
    """Return a JSON-friendly golden comparison report."""

    return {
        "schema_version": 1,
        "passed": report.passed,
        "issues": list(report.issues),
        "duration_drift_seconds": report.duration_drift_seconds,
        "peak_delta": report.peak_delta,
        "rms_delta": report.rms_delta,
        "log_mel_l1": report.log_mel_l1,
        "hf_energy_ratio_delta": report.hf_energy_ratio_delta,
        "thresholds": asdict(report.thresholds),
        "actual": _stats_to_dict(report.actual),
        "reference": _stats_to_dict(report.reference),
    }


def write_golden_report(path: str | Path, report: GoldenComparisonReport) -> Path:
    """Write a golden comparison report and return the output path."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(golden_report_to_dict(report), indent=2), encoding="utf-8")
    return report_path


def _golden_issues(
    *,
    actual: GoldenAudioStats,
    reference: GoldenAudioStats,
    thresholds: GoldenThresholds,
    duration_drift: float,
    peak_delta: float,
    rms_delta: float,
    log_mel_l1: float | None,
    hf_delta: float,
) -> list[str]:
    issues: list[str] = []
    if actual.sample_rate != reference.sample_rate:
        issues.append(f"sample rate {actual.sample_rate} does not match reference {reference.sample_rate}")
    if duration_drift > thresholds.max_duration_drift_seconds:
        issues.append(f"duration drift {duration_drift:.6f}s exceeds {thresholds.max_duration_drift_seconds:.6f}s")
    if peak_delta > thresholds.max_peak_delta:
        issues.append(f"peak delta {peak_delta:.6f} exceeds {thresholds.max_peak_delta:.6f}")
    if rms_delta > thresholds.max_rms_delta:
        issues.append(f"RMS delta {rms_delta:.6f} exceeds {thresholds.max_rms_delta:.6f}")
    if log_mel_l1 is not None and log_mel_l1 > thresholds.max_log_mel_l1:
        issues.append(f"log-mel L1 {log_mel_l1:.6f} exceeds {thresholds.max_log_mel_l1:.6f}")
    if hf_delta > thresholds.max_hf_energy_ratio_delta:
        issues.append(
            f"high-frequency energy ratio delta {hf_delta:.6f} exceeds {thresholds.max_hf_energy_ratio_delta:.6f}"
        )
    return issues


def _log_mel_l1(actual_path: Path, reference_path: Path) -> float:
    actual, sample_rate = sf.read(actual_path, always_2d=True)
    reference, reference_sample_rate = sf.read(reference_path, always_2d=True)
    if sample_rate != reference_sample_rate:
        raise ValueError("sample rates must match before computing log-mel difference")

    actual_log_mel = _log_mel(_mixdown(actual), sample_rate=sample_rate)
    reference_log_mel = _log_mel(_mixdown(reference), sample_rate=sample_rate)
    mel_bins = min(actual_log_mel.shape[0], reference_log_mel.shape[0])
    frames = min(actual_log_mel.shape[1], reference_log_mel.shape[1])
    if mel_bins == 0 or frames == 0:
        return 0.0
    diff = actual_log_mel[:mel_bins, :frames] - reference_log_mel[:mel_bins, :frames]
    return float(np.mean(np.abs(diff)))


def _log_mel(audio: np.ndarray, *, sample_rate: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((64, 0))

    frequencies, magnitude = _stft_magnitude(audio, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length)
    mel_filters = _mel_filterbank(frequencies, n_mels=64, sample_rate=sample_rate)
    mel = np.empty((mel_filters.shape[0], magnitude.shape[1]), dtype=np.float64)
    for mel_index, weights in enumerate(mel_filters):
        mel[mel_index] = np.sum(weights[:, None] * magnitude, axis=0)
    return np.log(np.clip(mel, 1e-8, None))


def _mel_filterbank(frequencies: np.ndarray, *, n_mels: int, sample_rate: int) -> np.ndarray:
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    filters = np.zeros((n_mels, len(frequencies)))
    for mel_index in range(n_mels):
        lower = hz_points[mel_index]
        center = hz_points[mel_index + 1]
        upper = hz_points[mel_index + 2]
        lower_slope = (frequencies - lower) / max(center - lower, 1e-12)
        upper_slope = (upper - frequencies) / max(upper - center, 1e-12)
        filters[mel_index] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    return filters


def _high_frequency_energy_ratio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    high_frequency_start_hz: float,
) -> float:
    if audio.size == 0:
        return 0.0

    frequencies, magnitude = _stft_magnitude(audio, sample_rate=sample_rate, n_fft=2048, hop_length=512)
    power = np.square(magnitude)
    total_energy = float(np.sum(power))
    if total_energy <= 0:
        return 0.0
    high_energy = float(np.sum(power[frequencies >= high_frequency_start_hz]))
    return high_energy / total_energy


def _mixdown(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _stft_magnitude(
    audio: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    segment_length = min(n_fft, audio.size)
    if segment_length == 0:
        return np.zeros(0), np.zeros((0, 0))

    frame_starts = list(range(0, max(audio.size - segment_length + 1, 1), hop_length))
    if not frame_starts:
        frame_starts = [0]

    window = np.hanning(segment_length)
    frames = np.stack([audio[start : start + segment_length] * window for start in frame_starts])
    spectrum = np.fft.rfft(frames, n=segment_length, axis=1)
    frequencies = np.fft.rfftfreq(segment_length, d=1 / sample_rate)
    return frequencies, np.abs(spectrum).T


def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (np.power(10.0, np.asarray(mel) / 2595.0) - 1.0)


def _stats_to_dict(stats: GoldenAudioStats) -> dict[str, str | int | float]:
    data = asdict(stats)
    data["path"] = str(stats.path)
    return data


def _fixture_section(fixture: dict[str, Any], key: str) -> dict[str, Any]:
    value = fixture.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"golden fixture field {key!r} must be an object")
    if not isinstance(value.get("path"), str):
        raise ValueError(f"golden fixture field {key!r}.path must be a string")
    return value


def _thresholds_from_fixture(fixture: dict[str, Any]) -> GoldenThresholds:
    raw_thresholds = fixture.get("thresholds", {})
    if not isinstance(raw_thresholds, dict):
        raise ValueError("golden fixture field 'thresholds' must be an object")
    allowed = GoldenThresholds.__dataclass_fields__
    return GoldenThresholds(**{key: value for key, value in raw_thresholds.items() if key in allowed})


def _resolve_fixture_audio_path(fixture_path: str | Path, audio_path: str) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return Path(fixture_path).expanduser().parent / path
