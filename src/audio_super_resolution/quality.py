from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioQualityReport:
    """Objective checks for a rendered audio file."""

    path: Path
    sample_rate: int
    duration_seconds: float
    channels: int
    peak_level: float
    clipped_samples: int
    clipped_fraction: float
    expected_sample_rate: int | None
    expected_duration_seconds: float | None
    duration_drift_seconds: float | None
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues


def inspect_audio_quality(
    path: str | Path,
    expected_sample_rate: int | None = None,
    expected_duration_seconds: float | None = None,
    max_duration_drift_seconds: float = 0.05,
    clipping_threshold: float = 0.999,
    max_peak_level: float = 1.0,
) -> AudioQualityReport:
    """Inspect sample rate, duration drift, clipping, and peak level."""

    audio_path = Path(path)
    audio, sample_rate = sf.read(audio_path, always_2d=True)

    duration_seconds = audio.shape[0] / sample_rate
    absolute = np.abs(audio)
    peak_level = float(absolute.max()) if absolute.size else 0.0
    clipped_samples = int(np.count_nonzero(absolute >= clipping_threshold))
    clipped_fraction = clipped_samples / absolute.size if absolute.size else 0.0
    duration_drift_seconds = (
        abs(duration_seconds - expected_duration_seconds) if expected_duration_seconds is not None else None
    )

    issues: list[str] = []
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        issues.append(f"sample rate {sample_rate} does not match expected {expected_sample_rate}")
    if duration_drift_seconds is not None and duration_drift_seconds > max_duration_drift_seconds:
        issues.append(f"duration drift {duration_drift_seconds:.6f}s exceeds {max_duration_drift_seconds:.6f}s")
    if clipped_samples:
        issues.append(f"{clipped_samples} samples are at or above clipping threshold {clipping_threshold}")
    if peak_level > max_peak_level:
        issues.append(f"peak level {peak_level:.6f} exceeds {max_peak_level:.6f}")

    return AudioQualityReport(
        path=audio_path,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        channels=audio.shape[1],
        peak_level=peak_level,
        clipped_samples=clipped_samples,
        clipped_fraction=clipped_fraction,
        expected_sample_rate=expected_sample_rate,
        expected_duration_seconds=expected_duration_seconds,
        duration_drift_seconds=duration_drift_seconds,
        issues=tuple(issues),
    )


def format_quality_report(report: AudioQualityReport) -> str:
    """Return a compact one-line quality report for CLI output."""

    status = "ok" if report.passed else "issues"
    drift = "n/a" if report.duration_drift_seconds is None else f"{report.duration_drift_seconds:.6f}s"
    return (
        f"{report.path}: {status}; sample_rate={report.sample_rate}; duration={report.duration_seconds:.3f}s; "
        f"drift={drift}; peak={report.peak_level:.6f}; clipped={report.clipped_samples}"
    )


def quality_report_to_dict(report: AudioQualityReport) -> dict[str, str | int | float | bool | list[str] | None]:
    """Return a JSON-friendly quality report."""

    return {
        "path": str(report.path),
        "sample_rate": report.sample_rate,
        "duration_seconds": report.duration_seconds,
        "channels": report.channels,
        "peak_level": report.peak_level,
        "clipped_samples": report.clipped_samples,
        "clipped_fraction": report.clipped_fraction,
        "expected_sample_rate": report.expected_sample_rate,
        "expected_duration_seconds": report.expected_duration_seconds,
        "duration_drift_seconds": report.duration_drift_seconds,
        "issues": list(report.issues),
        "passed": report.passed,
    }


def build_quality_report_bundle(reports: list[AudioQualityReport]) -> dict[str, object]:
    """Build a combined JSON report for one or more audio quality checks."""

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "passed": all(report.passed for report in reports),
        "report_count": len(reports),
        "issue_count": sum(len(report.issues) for report in reports),
        "reports": [quality_report_to_dict(report) for report in reports],
    }


def write_quality_report_bundle(path: str | Path, reports: list[AudioQualityReport]) -> Path:
    """Write a combined JSON quality report and return the path."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(build_quality_report_bundle(reports), indent=2), encoding="utf-8")
    return report_path
