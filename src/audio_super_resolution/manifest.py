from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .config import InferenceConfig
from .quality import AudioQualityReport
from .resolver import EnhancementResult, PlannedEnhancement


def build_manifest(
    mode: str,
    jobs: list[PlannedEnhancement],
    config: InferenceConfig,
    backend: str,
    target_sample_rate: int,
    results: list[EnhancementResult] | None = None,
    quality_reports: list[AudioQualityReport] | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable manifest for planned or completed enhancements."""

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "backend": backend,
        "target_sample_rate": target_sample_rate,
        "config": config.as_dict(),
        "jobs": [_planned_enhancement_to_dict(job) for job in jobs],
        "results": [_enhancement_result_to_dict(result) for result in results or []],
        "quality_reports": [_quality_report_to_dict(report) for report in quality_reports or []],
    }


def write_manifest(path: str | Path, manifest: dict[str, object]) -> Path:
    """Write a manifest to disk and return the resolved path."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _planned_enhancement_to_dict(job: PlannedEnhancement) -> dict[str, str]:
    return {
        "input_path": str(job.input_path),
        "output_path": str(job.output_path),
    }


def _enhancement_result_to_dict(result: EnhancementResult) -> dict[str, str | int | float]:
    return {
        "input_path": str(result.input_path),
        "output_path": str(result.output_path),
        "input_sample_rate": result.input_sample_rate,
        "sample_rate": result.sample_rate,
        "input_duration_seconds": result.input_duration_seconds,
        "duration_seconds": result.duration_seconds,
        "channels": result.channels,
        "backend": result.backend,
    }


def _quality_report_to_dict(report: AudioQualityReport) -> dict[str, str | int | float | bool | list[str] | None]:
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
