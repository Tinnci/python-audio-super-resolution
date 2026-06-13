from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from typing import Any

from .config import InferenceConfig
from .quality import AudioQualityReport
from .resolver import EnhancementResult, PlannedEnhancement


@dataclass(frozen=True)
class ManifestDifference:
    """A single difference found when comparing two run manifests."""

    key: str
    field: str
    expected: object
    actual: object
    message: str


@dataclass(frozen=True)
class ManifestComparison:
    """Result of comparing an expected manifest with an actual manifest."""

    differences: tuple[ManifestDifference, ...]

    @property
    def passed(self) -> bool:
        return not self.differences


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


def load_manifest(path: str | Path) -> dict[str, object]:
    """Load a JSON run manifest from disk."""

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("manifest root must be a JSON object")
    return loaded


def compare_manifests(
    expected: dict[str, object],
    actual: dict[str, object],
    duration_tolerance_seconds: float = 0.05,
    check_output_files: bool = False,
) -> ManifestComparison:
    """Compare two manifests for CI-friendly regression checks."""

    if duration_tolerance_seconds < 0:
        raise ValueError("duration_tolerance_seconds must be greater than or equal to zero")

    differences: list[ManifestDifference] = []

    for field in ("schema_version", "backend", "target_sample_rate"):
        _compare_exact(differences, "manifest", field, expected.get(field), actual.get(field))

    expected_results = _index_records(_dict_records(expected.get("results")), preferred_key="input_path")
    actual_results = _index_records(_dict_records(actual.get("results")), preferred_key="input_path")

    for key in sorted(expected_results.keys() - actual_results.keys()):
        differences.append(
            ManifestDifference(
                key=key,
                field="results",
                expected="present",
                actual="missing",
                message="result is missing from actual manifest",
            )
        )

    for key in sorted(actual_results.keys() - expected_results.keys()):
        differences.append(
            ManifestDifference(
                key=key,
                field="results",
                expected="missing",
                actual="present",
                message="actual manifest contains an unexpected result",
            )
        )

    expected_quality = _index_records(_dict_records(expected.get("quality_reports")), preferred_key="path")
    actual_quality = _index_records(_dict_records(actual.get("quality_reports")), preferred_key="path")

    for key in sorted(expected_results.keys() & actual_results.keys()):
        expected_result = expected_results[key]
        actual_result = actual_results[key]

        for field in ("input_sample_rate", "sample_rate", "channels", "backend"):
            _compare_exact(differences, key, field, expected_result.get(field), actual_result.get(field))

        for field in ("input_duration_seconds", "duration_seconds"):
            _compare_float(
                differences,
                key,
                field,
                expected_result.get(field),
                actual_result.get(field),
                tolerance=duration_tolerance_seconds,
            )

        _compare_output_presence(differences, key, expected_result, actual_result, check_output_files)
        _compare_quality_status(differences, key, expected_result, actual_result, expected_quality, actual_quality)

    return ManifestComparison(differences=tuple(differences))


def manifest_comparison_to_dict(comparison: ManifestComparison) -> dict[str, object]:
    """Return a JSON-friendly comparison summary."""

    return {
        "passed": comparison.passed,
        "difference_count": len(comparison.differences),
        "differences": [asdict(difference) for difference in comparison.differences],
    }


def format_manifest_comparison(comparison: ManifestComparison) -> str:
    """Return a compact human-readable manifest comparison report."""

    if comparison.passed:
        return "Manifest comparison passed."

    lines = [f"Manifest comparison found {len(comparison.differences)} difference(s):"]
    for difference in comparison.differences:
        lines.append(
            f"- {difference.key}: {difference.field}: {difference.message} "
            f"(expected={difference.expected!r}, actual={difference.actual!r})"
        )
    return "\n".join(lines)


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


def _dict_records(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def _index_records(records: list[dict[str, object]], preferred_key: str) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for index, record in enumerate(records):
        key = record.get(preferred_key) or record.get("output_path") or f"#{index}"
        indexed[str(key)] = record
    return indexed


def _compare_exact(
    differences: list[ManifestDifference],
    key: str,
    field: str,
    expected: object,
    actual: object,
) -> None:
    if expected == actual:
        return

    differences.append(
        ManifestDifference(
            key=key,
            field=field,
            expected=expected,
            actual=actual,
            message=f"{field} differs",
        )
    )


def _compare_float(
    differences: list[ManifestDifference],
    key: str,
    field: str,
    expected: object,
    actual: object,
    tolerance: float,
) -> None:
    if _is_number(expected) and _is_number(actual):
        if isclose(float(expected), float(actual), abs_tol=tolerance):
            return
        message = f"{field} differs by more than {tolerance:.6f}s"
    else:
        if expected == actual:
            return
        message = f"{field} differs"

    differences.append(
        ManifestDifference(
            key=key,
            field=field,
            expected=expected,
            actual=actual,
            message=message,
        )
    )


def _compare_output_presence(
    differences: list[ManifestDifference],
    key: str,
    expected_result: dict[str, object],
    actual_result: dict[str, object],
    check_output_files: bool,
) -> None:
    expected_has_output = bool(expected_result.get("output_path"))
    actual_has_output = bool(actual_result.get("output_path"))
    if expected_has_output != actual_has_output:
        differences.append(
            ManifestDifference(
                key=key,
                field="output_path",
                expected=expected_has_output,
                actual=actual_has_output,
                message="output path presence differs",
            )
        )

    if check_output_files and actual_has_output and not Path(str(actual_result["output_path"])).exists():
        differences.append(
            ManifestDifference(
                key=key,
                field="output_file",
                expected="exists",
                actual="missing",
                message="actual output file does not exist",
            )
        )


def _compare_quality_status(
    differences: list[ManifestDifference],
    key: str,
    expected_result: dict[str, object],
    actual_result: dict[str, object],
    expected_quality: dict[str, dict[str, object]],
    actual_quality: dict[str, dict[str, object]],
) -> None:
    expected_output_path = expected_result.get("output_path")
    actual_output_path = actual_result.get("output_path")
    expected_report = expected_quality.get(str(expected_output_path)) if expected_output_path is not None else None
    actual_report = actual_quality.get(str(actual_output_path)) if actual_output_path is not None else None

    if expected_report is None and actual_report is None:
        return

    if expected_report is None or actual_report is None:
        differences.append(
            ManifestDifference(
                key=key,
                field="quality_report",
                expected="present" if expected_report is not None else "missing",
                actual="present" if actual_report is not None else "missing",
                message="quality report presence differs",
            )
        )
        return

    _compare_exact(differences, key, "quality_passed", expected_report.get("passed"), actual_report.get("passed"))


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
