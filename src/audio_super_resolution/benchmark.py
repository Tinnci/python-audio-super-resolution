from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import InferenceConfig
from .quality import AudioQualityReport, quality_report_to_dict
from .resolver import EnhancementResult
from .runtime_stats import peak_rss_snapshot


def build_benchmark_report(
    *,
    backend: str,
    target_sample_rate: int,
    config: InferenceConfig,
    results: list[EnhancementResult],
    quality_reports: list[AudioQualityReport],
    backend_init_seconds: float = 0.0,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Build a machine-readable runtime benchmark summary."""

    total_input_duration = sum(result.input_duration_seconds for result in results)
    total_output_duration = sum(result.duration_seconds for result in results)
    total_elapsed_seconds = backend_init_seconds + elapsed_seconds
    rtf = elapsed_seconds / total_output_duration if total_output_duration > 0 else None
    realtime_factor = total_output_duration / elapsed_seconds if elapsed_seconds > 0 else None
    memory = peak_rss_snapshot()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "device": config.device,
        "runtime_provider": config.runtime_provider,
        "target_sample_rate": target_sample_rate,
        "job_count": len(results),
        "load_time_seconds": backend_init_seconds,
        "backend_init_seconds": backend_init_seconds,
        "elapsed_seconds": elapsed_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
        "total_input_duration_seconds": total_input_duration,
        "total_output_duration_seconds": total_output_duration,
        "rtf": rtf,
        "realtime_factor": realtime_factor,
        "memory": memory,
        "peak_rss_mb": memory["peak_rss_mb"],
        "results": [_enhancement_result_to_dict(result) for result in results],
        "quality_reports": [quality_report_to_dict(report) for report in quality_reports],
    }


def write_benchmark_report(path: str | Path, report: dict[str, object]) -> Path:
    """Write a benchmark report and return the path."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


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
