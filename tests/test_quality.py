from pathlib import Path

import numpy as np
import soundfile as sf

from audio_super_resolution import (
    build_quality_report_bundle,
    format_quality_report,
    inspect_audio_quality,
    quality_report_to_dict,
    write_quality_report_bundle,
)


def test_inspect_audio_quality_passes_for_clean_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "clean.wav"
    sample_rate = 16000
    audio = 0.25 * np.sin(2 * np.pi * 440 * np.arange(sample_rate // 10) / sample_rate)
    sf.write(audio_path, audio, sample_rate)

    report = inspect_audio_quality(
        audio_path,
        expected_sample_rate=sample_rate,
        expected_duration_seconds=0.1,
    )

    assert report.passed
    assert report.sample_rate == sample_rate
    assert report.duration_drift_seconds == 0
    assert report.peak_level > 0
    assert report.clipped_samples == 0
    assert "ok;" in format_quality_report(report)


def test_inspect_audio_quality_reports_clipping(tmp_path: Path) -> None:
    audio_path = tmp_path / "clipped.wav"
    sf.write(audio_path, np.ones(100), 1000)

    report = inspect_audio_quality(audio_path)

    assert not report.passed
    assert report.clipped_samples > 0
    assert any("clipping threshold" in issue for issue in report.issues)


def test_inspect_audio_quality_reports_sample_rate_and_duration_drift(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, np.zeros(1000), 1000)

    report = inspect_audio_quality(
        audio_path,
        expected_sample_rate=2000,
        expected_duration_seconds=2.0,
        max_duration_drift_seconds=0.1,
    )

    assert not report.passed
    assert any("sample rate" in issue for issue in report.issues)
    assert any("duration drift" in issue for issue in report.issues)


def test_quality_report_serializes_to_json_friendly_dict(tmp_path: Path) -> None:
    audio_path = tmp_path / "clean.wav"
    sf.write(audio_path, np.zeros(1000), 1000)

    report = inspect_audio_quality(audio_path, expected_sample_rate=1000)
    serialized = quality_report_to_dict(report)

    assert serialized["path"] == str(audio_path)
    assert serialized["sample_rate"] == 1000
    assert serialized["passed"] is True
    assert serialized["issues"] == []


def test_write_quality_report_bundle_creates_combined_report(tmp_path: Path) -> None:
    clean_path = tmp_path / "clean.wav"
    clipped_path = tmp_path / "clipped.wav"
    report_path = tmp_path / "reports" / "quality.json"
    sf.write(clean_path, np.zeros(1000), 1000)
    sf.write(clipped_path, np.ones(1000), 1000)

    reports = [inspect_audio_quality(clean_path), inspect_audio_quality(clipped_path)]
    bundle = build_quality_report_bundle(reports)
    written_path = write_quality_report_bundle(report_path, reports)

    assert bundle["passed"] is False
    assert bundle["report_count"] == 2
    assert bundle["issue_count"] > 0
    assert written_path == report_path
    assert '"reports"' in report_path.read_text(encoding="utf-8")
