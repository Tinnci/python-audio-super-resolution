from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_super_resolution import (
    GoldenThresholds,
    compare_golden_fixture,
    compare_golden_outputs,
    golden_report_to_dict,
    inspect_golden_audio,
    load_golden_fixture,
    write_golden_report,
)


def test_compare_golden_outputs_passes_for_matching_audio(tmp_path: Path) -> None:
    sample_rate = 48000
    audio = _tone(sample_rate=sample_rate, frequency=440, duration_seconds=0.05)
    actual_path = tmp_path / "actual.wav"
    reference_path = tmp_path / "reference.wav"
    sf.write(actual_path, audio, sample_rate)
    sf.write(reference_path, audio, sample_rate)

    report = compare_golden_outputs(actual_path, reference_path)

    assert report.passed
    assert report.log_mel_l1 == 0
    assert report.duration_drift_seconds == 0
    assert golden_report_to_dict(report)["passed"] is True


def test_compare_golden_outputs_reports_spectral_and_level_differences(tmp_path: Path) -> None:
    sample_rate = 48000
    reference = 0.1 * _tone(sample_rate=sample_rate, frequency=440, duration_seconds=0.05)
    actual = 0.5 * _tone(sample_rate=sample_rate, frequency=12000, duration_seconds=0.06)
    actual_path = tmp_path / "actual.wav"
    reference_path = tmp_path / "reference.wav"
    sf.write(actual_path, actual, sample_rate)
    sf.write(reference_path, reference, sample_rate)

    report = compare_golden_outputs(
        actual_path,
        reference_path,
        thresholds=GoldenThresholds(
            max_duration_drift_seconds=0.001,
            max_peak_delta=0.01,
            max_rms_delta=0.01,
            max_log_mel_l1=0.01,
            max_hf_energy_ratio_delta=0.01,
            high_frequency_start_hz=8000,
        ),
    )

    assert not report.passed
    assert any("duration drift" in issue for issue in report.issues)
    assert any("peak delta" in issue for issue in report.issues)
    assert any("RMS delta" in issue for issue in report.issues)
    assert any("log-mel L1" in issue for issue in report.issues)
    assert any("high-frequency energy ratio" in issue for issue in report.issues)


def test_compare_golden_outputs_reports_sample_rate_mismatch(tmp_path: Path) -> None:
    actual_path = tmp_path / "actual.wav"
    reference_path = tmp_path / "reference.wav"
    sf.write(actual_path, np.zeros(100), 48000)
    sf.write(reference_path, np.zeros(100), 44100)

    report = compare_golden_outputs(actual_path, reference_path)

    assert not report.passed
    assert report.log_mel_l1 is None
    assert any("sample rate" in issue for issue in report.issues)


def test_inspect_golden_audio_reports_hf_energy_ratio(tmp_path: Path) -> None:
    sample_rate = 48000
    low_path = tmp_path / "low.wav"
    high_path = tmp_path / "high.wav"
    sf.write(low_path, _tone(sample_rate=sample_rate, frequency=440, duration_seconds=0.05), sample_rate)
    sf.write(high_path, _tone(sample_rate=sample_rate, frequency=12000, duration_seconds=0.05), sample_rate)

    low_stats = inspect_golden_audio(low_path, thresholds=GoldenThresholds(high_frequency_start_hz=8000))
    high_stats = inspect_golden_audio(high_path, thresholds=GoldenThresholds(high_frequency_start_hz=8000))

    assert high_stats.high_frequency_energy_ratio > low_stats.high_frequency_energy_ratio


def test_load_golden_fixture_validates_schema(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "demo",
                "backend": "lavasr-compat",
                "input": {"path": "input.wav"},
                "reference": {"path": "reference.wav"},
                "thresholds": {"max_log_mel_l1": 0.5},
            }
        ),
        encoding="utf-8",
    )

    fixture = load_golden_fixture(fixture_path)

    assert fixture["id"] == "demo"
    assert fixture["backend"] == "lavasr-compat"


def test_compare_golden_fixture_resolves_reference_and_thresholds(tmp_path: Path) -> None:
    sample_rate = 48000
    actual_path = tmp_path / "actual.wav"
    reference_path = tmp_path / "reference.wav"
    fixture_path = tmp_path / "fixture.json"
    sf.write(actual_path, np.zeros(100), sample_rate)
    sf.write(reference_path, np.zeros(100), sample_rate)
    fixture_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "demo",
                "backend": "lavasr-compat",
                "input": {"path": "input.wav"},
                "reference": {"path": "reference.wav"},
                "thresholds": {"max_log_mel_l1": 0.25},
            }
        ),
        encoding="utf-8",
    )

    report = compare_golden_fixture(fixture_path, actual_path)

    assert report.passed
    assert report.thresholds.max_log_mel_l1 == 0.25


def test_write_golden_report_creates_json_report(tmp_path: Path) -> None:
    sample_rate = 48000
    actual_path = tmp_path / "actual.wav"
    reference_path = tmp_path / "reference.wav"
    report_path = tmp_path / "reports" / "golden.json"
    sf.write(actual_path, np.zeros(100), sample_rate)
    sf.write(reference_path, np.zeros(100), sample_rate)
    report = compare_golden_outputs(actual_path, reference_path)

    written = write_golden_report(report_path, report)

    assert written == report_path
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True


def _tone(*, sample_rate: int, frequency: float, duration_seconds: float) -> np.ndarray:
    return np.sin(2 * np.pi * frequency * np.arange(int(sample_rate * duration_seconds)) / sample_rate)
