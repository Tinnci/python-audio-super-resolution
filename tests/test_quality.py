from pathlib import Path

import numpy as np
import soundfile as sf

from audio_super_resolution import format_quality_report, inspect_audio_quality


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
