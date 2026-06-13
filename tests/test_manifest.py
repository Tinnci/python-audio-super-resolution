from pathlib import Path

from audio_super_resolution import (
    InferenceConfig,
    build_manifest,
    compare_manifests,
    format_manifest_comparison,
    manifest_comparison_to_dict,
    plan_enhancements,
    write_manifest,
)


def test_build_manifest_serializes_planned_jobs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")
    jobs = plan_enhancements(input_path, target_sr=48000)

    manifest = build_manifest(
        mode="dry-run",
        jobs=jobs,
        config=InferenceConfig(model_cache_dir=tmp_path / "models"),
        backend="sinc-resample",
        target_sample_rate=48000,
    )

    assert manifest["schema_version"] == 1
    assert manifest["mode"] == "dry-run"
    assert manifest["jobs"] == [
        {
            "input_path": str(input_path),
            "output_path": str(tmp_path / "input-sr48000.wav"),
        }
    ]
    assert manifest["results"] == []


def test_write_manifest_creates_parent_directories(tmp_path: Path) -> None:
    manifest_path = tmp_path / "reports" / "manifest.json"

    written_path = write_manifest(manifest_path, {"schema_version": 1})

    assert written_path == manifest_path
    assert manifest_path.read_text(encoding="utf-8").strip() == '{\n  "schema_version": 1\n}'


def test_compare_manifests_passes_for_matching_results() -> None:
    expected = _completed_manifest()
    actual = _completed_manifest(duration_seconds=1.02)

    comparison = compare_manifests(expected, actual, duration_tolerance_seconds=0.05)

    assert comparison.passed
    assert manifest_comparison_to_dict(comparison) == {
        "passed": True,
        "difference_count": 0,
        "differences": [],
    }
    assert format_manifest_comparison(comparison) == "Manifest comparison passed."


def test_compare_manifests_reports_audio_and_quality_regressions() -> None:
    expected = _completed_manifest()
    actual = _completed_manifest(sample_rate=44100, duration_seconds=1.2, channels=2, quality_passed=False)

    comparison = compare_manifests(expected, actual, duration_tolerance_seconds=0.05)
    fields = {difference.field for difference in comparison.differences}

    assert not comparison.passed
    assert {"sample_rate", "duration_seconds", "channels", "quality_passed"} <= fields


def test_compare_manifests_reports_missing_actual_output_file(tmp_path: Path) -> None:
    expected = _completed_manifest(output_path=str(tmp_path / "expected.wav"))
    actual = _completed_manifest(output_path=str(tmp_path / "missing.wav"))

    comparison = compare_manifests(expected, actual, check_output_files=True)

    assert not comparison.passed
    assert any(difference.field == "output_file" for difference in comparison.differences)


def _completed_manifest(
    sample_rate: int = 48000,
    duration_seconds: float = 1.0,
    channels: int = 1,
    output_path: str = "output.wav",
    quality_passed: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "completed",
        "backend": "sinc-resample",
        "target_sample_rate": 48000,
        "results": [
            {
                "input_path": "input.wav",
                "output_path": output_path,
                "input_sample_rate": 16000,
                "sample_rate": sample_rate,
                "input_duration_seconds": 1.0,
                "duration_seconds": duration_seconds,
                "channels": channels,
                "backend": "sinc-resample",
            }
        ],
        "quality_reports": [
            {
                "path": output_path,
                "passed": quality_passed,
            }
        ],
    }
