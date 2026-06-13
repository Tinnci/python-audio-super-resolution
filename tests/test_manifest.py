from pathlib import Path

from audio_super_resolution import InferenceConfig, build_manifest, plan_enhancements, write_manifest


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
