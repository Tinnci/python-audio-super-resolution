import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_super_resolution.cli import main


def test_env_info(capsys) -> None:
    assert main(["--env-info"]) == 0
    output = capsys.readouterr().out
    assert "audio-super-resolution:" in output
    assert "python:" in output
    assert "model_cache_dir:" in output


def test_list_backends(capsys) -> None:
    assert main(["--list-backends"]) == 0
    output = capsys.readouterr().out
    assert "Backend" in output
    assert "sinc-resample" in output
    assert "audiosr" in output


def test_list_backends_json(capsys) -> None:
    assert main(["--list-backends", "--list-format", "json"]) == 0
    backends = json.loads(capsys.readouterr().out)

    assert {backend["name"] for backend in backends} == {"audiosr", "sinc-resample", "lavasr-compat"}
    assert all("installed" in backend for backend in backends)


def test_list_models_json(capsys) -> None:
    assert main(["--list-models", "--list-format", "json"]) == 0
    models = json.loads(capsys.readouterr().out)

    assert {model["id"] for model in models} == {
        "audiosr-basic",
        "audiosr-speech",
        "lavasr-v2-bwe",
        "sinc-resample",
    }
    assert next(model for model in models if model["id"] == "lavasr-v2-bwe")["requires_weights"] is True


def test_list_models_filter(capsys) -> None:
    assert main(["--list-models", "--list-filter", "speech"]) == 0
    output = capsys.readouterr().out

    assert "audiosr-speech" in output
    assert "audiosr-basic" not in output


def test_config_info_uses_cli_options(tmp_path: Path, capsys) -> None:
    cache_dir = tmp_path / "cache"

    assert (
        main(["--config-info", "--device", "cpu", "--precision", "float32", "--model-cache-dir", str(cache_dir)]) == 0
    )
    output = capsys.readouterr().out

    assert "device: cpu" in output
    assert "model_name: basic" in output
    assert f"model_cache_dir: {cache_dir}" in output


def test_prepare_model_cache_creates_directory(tmp_path: Path, capsys) -> None:
    cache_dir = tmp_path / "cache"

    assert main(["--prepare-model-cache", "--model-cache-dir", str(cache_dir)]) == 0
    output = capsys.readouterr().out

    assert str(cache_dir) in output
    assert cache_dir.is_dir()


def test_prepare_model_cache_can_download_weights_without_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    cache_dir = tmp_path / "cache"
    calls: dict[str, object] = {}

    def download_model_weights(model_spec, cache_dir: Path, revision: str | None = None, force: bool = False):
        calls["model_id"] = model_spec.id
        calls["cache_dir"] = cache_dir
        calls["revision"] = revision
        calls["force"] = force
        output_dir = cache_dir / model_spec.id
        output_dir.mkdir(parents=True)
        return output_dir

    monkeypatch.setattr("audio_super_resolution.cli.download_model_weights", download_model_weights)

    assert (
        main(
            [
                "--backend",
                "lavasr-compat",
                "--download-weights",
                "--prepare-model-cache",
                "--model-cache-dir",
                str(cache_dir),
                "--weight-revision",
                "test-revision",
                "--force-download",
            ]
        )
        == 0
    )

    assert calls == {
        "model_id": "lavasr-v2-bwe",
        "cache_dir": cache_dir,
        "revision": "test-revision",
        "force": True,
    }
    assert str(cache_dir / "lavasr-v2-bwe") in capsys.readouterr().out


def test_prepare_model_cache_rejects_unknown_model_name_for_single_model_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--backend",
                "lavasr-compat",
                "--model-name",
                "basic",
                "--download-weights",
                "--prepare-model-cache",
                "--model-cache-dir",
                str(tmp_path / "cache"),
            ]
        )

    assert "Unknown model 'basic' for backend 'lavasr-compat'" in capsys.readouterr().err


def test_cli_verifies_explicit_weight_manifest(tmp_path: Path, capsys) -> None:
    weight_path = tmp_path / "weights.bin"
    weight_path.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "sinc-resample",
                "files": [{"path": "weights.bin", "sha256": digest, "size": len(b"weights")}],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--backend",
                "sinc-resample",
                "--verify-weights",
                "--weights-manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    assert "Verified weights" in capsys.readouterr().out


def test_cli_verify_weights_returns_one_for_invalid_manifest(tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "id": "demo", "files": [{"path": "missing.bin"}]}),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--backend",
                "lavasr-compat",
                "--verify-weights",
                "--weights-manifest",
                str(manifest_path),
            ]
        )
        == 1
    )
    assert "Weight verification failed" in capsys.readouterr().err


def test_dry_run_for_file_uses_default_output_path(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")

    assert main([str(input_path), "--dry-run", "--target-sr", "44100"]) == 0
    output = capsys.readouterr().out

    assert str(input_path) in output
    assert str(tmp_path / "input-sr44100.wav") in output
    assert not (tmp_path / "input-sr44100.wav").exists()


def test_dry_run_writes_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    manifest_path = tmp_path / "manifest.json"
    input_path.write_bytes(b"placeholder")

    assert main([str(input_path), "--dry-run", "--target-sr", "44100", "--manifest", str(manifest_path)]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["mode"] == "dry-run"
    assert manifest["jobs"][0]["output_path"] == str(tmp_path / "input-sr44100.wav")
    assert manifest["results"] == []


def test_cli_rejects_non_positive_target_sample_rate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")

    with pytest.raises(SystemExit):
        main([str(input_path), "--target-sr", "0"])


def test_cli_reports_missing_input_without_traceback(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path / "missing.wav")])

    assert "missing.wav" in capsys.readouterr().err


def test_cli_reports_missing_lavasr_weights_with_download_hint(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sf.write(input_path, np.zeros(1000), 1000)

    with pytest.raises(SystemExit):
        main(
            [
                str(input_path),
                str(output_path),
                "--backend",
                "lavasr-compat",
                "--model-cache-dir",
                str(tmp_path / "models"),
            ]
        )

    error = capsys.readouterr().err
    assert "--download-weights --prepare-model-cache" in error


def test_cli_reports_lavasr_denoise_as_reserved(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sf.write(input_path, np.zeros(1000), 1000)

    with pytest.raises(SystemExit):
        main(
            [
                str(input_path),
                str(output_path),
                "--backend",
                "lavasr-compat",
                "--denoise",
            ]
        )

    assert "denoise is reserved" in capsys.readouterr().err


def test_cli_processes_single_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sample_rate = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate // 20) / sample_rate)
    sf.write(input_path, tone, sample_rate)

    assert main([str(input_path), str(output_path), "--target-sr", "32000"]) == 0
    _, written_sr = sf.read(output_path)

    assert written_sr == 32000


def test_cli_processes_single_file_in_chunks(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sample_rate = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate // 5) / sample_rate)
    sf.write(input_path, tone, sample_rate)

    assert (
        main(
            [
                str(input_path),
                str(output_path),
                "--target-sr",
                "32000",
                "--chunked",
                "--chunk-seconds",
                "0.05",
                "--overlap-seconds",
                "0.01",
            ]
        )
        == 0
    )
    info = sf.info(output_path)

    assert info.samplerate == 32000
    assert info.frames / info.samplerate == pytest.approx(0.2, abs=1 / info.samplerate)


def test_cli_writes_completed_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    manifest_path = tmp_path / "manifest.json"
    sample_rate = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate // 20) / sample_rate)
    sf.write(input_path, tone, sample_rate)

    assert main([str(input_path), str(output_path), "--target-sr", "32000", "--manifest", str(manifest_path)]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["mode"] == "completed"
    assert manifest["results"][0]["input_sample_rate"] == sample_rate
    assert manifest["results"][0]["sample_rate"] == 32000


def test_cli_prints_quality_report(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sample_rate = 16000
    tone = 0.25 * np.sin(2 * np.pi * 440 * np.arange(sample_rate // 20) / sample_rate)
    sf.write(input_path, tone, sample_rate)

    assert main([str(input_path), str(output_path), "--target-sr", "32000", "--quality-report"]) == 0
    output = capsys.readouterr().out

    assert str(output_path) in output
    assert "peak=" in output
    assert "clipped=0" in output


def test_cli_writes_quality_report_json(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    report_path = tmp_path / "reports" / "quality.json"
    sample_rate = 16000
    tone = 0.25 * np.sin(2 * np.pi * 440 * np.arange(sample_rate // 20) / sample_rate)
    sf.write(input_path, tone, sample_rate)

    assert (
        main([str(input_path), str(output_path), "--target-sr", "32000", "--quality-report-json", str(report_path)])
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output = capsys.readouterr().out

    assert "Wrote quality report" in output
    assert report["passed"] is True
    assert report["report_count"] == 1
    assert report["reports"][0]["path"] == str(output_path)


def test_cli_can_fail_on_quality_issue(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sf.write(input_path, np.ones(1000), 1000)

    assert main([str(input_path), str(output_path), "--fail-on-quality-issue"]) == 1


def test_cli_compares_manifests_as_json(tmp_path: Path, capsys) -> None:
    expected_path = tmp_path / "expected.json"
    actual_path = tmp_path / "actual.json"
    expected_path.write_text(json.dumps(_minimal_manifest(sample_rate=48000)), encoding="utf-8")
    actual_path.write_text(json.dumps(_minimal_manifest(sample_rate=44100)), encoding="utf-8")

    assert main(["--compare-manifests", str(expected_path), str(actual_path), "--compare-format", "json"]) == 1
    comparison = json.loads(capsys.readouterr().out)

    assert comparison["passed"] is False
    assert comparison["difference_count"] == 1
    assert comparison["differences"][0]["field"] == "sample_rate"


def _minimal_manifest(sample_rate: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "completed",
        "backend": "sinc-resample",
        "target_sample_rate": 48000,
        "results": [
            {
                "input_path": "input.wav",
                "output_path": "output.wav",
                "input_sample_rate": 16000,
                "sample_rate": sample_rate,
                "input_duration_seconds": 1.0,
                "duration_seconds": 1.0,
                "channels": 1,
                "backend": "sinc-resample",
            }
        ],
        "quality_reports": [],
    }
