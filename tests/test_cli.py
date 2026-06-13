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
    assert "sinc-resample:" in output
    assert "audiosr:" in output


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


def test_dry_run_for_file_uses_default_output_path(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")

    assert main([str(input_path), "--dry-run", "--target-sr", "44100"]) == 0
    output = capsys.readouterr().out

    assert str(input_path) in output
    assert str(tmp_path / "input-sr44100.wav") in output
    assert not (tmp_path / "input-sr44100.wav").exists()


def test_cli_rejects_non_positive_target_sample_rate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")

    with pytest.raises(SystemExit):
        main([str(input_path), "--target-sr", "0"])


def test_cli_processes_single_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sample_rate = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate // 20) / sample_rate)
    sf.write(input_path, tone, sample_rate)

    assert main([str(input_path), str(output_path), "--target-sr", "32000"]) == 0
    _, written_sr = sf.read(output_path)

    assert written_sr == 32000


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


def test_cli_can_fail_on_quality_issue(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sf.write(input_path, np.ones(1000), 1000)

    assert main([str(input_path), str(output_path), "--fail-on-quality-issue"]) == 1
