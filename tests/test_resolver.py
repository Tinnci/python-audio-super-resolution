from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_super_resolution import AudioSuperResolver, InferenceConfig, available_backends, plan_enhancements


def test_enhance_writes_target_sample_rate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sample_rate = 16000
    target_sr = 48000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate // 10) / sample_rate)

    sf.write(input_path, tone, sample_rate)

    result = AudioSuperResolver(target_sr=target_sr).enhance(input_path, output_path)
    _, written_sr = sf.read(output_path)

    assert output_path.exists()
    assert result.sample_rate == target_sr
    assert result.input_sample_rate == sample_rate
    assert result.input_duration_seconds == pytest.approx(0.1)
    assert written_sr == target_sr
    assert result.backend == "sinc-resample"


def test_plan_enhancements_for_single_file_uses_default_output_name(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")

    jobs = plan_enhancements(input_path, target_sr=44100)

    assert len(jobs) == 1
    assert jobs[0].input_path == input_path
    assert jobs[0].output_path == tmp_path / "input-sr44100.wav"


def test_plan_enhancements_for_single_file_can_target_output_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_dir = tmp_path / "enhanced"
    input_path.write_bytes(b"placeholder")

    jobs = plan_enhancements(input_path, output_dir, target_sr=44100)

    assert jobs[0].output_path == output_dir / "input-sr44100.wav"


def test_target_sample_rate_must_be_positive(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="target_sr must be greater than zero"):
        plan_enhancements(input_path, target_sr=0)


def test_enhance_many_preserves_relative_directory_structure(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    nested_dir = input_dir / "nested"
    output_dir = tmp_path / "output"
    nested_dir.mkdir(parents=True)

    sample_rate = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate // 20) / sample_rate)
    sf.write(input_dir / "root.wav", tone, sample_rate)
    sf.write(nested_dir / "child.wav", tone, sample_rate)

    results = AudioSuperResolver(target_sr=32000).enhance_many(input_dir, output_dir, recursive=True)

    assert [result.output_path.relative_to(output_dir) for result in results] == [
        Path("nested/child-sr32000.wav"),
        Path("root-sr32000.wav"),
    ]
    assert (output_dir / "root-sr32000.wav").exists()
    assert (output_dir / "nested" / "child-sr32000.wav").exists()


def test_backend_registry_lists_sinc_resample() -> None:
    assert any(backend.name == "sinc-resample" for backend in available_backends())


def test_backend_receives_inference_config(tmp_path: Path) -> None:
    config = InferenceConfig(model_cache_dir=tmp_path / "models")
    resolver = AudioSuperResolver(backend="sinc-resample", config=config)

    assert resolver.config == config
    assert resolver.backend.config == config
