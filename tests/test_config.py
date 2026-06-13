from pathlib import Path

import pytest

from audio_super_resolution import InferenceConfig, default_model_cache_dir


def test_inference_config_normalizes_device_precision_and_cache_path(tmp_path: Path) -> None:
    config = InferenceConfig(
        device="CPU",
        precision="FLOAT32",
        model_cache_dir=tmp_path / "models",
        model_name="BASIC",
    )

    assert config.device == "cpu"
    assert config.precision == "float32"
    assert config.model_cache_dir == tmp_path / "models"
    assert config.model_name == "basic"


def test_inference_config_creates_model_cache_dir(tmp_path: Path) -> None:
    config = InferenceConfig(model_cache_dir=tmp_path / "models")

    assert config.ensure_model_cache_dir() == tmp_path / "models"
    assert (tmp_path / "models").is_dir()


def test_inference_config_rejects_invalid_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlap_seconds must be less than chunk_seconds"):
        InferenceConfig(chunk_seconds=2.0, overlap_seconds=2.0, model_cache_dir=tmp_path)


def test_inference_config_rejects_invalid_diffusion_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ddim_steps must be greater than zero"):
        InferenceConfig(ddim_steps=0, model_cache_dir=tmp_path)

    with pytest.raises(ValueError, match="guidance_scale must be greater than zero"):
        InferenceConfig(guidance_scale=0, model_cache_dir=tmp_path)


def test_inference_config_rejects_invalid_preprocessing_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="preprocess must be one of"):
        InferenceConfig(preprocess="normalize", model_cache_dir=tmp_path)

    with pytest.raises(ValueError, match="lowpass_cutoff_hz must be greater than zero"):
        InferenceConfig(lowpass_cutoff_hz=0, model_cache_dir=tmp_path)

    with pytest.raises(ValueError, match="lowpass_order must be greater than zero"):
        InferenceConfig(lowpass_order=0, model_cache_dir=tmp_path)


def test_default_model_cache_dir_uses_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUDIO_SUPER_RESOLUTION_CACHE", str(tmp_path / "cache"))

    assert default_model_cache_dir() == tmp_path / "cache"
