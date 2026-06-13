from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_super_resolution import AudioSuperResolver, InferenceConfig
from audio_super_resolution.audiosr_backend import AUDIOSR_SAMPLE_RATE, AudiosrBackend


def test_audiosr_backend_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "audiosr", None)
    backend = AudiosrBackend(config=InferenceConfig(model_cache_dir=tmp_path / "models"))

    with pytest.raises(RuntimeError, match="optional audiosr dependency"):
        backend.enhance_file(tmp_path / "input.wav", tmp_path / "output.wav", AUDIOSR_SAMPLE_RATE)


def test_audiosr_availability_handles_injected_module_without_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "audiosr", types.SimpleNamespace())

    assert not AudiosrBackend.is_available()


def test_audiosr_backend_rejects_unsupported_target_sample_rate(tmp_path: Path) -> None:
    backend = AudiosrBackend(config=InferenceConfig(model_cache_dir=tmp_path / "models"))

    with pytest.raises(ValueError, match="outputs 48000 Hz"):
        backend.enhance_file(tmp_path / "input.wav", tmp_path / "output.wav", 44100)


def test_audiosr_backend_writes_fake_model_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sf.write(input_path, np.zeros(1000), 1000)

    calls: dict[str, object] = {}

    def build_model(model_name: str, device: str):
        calls["model_name"] = model_name
        calls["device"] = device
        return object()

    def super_resolution(model, input_file: str, seed: int, guidance_scale: float, ddim_steps: int):
        calls["input_file"] = input_file
        calls["seed"] = seed
        calls["guidance_scale"] = guidance_scale
        calls["ddim_steps"] = ddim_steps
        return np.zeros((1, 1, 480), dtype=np.float32)

    fake_audiosr = types.SimpleNamespace(build_model=build_model, super_resolution=super_resolution)
    monkeypatch.setitem(sys.modules, "audiosr", fake_audiosr)
    monkeypatch.delenv("HF_HOME", raising=False)

    config = InferenceConfig(
        device="cpu",
        model_cache_dir=tmp_path / "models",
        model_name="speech",
        seed=123,
        ddim_steps=8,
        guidance_scale=2.5,
    )
    backend = AudiosrBackend(config=config)
    backend.enhance_file(input_path, output_path, AUDIOSR_SAMPLE_RATE)

    info = sf.info(output_path)
    assert info.samplerate == AUDIOSR_SAMPLE_RATE
    assert calls == {
        "model_name": "speech",
        "device": "cpu",
        "input_file": str(input_path),
        "seed": 123,
        "guidance_scale": 2.5,
        "ddim_steps": 8,
    }
    assert (tmp_path / "models").is_dir()


def test_resolver_can_use_audiosr_file_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sf.write(input_path, np.zeros(1000), 1000)

    fake_audiosr = types.SimpleNamespace(
        build_model=lambda model_name, device: object(),
        super_resolution=lambda model, input_file, seed, guidance_scale, ddim_steps: np.zeros(
            (1, 1, 480), dtype=np.float32
        ),
    )
    monkeypatch.setitem(sys.modules, "audiosr", fake_audiosr)

    result = AudioSuperResolver(
        backend="audiosr",
        config=InferenceConfig(model_cache_dir=tmp_path / "models"),
    ).enhance(input_path, output_path)

    assert result.backend == "audiosr"
    assert result.input_sample_rate == 1000
    assert result.sample_rate == AUDIOSR_SAMPLE_RATE
