from __future__ import annotations

import numpy as np
import pytest

from audio_super_resolution import InferenceConfig
from audio_super_resolution.backends.lavasr_compat import LavaSRCompatBackend


def test_lavasr_availability_depends_on_torch_not_yaml(monkeypatch) -> None:
    def find_spec(name: str):
        return object() if name == "torch" else None

    monkeypatch.setattr("audio_super_resolution.backends.lavasr_compat.importlib.util.find_spec", find_spec)

    assert LavaSRCompatBackend.is_available()


def test_lavasr_availability_is_false_without_torch(monkeypatch) -> None:
    monkeypatch.setattr("audio_super_resolution.backends.lavasr_compat.importlib.util.find_spec", lambda name: None)

    assert not LavaSRCompatBackend.is_available()


def test_lavasr_backend_rejects_unsupported_precision_before_weight_resolution(tmp_path) -> None:
    backend = LavaSRCompatBackend(
        config=InferenceConfig(
            model_cache_dir=tmp_path / "models",
            precision="float16",
        )
    )

    with pytest.raises(ValueError, match="precision modes"):
        backend.enhance(np.zeros(100, dtype=np.float32), 16000, 48000)


def test_lavasr_backend_preserves_exact_digital_silence(monkeypatch, tmp_path) -> None:
    resolved_weights = object()
    bundle_info = object()
    monkeypatch.setattr(
        "audio_super_resolution.weight_store.resolve_weights_for_spec",
        lambda *args, **kwargs: resolved_weights,
    )
    monkeypatch.setattr(
        "audio_super_resolution.backends.lavasr_validation.validate_lavasr_v2_weight_bundle",
        lambda resolved: bundle_info,
    )
    monkeypatch.setattr(
        "audio_super_resolution.backends.lavasr_compat.resolve_runtime_provider",
        lambda *args: None,
    )
    monkeypatch.setattr("audio_super_resolution.backends.lavasr_compat.resolve_device", lambda *args, **kwargs: "cuda")
    monkeypatch.setattr("audio_super_resolution.backends.lavasr_compat._require_torch_runtime", lambda: None)

    backend = LavaSRCompatBackend(
        config=InferenceConfig(
            device="cuda",
            runtime_provider="torch-eager",
            model_cache_dir=tmp_path / "models",
        )
    )
    monkeypatch.setattr(backend, "_load_model", lambda *args, **kwargs: pytest.fail("model should not load"))

    enhanced = backend.enhance(np.zeros(100, dtype=np.float32), 16000, 48000)

    assert enhanced.shape == (300,)
    assert np.count_nonzero(enhanced) == 0
