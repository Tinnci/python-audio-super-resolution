from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from audio_super_resolution import InferenceConfig, download_model_weights, verify_model_weights
from audio_super_resolution.backends.lavasr_compat import LAVASR_MODEL_ID, LAVASR_SAMPLE_RATE, LavaSRCompatBackend
from audio_super_resolution.backends.lavasr_validation import (
    LAVASR_CONFIG_PATH,
    LAVASR_WEIGHTS_PATH,
    expected_lavasr_v2_state_keys,
    validate_lavasr_v2_weight_bundle,
)

RUN_WEIGHT_DOWNLOAD = os.environ.get("AUDIO_SUPER_RESOLUTION_RUN_WEIGHT_DOWNLOAD") == "1"
RUN_LAVASR_SMOKE = os.environ.get("AUDIO_SUPER_RESOLUTION_RUN_LAVASR_SMOKE") == "1"


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_WEIGHT_DOWNLOAD,
    reason="set AUDIO_SUPER_RESOLUTION_RUN_WEIGHT_DOWNLOAD=1 to download and verify real LavaSR weights",
)
def test_lavasr_real_weight_download_and_verify(tmp_path: Path) -> None:
    pytest.importorskip(
        "huggingface_hub",
        reason="install audio-super-resolution[download] to download real LavaSR weights",
    )

    cache_dir = _lavasr_cache_dir(tmp_path)
    weight_dir = download_model_weights(
        LAVASR_MODEL_ID,
        cache_dir=cache_dir,
        force=_force_weight_download(),
    )
    resolved = verify_model_weights(LAVASR_MODEL_ID, cache_dir=cache_dir)
    bundle_info = validate_lavasr_v2_weight_bundle(resolved)

    assert weight_dir == cache_dir / LAVASR_MODEL_ID
    assert resolved.path_for(LAVASR_CONFIG_PATH).is_file()
    assert resolved.path_for(LAVASR_WEIGHTS_PATH).is_file()
    assert bundle_info.config.head.n_fft == 2048
    assert bundle_info.state_key_count >= len(expected_lavasr_v2_state_keys())


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_LAVASR_SMOKE,
    reason="set AUDIO_SUPER_RESOLUTION_RUN_LAVASR_SMOKE=1 to run real LavaSR torch inference",
)
def test_lavasr_real_weight_torch_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch", reason="install audio-super-resolution[lavasr] to run LavaSR inference")

    cache_dir = _lavasr_cache_dir(tmp_path)
    _ensure_lavasr_weights(cache_dir, allow_download=RUN_WEIGHT_DOWNLOAD)

    sample_rate = int(os.environ.get("AUDIO_SUPER_RESOLUTION_LAVASR_SMOKE_SR", "16000"))
    duration_seconds = float(os.environ.get("AUDIO_SUPER_RESOLUTION_LAVASR_SMOKE_SECONDS", "0.05"))
    frame_count = max(1, int(sample_rate * duration_seconds))
    tone = 0.05 * np.sin(2 * np.pi * 440 * np.arange(frame_count) / sample_rate)

    backend = LavaSRCompatBackend(
        config=InferenceConfig(
            device=os.environ.get("AUDIO_SUPER_RESOLUTION_LAVASR_DEVICE", "cpu"),
            model_cache_dir=cache_dir,
            download_weights=RUN_WEIGHT_DOWNLOAD,
            force_download=_force_weight_download(),
        )
    )

    enhanced = backend.enhance(tone.astype(np.float32), sample_rate, LAVASR_SAMPLE_RATE)

    assert enhanced.ndim == 1
    assert enhanced.size > 0
    assert np.isfinite(enhanced).all()


def _ensure_lavasr_weights(cache_dir: Path, *, allow_download: bool) -> None:
    if allow_download:
        pytest.importorskip(
            "huggingface_hub",
            reason="install audio-super-resolution[download] to download real LavaSR weights",
        )
        download_model_weights(
            LAVASR_MODEL_ID,
            cache_dir=cache_dir,
            force=_force_weight_download(),
        )

    try:
        resolved = verify_model_weights(LAVASR_MODEL_ID, cache_dir=cache_dir)
        validate_lavasr_v2_weight_bundle(resolved)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        pytest.fail(
            "LavaSR weights are missing or invalid. Run once with "
            "AUDIO_SUPER_RESOLUTION_RUN_WEIGHT_DOWNLOAD=1 and install "
            "audio-super-resolution[download], or set AUDIO_SUPER_RESOLUTION_LAVASR_CACHE "
            f"to an existing verified cache. Details: {exc}"
        )


def _lavasr_cache_dir(tmp_path: Path) -> Path:
    configured = os.environ.get("AUDIO_SUPER_RESOLUTION_LAVASR_CACHE")
    return Path(configured).expanduser() if configured else tmp_path / "models"


def _force_weight_download() -> bool:
    return os.environ.get("AUDIO_SUPER_RESOLUTION_FORCE_WEIGHT_DOWNLOAD") == "1"
