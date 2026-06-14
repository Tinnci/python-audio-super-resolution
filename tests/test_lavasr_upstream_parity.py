from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_super_resolution import (
    GoldenThresholds,
    InferenceConfig,
    compare_golden_outputs,
    download_model_weights,
    verify_model_weights,
)
from audio_super_resolution.backends.lavasr_compat import (
    LAVASR_MODEL_ID,
    LAVASR_SAMPLE_RATE,
    LavaSRCompatBackend,
    _merge_cutoff_hz,
    _prepare_lavasr_input,
)
from audio_super_resolution.backends.lavasr_validation import (
    LAVASR_WEIGHTS_PATH,
    validate_lavasr_v2_weight_bundle,
)

RUN_UPSTREAM_PARITY = os.environ.get("AUDIO_SUPER_RESOLUTION_RUN_LAVASR_UPSTREAM_PARITY") == "1"
RUN_WEIGHT_DOWNLOAD = os.environ.get("AUDIO_SUPER_RESOLUTION_RUN_WEIGHT_DOWNLOAD") == "1"


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_UPSTREAM_PARITY,
    reason="set AUDIO_SUPER_RESOLUTION_RUN_LAVASR_UPSTREAM_PARITY=1 to compare against upstream LavaSR",
)
def test_lavasr_upstream_bwe_parity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch", reason="install audio-super-resolution[lavasr]")
    upstream_enhancer = pytest.importorskip(
        "LavaSR.enhancer.enhancer",
        reason="install upstream LavaSR to run parity validation",
    )
    upstream_merge = pytest.importorskip(
        "LavaSR.enhancer.linkwitz_merge",
        reason="install upstream LavaSR to run parity validation",
    )

    input_sample_rate = int(os.environ.get("AUDIO_SUPER_RESOLUTION_LAVASR_PARITY_SR", "16000"))
    duration_seconds = float(os.environ.get("AUDIO_SUPER_RESOLUTION_LAVASR_PARITY_SECONDS", "0.08"))
    device = os.environ.get("AUDIO_SUPER_RESOLUTION_LAVASR_DEVICE", "cpu")
    source = _parity_source(sample_rate=input_sample_rate, duration_seconds=duration_seconds)
    resolved = _ensure_lavasr_weights(_lavasr_cache_dir(tmp_path))

    actual = LavaSRCompatBackend(
        config=InferenceConfig(
            device=device,
            model_cache_dir=resolved.root_dir.parent,
        )
    ).enhance(source, input_sample_rate, LAVASR_SAMPLE_RATE)
    reference = _run_upstream_lavasr_bwe(
        upstream_enhancer.LavaBWE,
        upstream_merge.FastLRMerge,
        torch=torch,
        source=source,
        sample_rate=input_sample_rate,
        enhancer_dir=resolved.path_for(LAVASR_WEIGHTS_PATH).parent,
        device=device,
    )

    actual_path = tmp_path / "actual.wav"
    reference_path = tmp_path / "reference.wav"
    sf.write(actual_path, actual, LAVASR_SAMPLE_RATE)
    sf.write(reference_path, reference, LAVASR_SAMPLE_RATE)
    report = compare_golden_outputs(actual_path, reference_path, thresholds=_parity_thresholds())

    assert report.passed, tuple(report.issues)


def _run_upstream_lavasr_bwe(
    lava_bwe_type,
    fast_lr_merge_type,
    *,
    torch,
    source: np.ndarray,
    sample_rate: int,
    enhancer_dir: Path,
    device: str,
) -> np.ndarray:
    prepared_audio, was_mono = _prepare_lavasr_input(source, sample_rate)
    waveform = torch.from_numpy(prepared_audio.T.copy()).to(device=device, dtype=torch.float32)

    model = lava_bwe_type(str(enhancer_dir), device=device)
    model.lr_refiner = fast_lr_merge_type(
        device=device,
        cutoff=_merge_cutoff_hz(sample_rate),
        transition_bins=1024,
    )
    with torch.inference_mode():
        enhanced = model.infer(waveform).detach().cpu().numpy().T

    if was_mono:
        return enhanced[:, 0]
    return enhanced


def _ensure_lavasr_weights(cache_dir: Path):
    if RUN_WEIGHT_DOWNLOAD:
        pytest.importorskip(
            "huggingface_hub",
            reason="install audio-super-resolution[download] to download real LavaSR weights",
        )
        download_model_weights(
            LAVASR_MODEL_ID,
            cache_dir=cache_dir,
            force=os.environ.get("AUDIO_SUPER_RESOLUTION_FORCE_WEIGHT_DOWNLOAD") == "1",
        )

    try:
        resolved = verify_model_weights(LAVASR_MODEL_ID, cache_dir=cache_dir)
        validate_lavasr_v2_weight_bundle(resolved)
        return resolved
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        pytest.fail(
            "LavaSR parity validation requires a verified local LavaSR cache. Run once with "
            "AUDIO_SUPER_RESOLUTION_RUN_WEIGHT_DOWNLOAD=1, or set AUDIO_SUPER_RESOLUTION_LAVASR_CACHE "
            f"to an existing cache. Details: {exc}"
        )


def _lavasr_cache_dir(tmp_path: Path) -> Path:
    configured = os.environ.get("AUDIO_SUPER_RESOLUTION_LAVASR_CACHE")
    return Path(configured).expanduser() if configured else tmp_path / "models"


def _parity_source(*, sample_rate: int, duration_seconds: float) -> np.ndarray:
    frame_count = max(1, int(sample_rate * duration_seconds))
    time = np.arange(frame_count) / sample_rate
    return (
        0.04 * np.sin(2 * np.pi * 220 * time)
        + 0.03 * np.sin(2 * np.pi * 1234 * time)
        + 0.02 * np.sin(2 * np.pi * 3400 * time)
    ).astype(np.float32)


def _parity_thresholds() -> GoldenThresholds:
    return GoldenThresholds(
        max_duration_drift_seconds=0.01,
        max_peak_delta=0.05,
        max_rms_delta=0.05,
        max_log_mel_l1=1.0,
        max_hf_energy_ratio_delta=0.10,
        high_frequency_start_hz=8000,
    )
