from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_super_resolution import AUDIOSR_MODEL_NAMES, AUDIOSR_SAMPLE_RATE, AudioSuperResolver, InferenceConfig

RUN_AUDIOSR_INTEGRATION = os.environ.get("AUDIO_SUPER_RESOLUTION_RUN_AUDIOSR_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_AUDIOSR_INTEGRATION,
    reason="set AUDIO_SUPER_RESOLUTION_RUN_AUDIOSR_INTEGRATION=1 to run real AudioSR inference",
)
def test_audiosr_backend_real_inference(tmp_path: Path) -> None:
    pytest.importorskip("audiosr")

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sample_rate = 16000
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(sample_rate // 2) / sample_rate)
    sf.write(input_path, tone, sample_rate)

    model_name = os.environ.get("AUDIO_SUPER_RESOLUTION_AUDIOSR_MODEL", "basic")
    if model_name not in AUDIOSR_MODEL_NAMES:
        choices = ", ".join(AUDIOSR_MODEL_NAMES)
        pytest.fail(f"AUDIO_SUPER_RESOLUTION_AUDIOSR_MODEL must be one of: {choices}")

    config = InferenceConfig(
        device=os.environ.get("AUDIO_SUPER_RESOLUTION_AUDIOSR_DEVICE", "cpu"),
        model_cache_dir=Path(os.environ.get("AUDIO_SUPER_RESOLUTION_AUDIOSR_CACHE", tmp_path / "models")),
        model_name=model_name,
        seed=int(os.environ.get("AUDIO_SUPER_RESOLUTION_AUDIOSR_SEED", "0")),
        ddim_steps=int(os.environ.get("AUDIO_SUPER_RESOLUTION_AUDIOSR_DDIM_STEPS", "2")),
        guidance_scale=float(os.environ.get("AUDIO_SUPER_RESOLUTION_AUDIOSR_GUIDANCE_SCALE", "3.5")),
    )

    result = AudioSuperResolver(target_sr=AUDIOSR_SAMPLE_RATE, backend="audiosr", config=config).enhance(
        input_path,
        output_path,
    )
    info = sf.info(output_path)

    assert output_path.exists()
    assert result.backend == "audiosr"
    assert result.sample_rate == AUDIOSR_SAMPLE_RATE
    assert info.samplerate == AUDIOSR_SAMPLE_RATE
    assert info.frames > 0
