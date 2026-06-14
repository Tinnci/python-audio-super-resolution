from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from audio_super_resolution.backends.lavasr_torch import (  # noqa: E402
    build_lavasr_v2_model,
    load_lavasr_v2_state_dict,
)
from audio_super_resolution.backends.lavasr_validation import (  # noqa: E402
    BACKBONE_CLASS,
    FEATURE_EXTRACTOR_CLASS,
    HEAD_CLASS,
    LavaSRBackboneConfig,
    LavaSRConfig,
    LavaSRFeatureConfig,
    LavaSRHeadConfig,
    expected_lavasr_v2_state_keys,
)


def test_lavasr_torch_model_state_keys_match_validation_contract() -> None:
    model = build_lavasr_v2_model(_tiny_lavasr_config(num_layers=8))

    assert set(model.state_dict()) == expected_lavasr_v2_state_keys(num_layers=8)


def test_lavasr_torch_model_loads_matching_state_dict(tmp_path: Path) -> None:
    model = build_lavasr_v2_model(_tiny_lavasr_config(num_layers=2))
    checkpoint_path = tmp_path / "pytorch_model.bin"
    torch.save(model.state_dict(), checkpoint_path)

    reloaded = build_lavasr_v2_model(_tiny_lavasr_config(num_layers=2))
    load_lavasr_v2_state_dict(reloaded, checkpoint_path)

    assert set(reloaded.state_dict()) == expected_lavasr_v2_state_keys(num_layers=2)


def test_lavasr_torch_model_runs_minimal_forward() -> None:
    model = build_lavasr_v2_model(_tiny_lavasr_config(num_layers=1))
    model.feature_extractor.mel_spec.mel_scale.fb.fill_(1)

    with torch.inference_mode():
        output = model(torch.zeros(1, 64))

    assert output.ndim == 2
    assert output.shape[0] == 1
    assert torch.isfinite(output).all()


def _tiny_lavasr_config(*, num_layers: int) -> LavaSRConfig:
    return LavaSRConfig(
        feature_extractor=LavaSRFeatureConfig(
            class_path=FEATURE_EXTRACTOR_CLASS,
            sample_rate=44100,
            n_fft=16,
            hop_length=4,
            n_mels=4,
            padding="same",
            f_min=0,
            f_max=8000,
            norm="slaney",
            mel_scale="slaney",
        ),
        backbone=LavaSRBackboneConfig(
            class_path=BACKBONE_CLASS,
            input_channels=4,
            dim=8,
            intermediate_dim=16,
            num_layers=num_layers,
        ),
        head=LavaSRHeadConfig(
            class_path=HEAD_CLASS,
            dim=8,
            n_fft=16,
            hop_length=4,
            padding="same",
        ),
    )
