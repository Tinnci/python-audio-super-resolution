from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import pytest

from audio_super_resolution import ResolvedWeights, WeightManifest
from audio_super_resolution.backends.lavasr_validation import (
    LAVASR_CONFIG_PATH,
    LAVASR_WEIGHTS_PATH,
    expected_lavasr_v2_state_keys,
    extract_torch_state_dict_keys,
    read_lavasr_config,
    validate_lavasr_v2_config,
    validate_lavasr_v2_state_keys,
    validate_lavasr_v2_weight_bundle,
)

LAVASR_V2_CONFIG = """
feature_extractor:
  class_path: vocos.feature_extractors.MelSpectrogramFeatures
  init_args:
    sample_rate: 44100
    n_fft: 2048
    hop_length: 512
    n_mels: 80
    padding: same
    f_min: 0
    f_max: 8000
    norm: "slaney"
    mel_scale: "slaney"

backbone:
  class_path: vocos.models.VocosBackbone
  init_args:
    input_channels: 80
    dim: 512
    intermediate_dim: 1536
    num_layers: 8

head:
  class_path: vocos.heads.ISTFTHead
  init_args:
    dim: 512
    n_fft: 2048
    hop_length: 512
    padding: same
"""


def test_read_lavasr_config_parses_v2_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(LAVASR_V2_CONFIG, encoding="utf-8")

    config = read_lavasr_config(config_path)

    validate_lavasr_v2_config(config)
    assert config.feature_extractor.sample_rate == 44100
    assert config.backbone.num_layers == 8
    assert config.head.n_fft == 2048


def test_lavasr_config_rejects_unexpected_architecture(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(LAVASR_V2_CONFIG.replace("num_layers: 8", "num_layers: 4"), encoding="utf-8")

    config = read_lavasr_config(config_path)

    with pytest.raises(ValueError, match="backbone.num_layers mismatch"):
        validate_lavasr_v2_config(config)


def test_extract_torch_state_dict_keys_reads_checkpoint_without_torch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "pytorch_model.bin"
    expected_keys = expected_lavasr_v2_state_keys()
    _write_fake_torch_checkpoint(checkpoint_path, expected_keys | {"backbone.convnext.0"})

    keys = extract_torch_state_dict_keys(checkpoint_path)

    assert keys == expected_keys
    validate_lavasr_v2_state_keys(keys)


def test_lavasr_state_key_validation_rejects_missing_required_key() -> None:
    keys = expected_lavasr_v2_state_keys()
    keys.remove("head.out.weight")

    with pytest.raises(ValueError, match="head.out.weight"):
        validate_lavasr_v2_state_keys(keys)


def test_validate_lavasr_v2_weight_bundle(tmp_path: Path) -> None:
    config_path = tmp_path / LAVASR_CONFIG_PATH
    checkpoint_path = tmp_path / LAVASR_WEIGHTS_PATH
    config_path.parent.mkdir(parents=True)
    config_path.write_text(LAVASR_V2_CONFIG, encoding="utf-8")
    _write_fake_torch_checkpoint(checkpoint_path, expected_lavasr_v2_state_keys())
    resolved_weights = ResolvedWeights(
        manifest=WeightManifest(id="lavasr-v2-bwe"),
        manifest_path=tmp_path / "manifest.json",
        root_dir=tmp_path,
        files={
            LAVASR_CONFIG_PATH: config_path,
            LAVASR_WEIGHTS_PATH: checkpoint_path,
        },
    )

    info = validate_lavasr_v2_weight_bundle(resolved_weights)

    assert info.config.backbone.dim == 512
    assert info.state_key_count == len(expected_lavasr_v2_state_keys())


def _write_fake_torch_checkpoint(path: Path, keys: set[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(sorted(keys)))
