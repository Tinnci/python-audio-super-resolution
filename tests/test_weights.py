from pathlib import Path

import pytest

from audio_super_resolution import (
    WeightFile,
    WeightManifest,
    read_weight_manifest,
    resolve_manifest_file_paths,
    resolve_weight_path,
    sha256_file,
    verify_weight_file,
    verify_weight_manifest,
    write_weight_manifest,
)


def test_weight_manifest_round_trips_and_resolves_cache_path(tmp_path: Path) -> None:
    manifest = WeightManifest(
        id="demo-model",
        filename="weights.safetensors",
        source="https://example.invalid/demo-model",
        sha256="0" * 64,
        license="MIT",
        architecture="demo-net",
        target_sample_rate=48000,
    )
    manifest_path = tmp_path / "manifests" / "demo.json"

    assert write_weight_manifest(manifest_path, manifest) == manifest_path
    loaded = read_weight_manifest(manifest_path)

    assert loaded == manifest
    assert resolve_weight_path(tmp_path / "cache", manifest) == (
        tmp_path / "cache" / "demo-model" / "weights.safetensors"
    )


def test_verify_weight_file_accepts_matching_sha256(tmp_path: Path) -> None:
    weight_path = tmp_path / "weights.bin"
    weight_path.write_bytes(b"weights")
    digest = sha256_file(weight_path)

    assert verify_weight_file(weight_path, digest) == weight_path


def test_verify_weight_file_rejects_hash_mismatch(tmp_path: Path) -> None:
    weight_path = tmp_path / "weights.bin"
    weight_path.write_bytes(b"weights")

    with pytest.raises(ValueError, match="weight hash mismatch"):
        verify_weight_file(weight_path, "0" * 64)


def test_multi_file_weight_manifest_round_trips_and_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "enhancer_v2" / "config.yaml"
    model_path = tmp_path / "enhancer_v2" / "pytorch_model.bin"
    config_path.parent.mkdir()
    config_path.write_bytes(b"config")
    model_path.write_bytes(b"model")
    manifest = WeightManifest(
        id="lavasr-v2-bwe",
        provider="huggingface",
        source="YatharthS/LavaSR",
        revision="abc123",
        license="Apache-2.0",
        architecture="vocos-istft-bwe",
        target_sample_rate=48000,
        files=(
            WeightFile(path="enhancer_v2/config.yaml", sha256=sha256_file(config_path), size=len(b"config")),
            WeightFile(path="enhancer_v2/pytorch_model.bin", sha256=sha256_file(model_path), size=len(b"model")),
        ),
    )
    manifest_path = tmp_path / "manifest.json"

    write_weight_manifest(manifest_path, manifest)
    loaded = verify_weight_manifest(manifest_path)
    paths = resolve_manifest_file_paths(manifest_path, loaded)

    assert loaded == manifest
    assert paths["enhancer_v2/config.yaml"] == config_path
    assert paths["enhancer_v2/pytorch_model.bin"] == model_path


def test_verify_weight_file_rejects_size_mismatch(tmp_path: Path) -> None:
    weight_path = tmp_path / "weights.bin"
    weight_path.write_bytes(b"weights")

    with pytest.raises(ValueError, match="weight size mismatch"):
        verify_weight_file(weight_path, expected_sha256=None, expected_size=999)
