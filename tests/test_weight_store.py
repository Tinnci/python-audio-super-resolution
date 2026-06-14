from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from audio_super_resolution import (
    InferenceConfig,
    ModelSpec,
    WeightFile,
    WeightFileSpec,
    WeightManifest,
    download_model_weights,
    register_weight_provider,
    resolve_model_weights,
    sha256_file,
    verify_model_weights,
    write_weight_manifest,
)


def test_resolve_model_weights_prefers_explicit_manifest(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "explicit", b"explicit")
    cache_manifest_path = _write_manifest(tmp_path / "cache" / "demo-model", b"cache")
    (cache_manifest_path.parent / ".complete").write_text("ok\n", encoding="utf-8")
    spec = _spec()
    config = InferenceConfig(model_cache_dir=tmp_path / "cache", weights_manifest=manifest_path)

    resolved = resolve_model_weights(spec, config)

    assert resolved.manifest_path == manifest_path
    assert resolved.files["weights.bin"].read_bytes() == b"explicit"


def test_resolve_model_weights_uses_verified_cache_manifest(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "cache" / "demo-model", b"cache")
    (manifest_path.parent / ".complete").write_text("ok\n", encoding="utf-8")
    spec = _spec()
    config = InferenceConfig(model_cache_dir=tmp_path / "cache")

    resolved = resolve_model_weights(spec, config)

    assert resolved.manifest_path == manifest_path


def test_resolve_model_weights_missing_cache_does_not_download_by_default(tmp_path: Path) -> None:
    spec = _spec()
    config = InferenceConfig(model_cache_dir=tmp_path / "cache")

    with pytest.raises(RuntimeError, match="--download-weights"):
        resolve_model_weights(spec, config)


def test_download_model_weights_uses_provider_temp_dir_and_verified_cache(tmp_path: Path) -> None:
    register_weight_provider("fake", FakeProvider, replace=True)
    spec = _spec()

    output_dir = download_model_weights(spec, cache_dir=tmp_path / "cache")

    assert output_dir == tmp_path / "cache" / "demo-model"
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / ".complete").is_file()
    assert not (tmp_path / "cache" / ".demo-model.download.tmp").exists()
    assert verify_model_weights(spec, cache_dir=tmp_path / "cache").files["weights.bin"].read_bytes() == b"weights"


def test_download_model_weights_cleans_temp_and_preserves_existing_cache_on_failure(tmp_path: Path) -> None:
    register_weight_provider("fake", FakeProvider, replace=True)
    spec = _spec()
    output_dir = download_model_weights(spec, cache_dir=tmp_path / "cache")
    (output_dir / "sentinel.txt").write_text("keep", encoding="utf-8")

    register_weight_provider("fake", BadProvider, replace=True)
    with pytest.raises(ValueError, match="weight (hash|size) mismatch"):
        download_model_weights(spec, cache_dir=tmp_path / "cache", force=True)

    assert (output_dir / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "cache" / ".demo-model.download.tmp").exists()


class FakeProvider:
    name = "fake"

    def __init__(self, source: str) -> None:
        self.source = source

    def fetch_manifest(self, spec: ModelSpec, revision: str | None) -> WeightManifest:
        return WeightManifest(
            id=spec.id,
            provider="fake",
            source=self.source,
            revision=revision or spec.default_weight_revision,
            files=(WeightFile(path="weights.bin", sha256=_fixture_digest(), size=len(b"weights")),),
        )

    def download_file(self, remote_path: str, destination: Path, revision: str | None) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")


class BadProvider(FakeProvider):
    def download_file(self, remote_path: str, destination: Path, revision: str | None) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bad")


def _spec() -> ModelSpec:
    return ModelSpec(
        id="demo-model",
        backend="demo",
        name="Demo",
        description="Demo model",
        implementation="self_torch",
        domain=("speech",),
        architecture="demo-net",
        target_sample_rates=(48000,),
        weights_source="demo/source",
        weight_provider="fake",
        default_weight_revision="main",
        requires_weights=True,
        weight_files=(WeightFileSpec(path="weights.bin", sha256=_fixture_digest(), size=len(b"weights")),),
    )


def _write_manifest(directory: Path, content: bytes) -> Path:
    directory.mkdir(parents=True)
    weight_path = directory / "weights.bin"
    weight_path.write_bytes(content)
    manifest_path = directory / "manifest.json"
    write_weight_manifest(
        manifest_path,
        WeightManifest(
            id="demo-model",
            files=(WeightFile(path="weights.bin", sha256=sha256_file(weight_path), size=len(content)),),
        ),
    )
    return manifest_path


def _fixture_digest() -> str:
    return hashlib.sha256(b"weights").hexdigest()
