from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import InferenceConfig, default_model_cache_dir
from .downloads import download_weights_for_spec
from .models import find_model_spec, get_model_spec
from .specs import ModelSpec, WeightFileSpec
from .weights import (
    WeightFile,
    WeightManifest,
    resolve_manifest_file_paths,
    validate_weight_file_path,
    verify_weight_manifest,
)


@dataclass(frozen=True)
class ResolvedWeights:
    """Verified local paths for a model's weight files."""

    manifest: WeightManifest
    manifest_path: Path
    root_dir: Path
    files: dict[str, Path]

    def path_for(self, file_path: str) -> Path:
        """Return a verified local path by manifest-relative file path."""

        safe_path = validate_weight_file_path(file_path)
        try:
            return self.files[safe_path]
        except KeyError as exc:
            choices = ", ".join(sorted(self.files))
            raise KeyError(f"Weight file {safe_path!r} was not resolved. Available files: {choices}") from exc


def resolve_model_weights(
    model: str | ModelSpec,
    config: InferenceConfig,
    *,
    allow_download: bool = False,
    force_download: bool = False,
) -> ResolvedWeights:
    """Resolve verified model weights from an explicit manifest, cache, or explicit download."""

    spec = get_model_spec(model) if isinstance(model, str) else model
    if config.weights_manifest is not None:
        return _resolve_manifest_path(config.weights_manifest, spec=spec)

    cache_manifest_path = Path(config.model_cache_dir).expanduser() / spec.id / "manifest.json"
    if cache_manifest_path.is_file() and (cache_manifest_path.parent / ".complete").is_file():
        return _resolve_manifest_path(cache_manifest_path, spec=spec)

    should_download = allow_download or config.download_weights
    if should_download:
        downloaded_dir = download_weights_for_spec(
            spec,
            config.model_cache_dir,
            revision=config.weight_revision,
            force=force_download or config.force_download,
        )
        return _resolve_manifest_path(downloaded_dir / "manifest.json", spec=spec)

    raise RuntimeError(
        f"Weights for model {spec.id!r} were not found. Provide --weights-manifest or run "
        f"`audio-super-res --backend {spec.backend} --model-name {spec.id} "
        f"--download-weights --prepare-model-cache`."
    )


def download_model_weights(
    model_id: str | ModelSpec,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Download a registered model's weights into the cache directory."""

    spec = get_model_spec(model_id)
    return download_weights_for_spec(spec, cache_dir or default_model_cache_dir(), revision=revision, force=force)


def verify_model_weights(
    model_id: str | ModelSpec,
    cache_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> ResolvedWeights:
    """Verify a registered model's local weights."""

    spec = get_model_spec(model_id)
    if manifest_path is not None:
        return _resolve_manifest_path(manifest_path, spec=spec)

    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_model_cache_dir()
    return _resolve_manifest_path(resolved_cache_dir / spec.id / "manifest.json", spec=spec)


def resolve_backend_model_weights(
    backend: str,
    model_name: str | None,
    config: InferenceConfig,
    *,
    allow_download: bool = False,
) -> ResolvedWeights:
    """Resolve weights for a backend/model-name pair."""

    spec = find_model_spec(backend=backend, model_name=model_name)
    return resolve_model_weights(spec, config, allow_download=allow_download)


def validate_weight_manifest_matches_spec(manifest: WeightManifest, spec: ModelSpec) -> None:
    """Validate that a verified manifest belongs to the requested model spec."""

    if manifest.id != spec.id:
        raise ValueError(f"weight manifest id mismatch: expected {spec.id!r}, got {manifest.id!r}")

    _validate_manifest_metadata(manifest, spec)
    _validate_manifest_files(manifest, spec)


def _validate_manifest_metadata(manifest: WeightManifest, spec: ModelSpec) -> None:
    _validate_optional_match("provider", manifest.provider, spec.weight_provider)
    _validate_optional_match("source", manifest.source, spec.weights_source)
    _validate_optional_match("architecture", manifest.architecture, spec.architecture)
    _validate_optional_match("target_sample_rate", manifest.target_sample_rate, spec.target_sample_rate)


def _validate_manifest_files(manifest: WeightManifest, spec: ModelSpec) -> None:
    expected_files = {validate_weight_file_path(file_spec.path): file_spec for file_spec in spec.weight_files}
    if not expected_files:
        return

    actual_files = {file_entry.path: file_entry for file_entry in manifest.file_entries}
    missing = sorted(set(expected_files) - set(actual_files))
    if missing:
        raise ValueError(f"weight manifest is missing required files for {spec.id!r}: {', '.join(missing)}")

    for file_path, expected_file in expected_files.items():
        _validate_manifest_file_metadata(file_path, expected_file, actual_files[file_path])


def _validate_manifest_file_metadata(file_path: str, expected_file: WeightFileSpec, actual_file: WeightFile) -> None:
    if expected_file.sha256 is not None:
        _validate_required_sha256(file_path, expected_file.sha256, actual_file.sha256)
    if expected_file.size is not None:
        _validate_required_size(file_path, expected_file.size, actual_file.size)


def _validate_required_sha256(file_path: str, expected_sha256: str, actual_sha256: str | None) -> None:
    if actual_sha256 is None:
        raise ValueError(f"weight manifest is missing sha256 for required file {file_path!r}")
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"weight manifest sha256 mismatch for {file_path!r}: expected {expected_sha256}, got {actual_sha256}"
        )


def _validate_required_size(file_path: str, expected_size: int, actual_size: int | None) -> None:
    if actual_size is None:
        raise ValueError(f"weight manifest is missing size for required file {file_path!r}")
    if actual_size != expected_size:
        raise ValueError(
            f"weight manifest size mismatch for {file_path!r}: expected {expected_size}, got {actual_size}"
        )


def _resolve_manifest_path(path: str | Path, *, spec: ModelSpec | None = None) -> ResolvedWeights:
    manifest_path = Path(path).expanduser()
    manifest = verify_weight_manifest(manifest_path)
    if spec is not None:
        validate_weight_manifest_matches_spec(manifest, spec)
    return ResolvedWeights(
        manifest=manifest,
        manifest_path=manifest_path,
        root_dir=manifest_path.parent,
        files=resolve_manifest_file_paths(manifest_path, manifest),
    )


def _validate_optional_match(field: str, actual: object | None, expected: object | None) -> None:
    if actual is not None and expected is not None and actual != expected:
        raise ValueError(f"weight manifest {field} mismatch: expected {expected!r}, got {actual!r}")
