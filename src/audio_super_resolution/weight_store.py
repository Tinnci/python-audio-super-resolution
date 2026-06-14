from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import InferenceConfig, default_model_cache_dir
from .downloads import download_model_weights as _download_model_weights
from .models import find_model_spec, get_model_spec
from .specs import ModelSpec
from .weights import (
    WeightManifest,
    resolve_manifest_file_paths,
    verify_weight_manifest,
)


@dataclass(frozen=True)
class ResolvedWeights:
    """Verified local paths for a model's weight files."""

    manifest: WeightManifest
    manifest_path: Path
    root_dir: Path
    files: dict[str, Path]


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
        return _resolve_manifest_path(config.weights_manifest)

    cache_manifest_path = Path(config.model_cache_dir).expanduser() / spec.id / "manifest.json"
    if cache_manifest_path.is_file() and (cache_manifest_path.parent / ".complete").is_file():
        return _resolve_manifest_path(cache_manifest_path)

    should_download = allow_download or config.download_weights
    if should_download:
        downloaded_dir = _download_model_weights(
            spec,
            config.model_cache_dir,
            revision=config.weight_revision,
            force=force_download or config.force_download,
        )
        return _resolve_manifest_path(downloaded_dir / "manifest.json")

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
    return _download_model_weights(spec, cache_dir or default_model_cache_dir(), revision=revision, force=force)


def verify_model_weights(
    model_id: str | ModelSpec,
    cache_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> ResolvedWeights:
    """Verify a registered model's local weights."""

    if manifest_path is not None:
        return _resolve_manifest_path(manifest_path)

    spec = get_model_spec(model_id)
    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_model_cache_dir()
    return _resolve_manifest_path(resolved_cache_dir / spec.id / "manifest.json")


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


def _resolve_manifest_path(path: str | Path) -> ResolvedWeights:
    manifest_path = Path(path).expanduser()
    manifest = verify_weight_manifest(manifest_path)
    return ResolvedWeights(
        manifest=manifest,
        manifest_path=manifest_path,
        root_dir=manifest_path.parent,
        files=resolve_manifest_file_paths(manifest_path, manifest),
    )
