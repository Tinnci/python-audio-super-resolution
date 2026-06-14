from __future__ import annotations

from pathlib import Path

from .config import InferenceConfig, default_model_cache_dir
from .downloads import download_weights_for_spec
from .models import find_model_spec, get_model_spec
from .specs import ModelSpec
from .weight_store import ResolvedWeights, resolve_weights_for_spec, verify_weights_for_spec


def resolve_model_weights(
    model_id: str | ModelSpec,
    config: InferenceConfig,
    *,
    allow_download: bool = False,
    force_download: bool = False,
) -> ResolvedWeights:
    """Resolve verified weights for a registered model id or model spec."""

    spec = get_model_spec(model_id)
    return resolve_weights_for_spec(
        spec,
        config,
        allow_download=allow_download,
        force_download=force_download,
    )


def download_model_weights(
    model_id: str | ModelSpec,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Download a registered model's weights into the cache directory."""

    spec = get_model_spec(model_id)
    resolved_cache_dir = cache_dir if cache_dir is not None else default_model_cache_dir()
    return download_weights_for_spec(
        spec,
        resolved_cache_dir,
        revision=revision,
        force=force,
    )


def verify_model_weights(
    model_id: str | ModelSpec,
    cache_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> ResolvedWeights:
    """Verify local weights for a registered model id or model spec."""

    spec = get_model_spec(model_id)
    return verify_weights_for_spec(spec, cache_dir=cache_dir, manifest_path=manifest_path)


def resolve_backend_model_weights(
    backend: str,
    model_name: str | None,
    config: InferenceConfig,
    *,
    allow_download: bool = False,
) -> ResolvedWeights:
    """Resolve weights for a backend/model-name pair."""

    spec = find_model_spec(backend=backend, model_name=model_name)
    return resolve_weights_for_spec(spec, config, allow_download=allow_download)
