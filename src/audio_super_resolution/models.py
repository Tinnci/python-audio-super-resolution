from __future__ import annotations

from dataclasses import dataclass

from .backends import available_backends, list_backend_model_specs
from .specs import ModelSpec


@dataclass(frozen=True)
class ModelInfo:
    """User-facing metadata for a selectable enhancement model."""

    id: str
    backend: str
    name: str
    description: str
    target_sample_rate: int | None
    installed: bool
    package_extra: str | None = None
    model_name: str | None = None
    implementation: str = "unknown"
    domain: tuple[str, ...] = ()
    architecture: str | None = None
    target_sample_rates: tuple[int, ...] | None = None
    fixed_target_sr: bool = False
    weights_license: str | None = None
    weights_source: str | None = None
    weights_hash: str | None = None
    weight_provider: str | None = None
    weight_manifest_url: str | None = None
    default_weight_revision: str | None = None
    requires_weights: bool = False
    maturity: str = "unknown"


def list_models(filter_text: str | None = None) -> list[ModelInfo]:
    """Return known enhancement models, optionally filtered by text."""

    backend_map = {backend.name: backend for backend in available_backends()}
    models = [
        _model_info_from_spec(
            spec,
            installed=backend_map[spec.backend].installed,
            package_extra=backend_map[spec.backend].package_extra,
        )
        for spec in list_backend_model_specs()
    ]

    if filter_text is None:
        return models

    needle = filter_text.lower()
    return [
        model
        for model in models
        if needle in model.id.lower()
        or needle in model.backend.lower()
        or needle in model.name.lower()
        or needle in model.description.lower()
    ]


def get_model_spec(model_id: str | ModelSpec) -> ModelSpec:
    """Return a model spec by id or model_name."""

    if isinstance(model_id, ModelSpec):
        return model_id

    for spec in list_backend_model_specs():
        if spec.id == model_id or spec.model_name == model_id:
            return spec
    raise ValueError(f"Unknown model {model_id!r}")


def find_model_spec(backend: str, model_name: str | None = None) -> ModelSpec:
    """Return the selected model spec for a backend and optional model name."""

    specs = [spec for spec in list_backend_model_specs() if spec.backend == backend]
    if not specs:
        raise ValueError(f"Backend {backend!r} does not expose model specs")

    if model_name is not None:
        for spec in specs:
            if spec.id == model_name or spec.model_name == model_name:
                return spec

    if len(specs) == 1:
        return specs[0]

    choices = ", ".join(spec.id for spec in specs)
    raise ValueError(f"Model name is required for backend {backend!r}. Available models: {choices}")


def _model_info_from_spec(spec: ModelSpec, installed: bool, package_extra: str | None) -> ModelInfo:
    return ModelInfo(
        id=spec.id,
        backend=spec.backend,
        name=spec.name,
        description=spec.description,
        target_sample_rate=spec.target_sample_rate,
        installed=installed,
        package_extra=package_extra,
        model_name=spec.model_name,
        implementation=spec.implementation,
        domain=spec.domain,
        architecture=spec.architecture,
        target_sample_rates=spec.target_sample_rates,
        fixed_target_sr=spec.fixed_target_sr,
        weights_license=spec.weights_license,
        weights_source=spec.weights_source,
        weights_hash=spec.weights_hash,
        weight_provider=spec.weight_provider,
        weight_manifest_url=spec.weight_manifest_url,
        default_weight_revision=spec.default_weight_revision,
        requires_weights=spec.requires_weights,
        maturity=spec.maturity,
    )
