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
    tasks: tuple[str, ...] = ()
    architecture: str | None = None
    input_sample_rates: tuple[int, ...] | None = None
    input_sample_rate_range: tuple[int, int] | None = None
    target_sample_rates: tuple[int, ...] | None = None
    fixed_target_sr: bool = False
    code_license: str | None = None
    weights_license: str | None = None
    weights_source: str | None = None
    weights_hash: str | None = None
    weight_provider: str | None = None
    weight_file_count: int = 0
    weight_size_bytes: int | None = None
    weight_manifest_url: str | None = None
    default_weight_revision: str | None = None
    requires_weights: bool = False
    upstream_url: str | None = None
    recommended_for: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    attribution_required: bool = False
    supports_array_io: bool = False
    supports_file_io: bool = False
    supports_chunking: bool = False
    deterministic: bool = False
    supports_cpu: bool = False
    supports_cuda: bool = False
    supports_mps: bool = False
    requires_gpu: bool = False
    precision_modes: tuple[str, ...] = ()
    accelerators: tuple[str, ...] = ()
    runtime_providers: tuple[str, ...] = ()
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
    return [model for model in models if needle in _model_search_text(model)]


def get_model_spec(model_id: str | ModelSpec) -> ModelSpec:
    """Return a model spec by id or model_name."""

    if isinstance(model_id, ModelSpec):
        return model_id

    requested = model_id.lower()
    for spec in list_backend_model_specs():
        if spec.id.lower() == requested or (spec.model_name is not None and spec.model_name.lower() == requested):
            return spec
    raise ValueError(f"Unknown model {model_id!r}")


def find_model_spec(backend: str, model_name: str | None = None) -> ModelSpec:
    """Return the selected model spec for a backend and optional model name."""

    specs = [spec for spec in list_backend_model_specs() if spec.backend == backend]
    if not specs:
        raise ValueError(f"Backend {backend!r} does not expose model specs")

    if model_name is not None:
        requested = model_name.lower()
        for spec in specs:
            if spec.id.lower() == requested or (spec.model_name is not None and spec.model_name.lower() == requested):
                return spec
        choices = ", ".join(spec.id for spec in specs)
        raise ValueError(f"Unknown model {model_name!r} for backend {backend!r}. Available models: {choices}")

    if len(specs) == 1:
        return specs[0]

    choices = ", ".join(spec.id for spec in specs)
    raise ValueError(f"Model name is required for backend {backend!r}. Available models: {choices}")


def _model_info_from_spec(spec: ModelSpec, installed: bool, package_extra: str | None) -> ModelInfo:
    capability = spec.capability
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
        tasks=spec.tasks,
        architecture=spec.architecture,
        input_sample_rates=spec.input_sample_rates,
        input_sample_rate_range=spec.input_sample_rate_range,
        target_sample_rates=spec.target_sample_rates,
        fixed_target_sr=spec.fixed_target_sr,
        code_license=spec.code_license,
        weights_license=spec.weights_license,
        weights_source=spec.weights_source,
        weights_hash=spec.weights_hash,
        weight_provider=spec.weight_provider,
        weight_file_count=len(spec.weight_files),
        weight_size_bytes=_total_weight_size_bytes(spec),
        weight_manifest_url=spec.weight_manifest_url,
        default_weight_revision=spec.default_weight_revision,
        requires_weights=spec.requires_weights,
        upstream_url=spec.upstream_url,
        recommended_for=spec.recommended_for,
        known_limitations=spec.known_limitations,
        validation=spec.validation,
        attribution_required=spec.attribution_required,
        supports_array_io=capability.supports_array_io if capability is not None else False,
        supports_file_io=capability.supports_file_io if capability is not None else False,
        supports_chunking=capability.supports_chunking if capability is not None else False,
        deterministic=capability.deterministic if capability is not None else False,
        supports_cpu=capability.supports_cpu if capability is not None else False,
        supports_cuda=capability.supports_cuda if capability is not None else False,
        supports_mps=capability.supports_mps if capability is not None else False,
        requires_gpu=capability.requires_gpu if capability is not None else False,
        precision_modes=capability.precision_modes if capability is not None else (),
        accelerators=capability.accelerators if capability is not None else (),
        runtime_providers=capability.runtime_providers if capability is not None else (),
        maturity=spec.maturity,
    )


def _total_weight_size_bytes(spec: ModelSpec) -> int | None:
    if not spec.weight_files:
        return None
    sizes = [file_spec.size for file_spec in spec.weight_files]
    if any(size is None for size in sizes):
        return None
    return sum(size for size in sizes if size is not None)


def _model_search_text(model: ModelInfo) -> str:
    values: list[str] = [
        model.id,
        model.backend,
        model.name,
        model.description,
        model.implementation,
        model.architecture or "",
        model.maturity,
    ]
    values.extend(model.domain)
    values.extend(model.tasks)
    values.extend(model.recommended_for)
    values.extend(model.known_limitations)
    values.extend(model.validation)
    return " ".join(values).lower()
