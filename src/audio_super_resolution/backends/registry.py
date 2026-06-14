from __future__ import annotations

from ..config import InferenceConfig
from ..specs import ModelSpec
from .audiosr_external import AudiosrBackend
from .base import BackendInfo, EnhancementBackend, backend_model_specs
from .lavasr_compat import LavaSRCompatBackend
from .sinc import SincResampleBackend

_BACKENDS: dict[str, type[EnhancementBackend]] = {}


def register_backend(backend_type: type[EnhancementBackend], *, replace: bool = False) -> None:
    """Register an enhancement backend type by its public name."""

    name = getattr(backend_type, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("backend_type must define a non-empty string name")
    if name in _BACKENDS and not replace:
        raise ValueError(f"Backend {name!r} is already registered")
    _BACKENDS[name] = backend_type


def registered_backend_types() -> dict[str, type[EnhancementBackend]]:
    """Return a copy of the backend registry."""

    return dict(_BACKENDS)


def available_backends() -> list[BackendInfo]:
    """Return the registered enhancement backends."""

    return [
        BackendInfo(
            name=name,
            description=backend.description,
            installed=_backend_is_available(backend),
            optional_dependency=getattr(backend, "optional_dependency", None),
            package_extra=getattr(backend, "package_extra", None),
        )
        for name, backend in sorted(_BACKENDS.items(), key=lambda item: item[0])
    ]


def get_backend(name: str, config: InferenceConfig | None = None) -> EnhancementBackend:
    """Create a backend by name."""

    try:
        backend_type = _BACKENDS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unknown backend {name!r}. Available backends: {choices}") from exc

    return backend_type(config=config)


def list_backend_model_specs() -> list[ModelSpec]:
    """Return model specs exposed by every registered backend."""

    specs: list[ModelSpec] = []
    for _, backend_type in sorted(_BACKENDS.items(), key=lambda item: item[0]):
        specs.extend(backend_model_specs(backend_type))
    return specs


def _backend_is_available(backend_type: type[EnhancementBackend]) -> bool:
    checker = getattr(backend_type, "is_available", None)
    if checker is None:
        return True
    return bool(checker())


register_backend(AudiosrBackend)
register_backend(LavaSRCompatBackend)
register_backend(SincResampleBackend)
