from __future__ import annotations

import importlib.util
from dataclasses import dataclass

VALID_RUNTIME_PROVIDERS = ("auto", "python", "torch-eager", "onnxruntime", "external-package")


@dataclass(frozen=True)
class RuntimeProviderInfo:
    """User-facing runtime provider metadata."""

    name: str
    description: str
    installed: bool
    devices: tuple[str, ...]
    reason: str | None = None
    package_extra: str | None = None


@dataclass(frozen=True)
class RuntimeProviderSpec:
    """Static runtime provider declaration."""

    name: str
    description: str
    devices: tuple[str, ...]
    module_name: str | None = None
    package_extra: str | None = None


_RUNTIME_PROVIDER_SPECS = {
    "python": RuntimeProviderSpec(
        name="python",
        description="Pure Python/NumPy/SciPy runtime used by lightweight CPU backends.",
        devices=("cpu",),
    ),
    "torch-eager": RuntimeProviderSpec(
        name="torch-eager",
        description="PyTorch eager runtime for package-owned torch modules.",
        devices=("cpu", "cuda", "rocm", "xpu", "mps"),
        module_name="torch",
        package_extra="lavasr",
    ),
    "onnxruntime": RuntimeProviderSpec(
        name="onnxruntime",
        description="ONNX Runtime provider family for exported graphs.",
        devices=("cpu", "cuda", "rocm", "directml", "openvino"),
        module_name="onnxruntime",
    ),
    "external-package": RuntimeProviderSpec(
        name="external-package",
        description="Runtime owned by a selected external package backend.",
        devices=("cpu", "cuda", "mps"),
    ),
}


def list_runtime_providers() -> list[RuntimeProviderInfo]:
    """Return known runtime providers without importing heavy runtimes."""

    return [runtime_provider_info(name) for name in _RUNTIME_PROVIDER_SPECS]


def runtime_provider_info(name: str) -> RuntimeProviderInfo:
    """Return runtime provider metadata by name."""

    try:
        spec = _RUNTIME_PROVIDER_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(VALID_RUNTIME_PROVIDERS)
        raise ValueError(f"Unknown runtime provider {name!r}. Choices: {choices}") from exc

    installed = spec.module_name is None or _module_available(spec.module_name)
    reason = None if installed else f"{spec.module_name} is not installed"
    return RuntimeProviderInfo(
        name=spec.name,
        description=spec.description,
        installed=installed,
        devices=spec.devices,
        reason=reason,
        package_extra=spec.package_extra,
    )


def resolve_runtime_provider(requested: str, supported_providers: tuple[str, ...]) -> RuntimeProviderInfo:
    """Resolve a runtime provider request for one backend/model."""

    normalized = requested.lower()
    if normalized not in VALID_RUNTIME_PROVIDERS:
        choices = ", ".join(VALID_RUNTIME_PROVIDERS)
        raise ValueError(f"runtime_provider must be one of: {choices}")

    supported = _normalize_supported_providers(supported_providers)
    if normalized == "auto":
        for provider_name in supported:
            provider = runtime_provider_info(provider_name)
            if provider.installed:
                return provider
        details = "; ".join(_provider_unavailable_message(runtime_provider_info(name)) for name in supported)
        raise RuntimeError(f"No supported runtime provider is installed for the selected backend/model: {details}")

    if normalized not in supported:
        choices = ", ".join(supported)
        raise ValueError(
            f"Runtime provider {normalized!r} is not supported by the selected backend; supported providers: {choices}"
        )

    provider = runtime_provider_info(normalized)
    if not provider.installed:
        raise RuntimeError(_provider_unavailable_message(provider))
    return provider


def _provider_unavailable_message(provider: RuntimeProviderInfo) -> str:
    reason = provider.reason or "provider unavailable"
    install_hint = (
        f" Install it with `uv pip install audio-super-resolution[{provider.package_extra}]`."
        if provider.package_extra
        else ""
    )
    return f"Runtime provider {provider.name!r} is not installed: {reason}.{install_hint}"


def _normalize_supported_providers(supported_providers: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(provider.lower() for provider in supported_providers)
    unknown = [provider for provider in normalized if provider not in _RUNTIME_PROVIDER_SPECS]
    if unknown:
        choices = ", ".join(_RUNTIME_PROVIDER_SPECS)
        raise ValueError(f"Supported providers contain unknown provider(s): {', '.join(unknown)}. Choices: {choices}")
    return normalized


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False
