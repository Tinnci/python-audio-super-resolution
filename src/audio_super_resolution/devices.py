from __future__ import annotations

import importlib.util
from dataclasses import dataclass

LOGICAL_DEVICES = ("cpu", "cuda", "rocm", "xpu", "mps", "directml")
VALID_DEVICES = (*LOGICAL_DEVICES, "auto")
AUTO_DEVICE_ORDER = ("cuda", "rocm", "mps", "xpu", "directml", "cpu")


@dataclass(frozen=True)
class DeviceInfo:
    """Detected runtime support for one inference device."""

    name: str
    available: bool
    reason: str | None = None
    runtime_device: str | None = None


def available_devices() -> list[DeviceInfo]:
    """Return CPU and optional accelerator availability without requiring torch."""

    devices = [DeviceInfo(name="cpu", available=True, runtime_device="cpu")]
    torch = _import_torch()
    if torch is None:
        return [
            *devices,
            DeviceInfo(name="cuda", available=False, reason="torch is not installed", runtime_device="cuda"),
            DeviceInfo(name="rocm", available=False, reason="torch is not installed", runtime_device="cuda"),
            DeviceInfo(name="xpu", available=False, reason="torch is not installed", runtime_device="xpu"),
            DeviceInfo(name="mps", available=False, reason="torch is not installed", runtime_device="mps"),
            _directml_info(),
        ]

    torch_cuda_available = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    cuda_available = torch_cuda_available and not hip_version
    devices.append(
        DeviceInfo(
            name="cuda",
            available=cuda_available,
            reason=None if cuda_available else "cuda unavailable",
            runtime_device="cuda",
        )
    )
    rocm_available = torch_cuda_available and bool(hip_version)
    devices.append(
        DeviceInfo(
            name="rocm",
            available=rocm_available,
            reason=None if rocm_available else "rocm unavailable",
            runtime_device="cuda",
        )
    )

    xpu_backend = getattr(torch, "xpu", None)
    xpu_available = bool(getattr(xpu_backend, "is_available", lambda: False)())
    devices.append(DeviceInfo(name="xpu", available=xpu_available, reason=None if xpu_available else "xpu unavailable"))

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(getattr(mps_backend, "is_available", lambda: False)())
    devices.append(
        DeviceInfo(
            name="mps",
            available=mps_available,
            reason=None if mps_available else "mps unavailable",
            runtime_device="mps",
        )
    )
    devices.append(_directml_info())
    return devices


def resolve_device(requested: str, supported_devices: tuple[str, ...] | None = None) -> str:
    """Resolve a requested device, using the best available accelerator for auto."""

    normalized = requested.lower()
    devices = {device.name: device for device in available_devices()}
    supported = _normalize_supported_devices(supported_devices)
    if normalized == "auto":
        for candidate in AUTO_DEVICE_ORDER:
            if candidate in supported and devices[candidate].available:
                return devices[candidate].runtime_device or candidate
        choices = ", ".join(supported)
        raise ValueError(f"No supported devices are available for the selected backend: {choices}")

    if normalized not in devices:
        raise ValueError(f"Unsupported device: {requested}")
    if normalized not in supported:
        choices = ", ".join(supported)
        raise ValueError(
            f"Device {normalized!r} is not supported by the selected backend; supported devices: {choices}"
        )
    if not devices[normalized].available:
        reason = devices[normalized].reason or "device unavailable"
        raise ValueError(f"Requested device {normalized!r} is not available: {reason}")
    return devices[normalized].runtime_device or normalized


def _import_torch():
    if importlib.util.find_spec("torch") is None:
        return None

    try:
        import torch
    except ImportError:
        return None
    return torch


def _directml_info() -> DeviceInfo:
    installed = _module_available("torch_directml")
    return DeviceInfo(
        name="directml",
        available=installed,
        reason=None if installed else "torch-directml is not installed",
        runtime_device="directml",
    )


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _normalize_supported_devices(supported_devices: tuple[str, ...] | None) -> tuple[str, ...]:
    if supported_devices is None:
        return LOGICAL_DEVICES
    normalized = tuple(device.lower() for device in supported_devices)
    unknown = [device for device in normalized if device not in LOGICAL_DEVICES]
    if unknown:
        choices = ", ".join(LOGICAL_DEVICES)
        raise ValueError(f"Supported devices contain unknown device(s): {', '.join(unknown)}. Choices: {choices}")
    return normalized
