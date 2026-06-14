from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """Detected runtime support for one inference device."""

    name: str
    available: bool
    reason: str | None = None


def available_devices() -> list[DeviceInfo]:
    """Return CPU and optional accelerator availability without requiring torch."""

    devices = [DeviceInfo(name="cpu", available=True)]
    torch = _import_torch()
    if torch is None:
        return devices + [
            DeviceInfo(name="cuda", available=False, reason="torch is not installed"),
            DeviceInfo(name="mps", available=False, reason="torch is not installed"),
        ]

    cuda_available = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
    devices.append(
        DeviceInfo(name="cuda", available=cuda_available, reason=None if cuda_available else "cuda unavailable")
    )

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(getattr(mps_backend, "is_available", lambda: False)())
    devices.append(DeviceInfo(name="mps", available=mps_available, reason=None if mps_available else "mps unavailable"))
    return devices


def resolve_device(requested: str) -> str:
    """Resolve a requested device, using the best available accelerator for auto."""

    normalized = requested.lower()
    devices = {device.name: device for device in available_devices()}
    if normalized == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if devices[candidate].available:
                return candidate

    if normalized not in devices:
        raise ValueError(f"Unsupported device: {requested}")
    if not devices[normalized].available:
        reason = devices[normalized].reason or "device unavailable"
        raise ValueError(f"Requested device {normalized!r} is not available: {reason}")
    return normalized


def _import_torch():
    if importlib.util.find_spec("torch") is None:
        return None

    try:
        import torch
    except ImportError:
        return None
    return torch
