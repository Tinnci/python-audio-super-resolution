from __future__ import annotations

import platform
from importlib import import_module
from typing import Any, cast

_resource_module: Any
try:
    import resource as _resource_module
except ImportError:  # pragma: no cover - exercised on platforms without resource.
    _resource_module = None

_resource: Any | None = _resource_module


def peak_rss_snapshot() -> dict[str, object]:
    """Return a JSON-friendly peak RSS snapshot with platform unit notes."""

    system = platform.system()
    if _resource is None:
        return {
            "strategy": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "platform": system,
            "available": False,
            "peak_rss_mb": None,
            "unit_note": None,
            "fallback": "peak RSS unavailable because the platform does not provide the resource module",
        }

    usage = _resource.getrusage(_resource.RUSAGE_SELF)
    raw_peak_rss = float(usage.ru_maxrss)
    if system == "Darwin":
        peak_rss_mb = raw_peak_rss / (1024 * 1024)
        unit_note = "ru_maxrss reports bytes on Darwin"
    else:
        peak_rss_mb = raw_peak_rss / 1024
        unit_note = "ru_maxrss reports kilobytes on Linux and most Unix platforms"

    return {
        "strategy": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "platform": system,
        "available": True,
        "peak_rss_mb": peak_rss_mb,
        "unit_note": unit_note,
        "fallback": None,
    }


def accelerator_memory_snapshot() -> dict[str, object]:
    """Return optional accelerator memory data without requiring torch."""

    try:
        torch_module = import_module("torch")
    except ImportError:
        return {
            "strategy": "torch.cuda.memory_*",
            "available": False,
            "device": None,
            "allocated_mb": None,
            "reserved_mb": None,
            "fallback": "torch is not installed",
        }
    torch = cast(Any, torch_module)
    cuda = torch.cuda
    if not cuda.is_available():
        return {
            "strategy": "torch.cuda.memory_*",
            "available": False,
            "device": None,
            "allocated_mb": None,
            "reserved_mb": None,
            "fallback": "CUDA is not available",
        }
    device = cuda.current_device()
    return {
        "strategy": "torch.cuda.memory_*",
        "available": True,
        "device": cuda.get_device_name(device),
        "allocated_mb": float(cuda.memory_allocated(device)) / (1024 * 1024),
        "reserved_mb": float(cuda.memory_reserved(device)) / (1024 * 1024),
        "fallback": None,
    }
