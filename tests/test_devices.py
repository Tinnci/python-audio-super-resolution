import pytest

from audio_super_resolution import available_devices, resolve_device


def test_available_devices_always_includes_cpu() -> None:
    devices = {device.name: device for device in available_devices()}

    assert devices["cpu"].available is True


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("quantum")
