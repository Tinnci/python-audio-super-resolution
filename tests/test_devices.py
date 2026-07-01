import pytest

from audio_super_resolution import available_devices, resolve_device


def test_available_devices_reports_extended_accelerators_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audio_super_resolution.devices._import_torch", lambda: None)
    monkeypatch.setattr("audio_super_resolution.devices._module_available", lambda name: False)

    devices = {device.name: device for device in available_devices()}

    assert set(devices) == {"cpu", "cuda", "rocm", "xpu", "mps", "directml"}
    assert devices["cpu"].available is True
    assert devices["rocm"].available is False
    assert devices["rocm"].reason == "torch is not installed"


def test_available_devices_always_includes_cpu() -> None:
    devices = {device.name: device for device in available_devices()}

    assert devices["cpu"].available is True


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_respects_backend_supported_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audio_super_resolution.devices._import_torch", lambda: None)
    monkeypatch.setattr("audio_super_resolution.devices._module_available", lambda name: False)

    assert resolve_device("auto", supported_devices=("cpu",)) == "cpu"


def test_resolve_device_rejects_backend_unsupported_device() -> None:
    with pytest.raises(ValueError, match="not supported by the selected backend"):
        resolve_device("cuda", supported_devices=("cpu",))


def test_resolve_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("quantum")
