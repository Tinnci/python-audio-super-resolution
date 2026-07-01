import pytest

from audio_super_resolution.runtime import list_runtime_providers, resolve_runtime_provider


def test_list_runtime_providers_uses_mocked_import_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audio_super_resolution.runtime._module_available", lambda name: name == "torch")

    providers = {provider.name: provider for provider in list_runtime_providers()}

    assert providers["python"].installed is True
    assert providers["external-package"].installed is True
    assert providers["torch-eager"].installed is True
    assert providers["onnxruntime"].installed is False
    assert "cuda" in providers["torch-eager"].devices


def test_resolve_runtime_provider_auto_picks_first_installed_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audio_super_resolution.runtime._module_available", lambda name: name == "torch")

    provider = resolve_runtime_provider("auto", supported_providers=("torch-eager", "onnxruntime"))

    assert provider.name == "torch-eager"


def test_resolve_runtime_provider_rejects_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="not supported by the selected backend"):
        resolve_runtime_provider("onnxruntime", supported_providers=("python",))


def test_resolve_runtime_provider_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audio_super_resolution.runtime._module_available", lambda name: False)

    with pytest.raises(RuntimeError, match="audio-super-resolution\\[lavasr\\]"):
        resolve_runtime_provider("torch-eager", supported_providers=("torch-eager",))
