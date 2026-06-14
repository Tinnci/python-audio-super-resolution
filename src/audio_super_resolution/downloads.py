from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .specs import ModelSpec
from .weights import WeightFile, WeightManifest, verify_weight_manifest_files, write_weight_manifest

ProviderFactory = Callable[[str], "WeightProvider"]


class WeightProvider(Protocol):
    """Provider capable of describing and downloading model weights."""

    name: str

    def fetch_manifest(self, spec: ModelSpec, revision: str | None) -> WeightManifest:
        """Return the expected manifest for a model spec."""

    def download_file(self, remote_path: str, destination: Path, revision: str | None) -> None:
        """Download one remote file to destination."""


class HuggingFaceProvider:
    """Hugging Face Hub weight provider."""

    name = "huggingface"

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id

    def fetch_manifest(self, spec: ModelSpec, revision: str | None) -> WeightManifest:
        resolved_revision = revision or spec.default_weight_revision
        if not spec.weight_files:
            raise ValueError(f"Model {spec.id!r} does not declare downloadable weight files")

        return WeightManifest(
            schema_version=1,
            id=spec.id,
            provider=self.name,
            source=spec.weights_source,
            revision=resolved_revision,
            license=spec.weights_license,
            architecture=spec.architecture,
            target_sample_rate=spec.target_sample_rate,
            files=tuple(
                WeightFile(path=file_spec.path, sha256=file_spec.sha256, size=file_spec.size)
                for file_spec in spec.weight_files
            ),
        )

    def download_file(self, remote_path: str, destination: Path, revision: str | None) -> None:
        if importlib.util.find_spec("huggingface_hub") is None:
            raise RuntimeError(
                "Downloading Hugging Face weights requires `pip install audio-super-resolution[download]`."
            )

        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(repo_id=self.repo_id, filename=remote_path, revision=revision)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded_path, destination)


_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    HuggingFaceProvider.name: HuggingFaceProvider,
}


def register_weight_provider(name: str, factory: ProviderFactory, *, replace: bool = False) -> None:
    """Register a weight download provider factory."""

    if name in _PROVIDER_FACTORIES and not replace:
        raise ValueError(f"Weight provider {name!r} is already registered")
    _PROVIDER_FACTORIES[name] = factory


def download_model_weights(
    model_spec: ModelSpec,
    cache_dir: str | Path,
    *,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Download and verify model weights into the model cache directory."""

    if not model_spec.requires_weights:
        raise ValueError(f"Model {model_spec.id!r} does not require managed weights")
    if model_spec.weight_provider is None:
        raise ValueError(f"Model {model_spec.id!r} does not declare a weight provider")
    if model_spec.weights_source is None:
        raise ValueError(f"Model {model_spec.id!r} does not declare a weight source")

    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    target_dir = cache_root / model_spec.id
    manifest_path = target_dir / "manifest.json"
    if manifest_path.is_file() and (target_dir / ".complete").is_file() and not force:
        verify_weight_manifest_files(WeightManifest.from_dict(_read_json_manifest(manifest_path)), target_dir)
        return target_dir

    lock_path = cache_root / f".{model_spec.id}.download.lock"
    lock_handle = _acquire_lock(lock_path)
    temp_dir = cache_root / f".{model_spec.id}.download.tmp"
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        provider = _build_provider(model_spec)
        manifest = provider.fetch_manifest(model_spec, revision=revision)
        for file_entry in manifest.file_entries:
            provider.download_file(file_entry.path, temp_dir / file_entry.path, revision=manifest.revision)

        verify_weight_manifest_files(manifest, temp_dir)
        write_weight_manifest(temp_dir / "manifest.json", manifest)
        (temp_dir / ".complete").write_text("ok\n", encoding="utf-8")

        if target_dir.exists():
            if not force:
                raise FileExistsError(f"weight cache already exists: {target_dir}")
            shutil.rmtree(target_dir)
        temp_dir.rename(target_dir)
        return target_dir
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    finally:
        lock_handle.close()
        if lock_path.exists():
            lock_path.unlink()


def _build_provider(model_spec: ModelSpec) -> WeightProvider:
    try:
        factory = _PROVIDER_FACTORIES[str(model_spec.weight_provider)]
    except KeyError as exc:
        raise ValueError(f"Unknown weight provider {model_spec.weight_provider!r}") from exc
    return factory(str(model_spec.weights_source))


def _acquire_lock(path: Path):
    try:
        return path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Weight download is already in progress for this model: {path}") from exc


def _read_json_manifest(path: Path) -> dict[str, object]:
    import json

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("weight manifest root must be a JSON object")
    return loaded
