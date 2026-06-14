from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


@dataclass(frozen=True)
class WeightFile:
    """One file required by a model weight manifest."""

    path: str
    sha256: str | None = None
    size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", validate_weight_file_path(self.path))
        if self.size is not None and self.size < 0:
            raise ValueError("weight file size cannot be negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightFile:
        return cls(
            path=str(data["path"]),
            sha256=_optional_str(data.get("sha256")),
            size=_optional_int(data.get("size")),
        )

    def as_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class WeightManifest:
    """Metadata needed to verify and attribute model weight artifacts.

    `filename`/`sha256` are kept for the original single-file manifest shape.
    New model backends should prefer `files`.
    """

    id: str
    filename: str | None = None
    source: str | None = None
    sha256: str | None = None
    license: str | None = None
    architecture: str | None = None
    target_sample_rate: int | None = None
    schema_version: int = 1
    provider: str | None = None
    revision: str | None = None
    files: tuple[WeightFile, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightManifest:
        raw_files = data.get("files", ())
        if raw_files is None:
            raw_files = ()
        if not isinstance(raw_files, list | tuple):
            raise ValueError("weight manifest files must be a list")

        return cls(
            schema_version=_optional_int(data.get("schema_version")) or 1,
            id=str(data["id"]),
            filename=_optional_str(data.get("filename")),
            source=_optional_str(data.get("source")),
            sha256=_optional_str(data.get("sha256")),
            license=_optional_str(data.get("license")),
            architecture=_optional_str(data.get("architecture")),
            target_sample_rate=_optional_int(data.get("target_sample_rate")),
            provider=_optional_str(data.get("provider")),
            revision=_optional_str(data.get("revision")),
            files=tuple(WeightFile.from_dict(file_data) for file_data in raw_files),
        )

    @classmethod
    def from_single_file(
        cls,
        *,
        id: str,
        filename: str,
        source: str | None = None,
        sha256: str | None = None,
        license: str | None = None,
        architecture: str | None = None,
        target_sample_rate: int | None = None,
    ) -> WeightManifest:
        return cls(
            id=id,
            filename=filename,
            source=source,
            sha256=sha256,
            license=license,
            architecture=architecture,
            target_sample_rate=target_sample_rate,
        )

    @property
    def file_entries(self) -> tuple[WeightFile, ...]:
        """Return all files represented by this manifest, including legacy single-file manifests."""

        if self.files:
            return self.files
        if self.filename is not None:
            return (WeightFile(path=self.filename, sha256=self.sha256),)
        return ()

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "source": self.source,
            "license": self.license,
            "architecture": self.architecture,
            "target_sample_rate": self.target_sample_rate,
            "provider": self.provider,
            "revision": self.revision,
        }
        if self.filename is not None:
            data["filename"] = self.filename
            data["sha256"] = self.sha256
        if self.files:
            data["files"] = [file_entry.as_dict() for file_entry in self.files]
        return data


def read_weight_manifest(path: str | Path) -> WeightManifest:
    """Read a weight manifest JSON file."""

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("weight manifest root must be a JSON object")
    return WeightManifest.from_dict(loaded)


def write_weight_manifest(path: str | Path, manifest: WeightManifest) -> Path:
    """Write a weight manifest JSON file and return the path."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2), encoding="utf-8")
    return manifest_path


def resolve_weight_path(cache_dir: str | Path, manifest: WeightManifest) -> Path:
    """Return the expected local path for a legacy single-file manifest."""

    if manifest.filename is None:
        raise ValueError("manifest does not contain a legacy filename")
    return Path(cache_dir).expanduser() / manifest.id / manifest.filename


def resolve_manifest_file_paths(manifest_path: str | Path, manifest: WeightManifest) -> dict[str, Path]:
    """Return every manifest file path resolved relative to the manifest location."""

    base_dir = Path(manifest_path).expanduser().parent
    return {
        file_entry.path: resolve_weight_file_path(base_dir, file_entry.path) for file_entry in manifest.file_entries
    }


def verify_weight_manifest(path: str | Path) -> WeightManifest:
    """Read and verify every file in a weight manifest."""

    manifest_path = Path(path).expanduser()
    manifest = read_weight_manifest(manifest_path)
    verify_weight_manifest_files(manifest, manifest_path.parent)
    return manifest


def verify_weight_manifest_files(manifest: WeightManifest, base_dir: str | Path) -> None:
    """Verify all files declared by a manifest relative to base_dir."""

    for file_entry in manifest.file_entries:
        verify_weight_file(
            resolve_weight_file_path(base_dir, file_entry.path),
            expected_sha256=file_entry.sha256,
            expected_size=file_entry.size,
        )


def validate_weight_file_path(path: str | Path) -> str:
    """Return a normalized safe relative manifest path.

    Weight manifests describe files inside a model cache directory. Absolute paths,
    Windows drive paths, and parent-directory traversal would let a manifest point
    outside that directory, so they are rejected before any filesystem access.
    """

    raw_path = str(path)
    normalized = raw_path.replace("\\", "/")
    windows_path = PureWindowsPath(raw_path)
    posix_path = PurePosixPath(normalized)
    if (
        normalized in {"", "."}
        or windows_path.is_absolute()
        or posix_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or posix_path.root
    ):
        raise ValueError(f"weight file path must be relative to the manifest directory: {raw_path!r}")

    parts = posix_path.parts
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"weight file path cannot escape the manifest directory: {raw_path!r}")
    return str(posix_path)


def resolve_weight_file_path(base_dir: str | Path, file_path: str | Path) -> Path:
    """Resolve a manifest file path under base_dir and reject directory escape."""

    safe_path = validate_weight_file_path(file_path)
    root = Path(base_dir).expanduser()
    resolved = root.joinpath(*PurePosixPath(safe_path).parts)
    try:
        resolved.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"weight file path escapes the manifest directory: {file_path!r}") from exc
    return resolved


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weight_file(
    path: str | Path,
    expected_sha256: str | None,
    expected_size: int | None = None,
) -> Path:
    """Verify that a weight file exists and optionally matches size and SHA256."""

    weight_path = Path(path)
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)

    if expected_size is not None:
        actual_size = weight_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"weight size mismatch for {weight_path}: expected {expected_size}, got {actual_size}")

    if expected_sha256 is not None:
        actual_sha256 = sha256_file(weight_path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(f"weight hash mismatch for {weight_path}: expected {expected_sha256}, got {actual_sha256}")
    return weight_path


def load_safetensors(path: str | Path) -> dict[str, Any]:
    """Load a safetensors file when the optional safetensors package is installed."""

    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise RuntimeError("Loading safetensors weights requires `pip install safetensors`.") from exc

    return dict(load_file(str(path)))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("integer manifest fields cannot be booleans")
    return int(value)
