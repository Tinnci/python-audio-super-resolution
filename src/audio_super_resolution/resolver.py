from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audiosr_backend import AudiosrBackend
from .config import InferenceConfig

DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif")


@dataclass(frozen=True)
class EnhancementResult:
    """Metadata returned after writing an enhanced audio file."""

    input_path: Path
    output_path: Path
    input_sample_rate: int
    sample_rate: int
    input_duration_seconds: float
    duration_seconds: float
    channels: int
    backend: str


@dataclass(frozen=True)
class PlannedEnhancement:
    """Input and output paths for an enhancement job."""

    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class BackendInfo:
    """User-facing metadata for an enhancement backend."""

    name: str
    description: str


class EnhancementBackend(Protocol):
    """Interface implemented by audio super-resolution backends."""

    name: str
    description: str
    config: InferenceConfig

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        """Return audio enhanced to target_sample_rate."""


class SincResampleBackend:
    """Deterministic baseline backend using polyphase sinc resampling."""

    name = "sinc-resample"
    description = "Deterministic polyphase sinc resampling baseline."

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        if sample_rate == target_sample_rate:
            return np.asarray(audio)

        gcd = np.gcd(sample_rate, target_sample_rate)
        up = target_sample_rate // gcd
        down = sample_rate // gcd
        return resample_poly(audio, up, down, axis=0)


_BACKENDS: dict[str, type[EnhancementBackend]] = {
    AudiosrBackend.name: AudiosrBackend,
    SincResampleBackend.name: SincResampleBackend,
}


def available_backends() -> list[BackendInfo]:
    """Return the registered enhancement backends."""

    return [
        BackendInfo(name=name, description=backend.description)
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


def discover_audio_files(
    input_path: str | Path,
    recursive: bool = False,
    extensions: tuple[str, ...] | None = None,
) -> list[Path]:
    """Return audio files represented by input_path."""

    path = Path(input_path)
    requested_extensions = _normalize_extensions(extensions)

    if path.is_file():
        return [path]

    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_dir():
        raise ValueError(f"Input path is neither a file nor directory: {path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        child for child in path.glob(pattern) if child.is_file() and child.suffix.lower() in requested_extensions
    )


def plan_enhancements(
    input_path: str | Path,
    output_path: str | Path | None = None,
    target_sr: int = 48000,
    recursive: bool = False,
    extensions: tuple[str, ...] | None = None,
    suffix: str = "-sr",
) -> list[PlannedEnhancement]:
    """Plan output paths for one file or a directory of audio files."""

    if target_sr <= 0:
        raise ValueError("target_sr must be greater than zero")

    source = Path(input_path)
    output = Path(output_path) if output_path is not None else None
    files = discover_audio_files(source, recursive=recursive, extensions=extensions)

    if source.is_file():
        destination = _single_file_output_path(source, output, suffix=suffix, target_sr=target_sr)
        return [PlannedEnhancement(input_path=source, output_path=destination)]

    output_dir = output or source / "enhanced"
    return [
        PlannedEnhancement(
            input_path=file_path,
            output_path=output_dir / _relative_output_path(file_path, source, suffix=suffix, target_sr=target_sr),
        )
        for file_path in files
    ]


def _normalize_extensions(extensions: tuple[str, ...] | None) -> set[str]:
    raw_extensions = extensions or DEFAULT_AUDIO_EXTENSIONS
    return {extension.lower() if extension.startswith(".") else f".{extension.lower()}" for extension in raw_extensions}


def _single_file_output_path(input_path: Path, output_path: Path | None, suffix: str, target_sr: int) -> Path:
    if output_path is None:
        return input_path.with_name(_enhanced_file_name(input_path, suffix=suffix, target_sr=target_sr))

    if (output_path.exists() and output_path.is_dir()) or output_path.suffix == "":
        return output_path / _enhanced_file_name(input_path, suffix=suffix, target_sr=target_sr)

    return output_path


def _relative_output_path(input_path: Path, root: Path, suffix: str, target_sr: int) -> Path:
    relative = input_path.relative_to(root)
    return relative.with_name(_enhanced_file_name(input_path, suffix=suffix, target_sr=target_sr))


def _enhanced_file_name(input_path: Path, suffix: str, target_sr: int) -> str:
    return f"{input_path.stem}{suffix}{target_sr}{input_path.suffix}"


class AudioSuperResolver:
    """High-level API for audio super-resolution workflows."""

    def __init__(
        self,
        target_sr: int = 48000,
        backend: EnhancementBackend | str | None = None,
        config: InferenceConfig | None = None,
    ) -> None:
        if target_sr <= 0:
            raise ValueError("target_sr must be greater than zero")

        self.target_sr = target_sr
        self.config = config or InferenceConfig()
        self.backend = get_backend(backend, config=self.config) if isinstance(backend, str) else backend
        if self.backend is None:
            self.backend = SincResampleBackend(config=self.config)

    def enhance(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_sr: int | None = None,
    ) -> EnhancementResult:
        input_path = Path(input_path)
        output_path = Path(output_path)
        requested_sr = self.target_sr if target_sr is None else target_sr

        if requested_sr <= 0:
            raise ValueError("target_sr must be greater than zero")

        input_info = sf.info(input_path)
        file_enhancer = getattr(self.backend, "enhance_file", None)
        if callable(file_enhancer):
            file_enhancer(input_path, output_path, requested_sr)
        else:
            audio, sample_rate = sf.read(input_path, always_2d=True)
            enhanced = self.backend.enhance(audio, sample_rate, requested_sr)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, enhanced, requested_sr)

        output_info = sf.info(output_path)
        return EnhancementResult(
            input_path=input_path,
            output_path=output_path,
            input_sample_rate=input_info.samplerate,
            sample_rate=output_info.samplerate,
            input_duration_seconds=input_info.frames / input_info.samplerate,
            duration_seconds=output_info.frames / output_info.samplerate,
            channels=output_info.channels,
            backend=self.backend.name,
        )

    def enhance_many(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        target_sr: int | None = None,
        recursive: bool = False,
        extensions: tuple[str, ...] | None = None,
        suffix: str = "-sr",
    ) -> list[EnhancementResult]:
        """Enhance one file or every supported audio file in a directory."""

        requested_sr = self.target_sr if target_sr is None else target_sr
        jobs = plan_enhancements(
            input_path=input_path,
            output_path=output_path,
            target_sr=requested_sr,
            recursive=recursive,
            extensions=extensions,
            suffix=suffix,
        )
        return [
            self.enhance(
                input_path=job.input_path,
                output_path=job.output_path,
                target_sr=requested_sr,
            )
            for job in jobs
        ]
