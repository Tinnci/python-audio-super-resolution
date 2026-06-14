from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

from .backends import (
    BackendInfo,
    EnhancementBackend,
    SincResampleBackend,
    available_backends,
    get_backend,
)
from .chunking import iter_audio_chunks, write_crossfaded_chunks
from .config import InferenceConfig
from .preprocess import apply_preprocessing

DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif")

__all__ = [
    "DEFAULT_AUDIO_EXTENSIONS",
    "AudioSuperResolver",
    "BackendInfo",
    "EnhancementBackend",
    "EnhancementResult",
    "PlannedEnhancement",
    "SincResampleBackend",
    "available_backends",
    "discover_audio_files",
    "get_backend",
    "plan_enhancements",
]


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
        if self.config.chunked:
            self._enhance_chunked(input_path, output_path, requested_sr, file_enhancer)
        elif callable(file_enhancer) and self.config.preprocess == "none":
            file_enhancer(input_path, output_path, requested_sr)
        elif callable(file_enhancer):
            audio, sample_rate = sf.read(input_path, always_2d=True)
            preprocessed = apply_preprocessing(
                audio,
                sample_rate=sample_rate,
                mode=self.config.preprocess,
                lowpass_cutoff_hz=self.config.lowpass_cutoff_hz,
                lowpass_order=self.config.lowpass_order,
            )
            with TemporaryDirectory(prefix="audio-super-resolution-") as temp_dir:
                preprocessed_input_path = Path(temp_dir) / f"{input_path.stem}-preprocessed.wav"
                sf.write(preprocessed_input_path, preprocessed, sample_rate, subtype="FLOAT")
                file_enhancer(preprocessed_input_path, output_path, requested_sr)
        else:
            audio, sample_rate = sf.read(input_path, always_2d=True)
            audio = apply_preprocessing(
                audio,
                sample_rate=sample_rate,
                mode=self.config.preprocess,
                lowpass_cutoff_hz=self.config.lowpass_cutoff_hz,
                lowpass_order=self.config.lowpass_order,
            )
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

    def _enhance_chunked(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        file_enhancer,
    ) -> None:
        if callable(file_enhancer):
            self._enhance_file_chunks(input_path, output_path, target_sample_rate, file_enhancer)
            return

        def enhanced_chunks():
            for audio, sample_rate in iter_audio_chunks(
                input_path,
                chunk_seconds=self.config.chunk_seconds,
                overlap_seconds=self.config.overlap_seconds,
            ):
                preprocessed = apply_preprocessing(
                    audio,
                    sample_rate=sample_rate,
                    mode=self.config.preprocess,
                    lowpass_cutoff_hz=self.config.lowpass_cutoff_hz,
                    lowpass_order=self.config.lowpass_order,
                )
                yield self.backend.enhance(preprocessed, sample_rate, target_sample_rate)

        write_crossfaded_chunks(
            enhanced_chunks(),
            output_path=output_path,
            sample_rate=target_sample_rate,
            overlap_seconds=self.config.overlap_seconds,
        )

    def _enhance_file_chunks(
        self,
        input_path: Path,
        output_path: Path,
        target_sample_rate: int,
        file_enhancer,
    ) -> None:
        with TemporaryDirectory(prefix="audio-super-resolution-chunks-") as temp_dir:

            def enhanced_chunks():
                for index, (audio, sample_rate) in enumerate(
                    iter_audio_chunks(
                        input_path,
                        chunk_seconds=self.config.chunk_seconds,
                        overlap_seconds=self.config.overlap_seconds,
                    )
                ):
                    preprocessed = apply_preprocessing(
                        audio,
                        sample_rate=sample_rate,
                        mode=self.config.preprocess,
                        lowpass_cutoff_hz=self.config.lowpass_cutoff_hz,
                        lowpass_order=self.config.lowpass_order,
                    )
                    chunk_input_path = Path(temp_dir) / f"chunk-{index:05d}.wav"
                    chunk_output_path = Path(temp_dir) / f"chunk-{index:05d}-enhanced.wav"
                    sf.write(chunk_input_path, preprocessed, sample_rate, subtype="FLOAT")
                    file_enhancer(chunk_input_path, chunk_output_path, target_sample_rate)

                    enhanced, enhanced_sample_rate = sf.read(chunk_output_path, always_2d=True)
                    if enhanced_sample_rate != target_sample_rate:
                        raise ValueError(
                            f"chunk output sample rate {enhanced_sample_rate} does not match target "
                            f"{target_sample_rate}"
                        )
                    yield enhanced

            write_crossfaded_chunks(
                enhanced_chunks(),
                output_path=output_path,
                sample_rate=target_sample_rate,
                overlap_seconds=self.config.overlap_seconds,
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
