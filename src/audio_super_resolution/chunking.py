from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import soundfile as sf


def iter_audio_chunks(
    path: str | Path,
    chunk_seconds: float,
    overlap_seconds: float,
) -> Iterator[tuple[np.ndarray, int]]:
    """Yield overlapping audio chunks and the source sample rate."""

    validate_chunking_options(chunk_seconds, overlap_seconds)

    audio_path = Path(path)
    info = sf.info(audio_path)
    chunk_frames = _seconds_to_frames(chunk_seconds, info.samplerate)
    overlap_frames = _seconds_to_frames(overlap_seconds, info.samplerate)

    for block in sf.blocks(
        audio_path,
        blocksize=chunk_frames,
        overlap=overlap_frames,
        always_2d=True,
    ):
        yield np.asarray(block), info.samplerate


def write_crossfaded_chunks(
    chunks: Iterable[np.ndarray],
    output_path: str | Path,
    sample_rate: int,
    overlap_seconds: float,
) -> Path:
    """Write enhanced chunks with linear crossfades across overlapped regions."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be greater than or equal to zero")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlap_frames = _seconds_to_frames(overlap_seconds, sample_rate)
    pending: np.ndarray | None = None
    writer: sf.SoundFile | None = None

    try:
        for chunk in chunks:
            chunk = _ensure_2d_audio(chunk)
            if writer is None:
                writer = sf.SoundFile(output, mode="w", samplerate=sample_rate, channels=chunk.shape[1])
            elif chunk.shape[1] != writer.channels:
                raise ValueError("all enhanced chunks must have the same channel count")

            if overlap_frames == 0:
                writer.write(chunk)
                continue

            if pending is None:
                pending = _write_all_but_tail(writer, chunk, overlap_frames)
                continue

            pending = _write_with_crossfade(writer, pending, chunk, overlap_frames)

        if writer is None:
            raise ValueError("no chunks were produced")
        if pending is not None and len(pending):
            writer.write(pending)
    finally:
        if writer is not None:
            writer.close()

    return output


def validate_chunking_options(chunk_seconds: float, overlap_seconds: float) -> None:
    """Validate chunk duration and overlap settings."""

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than zero")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be greater than or equal to zero")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be less than chunk_seconds")


def _write_all_but_tail(writer: sf.SoundFile, chunk: np.ndarray, tail_frames: int) -> np.ndarray:
    if len(chunk) > tail_frames:
        writer.write(chunk[:-tail_frames])
        return chunk[-tail_frames:]
    return chunk


def _write_with_crossfade(
    writer: sf.SoundFile,
    pending: np.ndarray,
    chunk: np.ndarray,
    overlap_frames: int,
) -> np.ndarray:
    current_overlap_frames = min(len(pending), len(chunk), overlap_frames)
    if current_overlap_frames == 0:
        writer.write(pending)
        return _write_all_but_tail(writer, chunk, overlap_frames)

    if len(pending) > current_overlap_frames:
        writer.write(pending[:-current_overlap_frames])

    writer.write(_crossfade(pending[-current_overlap_frames:], chunk[:current_overlap_frames]))
    remainder = chunk[current_overlap_frames:]
    return _write_all_but_tail(writer, remainder, overlap_frames)


def _crossfade(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    frame_count = len(previous)
    if frame_count != len(current):
        raise ValueError("crossfade inputs must have the same frame count")
    if frame_count == 0:
        return current

    fade_out = np.linspace(1.0, 0.0, frame_count, endpoint=False, dtype=np.float64)[:, np.newaxis]
    fade_in = 1.0 - fade_out
    return previous * fade_out + current * fade_in


def _ensure_2d_audio(audio: np.ndarray) -> np.ndarray:
    audio_array = np.asarray(audio)
    if audio_array.ndim == 1:
        return audio_array[:, np.newaxis]
    if audio_array.ndim == 2:
        return audio_array
    raise ValueError(f"audio chunks must be one- or two-dimensional, got shape {audio_array.shape}")


def _seconds_to_frames(seconds: float, sample_rate: int) -> int:
    return max(1, int(round(seconds * sample_rate)))
