from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_super_resolution.chunking import iter_audio_chunks, write_crossfaded_chunks


def test_iter_audio_chunks_yields_overlapping_blocks(tmp_path: Path) -> None:
    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, np.zeros(1000), 1000)

    chunks = list(iter_audio_chunks(audio_path, chunk_seconds=0.4, overlap_seconds=0.1))

    assert [chunk.shape[0] for chunk, _ in chunks] == [400, 400, 400]
    assert {sample_rate for _, sample_rate in chunks} == {1000}


def test_write_crossfaded_chunks_reduces_overlap_duration(tmp_path: Path) -> None:
    output_path = tmp_path / "output.wav"
    chunks = [
        np.ones((4, 1), dtype=np.float32),
        np.zeros((4, 1), dtype=np.float32),
    ]

    write_crossfaded_chunks(chunks, output_path, sample_rate=4, overlap_seconds=0.5)
    audio, sample_rate = sf.read(output_path, always_2d=True)

    assert sample_rate == 4
    assert audio.shape == (6, 1)
    assert audio[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert audio[-1, 0] == pytest.approx(0.0, abs=1e-4)


def test_write_crossfaded_chunks_rejects_empty_iterable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no chunks"):
        write_crossfaded_chunks([], tmp_path / "output.wav", sample_rate=1000, overlap_seconds=0.1)
