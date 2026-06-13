from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_super_resolution import AudioSuperResolver, InferenceConfig, lowpass_filter


def test_lowpass_filter_attenuates_high_frequency() -> None:
    sample_rate = 16000
    high_tone = np.sin(2 * np.pi * 6000 * np.arange(sample_rate) / sample_rate)

    filtered = lowpass_filter(high_tone, sample_rate=sample_rate, cutoff_hz=1000, order=8)

    assert _rms(filtered) < _rms(high_tone) * 0.1


def test_lowpass_filter_rejects_cutoff_above_nyquist() -> None:
    with pytest.raises(ValueError, match="below Nyquist"):
        lowpass_filter(np.zeros(1000), sample_rate=16000, cutoff_hz=9000)


def test_resolver_preprocesses_file_backend_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sample_rate = 16000
    high_tone = np.sin(2 * np.pi * 6000 * np.arange(sample_rate) / sample_rate)
    sf.write(input_path, high_tone, sample_rate)

    captured: dict[str, float] = {}

    class FileBackend:
        name = "file-backend"

        def enhance_file(self, input_path: Path, output_path: Path, target_sample_rate: int) -> None:
            audio, written_sample_rate = sf.read(input_path, always_2d=True)
            captured["rms"] = _rms(audio)
            captured["sample_rate"] = written_sample_rate
            sf.write(output_path, audio, target_sample_rate)

    config = InferenceConfig(
        preprocess="lowpass",
        lowpass_cutoff_hz=1000,
        model_cache_dir=tmp_path / "models",
    )

    AudioSuperResolver(target_sr=sample_rate, backend=FileBackend(), config=config).enhance(input_path, output_path)

    assert captured["sample_rate"] == sample_rate
    assert captured["rms"] < _rms(high_tone) * 0.1


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))
