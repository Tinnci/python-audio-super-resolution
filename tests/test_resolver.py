from pathlib import Path

import numpy as np
import soundfile as sf

from audio_super_resolution import AudioSuperResolver


def test_enhance_writes_target_sample_rate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sample_rate = 16000
    target_sr = 48000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate // 10) / sample_rate)

    sf.write(input_path, tone, sample_rate)

    result = AudioSuperResolver(target_sr=target_sr).enhance(input_path, output_path)
    _, written_sr = sf.read(output_path)

    assert output_path.exists()
    assert result.sample_rate == target_sr
    assert written_sr == target_sr
    assert result.backend == "sinc-resample"
