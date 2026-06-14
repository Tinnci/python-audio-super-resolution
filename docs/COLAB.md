# Colab And GPU Validation

This guide is for validating the current repository on a fresh Colab or GPU runtime. It is intentionally repository-based rather than PyPI-based so gated tests and unreleased validation helpers are available.

## Runtime Setup

Use a GPU runtime if available, but keep the commands valid on CPU:

```sh
git clone https://github.com/Tinnci/python-audio-super-resolution.git
cd python-audio-super-resolution
python -m pip install -U pip
python -m pip install -e ".[lavasr,download]"
```

Check the visible environment:

```sh
audio-super-res --env-info
audio-super-res --list-backends --list-format json
audio-super-res --list-models --list-format json
```

For a direct torch check:

```python
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
```

## Prepare LavaSR Weights

Downloads are explicit. Prepare and verify the LavaSR v2 BWE cache before inference:

```sh
audio-super-res \
  --backend lavasr-compat \
  --download-weights \
  --prepare-model-cache

audio-super-res \
  --backend lavasr-compat \
  --verify-weights
```

## Generate A Short Input

Use an uploaded file, mounted Drive file, or a generated speech-like smoke fixture:

```python
import numpy as np
import soundfile as sf

sample_rate = 16000
seconds = 1.0
time = np.arange(int(sample_rate * seconds)) / sample_rate
audio = (
    0.04 * np.sin(2 * np.pi * 220 * time)
    + 0.03 * np.sin(2 * np.pi * 1234 * time)
    + 0.02 * np.sin(2 * np.pi * 3400 * time)
).astype("float32")
sf.write("input.wav", audio, sample_rate)
```

## Run LavaSR Compat

Use `--device auto` for portable validation. Use `--device cuda` only when the runtime has a working CUDA torch install.

```sh
audio-super-res input.wav output.wav \
  --backend lavasr-compat \
  --target-sr 48000 \
  --device auto \
  --manifest run.json \
  --quality-report \
  --quality-report-json quality.json
```

Expected checks:

- `output.wav` exists and is non-empty.
- `run.json` is valid JSON and records a completed run.
- `quality.json` is valid JSON.
- The output sample rate is 48000 Hz.
- `lavasr-compat` remains marked experimental.

## Optional Upstream Parity

Run this only when validating compatibility against upstream LavaSR/Vocos. It installs upstream project dependencies and is not part of the default suite.

```sh
python -m pip install "git+https://github.com/ysharma3501/LavaSR.git"

export AUDIO_SUPER_RESOLUTION_RUN_LAVASR_UPSTREAM_PARITY=1
pytest tests/test_lavasr_upstream_parity.py -q
```

On Windows PowerShell, use:

```powershell
$env:AUDIO_SUPER_RESOLUTION_RUN_LAVASR_UPSTREAM_PARITY = "1"
pytest tests/test_lavasr_upstream_parity.py -q
```

## Optional AudioSR External Backend

AudioSR is a heavier external package backend and owns its upstream checkpoint behavior. Validate it separately from `lavasr-compat`:

```sh
python -m pip install -e ".[audiosr]"

audio-super-res input.wav audiosr-output.wav \
  --backend audiosr \
  --target-sr 48000 \
  --device auto \
  --model-name basic \
  --preprocess lowpass \
  --manifest audiosr-run.json \
  --quality-report-json audiosr-quality.json
```

## Evidence To Record

When closing validation work, record:

- runtime type: CPU, CUDA, MPS, or other
- Python and torch versions
- `audio-super-res --env-info`
- weight cache path and `--verify-weights` output
- command used for inference
- quality report summary
- optional upstream parity result

Advanced CUDA, ROCm, XPU, DirectML, OpenVINO, TensorRT, and ONNX Runtime provider support belongs to the `v0.4.0` accelerator milestone.
