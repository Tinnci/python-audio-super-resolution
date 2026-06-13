# Colab Plan

This repository should provide a Colab notebook after the optional AudioSR backend is validated with real model weights.

## Notebook Goals

- Install the package from GitHub with the `audiosr` extra.
- Upload or mount an input audio file.
- Prepare the model cache.
- Run `audio-super-res` with the `audiosr` backend.
- Write a manifest and JSON quality report.
- Download the enhanced audio and JSON artifacts.

## Draft Notebook Cells

```sh
pip install "audio-super-resolution[audiosr] @ git+https://github.com/Tinnci/python-audio-super-resolution.git"
```

```python
from google.colab import files

uploaded = files.upload()
input_file = next(iter(uploaded))
```

```sh
audio-super-res "$input_file" output.wav \
  --backend audiosr \
  --target-sr 48000 \
  --device auto \
  --model-name basic \
  --preprocess lowpass \
  --manifest run.json \
  --quality-report-json quality.json
```

```python
files.download("output.wav")
files.download("run.json")
files.download("quality.json")
```

## Validation Before Publishing

- Run on a fresh Colab runtime.
- Confirm model download succeeds.
- Confirm generated output is non-empty and 48000 Hz.
- Confirm manifest and quality report are valid JSON.
- Confirm runtime notes mention that AudioSR dependencies are heavier than the baseline package.
