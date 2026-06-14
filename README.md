<div align="center">

# Audio Super Resolution

[![PyPI version](https://badge.fury.io/py/audio-super-resolution.svg)](https://badge.fury.io/py/audio-super-resolution)
[![CI](https://github.com/Tinnci/python-audio-super-resolution/actions/workflows/ci.yml/badge.svg)](https://github.com/Tinnci/python-audio-super-resolution/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

Audio Super Resolution is a Python CLI and library for audio super-resolution and bandwidth extension. It ships with a deterministic `sinc-resample` baseline, optional external AudioSR support, and managed metadata for self-contained model backends.

The baseline package stays lightweight: normal inference is offline, model downloads are explicit, and heavyweight model dependencies live behind optional extras.

## Features

- CLI and Python API for single files, directory batches, dry runs, and recursive path-preserving output.
- Pluggable backend registry with `sinc-resample`, optional `audiosr`, and experimental `lavasr-compat`.
- Shared inference config for device, precision, chunking, preprocessing, seeds, and model cache paths.
- JSON run manifests, manifest comparison, and quality reports for regression workflows.
- Explicit local weight resolution with multi-file manifests, size/SHA256 checks, and opt-in Hugging Face downloads.
- Pixi tasks for repeatable test, lint, format, and build commands.

## Installation

Install from PyPI:

```sh
pip install audio-super-resolution
```

Install optional model/runtime extras only when needed:

```sh
pip install "audio-super-resolution[lavasr,download]"
```

Install the unreleased repository version from GitHub:

```sh
pip install "audio-super-resolution @ git+https://github.com/Tinnci/python-audio-super-resolution.git"
```

For local development:

```sh
git clone https://github.com/Tinnci/python-audio-super-resolution.git
cd python-audio-super-resolution
pixi install
```

Optional extras:

| Extra | Purpose |
| --- | --- |
| `audiosr` | External AudioSR wrapper. Use Python 3.10 because upstream dependencies are older. |
| `download` | Hugging Face model weight downloads. |
| `weights` | Optional safetensors loading helpers. |
| `lavasr` | Torch runtime for the experimental LavaSR-compatible backend. |

GitHub install with optional extras:

```sh
pip install "audio-super-resolution[lavasr,download] @ git+https://github.com/Tinnci/python-audio-super-resolution.git"
```

## CLI Quick Start

Enhance one file:

```sh
audio-super-res input.wav output.wav --target-sr 48000
```

If `output.wav` is omitted, the CLI writes next to the input as `input-sr48000.wav`.

Batch process a directory:

```sh
audio-super-res ./low-res-audio ./enhanced-audio --recursive --target-sr 48000
```

Preview or record a run:

```sh
audio-super-res ./low-res-audio ./enhanced-audio --recursive --dry-run --manifest plan.json
audio-super-res ./low-res-audio ./enhanced-audio --recursive --manifest run.json
audio-super-res --compare-manifests expected.json actual.json
```

List backends and models:

```sh
audio-super-res --list-backends
audio-super-res --list-models --list-format json
```

Run post-write quality checks:

```sh
audio-super-res input.wav output.wav --quality-report --fail-on-quality-issue
audio-super-res input.wav output.wav --quality-report-json quality.json
```

The shorter `audiosr` command is also available as an alias for `audio-super-res`.

## Models And Weights

Current backend status:

| Backend | Status |
| --- | --- |
| `sinc-resample` | Default deterministic baseline. |
| `audiosr` | Optional external package backend; upstream package owns its checkpoint behavior. |
| `lavasr-compat` | Experimental self-contained LavaSR v2 BWE path with managed weights. Gated real-weight download, torch smoke, and initial upstream parity validation pass. |

Use `audio-super-res --list-models --list-format json` for machine-readable comparison metadata, including task/domain, input and target sample rates, implementation family, I/O capabilities, accelerator declarations, weight source/size/license, validation evidence, recommended use, and known limitations.

Managed downloads are explicit. Normal enhancement only uses local verified files unless `--download-weights` is set:

```sh
audio-super-res --backend lavasr-compat --download-weights --prepare-model-cache
audio-super-res --backend lavasr-compat --verify-weights
```

Use an existing manifest:

```sh
audio-super-res input.wav output.wav \
  --backend lavasr-compat \
  --target-sr 48000 \
  --weights-manifest C:\path\to\lavasr-v2-bwe\manifest.json
```

Run the optional external AudioSR backend:

```sh
audio-super-res input.wav output.wav \
  --backend audiosr \
  --target-sr 48000 \
  --model-name basic \
  --device auto
```

## Python API

```python
from audio_super_resolution import AudioSuperResolver

resolver = AudioSuperResolver(target_sr=48000)
result = resolver.enhance("input.wav", "output.wav")

print(result.output_path)
print(result.sample_rate)
```

Batch planning and manifests:

```python
from audio_super_resolution import InferenceConfig, build_manifest, plan_enhancements

jobs = plan_enhancements("low-res-audio", "enhanced-audio", recursive=True)
manifest = build_manifest("dry-run", jobs, InferenceConfig(), backend="sinc-resample", target_sample_rate=48000)
```

Managed weights:

```python
from audio_super_resolution import InferenceConfig, download_model_weights, resolve_model_weights, verify_model_weights

download_model_weights("lavasr-v2-bwe")
verified = verify_model_weights("lavasr-v2-bwe")
weights = resolve_model_weights("lavasr-v2-bwe", InferenceConfig(model_cache_dir=verified.root_dir.parent))
model_path = weights.path_for("enhancer_v2/pytorch_model.bin")
```

## Development

```sh
pixi run test
pixi run lint
pixi run format
pixi run build
```

Run optional real AudioSR integration only when model inference and upstream checkpoint handling are intended:

```sh
set AUDIO_SUPER_RESOLUTION_RUN_AUDIOSR_INTEGRATION=1
pixi run pytest tests/test_audiosr_integration.py
```

Real LavaSR weight download and torch smoke tests are also gated; see [tests/README.md](tests/README.md).

## Docker

```sh
docker build -t audio-super-resolution .
docker run --rm -v "%cd%":/workdir audio-super-resolution input.wav output.wav --target-sr 48000
```

On Unix-like shells, use `-v "$PWD":/workdir`.

## Project Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): package layers, backend contract, and weight-management boundaries.
- [ROADMAP.md](ROADMAP.md): current status and next implementation work.
- [CHANGELOG.md](CHANGELOG.md): release history and unreleased changes.
- [docs/GOLDEN.md](docs/GOLDEN.md): golden-sample fixture format and comparison metrics.
- [docs/RELEASE.md](docs/RELEASE.md): release checklist and publishing notes.
- [docs/COLAB.md](docs/COLAB.md): draft notebook plan gated on real model validation.
- [tests/README.md](tests/README.md): default and optional test strategy.
- [examples/](examples/): Python examples and sample JSON artifacts.

## Requirements

- Python 3.10 or newer
- Pixi for development
- libsndfile-compatible audio files for the default reader/writer

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Credits

Inspired by the project structure and user experience of [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator).
