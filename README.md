<div align="center">

# Audio Super Resolution

[![PyPI version](https://badge.fury.io/py/audio-super-resolution.svg)](https://badge.fury.io/py/audio-super-resolution)
[![CI](https://github.com/Tinnci/python-audio-super-resolution/actions/workflows/ci.yml/badge.svg)](https://github.com/Tinnci/python-audio-super-resolution/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

**Summary:** Easy to use audio super-resolution and bandwidth extension from the command line or as a dependency in your own Python project.

Audio Super Resolution is a Python package for improving low-resolution audio by increasing sample rate and reconstructing high-frequency detail with pluggable enhancement backends.

The initial package provides a clean CLI, Python API, and Pixi-managed development environment for AudioSR-style workflows. Model backends can be added behind the same interface without changing downstream scripts.

<details>
<summary align="center"><b>Table of Contents</b></summary>

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Inference Configuration](#inference-configuration)
  - [Quality Reports](#quality-reports)
- [Examples](#examples)
- [Docker](#docker)
- [Roadmap](#roadmap)
- [Development](#development)
- [Release](#release)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [License](#license)
- [Credits](#credits)
- [Contact](#contact)

</details>

---

## Features

- Enhance audio to a target sample rate from the command line.
- Process a single file or batch process directories.
- Recursively scan audio folders while preserving relative output paths.
- Supports common audio formats handled by libsndfile, including WAV, FLAC, and OGG.
- Provides a Python API for batch processing and integration into larger pipelines.
- Exposes shared inference configuration for device, precision, chunking, seed, and model cache.
- Includes objective audio quality checks for sample rate, duration drift, clipping, and peak level.
- Uses a backend abstraction so model-based AudioSR implementations can be added cleanly.
- Managed with Pixi for reproducible development tasks and dependencies.

## Installation

Install from the repository:

```sh
pip install git+https://github.com/Tinnci/python-audio-super-resolution.git
```

For local development, use Pixi:

```sh
git clone https://github.com/Tinnci/python-audio-super-resolution.git
cd python-audio-super-resolution
pixi install
```

AudioSR model inference is optional because it pulls in heavy ML dependencies:

```sh
pip install "audio-super-resolution[audiosr] @ git+https://github.com/Tinnci/python-audio-super-resolution.git"
```

For local development with the optional AudioSR backend:

```sh
pip install -e ".[audiosr]"
```

Use Python 3.10 for the AudioSR backend because the upstream `audiosr` package pins older ML dependencies.

## Usage

### Command Line Interface

Enhance an audio file to a target sample rate:

```sh
audio-super-res input.wav output.wav --target-sr 48000
```

Configure inference options for backends that use model inference:

```sh
audio-super-res input.wav output.wav \
  --device cpu \
  --precision float32 \
  --chunk-seconds 30 \
  --overlap-seconds 1 \
  --seed 0
```

If the output path is omitted, the CLI writes next to the input file:

```sh
audio-super-res input.wav --target-sr 48000
```

This creates `input-sr48000.wav`.

Batch process a directory:

```sh
audio-super-res ./low-res-audio ./enhanced-audio --recursive --target-sr 48000
```

Preview planned outputs without writing files:

```sh
audio-super-res ./low-res-audio ./enhanced-audio --recursive --dry-run
```

List available enhancement backends:

```sh
audio-super-res --list-backends
```

Run the optional AudioSR model backend:

```sh
audio-super-res input.wav output.wav \
  --backend audiosr \
  --target-sr 48000 \
  --model-name basic \
  --device auto \
  --ddim-steps 50 \
  --guidance-scale 3.5
```

The AudioSR backend currently writes 48000 Hz audio and supports `basic` and `speech` models. Checkpoints are downloaded through Hugging Face into the configured model cache path.

Print the resolved inference configuration:

```sh
audio-super-res --config-info
```

Create the configured model cache directory:

```sh
audio-super-res --prepare-model-cache
```

Run post-write audio quality checks:

```sh
audio-super-res input.wav output.wav --quality-report --fail-on-quality-issue
```

The shorter alias is also available:

```sh
audiosr input.wav output.wav --target-sr 48000
```

Show environment information:

```sh
audio-super-res --env-info
```

### Python API

```python
from audio_super_resolution import AudioSuperResolver

resolver = AudioSuperResolver(target_sr=48000)
result = resolver.enhance("input.wav", "output.wav")

print(result.output_path)
print(result.sample_rate)
```

Batch processing:

```python
from audio_super_resolution import AudioSuperResolver

resolver = AudioSuperResolver(target_sr=48000, backend="sinc-resample")
results = resolver.enhance_many(
    "low-res-audio",
    "enhanced-audio",
    recursive=True,
)

for result in results:
    print(result.input_path, "->", result.output_path)
```

Plan output paths without writing files:

```python
from audio_super_resolution import plan_enhancements

jobs = plan_enhancements("low-res-audio", "enhanced-audio", recursive=True)
```

### Inference Configuration

```python
from audio_super_resolution import AudioSuperResolver, InferenceConfig

config = InferenceConfig(
    device="cpu",
    precision="float32",
    chunk_seconds=30.0,
    overlap_seconds=1.0,
    seed=0,
)

config.ensure_model_cache_dir()
resolver = AudioSuperResolver(target_sr=48000, backend="sinc-resample", config=config)
```

The default model cache directory can be overridden with:

```sh
set AUDIO_SUPER_RESOLUTION_CACHE=C:\path\to\models
```

On Unix-like shells:

```sh
export AUDIO_SUPER_RESOLUTION_CACHE=/path/to/models
```

### Quality Reports

```python
from audio_super_resolution import inspect_audio_quality

report = inspect_audio_quality(
    "output.wav",
    expected_sample_rate=48000,
    expected_duration_seconds=30.0,
)

if not report.passed:
    print(report.issues)
```

## Examples

- [examples/basic_usage.py](examples/basic_usage.py)
- [examples/batch_process.py](examples/batch_process.py)
- [examples/quality_check.py](examples/quality_check.py)

## Docker

Build the baseline CPU image:

```sh
docker build -t audio-super-resolution .
```

Run the CLI in the container:

```sh
docker run --rm -v "%cd%":/workdir audio-super-resolution input.wav output.wav --target-sr 48000
```

On Unix-like shells:

```sh
docker run --rm -v "$PWD":/workdir audio-super-resolution input.wav output.wav --target-sr 48000
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the staged plan. The short version:

- Stabilize the CLI/API contract around files, directories, and backend selection.
- Add model-backed AudioSR inference behind the existing backend interface.
- Add objective quality metrics and example notebooks.
- Publish release artifacts once the first model backend is usable.

## Development

Pixi manages local dependencies and common tasks:

```sh
pixi run test
pixi run lint
pixi run format
```

Run the CLI from the local checkout:

```sh
pixi run audio-super-res input.wav output.wav --target-sr 48000
```

Build the package:

```sh
pixi run build
```

## Release

See [docs/RELEASE.md](docs/RELEASE.md). Releases are built with Pixi and published through PyPI trusted publishing.

## Project Structure

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the package layers and backend contract.
- [tests/README.md](tests/README.md) describes the lightweight test strategy.
- `.github/ISSUE_TEMPLATE/` contains bug and feature templates.
- `.github/workflows/` contains CI, release, and security workflows.

## Requirements

- Python 3.10 or newer
- Pixi for development
- libsndfile-compatible audio files for the default reader/writer

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Credits

Inspired by the project structure and user experience of [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator).

## Contact

For questions, issues, or contributions, open an issue on GitHub:

https://github.com/Tinnci/python-audio-super-resolution/issues
