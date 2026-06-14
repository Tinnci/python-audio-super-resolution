<div align="center">

# Audio Super Resolution

[![PyPI version](https://badge.fury.io/py/audio-super-resolution.svg)](https://badge.fury.io/py/audio-super-resolution)
[![CI](https://github.com/Tinnci/python-audio-super-resolution/actions/workflows/ci.yml/badge.svg)](https://github.com/Tinnci/python-audio-super-resolution/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

**Summary:** Easy to use audio super-resolution and bandwidth extension from the command line or as a dependency in your own Python project.

Audio Super Resolution is a Python package for improving low-resolution audio by increasing sample rate and reconstructing high-frequency detail with pluggable enhancement backends.

The package provides a clean CLI, Python API, and Pixi-managed development environment for AudioSR-style workflows. Model backends can be added behind the same interface without changing downstream scripts.

<details>
<summary align="center"><b>Table of Contents</b></summary>

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Inference Configuration](#inference-configuration)
  - [Preprocessing](#preprocessing)
  - [Quality Reports](#quality-reports)
  - [Manifest Comparison](#manifest-comparison)
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

- CLI and Python API for single files, directory batches, and recursive path-preserving runs.
- Pluggable backend registry with a deterministic `sinc-resample` baseline, optional external AudioSR support, and managed LavaSR-compatible weight metadata.
- Shared inference configuration for device, precision, chunking, seeds, preprocessing, and model cache paths.
- JSON run manifests, manifest comparison, and standalone quality reports for regression workflows.
- Explicit managed-weight resolution with local manifests, SHA256 verification, and opt-in Hugging Face downloads.
- Pixi-managed development tasks for reproducible test, lint, format, and build commands.

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

Optional model and weight features are split into extras so the baseline package stays lightweight:

```sh
pip install "audio-super-resolution[audiosr] @ git+https://github.com/Tinnci/python-audio-super-resolution.git"
pip install "audio-super-resolution[weights] @ git+https://github.com/Tinnci/python-audio-super-resolution.git"
pip install "audio-super-resolution[download] @ git+https://github.com/Tinnci/python-audio-super-resolution.git"
pip install "audio-super-resolution[lavasr,download] @ git+https://github.com/Tinnci/python-audio-super-resolution.git"
```

Use `audiosr` for the external AudioSR wrapper, `download` for Hugging Face weight downloads, `weights` for optional safetensors support, and `lavasr` for the self-contained LavaSR-compatible runtime path.

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
  --chunked \
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

Write a JSON manifest for planned or completed work:

```sh
audio-super-res ./low-res-audio ./enhanced-audio --recursive --manifest run.json
audio-super-res ./low-res-audio ./enhanced-audio --recursive --dry-run --manifest plan.json
```

Compare two completed manifests for regressions:

```sh
audio-super-res --compare-manifests expected.json actual.json
audio-super-res --compare-manifests expected.json actual.json --compare-format json
```

List available enhancement backends:

```sh
audio-super-res --list-backends
audio-super-res --list-backends --list-format json
```

List known enhancement models:

```sh
audio-super-res --list-models
audio-super-res --list-models --list-filter speech --list-format json
```

Model listings include implementation type, domain, target sample rate, maturity, and weight metadata when the backend exposes it.

Managed weight downloads are explicit. Normal enhancement uses local verified files only unless `--download-weights` is provided:

```sh
audio-super-res \
  --backend lavasr-compat \
  --model-name lavasr-v2-bwe \
  --download-weights \
  --prepare-model-cache

audio-super-res \
  --backend lavasr-compat \
  --model-name lavasr-v2-bwe \
  --verify-weights
```

You can also point to an existing manifest:

```sh
audio-super-res input.wav output.wav \
  --backend lavasr-compat \
  --target-sr 48000 \
  --weights-manifest C:\path\to\lavasr-v2-bwe\manifest.json
```

Current backend status:

- `sinc-resample`: default deterministic baseline.
- `audiosr`: optional external package backend; upstream package owns its checkpoint behavior.
- `lavasr-compat`: LavaSR v2 BWE weight download and verification are wired; self-contained inference is still pending.

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

The AudioSR backend writes 48000 Hz audio and supports `basic` and `speech` models.

Apply optional low-pass preprocessing before enhancement:

```sh
audio-super-res input.wav output.wav \
  --backend audiosr \
  --target-sr 48000 \
  --preprocess lowpass \
  --lowpass-cutoff-hz 16000
```

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
audio-super-res input.wav output.wav --quality-report-json quality.json
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
from audio_super_resolution import InferenceConfig, build_manifest, list_models, plan_enhancements

jobs = plan_enhancements("low-res-audio", "enhanced-audio", recursive=True)
models = list_models(filter_text="audiosr")
manifest = build_manifest("dry-run", jobs, InferenceConfig(), backend="sinc-resample", target_sample_rate=48000)
```

### Inference Configuration

```python
from audio_super_resolution import AudioSuperResolver, InferenceConfig

config = InferenceConfig(
    device="cpu",
    precision="float32",
    chunked=True,
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

### Preprocessing

Low-pass preprocessing is optional and disabled by default. It can help when model backends are sensitive to unfamiliar cutoff patterns, MP3 artifacts, or aggressive prior filtering.

```python
from audio_super_resolution import InferenceConfig

config = InferenceConfig(
    preprocess="lowpass",
    lowpass_cutoff_hz=16000,
)
```

If `lowpass_cutoff_hz` is omitted, the cutoff defaults to `min(16000, 45% of input sample rate)`.

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

Write a combined JSON quality report from the CLI:

```sh
audio-super-res input.wav output.wav --quality-report-json quality.json
```

### Manifest Comparison

Manifest comparison checks backend, target sample rate, per-input result presence, sample rate, duration, channel count, output path presence, and quality status. The CLI exits with `1` when differences are found, which makes it suitable for CI regression checks.

```python
from audio_super_resolution import compare_manifests, load_manifest

comparison = compare_manifests(
    load_manifest("expected.json"),
    load_manifest("actual.json"),
)

if not comparison.passed:
    print(comparison.differences)
```

## Examples

- [examples/basic_usage.py](examples/basic_usage.py)
- [examples/batch_process.py](examples/batch_process.py)
- [examples/compare_manifests.py](examples/compare_manifests.py)
- [examples/quality_check.py](examples/quality_check.py)
- [examples/artifacts/](examples/artifacts/) contains sample manifest and quality report JSON artifacts.

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

See [ROADMAP.md](ROADMAP.md) for the current implementation plan. The active work is:

- Complete LavaSR-compatible self-contained inference on top of the managed weight store.
- Add golden-sample validation against source implementations for compatible backends.
- Validate optional real-weight integrations before release-facing notebooks and GPU images.

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

Run optional real AudioSR integration tests only when you explicitly want model inference and upstream checkpoint handling:

```sh
set AUDIO_SUPER_RESOLUTION_RUN_AUDIOSR_INTEGRATION=1
pixi run pytest tests/test_audiosr_integration.py
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
