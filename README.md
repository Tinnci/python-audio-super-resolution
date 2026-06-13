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
- [Development](#development)
- [Requirements](#requirements)
- [License](#license)
- [Credits](#credits)
- [Contact](#contact)

</details>

---

## Features

- Enhance audio to a target sample rate from the command line.
- Supports common audio formats handled by libsndfile, including WAV, FLAC, and OGG.
- Provides a Python API for batch processing and integration into larger pipelines.
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

## Usage

### Command Line Interface

Enhance an audio file to a target sample rate:

```sh
audio-super-res input.wav output.wav --target-sr 48000
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
