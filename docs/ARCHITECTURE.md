# Architecture

The package is organized around a stable CLI/API surface and pluggable enhancement backends.

## Layers

- `audio_super_resolution.cli`: command-line parsing, user-facing errors, and reporting.
- `audio_super_resolution.resolver`: path planning, file dispatch, backend registry, and high-level API.
- `audio_super_resolution.config`: shared inference configuration and model cache resolution.
- `audio_super_resolution.models`: model catalog for CLI and programmatic discovery.
- `audio_super_resolution.manifest`: JSON manifest generation and regression comparison for planned and completed jobs.
- `audio_super_resolution.preprocess`: optional input preprocessing before backend enhancement.
- `audio_super_resolution.quality`: objective quality checks for rendered audio.
- `audio_super_resolution.audiosr_backend`: optional wrapper around the upstream `audiosr` package.

## Backend Contract

Backends are registered by name and expose user-facing metadata through `available_backends()`.

Array-native backends implement:

```python
def enhance(audio, sample_rate, target_sample_rate):
    ...
```

File-native backends, such as the optional AudioSR wrapper, may implement:

```python
def enhance_file(input_path, output_path, target_sample_rate):
    ...
```

`AudioSuperResolver.enhance()` checks for `enhance_file()` first. This keeps file-based model packages behind the same public API without forcing every backend to accept temporary arrays.

If preprocessing is enabled in `InferenceConfig`, array-native backends receive the preprocessed array directly. File-native backends receive a temporary WAV created from the preprocessed input, keeping their file-oriented interface unchanged.

## Dependency Policy

The baseline package must remain lightweight and installable without model downloads or GPU dependencies.

- `sinc-resample` is the default backend and must run in CI.
- `audiosr` is optional and only imported when selected.
- Missing optional dependencies must raise clear user-facing errors.
- Normal CI must not download model weights.

## CLI Policy

The CLI is grouped by user task:

- info/debugging
- enhancement I/O
- backend and inference
- quality checks

Machine-readable output should be available for listing commands when useful. `--list-backends --list-format json` and `--list-models --list-format json` are the current examples.

Batch runs should be reproducible from their manifest. `--manifest` writes planned jobs, completed results, configuration, backend, and optional quality reports.

`--compare-manifests` compares completed manifests by backend, target sample rate, result presence, sample rate, duration, channel count, output presence, and quality status. It returns a non-zero exit code when differences are found.

## Release Policy

Releases are built with Pixi. PyPI publishing uses trusted publishing from `.github/workflows/release.yml`.

See [RELEASE.md](RELEASE.md) for the checklist.
