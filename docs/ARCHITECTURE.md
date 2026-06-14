# Architecture

The package is organized around a stable CLI/API surface and pluggable enhancement backends.

## Layers

- `audio_super_resolution.cli`: command-line parsing, user-facing errors, and reporting.
- `audio_super_resolution.resolver`: path planning, file dispatch, and high-level API.
- `audio_super_resolution.backends`: backend implementations and the runtime backend registry.
- `audio_super_resolution.chunking`: overlapping chunk iteration and crossfaded chunk writing.
- `audio_super_resolution.config`: shared inference configuration and model cache resolution.
- `audio_super_resolution.models`: model catalog for CLI and programmatic discovery.
- `audio_super_resolution.specs`: backend capability and model metadata used by listings and self-contained backends.
- `audio_super_resolution.weights`: weight manifest, local path, and hash verification helpers.
- `audio_super_resolution.downloads`: explicit provider-backed model weight downloads.
- `audio_super_resolution.weight_store`: shared local weight lookup, verification, and opt-in download resolution.
- `audio_super_resolution.devices`: lightweight runtime device discovery helpers.
- `audio_super_resolution.manifest`: JSON manifest generation and regression comparison for planned and completed jobs.
- `audio_super_resolution.preprocess`: optional input preprocessing before backend enhancement.
- `audio_super_resolution.quality`: objective quality checks for rendered audio.
- `audio_super_resolution.audiosr_backend`: compatibility import for the optional AudioSR external backend.

## Backend Contract

Backends are registered by name in `audio_super_resolution.backends.registry` and expose user-facing metadata through `available_backends()`. Built-in backends register at import time. Additional backends can be added with `register_backend()` as long as they implement the backend contract.

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

If `chunked=True`, `AudioSuperResolver.enhance()` reads overlapping chunks, applies preprocessing per chunk, calls the selected backend for each chunk, and writes the result with a linear crossfade across the overlap. This path supports both array-native and file-native backends while preserving the default non-chunked behavior.

Backends can expose static `ModelSpec` records through `model_specs()`. The model catalog converts those specs into `ModelInfo` records for `--list-models` and the Python API. This keeps model metadata close to the backend that owns validation and inference behavior.

Self-contained model backends should keep responsibilities separated:

- `specs` defines static model metadata and support boundaries.
- `weights` parses manifests, resolves relative paths, and verifies size/SHA256.
- `downloads` talks to remote providers only during explicit download commands.
- `weight_store` chooses between an explicit manifest, a verified cache entry, or an opt-in download.
- The backend validates architecture/device details and loads only verified local files.

Backends must not call remote providers directly. Downloads are explicit and centralized in `audio_super_resolution.downloads`, while local cache and manifest precedence are centralized in `audio_super_resolution.weight_store`.

## Weight Management

The default inference path is offline. Missing weights should produce a clear message with the matching `--download-weights --prepare-model-cache` command.

The API is layered by responsibility:

- `weights` is the low-level manifest layer. It parses JSON, rejects unsafe file paths, resolves manifest-relative files, and verifies size/SHA256.
- `downloads.download_weights_for_spec()` is the provider-facing layer. It accepts a `ModelSpec`, talks to the provider, writes a temporary cache, and publishes it only after verification.
- `weight_store.download_model_weights()` is the user-facing download API. It accepts a registered model id or `ModelSpec`.
- `weight_store.verify_model_weights()` and `weight_store.resolve_model_weights()` bind a verified manifest back to the selected `ModelSpec` before returning `ResolvedWeights`.

Managed weights use a per-model cache directory:

```text
<model_cache_dir>/<model_id>/
  manifest.json
  .complete
  ...
```

Downloads are written to a temporary directory first. The final cache directory is replaced only after every required file passes verification. Failed downloads remove the temporary directory and must not delete an existing verified cache.

Manifest file paths must be relative to the manifest directory. Absolute paths, Windows drive paths, and `..` traversal are rejected before any file access. A manifest must also match the selected `ModelSpec`: the model id must match, optional provider/source/architecture/target sample rate metadata cannot conflict, and every file declared by the spec must be present with matching size/hash metadata.

The first provider is Hugging Face through the optional `download` extra. Additional providers should implement the same provider interface without changing backend code.

## Dependency Policy

The baseline package must remain lightweight and installable without model downloads or GPU dependencies.

- `sinc-resample` is the default backend and must run in CI.
- `audiosr` is optional and only imported when selected through the `audiosr` external backend.
- Missing optional dependencies must raise clear user-facing errors.
- Normal CI and normal inference must not download model weights.
- Untrusted pickle-style weight loading should be avoided for self-contained backends. Prefer safetensors or hash-verified trusted state dicts.

## CLI Policy

The CLI is grouped by user task:

- info/debugging
- enhancement I/O
- backend and inference
- quality checks

Machine-readable output should be available for listing commands when useful. `--list-backends --list-format json` and `--list-models --list-format json` are the current examples.

- `--download-weights --prepare-model-cache` downloads and verifies the selected model without requiring input/output audio.
- `--verify-weights` verifies the selected model and returns non-zero when local files are missing or mismatched.
- `--manifest` writes planned jobs, completed results, configuration, backend, and optional quality reports.
- `--compare-manifests` compares completed manifests and exits non-zero on differences.
- `--quality-report-json` writes standalone quality report artifacts for CI workflows that do not need a full run manifest.

## Test Policy

Default tests must remain lightweight, CPU-friendly, and offline. Download tests use fake providers by default. Real provider or real model inference tests must be gated by environment variables, such as `AUDIO_SUPER_RESOLUTION_RUN_AUDIOSR_INTEGRATION=1`.

## Release Policy

Releases are built with Pixi. PyPI publishing uses trusted publishing from `.github/workflows/release.yml`.

See [RELEASE.md](RELEASE.md) for the checklist.
