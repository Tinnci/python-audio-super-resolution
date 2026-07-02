# Architecture

The package is organized around a small public CLI/API surface, a pluggable backend registry, and explicit model-weight management. The default path must stay lightweight, deterministic, and offline.

## Component Map

| Area | Modules | Responsibility |
| --- | --- | --- |
| User entry points | `cli`, `resolver` | Parse commands, plan files, run enhancement, and report results. |
| Backends | `backends.*`, `audiosr_backend` | Register enhancement implementations and expose backend-owned model specs. |
| Model catalog | `models`, `specs` | Convert backend `ModelSpec` records into user-facing listings and strict selection. |
| Weight management | `weights`, `downloads`, `weight_store`, `model_weights` | Verify local manifests, download explicit provider files, and resolve verified paths. |
| Runtime helpers | `config`, `devices`, `runtime`, `chunking`, `preprocess` | Shared inference options, device discovery, runtime-provider selection, chunking, and optional preprocessing. |
| Artifacts | `manifest`, `quality` | JSON run manifests, manifest comparison, and quality reports. |

High-level flow:

```text
CLI / Python API
  -> resolver
    -> backend registry
      -> selected backend
        -> optional spec-based weight resolution
```

## Backend Contract

Backends register by name in `audio_super_resolution.backends.registry`. Built-in backends register at import time; external code can call `register_backend()` with an object that follows the backend protocol.

Array-native backends implement:

```python
def enhance(audio, sample_rate, target_sample_rate):
    ...
```

File-native backends, such as the optional external AudioSR wrapper, may implement:

```python
def enhance_file(input_path, output_path, target_sample_rate):
    ...
```

`AudioSuperResolver.enhance()` prefers `enhance_file()` when present. Otherwise it reads audio into arrays, applies optional preprocessing, and calls `enhance()`.

If `chunked=True`, the resolver reads overlapping chunks, processes each chunk through the same backend path, and writes a crossfaded output. Chunking is resolver behavior; backends do not need to duplicate batch or file-planning logic.

Backends may expose static `ModelSpec` records through `model_specs()`. The catalog uses those specs for `--list-models`, Python discovery, and strict model selection. A backend that owns a `ModelSpec` should pass that spec directly to weight resolution instead of looking itself up through the catalog.

Model specs should describe stable comparison facts without importing heavy runtimes:

- task and domain, such as resampling, speech bandwidth extension, or general audio super-resolution
- architecture and implementation family, such as baseline, external package, or self-contained torch
- supported input and target sample-rate metadata
- managed weight source, provider, file count, size, hash, revision, and license metadata
- backend capability metadata for array/file I/O, chunking, determinism, accelerator support, runtime providers, and precision modes
- validation evidence, recommended use, and known limitations

Provider-specific accelerator routing belongs below the catalog in the runtime-provider layer. The catalog reports declared support through `accelerators` and `runtime_providers`, but it must not import CUDA, ROCm, XPU, DirectML, OpenVINO, TensorRT, ONNX Runtime, or other SDKs merely to list models.

Runtime-provider rules live in [ACCELERATORS.md](ACCELERATORS.md). Backends may request a supported
provider and logical device, but provider installation and global device detection stay in shared
runtime helpers.

Candidate admission rules and scorecard usage live in [MODEL_ADMISSION.md](MODEL_ADMISSION.md).

## External Adapter Protocol

External wrappers for ClearerVoice, Resemble Enhance, FlowHigh, or similar packages should use the
same backend contract as built-ins:

- register a backend object through `register_backend()`
- expose import-light `ModelSpec` metadata through `model_specs()`
- implement `enhance_file()` when the upstream package is file-native
- never download weights during normal inference
- fail with explicit install/cache guidance when optional packages or weights are missing
- declare `implementation="external_package"` unless inference is package-owned
- provide enough `BackendCapability` metadata for `eval matrix` to compare I/O, CPU/GPU support,
  determinism, runtime provider, and governance without importing the heavy runtime

An external adapter can enter `eval matrix` before becoming a self-contained backend. Promotion to a
package-owned compatibility backend still requires the admission gates in
[MODEL_ADMISSION.md](MODEL_ADMISSION.md).

## Weight Management

Default inference is offline. Missing weights should fail with a command that tells the user how to explicitly download and verify the selected model.

Layering:

| Layer | Input | May access network? | May import backend catalog? | Purpose |
| --- | --- | --- | --- | --- |
| `weights` | manifest paths | No | No | Parse manifests, reject unsafe paths, verify size/SHA256. |
| `downloads` | `ModelSpec` | Yes, only when called explicitly | No | Provider abstraction and atomic cache publication. |
| `weight_store` | `ModelSpec` + `InferenceConfig` | Only through explicit download flag | No | Resolve explicit manifest, verified cache, or opt-in download. |
| `model_weights` | model id/backend selection or `ModelSpec` | Delegates only | Yes | Public facade for CLI and Python API convenience. |

This split prevents the cycle `backend -> weight_store -> models -> backends`. Inference code depends on stable metadata (`ModelSpec`), while model-id lookup stays at the CLI/API edge.

Managed cache layout:

```text
<model_cache_dir>/<model_id>/
  manifest.json
  .complete
  ...
```

Download behavior:

- Downloads are explicit through CLI/API flags or calls.
- Files are written to a temporary directory first.
- Every required file must pass size/hash verification before publication.
- Existing verified cache directories are replaced only when `force=True` and the new cache validates.
- Failed downloads remove only the temporary directory.

Manifest rules:

- File paths must be relative to the manifest directory.
- Absolute paths, Windows drive paths, and `..` traversal are rejected before file access.
- A manifest must match the selected `ModelSpec` id.
- Optional provider/source/architecture/target sample rate metadata cannot conflict with the spec.
- Every file declared by the spec must be present with matching required size/hash metadata.

The first remote provider is Hugging Face through the optional `download` extra. Additional providers should implement the same provider interface without changing backend code.

## Dependency Policy

- `sinc-resample` is the default backend and must run in CI.
- `audiosr` is optional and imported only when its external backend is selected.
- `lavasr-compat` can require local verified files and optional torch runtime, but it must not call Hugging Face or other providers directly.
- Runtime provider adapters must be optional and import-light until selected.
- Normal CI and normal inference must not download model weights.
- Self-contained backends should prefer safetensors or hash-verified trusted state dicts over untrusted pickle-style loading.

## CLI Policy

The CLI groups options by user task:

- info/debugging
- enhancement I/O
- backend and inference
- quality checks

Machine-readable listing output should be available where useful, such as `--list-backends --list-format json` and `--list-models --list-format json`.

Key weight commands:

- `--download-weights --prepare-model-cache`: download and verify the selected model without requiring input/output audio.
- `--verify-weights`: verify local weights and return non-zero on missing or mismatched files.
- normal enhancement: use local verified files unless `--download-weights` is explicitly provided.

## Maintenance Policy

- Documentation ownership is summarized in [docs/README.md](README.md).
- Default tests must stay lightweight, CPU-friendly, and offline. See [tests/README.md](../tests/README.md).
- Model maturity and milestone state live in [ROADMAP.md](../ROADMAP.md); this document owns architecture boundaries.
- Release steps live in [RELEASE.md](RELEASE.md).
- User-facing commands and examples live in [README.md](../README.md) and [examples/](../examples/).
- Future notebooks and GPU images should be documented only after real model validation.
