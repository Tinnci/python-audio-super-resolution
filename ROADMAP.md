# Roadmap

The package has a stable lightweight baseline and is adding model-backed inference behind the same CLI/API surface. The default path must remain offline, deterministic, and small.

## Current Baseline

- `sinc-resample` is the default backend and works without model weights or network access.
- `audiosr` is available as an optional external-package backend; its checkpoint behavior is upstream-controlled.
- `lavasr-compat` is available as an experimental self-contained LavaSR v2 BWE backend with managed weights.
- Managed weight infrastructure is implemented: multi-file manifests, path safety, size/SHA256 verification, explicit Hugging Face downloads, and verified local cache resolution.
- Regression helpers are implemented: run manifests, manifest comparison, quality reports, preprocessing, chunking, and sample JSON artifacts.
- Release automation uses GitHub Actions with PyPI Trusted Publishing / OIDC.

## Completed: v0.1.x

`v0.1.0` shipped the first public alpha baseline. Its PyPI files were yanked after `v0.1.1` replaced it with Python 3.10-compatible metadata/code.

`v0.1.1` is the current published baseline:

- PyPI install and CLI smoke test were validated on Python 3.10.
- The `v0.1.0` milestone is closed.
- First-release dry-run notes remain in [docs/RELEASE_DRY_RUN_0.1.0.md](docs/RELEASE_DRY_RUN_0.1.0.md) as historical release evidence.

## Active: v0.2.0

Goal: make the first self-contained compatible model backend useful enough to validate with real weights.

Tracked work:

- `#16` Implement `lavasr-compat` self-contained inference. Completed: self-contained torch inference runs with real LavaSR v2 BWE weights.
- `#17` Add golden-sample validation for compatible backends. Completed: fixture format, metrics, docs, and offline tests are available.
- `#18` Add gated real-weight model validation. Completed: gated LavaSR download verification and torch smoke tests are available.
- `#19` Publish validated Colab and GPU documentation. In progress: repository-based Colab/GPU validation guide is available.
- `#25` Add LavaSR upstream golden parity validation. Completed: gated upstream LavaSR/Vocos parity harness passed locally with the same verified weights.

Current `lavasr-compat` status:

- LavaSR v2 BWE model spec and managed weight metadata are implemented.
- Local bundle validation checks config metadata and required checkpoint key layout without importing torch.
- A self-contained torch runtime is wired experimentally: mel features, Vocos-style ConvNeXt backbone, ISTFT head, strict state-dict loading, and low/high-frequency merge.
- Real LavaSR v2 BWE download and bundle verification has passed through the gated integration test.
- Real LavaSR v2 BWE torch inference smoke has passed through the gated integration test.
- Upstream LavaSR/Vocos parity has passed through the gated parity test for a fixed synthetic fixture and shared local weights.

Remaining before closing `v0.2.0`:

- Record a fresh Colab or GPU runtime validation result in `#19`.

Remaining before marking `lavasr-compat` stable:

- Broaden fixture coverage beyond the initial parity sample.
- Add stricter mel/STFT exactness tests if future changes touch `lavasr_torch`.
- Keep default installs and default tests CPU/offline.

## Next: v0.3.0

Goal: plan model expansion after the first compatible inference path has real validation evidence.

Tracked work:

- `#21` Improve model catalog metadata for backend comparison. Completed: model listings expose task/domain, I/O, accelerator declarations, weight metadata, validation evidence, recommended use, and limitations.
- `#22` Define model admission criteria and candidate scorecard.
- `#23` Evaluate next speech SR/BWE compatible backend candidates.
- `#24` Evaluate general-audio SR candidate backends.

Decision rule:

- `v0.2.0` is for inference framework hardening and validation.
- `v0.3.0` is for choosing what to add next and how to compare candidates.
- `v0.4.0` is for hardware acceleration and runtime-provider work after the model/backend abstractions are clear.
- New candidate backends should not move into implementation until their weight format, license, preprocessing, I/O shape, and validation path are clear.

## Later: v0.4.0

Goal: optimize execution across hardware and external runtimes without making the baseline install heavy.

Tracked work:

- `#26` Define accelerator capability model and fallback policy.
- `#27` Add runtime provider abstraction for optimized execution.
- `#28` Add gated accelerator validation and benchmark matrix.
- `#29` Document accelerator install strategy and optional extras.
- `#30` Evaluate LavaSR optimized runtime and export paths.

Decision rule:

- Accelerator support is a runtime layer, not a model-selection layer.
- Backend code should request capabilities from a runtime/provider abstraction instead of hard-coding CUDA, ROCm, XPU, DirectML, OpenVINO, TensorRT, or ONNX Runtime checks.
- GPU/SDK-specific tests must remain gated and should produce JSON evidence before they become release gates.

## Candidate Backends

| Backend | Role |
| --- | --- |
| `sinc-resample` | Implemented deterministic baseline. |
| `audiosr` | Implemented optional external AudioSR wrapper. |
| `lavasr-compat` | Experimental self-contained speech BWE backend; real-weight download, torch smoke, and initial upstream parity pass. |
| `mossformer-sr-compat` | Future speech super-resolution candidate. |
| `nuwave` | Future diffusion-based bandwidth extension candidate. |
| custom backend | User-provided backend implementing the package protocol. |

## Constraints

- Baseline installation must not require GPU libraries, model weights, or network access.
- Normal inference must remain offline unless the user explicitly opts into downloading weights.
- Backend inference code must use verified local weight paths and must not call provider APIs directly.
- Default tests must not require GPU access or large model downloads.
