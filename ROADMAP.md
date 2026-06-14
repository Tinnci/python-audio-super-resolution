# Roadmap

The package has a stable lightweight baseline and is adding model-backed inference behind the same CLI/API surface. The default path must remain offline, deterministic, and small.

## Current Baseline

- `sinc-resample` is the default backend and works without model weights or network access.
- `audiosr` is available as an optional external-package backend; its checkpoint behavior is upstream-controlled.
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

- `#16` Implement `lavasr-compat` self-contained inference.
- `#17` Add golden-sample validation for compatible backends. Completed: fixture format, metrics, docs, and offline tests are available.
- `#18` Add gated real-weight model validation. Completed: gated LavaSR download verification and torch smoke tests are available.
- `#19` Publish validated Colab and GPU documentation.

Current `lavasr-compat` status:

- LavaSR v2 BWE model spec and managed weight metadata are implemented.
- Local bundle validation checks config metadata and required checkpoint key layout without importing torch.
- A self-contained torch runtime is wired experimentally: mel features, Vocos-style ConvNeXt backbone, ISTFT head, strict state-dict loading, and low/high-frequency merge.

Remaining before marking `lavasr-compat` stable:

- Run the gated real-weight download/load/inference tests in an environment with `download` and `lavasr` extras installed.
- Use the golden fixture framework to compare LavaSR output against upstream LavaSR/Vocos samples.
- Add exactness tests for any mel/STFT behavior that differs from the reference implementation.
- Document Colab/GPU usage only after real model validation passes.

## Next: v0.3.0

Goal: plan model expansion after the first compatible inference path has real validation evidence.

Tracked work:

- `#21` Improve model catalog metadata for backend comparison.
- `#22` Define model admission criteria and candidate scorecard.
- `#23` Evaluate next speech SR/BWE compatible backend candidates.
- `#24` Evaluate general-audio SR candidate backends.

Decision rule:

- `v0.2.0` is for inference framework hardening and validation.
- `v0.3.0` is for choosing what to add next and how to compare candidates.
- New candidate backends should not move into implementation until their weight format, license, preprocessing, I/O shape, and validation path are clear.

## Candidate Backends

| Backend | Role |
| --- | --- |
| `sinc-resample` | Implemented deterministic baseline. |
| `audiosr` | Implemented optional external AudioSR wrapper. |
| `lavasr-compat` | Experimental self-contained speech BWE backend; real-weight execution and LavaSR-specific golden artifacts pending. |
| `mossformer-sr-compat` | Future speech super-resolution candidate. |
| `nuwave` | Future diffusion-based bandwidth extension candidate. |
| custom backend | User-provided backend implementing the package protocol. |

## Constraints

- Baseline installation must not require GPU libraries, model weights, or network access.
- Normal inference must remain offline unless the user explicitly opts into downloading weights.
- Backend inference code must use verified local weight paths and must not call provider APIs directly.
- Default tests must not require GPU access or large model downloads.
