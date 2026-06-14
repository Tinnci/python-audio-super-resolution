# Roadmap

The package is already usable through the deterministic `sinc-resample` backend. Model-backed inference is being added behind the same CLI/API surface so baseline workflows stay lightweight, offline, and testable.

## Implemented

- CLI and Python API for single-file runs, recursive batches, dry runs, and output planning.
- Backend registry, model metadata specs, backend/model listing, and strict model selection.
- Baseline `sinc-resample` backend and optional external `audiosr` wrapper.
- Managed weight infrastructure: multi-file manifests, path safety, size/SHA256 verification, explicit Hugging Face downloads, and local cache resolution.
- Regression helpers: manifests, manifest comparison, quality reports, low-pass preprocessing, and chunked processing.
- Maintenance infrastructure: Pixi tasks, CPU Dockerfile, GitHub workflows, release notes, architecture docs, and sample JSON artifacts.

## Active Work

1. Implement `lavasr-compat` self-contained inference using the verified LavaSR v2 BWE files.
2. Add golden-sample checks against source implementations for compatible self-contained backends.
3. Validate optional real-weight AudioSR and LavaSR paths on suitable hardware.
4. Convert the Colab plan into an executable notebook after real model validation.
5. Add GPU Docker documentation only after a CUDA image has been tested.
6. Confirm PyPI trusted publishing before the first public release tag.

## Candidate Backends

| Backend | Role |
| --- | --- |
| `sinc-resample` | Implemented deterministic baseline. |
| `audiosr` | Implemented optional external AudioSR wrapper. |
| `lavasr-compat` | Next self-contained speech BWE target; weights are managed, inference pending. |
| `mossformer-sr-compat` | Future speech super-resolution candidate. |
| `nuwave` | Future diffusion-based bandwidth extension candidate. |
| custom backend | User-provided backend implementing the package protocol. |

## Constraints

- Baseline installation must not require GPU libraries, model weights, or network access.
- Normal inference must remain offline unless the user explicitly opts into downloading weights.
- Backend inference code must use verified local weight paths and must not call provider APIs directly.
- Default tests must not require GPU access or large model downloads.
