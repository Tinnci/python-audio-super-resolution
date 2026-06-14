# Roadmap

The package is useful before heavyweight model inference is complete. The default `sinc-resample` backend keeps the CLI, Python API, manifests, quality checks, packaging, and tests deterministic while model backends mature behind the same interface.

## Current Status

- Baseline CLI/API: single-file enhancement, recursive batch planning, dry runs, manifests, and manifest comparison are implemented.
- Backend system: runtime backend registry, model metadata specs, and model/backend listing commands are implemented.
- Model wrappers: `audiosr` is available as an optional external-package backend.
- Managed weights: multi-file manifests, size/SHA256 verification, explicit Hugging Face downloads, local cache resolution, and LavaSR v2 BWE metadata are implemented.
- Evaluation helpers: low-pass preprocessing, chunked enhancement, quality reports, and release example artifacts are implemented.
- Packaging: Pixi tasks, CPU Dockerfile, release workflow, changelog, and release dry-run notes are present.

## Active Next Work

- Complete `lavasr-compat` self-contained inference using the managed LavaSR v2 BWE weights.
- Add golden-sample tests comparing compatible backends against source implementations for fixed checkpoints.
- Run real-weight AudioSR and LavaSR validation on a suitable machine before promoting notebooks or GPU images.
- Convert the Colab plan into an executable notebook after real model validation.
- Add CUDA-oriented Docker documentation only after a GPU image has been tested.
- Confirm PyPI trusted publishing in the PyPI project settings before the first public release.

## Candidate Backends

- `sinc-resample`: deterministic baseline, implemented.
- `audiosr`: optional external AudioSR wrapper, implemented.
- `lavasr-compat`: managed LavaSR v2 BWE weight metadata and verification are implemented; inference is pending.
- `mossformer-sr-compat`: candidate speech super-resolution backend for future compatible inference.
- `nuwave`: candidate diffusion-based bandwidth extension backend.
- `custom`: user-provided backend implementing the Python protocol.

## Design Constraints

- Baseline installation must not require GPU libraries, model weights, or network access.
- Normal inference must remain offline unless the user explicitly opts into downloading weights.
- Backends must use verified local weight paths from `weight_store`; provider-specific download code stays outside backend inference.
- Default tests must not require GPU access or large model downloads.
