# Roadmap

This repository should stay useful before model inference lands. The baseline backend gives deterministic behavior for CLI, API, packaging, and tests; model-backed super-resolution can then slot into the same interface.

## Phase 1: Stable Local Tooling

- Keep Pixi as the primary development workflow.
- Maintain `audio-super-res` and `audiosr` CLI entry points.
- Support single-file and directory batch processing.
- Preserve relative paths during recursive batch processing.
- Keep backend selection stable through `--backend` and `AudioSuperResolver(backend=...)`.
- Provide machine-readable model/backend discovery and run manifests.

Status: complete for the baseline package. The implementation is available through the `sinc-resample` backend.

## Phase 2: Model Backend Integration

- Add a model cache directory and download policy. Done for local cache path configuration; download policy remains backend-specific.
- Add a configuration object for device, precision, chunk size, overlap, and seed. Done.
- Implement the first AudioSR-style model backend behind the existing backend protocol. Done with the optional `audiosr` backend wrapper.
- Keep the baseline `sinc-resample` backend for tests and CPU-only environments.
- Add clear errors for missing model weights, unsupported devices, and unavailable accelerators.
- Add optional input preprocessing for model backends. Done with low-pass preprocessing.

## Phase 3: Evaluation and Examples

- Add objective checks for sample rate, clipping, peak level, and duration drift. Done.
- Add example scripts for single-file enhancement, batch folder enhancement, and quality checks. Done.
- Add a small fixture-based test set that can run in CI without model weights. Done with generated fixtures.
- Add optional long-running tests for model inference.
- Add manifest-based regression comparison for batch runs. Done.

## Phase 4: Release Readiness

- Publish wheels to PyPI. Release workflow added; PyPI trusted publishing setup remains external.
- Add GitHub release notes and changelog automation. Changelog and release workflow added.
- Add Docker images for CPU and CUDA workflows. Baseline CPU Dockerfile added; CUDA image remains backend-specific.
- Add a Colab notebook once a model backend is wired.
- Add release artifacts for example manifests and quality reports.

## Next Implementation Plan

- Add optional AudioSR integration tests that are skipped unless the `audiosr` dependency and cache are available.
- Add sample manifest and quality-report artifacts under `examples/` for release notes and CI documentation.
- Add a small notebook or Colab plan once the AudioSR backend is verified with real weights.
- Add CUDA-oriented Docker documentation after a real GPU image can be tested.

## Candidate Backends

- `sinc-resample`: deterministic baseline, already implemented.
- `audiosr`: latent diffusion audio super-resolution backend, implemented as an optional dependency wrapper.
- `nuwave`: diffusion-based bandwidth extension backend.
- `custom`: user-provided backend implementing the Python protocol.

## Design Constraints

- The CLI should work for both one-off files and batch jobs.
- The Python API should be stable enough for scripts before model dependencies are added.
- Heavy ML dependencies should be optional extras, not required for baseline installation.
- Tests should not require GPU access or large model downloads by default.
