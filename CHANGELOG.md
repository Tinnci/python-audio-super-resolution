# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning once published to PyPI.

## Unreleased

- Added offline LavaSR v2 bundle validation with optional PyYAML fallback parsing.
- Added an experimental self-contained LavaSR-compatible torch runtime with Vocos-style modules and strict checkpoint loading.
- Wired `lavasr-compat` past verified local weights into the experimental runtime while keeping provider downloads outside backend inference.
- Removed PyYAML from the `lavasr` extra; the extra now only provides the torch runtime dependency.
- Updated roadmap and release documentation for the completed `v0.1.x` release flow and active `v0.2.0` work.

## 0.1.1

- Fixed Python 3.10 import compatibility by replacing Python 3.11-only `datetime.UTC` usage.
- Superseded the yanked `0.1.0` PyPI files as the current public alpha baseline.

## 0.1.0

- Added the initial Pixi-managed Python package.
- Added `audio-super-res` and `audiosr` CLI entry points.
- Added single-file and recursive directory batch processing.
- Added explicit chunked processing with overlap crossfading for long files.
- Added backend selection with the baseline `sinc-resample` backend.
- Added inference configuration for device, precision, chunking, seed, and model cache path.
- Added audio quality checks for sample rate, duration drift, clipping, and peak level.
- Added standalone JSON quality report export through `--quality-report-json`.
- Added examples for single-file enhancement, batch processing, and quality checks.
- Added the optional `audiosr` backend wrapper for AudioSR latent diffusion inference.
- Added an environment-gated real AudioSR integration test.
- Added backend availability metadata and JSON output for `--list-backends`.
- Added model catalog output through `--list-models`.
- Added JSON manifests for dry-run plans and completed enhancement runs.
- Added manifest regression comparison through `--compare-manifests`.
- Added optional low-pass preprocessing for model-backed enhancement runs.
- Added managed weight manifests, local cache verification, and explicit Hugging Face download plumbing.
- Added LavaSR-compatible managed weight metadata while keeping self-contained LavaSR inference marked as pending.
- Added PyPI Trusted Publishing / GitHub OIDC release workflow configuration.
- Added release dry-run notes, a Colab plan, and sample JSON artifacts.
- Added GitHub issue templates, Dependabot configuration, CodeQL security workflow, and architecture/test documentation.
