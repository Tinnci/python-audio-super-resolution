# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning once published to PyPI.

## Unreleased

- Added offline LavaSR v2 bundle validation with optional PyYAML fallback parsing.
- Added an experimental self-contained LavaSR-compatible torch runtime with Vocos-style modules and strict checkpoint loading.
- Wired `lavasr-compat` past verified local weights into the experimental runtime while keeping provider downloads outside backend inference.
- Removed PyYAML from the `lavasr` extra; the extra now only provides the torch runtime dependency.
- Updated roadmap and release documentation for the completed `v0.1.x` release flow and active `v0.2.0` work.
- Added `v0.3.0` planning scope for model admission criteria and future backend candidates.
- Added environment-gated LavaSR real-weight download verification and torch smoke-test coverage.
- Added golden-sample comparison helpers, tests, and fixture documentation for compatible backends.
- Validated `lavasr-compat` real-weight torch smoke locally through the gated integration test.
- Reworked lightweight golden/preprocessing spectral paths to avoid fragile SciPy stateful filtering/STFT calls in default tests.
- Added a gated LavaSR upstream parity test that compares `lavasr-compat` with upstream `LavaSR.enhancer.LavaBWE` using the same verified local weights.
- Added `v0.4.0` planning scope for accelerator capability metadata, runtime providers, gated hardware benchmarks, accelerator install docs, and LavaSR optimized export paths.
- Replaced the Colab draft with a repository-based LavaSR/GPU validation guide and evidence checklist.
- Expanded model catalog metadata for backend comparison, including task/domain, input-rate metadata, I/O capabilities, accelerator declarations, weight size/source/license, validation evidence, recommended use, and limitations.
- Added model admission criteria documentation and a public scorecard API for catalog and self-contained backend candidates.
- Added speech SR/BWE candidate review selecting ClearerVoice `MossFormer2_SR_48K` for a feasibility spike before any new backend implementation.
- Added general-audio SR candidate review keeping AudioSR as the external baseline and deferring new self-contained general-audio work.

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
