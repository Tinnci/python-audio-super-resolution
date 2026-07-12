# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning once published to PyPI.

## Unreleased

### Fixed

- Preserved exact digital silence in `lavasr-compat` instead of passing it through the model and
  producing deterministic low-level output texture.
- Made `audio-super-res eval matrix` return a non-zero exit status when any matrix run fails, so
  remote evidence workflows cannot report success for failed stability checks.
- Corrected `load_time_seconds` and `total_elapsed_seconds` comparison direction to
  `lower_is_better` alongside the other runtime metrics.

### Added

- Added a versioned release regression threshold policy and a Colab CLI workflow for same-backend
  baseline/candidate matrix evidence.
- Added a pinned, license-recorded LibriSpeech `dev-clean` tiny baseline specification and
  Colab-only real-speech matrix workflow.
- Added a pinned external Whisper evaluator specification and Colab-only transcript workflow for
  precomputed downstream WER/CER evidence without adding an ASR package dependency.
- Added a Colab-only verified MUSHRA bundle export for the licensed LibriSpeech sinc/LavaSR matrix.
- Added a bounded Colab-only MossFormer2 feasibility workflow covering minimal checkpoints, safe
  conversion, offline CPU inference, channel behavior, alignment, and parity-fixture constraints.
- Added a Colab T4 LavaSR eager/`torch.compile` benchmark and recorded the evidence-based decision
  to keep eager as the only selectable provider.

## 0.6.0 - 2026-07-12

### Added

- Added an experimental self-contained LavaSR-compatible torch runtime with Vocos-style modules, strict checkpoint loading, and verified local weight resolution.
- Added offline LavaSR v2 bundle validation, environment-gated real-weight download verification, torch smoke coverage, and upstream LavaSR/Vocos parity testing.
- Added golden-sample comparison helpers, tests, and fixture documentation for compatible backends.
- Expanded model catalog metadata and added a public model admission scorecard API.
- Added `v0.3.0` model-planning scope and `v0.4.0` accelerator/runtime-provider planning scope.
- Added speech and general-audio candidate reviews for ClearerVoice, Resemble Enhance, AudioSR, FlowHigh, and deferred research candidates.
- Added a repository-based LavaSR/GPU Colab validation guide and evidence checklist.
- Added a fresh Colab T4 validation record for `lavasr-compat` real-weight CUDA inference.
- Added `format-check`, `typecheck`, package metadata, and wheel contents checks to the Pixi quality tasks and CI/release workflows.
- Added a PEP 561 `py.typed` marker for downstream type checkers.
- Completed `v0.3.0` feasibility conclusions for ClearerVoice `MossFormer2_SR_48K`, FlowHigh, and Resemble Enhance, deferring each until its validation, checkpoint, or runtime blockers are resolved.
- Added accelerator/runtime-provider metadata, provider resolution helpers, and benchmark JSON output for gated hardware validation.
- Added accelerator installation, validation-matrix, and LavaSR optimization guidance.
- Added the first lightweight backend evaluation harness with controlled degraders, full-reference metrics, eval manifests, and `audio-super-res eval run/compare`.
- Added engineering/stability fields to eval manifests and benchmark JSON, including failure status, failure-case classification, RTF/peak RSS reporting, and backend capability/governance profiles.
- Added threshold-based `audio-super-res eval compare` checks and comparison tables for audio quality, downstream, engineering, stability, and governance evidence.
- Documented optional PESQ/STOI/ESTOI/MCD and codec-degrader integration requirements for full-reference evaluation.
- Added `audio-super-res eval no-reference` with builtin CPU/offline signal-stat screening records and documented gated DNSMOS/NISQA/UTMOS/ViSQOL adapter requirements.
- Added `audio-super-res eval downstream` with builtin ASR transcript WER/CER delta evaluation and gated speaker/VAD/KWS adapter schema.
- Added `audio-super-res eval listening-export` for AB/ABX/MUSHRA-ready blind bundles with separate answer keys and explicit rating dimensions.
- Completed the `v0.5.0` evaluation/regression harness milestone scope across full-reference, no-reference, downstream, listening, engineering/stability, and regression workflows.
- Prepared source version metadata for `0.6.0` release hardening after the completed v0.2-v0.5 milestones.
- Added `audio-super-res eval init-speech-bwe` to generate a deterministic synthetic `speech_bwe_v1_tiny` evalset for smoke/regression workflows.
- Added deterministic `opus_16k_24kbps` and `mp3_32kbps` codec-like degraders for lightweight eval coverage without requiring ffmpeg.
- Added explicit optional full-reference adapter attempts through `eval run --optional-metric` for PESQ, STOI, ESTOI, and MCD, with skipped dependency records when unavailable.
- Added backend load/init time and total elapsed time fields to benchmark and eval performance reports.
- Added `audio-super-res eval matrix` to run backend/degrader grids and write a `matrix.json` index over individual eval manifests.
- Added matrix comparison, JSON threshold policies, Markdown eval reports, artifact bundles, dataset manifest validation, lightweight MCD, optional full-reference extras, and optional CUDA memory profiling metadata.
- Added matrix `--reuse-existing` / `--fail-fast`, synthetic failure-case evalset generation, explicit local torch checkpoint inspection, and external backend adapter protocol documentation.

### Changed

- Wired `lavasr-compat` past verified local weights into the experimental runtime while keeping provider downloads outside backend inference.
- Removed PyYAML from the `lavasr` extra; the extra now only provides the torch runtime dependency.
- Reworked lightweight golden/preprocessing spectral paths to avoid fragile SciPy stateful filtering/STFT calls in default tests.
- Updated roadmap and release documentation for the completed `v0.1.x` release flow and completed `v0.2.0` validation gates.
- Updated model and backend listing metadata to report declared accelerators and runtime providers without importing heavy runtimes.
- Simplified documentation ownership by adding a docs index and keeping README focused on user entry points.
- Consolidated GPU/Colab, golden validation, backend candidate, and release records into their
  long-lived owner documents; refocused the roadmap on post-v0.6 priorities.
- Updated Docker installation to use `uv pip install --system` instead of direct `pip install`.

### Fixed

- Fixed eval matrix run-manifest resolution when `matrix.json` was created from a relative output
  directory and later compared or bundled from a different working directory.

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
