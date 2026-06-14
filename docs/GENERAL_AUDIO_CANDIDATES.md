# General-Audio Backend Candidates

This document records the `v0.3.0` general-audio super-resolution candidate review. General-audio means speech, music, and sound effects rather than speech-only bandwidth extension.

## Decision

Do not start a new self-contained general-audio backend yet.

Keep AudioSR as the external high-quality baseline and wait for a simpler, officially reproducible general-audio model before implementing a self-contained backend.

Rationale:

- General-audio SR has a wider input distribution than speech-only BWE.
- Current strong candidates are either heavy diffusion systems or do not yet expose enough official implementation-grade metadata.
- A self-contained backend needs architecture, preprocessing, scheduler/inference parameters, weight files, license, CPU fallback, and golden validation. That bar is not met by the next general-audio candidates reviewed here.

## Candidate Summary

| Candidate | Track | Decision |
| --- | --- | --- |
| AudioSR `basic` | External package | Keep as existing external backend and high-quality baseline. Do not rewrite now. |
| AudioSR `speech` | External package | Keep as external baseline; speech-specific work should continue through LavaSR/ClearerVoice instead. |
| FlashSR | Research/watchlist | Promising direction because it targets faster AudioSR-style inference, but no official implementation-grade repo/weights were identified in this pass. |
| UniverSR | Research/watchlist | Promising vocoder-free direction, but no official implementation-grade repo/weights were identified in this pass. |
| NU-Wave2 | Research/watchlist | Defer; diffusion-style BWE is not a good next self-contained target without maintained code and validation artifacts. |

## AudioSR Status

Source facts:

- Repository: <https://github.com/haoheliu/versatile_audio_super_resolution>
- License: MIT
- Package: `audiosr==0.0.7`
- Upstream CLI supports `basic` and `speech` model names.
- Upstream describes the model as working on music, speech, sound effects, and broad input sampling rates.
- Inference uses stochastic generation controls such as DDIM steps, guidance scale, and seed.
- Upstream documents sensitivity to cutoff patterns and recommends low-pass preprocessing for some compressed inputs.

Admission status:

| Criterion | Status | Notes |
| --- | --- | --- |
| Catalog | Pass | Already wrapped as `external_package` models. |
| Self-contained implementation | Fail for now | Heavy latent diffusion path and upstream checkpoint behavior make this a poor first rewrite target. |
| Weight management | Not package-owned | Upstream package owns checkpoint downloads. |
| Golden validation | Needs gated external tests | Strict waveform parity is not expected for stochastic diffusion unless scheduler and seed are fully pinned. |

Decision:

- Keep `audiosr` as an optional external backend.
- Keep normal inference offline for package-owned managed weights; AudioSR remains the documented external-package exception.
- Do not implement `audiosr-compat` until a separate feasibility issue proves architecture, scheduler, preprocessing, and checkpoint conversion.

## Watchlist Criteria

A general-audio candidate can move from watchlist to feasibility only when it has:

- official source repository or reproducible package
- permissive code license
- clear checkpoint license
- public weight files with stable revisions, sizes, and hashes
- waveform-in/waveform-out or fully documented latent/vocoder boundaries
- fixed target sample rate or explicit target-rate contract
- deterministic or seed-controlled inference
- CPU fallback for default validation
- small enough fixture path for gated golden testing

## Deferred Candidates

FlashSR and UniverSR should stay on the watchlist. They are worth revisiting because they aim to reduce the cost or complexity of general-audio SR, but this pass did not identify enough official implementation metadata to start a package-owned backend.

NU-Wave2 should also stay deferred. It may be valuable for benchmark context, but diffusion-style BWE is not the next practical self-contained backend after LavaSR.

## Follow-Up

General-audio implementation should resume after one of these happens:

1. An official FlashSR/UniverSR implementation with clear weight files and permissive licensing becomes available.
2. A maintainer decides to run an AudioSR compatibility feasibility spike despite the expected complexity.
3. A simpler STFT/iSTFT or ONNX-friendly general-audio SR model appears with reproducible inference.

Until then, prioritize:

- `#31` ClearerVoice/MossFormer2 speech feasibility.
- `v0.4.0` runtime-provider and accelerator planning for existing backends.
