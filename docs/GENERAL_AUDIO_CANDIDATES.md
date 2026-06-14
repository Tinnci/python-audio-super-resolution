# General-Audio Backend Candidates

This document records the `v0.3.0` general-audio super-resolution candidate review. General-audio means speech, music, and sound effects rather than speech-only bandwidth extension.

## Decision

Do not start a new self-contained general-audio backend yet.

Keep AudioSR as the external high-quality baseline. Move FlowHigh into a feasibility spike because
it has an official implementation repository and a permissive code license, but do not implement it
until checkpoint distribution, vocoder boundaries, and validation mechanics are proven.

Rationale:

- General-audio SR has a wider input distribution than speech-only BWE.
- Current strong candidates are either heavy generation systems or do not yet expose enough
  implementation-grade metadata.
- A self-contained backend needs architecture, preprocessing, scheduler/inference parameters, weight files, license, CPU fallback, and golden validation. That bar is not met by the next general-audio candidates reviewed here.

## Candidate Summary

| Candidate | Track | Decision |
| --- | --- | --- |
| AudioSR `basic` | External package | Keep as existing external backend and high-quality baseline. Do not rewrite now. |
| AudioSR `speech` | External package | Keep as external baseline; speech-specific work should continue through LavaSR/ClearerVoice instead. |
| FlowHigh | Feasibility candidate | Open `#32` to verify checkpoints, BigVGAN/vocoder boundaries, CPU/offline validation, and managed-weight suitability. |
| SAGA-SR | Research/watchlist | Watch only until an official implementation, weight source, and license are identified. |
| Latent Bridge Models | Research/watchlist | Watch only until implementation-grade artifacts are available. |
| FlashSR | Research/watchlist | Promising direction because it targets faster AudioSR-style inference, but no official implementation-grade repo/weights were identified in this pass. |
| UniverSR | Research/watchlist | Promising vocoder-free direction, but no official implementation-grade repo/weights were identified in this pass. |
| NU-Wave2 / FastWave | Research/watchlist | Defer; speech diffusion/fast-diffusion BWE is not a general-audio self-contained target without maintained code and validation artifacts. |

Resemble Enhance is a strong speech-focused enhancement/BWE candidate, but it is not a
general-audio SR model. It is tracked in [SPEECH_BACKEND_CANDIDATES.md](SPEECH_BACKEND_CANDIDATES.md).

## AudioSR Status

Source facts:

- Project page repository: <https://github.com/AudioLDM/audiosr>
- Package/code repository: <https://github.com/haoheliu/versatile_audio_super_resolution>
- Package/code repository license: MIT
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

## FlowHigh Feasibility Candidate

Source facts:

- Repository: <https://github.com/resemble-ai/flowhigh>
- License: MIT
- Upstream describes it as an ICASSP 2025 single-step flow-matching audio
  super-resolution model.
- Public usage exposes `FlowHighSR.from_pretrained(...).generate(wav, sr_in, target_sr)`.
- Target sample rate is 48 kHz.
- The implementation includes transformer and BigVGAN/vocoder components.
- Public README references pretrained checkpoints distributed through Google Drive links.

Admission status:

| Criterion | Status | Notes |
| --- | --- | --- |
| Official implementation | Pass | The repository is public and implementation-oriented. |
| Code license | Pass | MIT. |
| Fixed target sample rate | Pass | 48 kHz target is explicit. |
| Weight management | Needs spike | Current checkpoint links are not yet a provider-backed manifest with stable revision, size, and SHA256 metadata. |
| Self-contained feasibility | Needs spike | The vocoder boundary and exact required files must be mapped before deciding whether to reimplement or wrap. |
| CPU/offline validation | Needs spike | Single-step generation helps, but default tests must stay CPU/offline and small. |
| Golden validation | Needs spike | Need deterministic fixture behavior or a statistical comparison strategy. |

Decision:

- Track as `#32` before any backend implementation.
- Prefer a feasibility report that separates three outcomes: external wrapper,
  self-contained compatibility backend, or continued watchlist.
- Do not add FlowHigh dependencies or checkpoint download behavior to the baseline package.

## Unverified SOTA Watchlist

The following names are relevant for research tracking, but they are not implementation targets yet:

- SAGA-SR: promising semantic/acoustic guided SR direction, but no official implementation-grade
  repository and weight source were identified in this pass.
- Latent Bridge Models for audio SR: promising latent bridge direction, but no implementation-grade
  source artifacts were identified in this pass.
- FastWave: promising lightweight speech BWE direction, but no implementation-grade source artifacts
  were identified in this pass.

These can move to feasibility only after source repository, license, checkpoint source, inference
contract, and validation artifacts are identified.

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

NU-Wave2/FastWave should also stay deferred for this general-audio track. They may be valuable for
speech benchmark context, but speech-specific BWE planning belongs in the speech candidate review.

## Follow-Up

General-audio implementation should resume after one of these happens:

1. An official FlashSR/UniverSR implementation with clear weight files and permissive licensing becomes available.
2. A maintainer decides to run an AudioSR compatibility feasibility spike despite the expected complexity.
3. A simpler STFT/iSTFT or ONNX-friendly general-audio SR model appears with reproducible inference.

Until then, prioritize:

- `#31` ClearerVoice/MossFormer2 speech feasibility.
- `#32` FlowHigh general-audio feasibility.
- `#33` Resemble Enhance speech enhancement/BWE feasibility.
- `v0.4.0` runtime-provider and accelerator planning for existing backends.
