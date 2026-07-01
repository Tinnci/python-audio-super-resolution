# General-Audio Backend Candidates

This document records the `v0.3.0` general-audio super-resolution candidate review. General-audio means speech, music, and sound effects rather than speech-only bandwidth extension.

## Decision

Do not start a new self-contained general-audio backend yet.

Keep AudioSR as the external high-quality baseline. The FlowHigh feasibility pass found an official
implementation repository and a Hugging Face checkpoint repository, but FlowHigh should remain
deferred until CPU/offline validation, checkpoint licensing, and provider-neutral runtime behavior
are proven.

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
| FlowHigh | Feasibility candidate | Deferred. HF files and hashes are now known, but current code is CUDA-first and checkpoint license/provider guarantees are not strong enough for implementation. |
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
- Source revision inspected: `resemble-ai/flowhigh@ce73b7a3701b16e86f69050729c41dfc2b24a35b`
- Upstream describes it as an ICASSP 2025 single-step flow-matching audio
  super-resolution model.
- Public usage exposes `FlowHighSR.from_pretrained(...).generate(wav, sr_in, target_sr)`.
- Target sample rate is 48 kHz.
- The implementation includes transformer and BigVGAN/vocoder components.
- Public README references pretrained checkpoints distributed through Google Drive links, but the
  current package code downloads `ResembleAI/FlowHigh` from Hugging Face.
- Hugging Face model revision inspected: `ResembleAI/FlowHigh@4281fe4119e5f3d209a9a893218fed85af2e5bbe`.

Known Hugging Face files:

| File | Size | SHA256 |
| --- | ---: | --- |
| `FLowHigh_basic_400k.json` | 1264 | n/a |
| `FLowHigh_basic_400k.pt` | 481490826 | `84688e90d09b6f0788aeabc351ebc5f8d86463adb7ecf7ca3ef9c548e3825b8f` |
| `bigvgan_48khz_256band.json` | 1012 | n/a |
| `bigvgan_48khz_256band.pt` | 56105238 | `40c9fbe33e8d9f4090b988733996984e899c2ba69e475435e49704c8378c14bb` |

Model config facts:

- 48 kHz target.
- STFT/mel: `n_fft=2048`, `hop_length=480`, `win_length=2048`, `n_mel_channels=256`,
  `mel_fmin=20`, `mel_fmax=24000`.
- Flow model: transformer, `dim=1024`, `n_layers=2`, `n_heads=16`, `dim_head=64`,
  `cfm_path=basic_cfm`, `sigma=1e-4`.
- Vocoder: BigVGAN 48 kHz 256-band checkpoint.

Admission status:

| Criterion | Status | Notes |
| --- | --- | --- |
| Official implementation | Pass | The repository is public and implementation-oriented. |
| Code license | Pass | MIT. |
| Fixed target sample rate | Pass | 48 kHz target is explicit. |
| Weight management | Partial pass | Current code uses Hugging Face files with stable sizes and hashes, but the HF repo has no explicit license tag. |
| Self-contained feasibility | Needs spike | The vocoder boundary and exact required files must be mapped before deciding whether to reimplement or wrap. |
| CPU/offline validation | Blocked | Current package code hard-codes `.cuda()` in model construction, post-processing, and sampling paths. |
| Golden validation | Needs spike | Need deterministic fixture behavior or a statistical comparison strategy. |

Decision:

- Defer implementation.
- Prefer continued watchlist or a future optional GPU-only external wrapper only after upstream
  removes hard-coded CUDA assumptions.
- Do not pursue a self-contained compatibility backend until CPU fallback, checkpoint license, and
  safe checkpoint loading/conversion are resolved.
- Do not add FlowHigh dependencies or checkpoint download behavior to the baseline package.

Blockers:

- `FlowHighSR.from_local()` calls `.cuda()` directly for the generator and wrapper.
- `PostProcessing` and CFM sampling paths also create CUDA tensors directly.
- Checkpoints are PyTorch `.pt` files loaded with `torch.load`; self-contained admission needs a
  conversion plan.
- HF checkpoint repository lacks an explicit license tag even though the source repository is MIT.

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
2. FlowHigh publishes provider-neutral CPU/CUDA inference and explicit checkpoint licensing.
3. A maintainer decides to run an AudioSR compatibility feasibility spike despite the expected complexity.
4. A simpler STFT/iSTFT or ONNX-friendly general-audio SR model appears with reproducible inference.

Until then, prioritize:

- `v0.4.0` runtime-provider and accelerator planning for existing backends.
