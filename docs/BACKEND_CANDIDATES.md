# Backend Candidates

This document owns implementation-grade evidence for model candidates. Admission rules live in
[MODEL_ADMISSION.md](MODEL_ADMISSION.md); milestone ordering lives in [../ROADMAP.md](../ROADMAP.md).

## Current Decision

Do not start another package-owned backend until v0.6.0 is released and broader LavaSR evidence is
recorded. When capacity returns, run one bounded ClearerVoice `MossFormer2_SR_48K` feasibility spike.
Do not implement MossFormer2, FlowHigh, and Resemble Enhance in parallel.

| Candidate | Domain | Decision |
| --- | --- | --- |
| `lavasr-v2-bwe` | Speech BWE | Implemented experimental baseline; broaden evidence before stable promotion. |
| ClearerVoice `MossFormer2_SR_48K` | Speech SR to 48 kHz | First future spike; defer implementation pending safe checkpoint and parity evidence. |
| Resemble Enhance | Speech enhancement/BWE to 44.1 kHz | Deferred; consider only as an optional external backend. |
| AudioSR `basic` / `speech` | General audio / speech | Keep the existing external-package backend; do not rewrite now. |
| FlowHigh | General audio SR to 48 kHz | Deferred; CUDA-first runtime and checkpoint-license/provider blockers remain. |
| AP-BWE, NU-Wave2, FastWave, SAGA-SR, FlashSR, UniverSR, Latent Bridge Models | Research/watchlist | Reopen only with official code, usable licenses, weights, and reproducible inference. |

## ClearerVoice MossFormer2_SR_48K

Source:

- repository: <https://github.com/modelscope/ClearerVoice-Studio>
- code license: Apache-2.0
- model repository: <https://huggingface.co/alibabasglab/MossFormer2_SR_48K>
- inspected source revision: `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`
- inspected model revision: `39eb1f25ea84f5e0315ade9ac0070fff216fc690`
- upstream task: clean-speech super-resolution to 48 kHz

Observed weight files:

| File | Size | SHA256 |
| --- | ---: | --- |
| `last_best_checkpoint` | 52 | Pointer metadata |
| `last_best_checkpoint_g.pt` | 220,712,702 | `0bdd13c21466f5963d9d1f86a9d84fc6196868318fe22c6b0a750f041805adda` |
| `last_best_checkpoint_m.pt` | 218,471,889 | `6cbadb2b6b839e444bb65223c69eea162c8ad08f36e9d0a64144672c4095ab36` |
| `do_03925000` | 1,744,458,369 | `549035d29a03928a854815b9c1b21c02d825845aa2e5f093af45694a236c1f19` |

The public SR loader appears to use the pointer plus `m` and `g` checkpoints; the larger file should
not enter a managed manifest unless the spike proves it is required.

Blockers:

- upstream loading can download implicitly when files are absent;
- checkpoints use PyTorch `.pt` loading and need a safe conversion/loading plan;
- package-owned module boundaries and key mapping are not reproduced;
- a short upstream CPU smoke and parity fixture have not been recorded;
- the clean-speech limitation must remain visible in catalog metadata.

Spike exit criteria:

1. Inspect tensor keys for the two required checkpoints from a temporary cache.
2. Run upstream CPU inference on a sub-second fixture without provider calls during inference.
3. Record input/output sample-rate, channel, alignment, and preprocessing behavior.
4. Decide whether conversion to a safer package-owned format is practical.
5. Define a parity fixture before opening backend implementation.

## Resemble Enhance

Source:

- repository: <https://github.com/resemble-ai/resemble-enhance>
- code license: MIT
- model repository: `ResembleAI/resemble-enhance`
- inspected source revision: `8e978149bfe8abab3eb77d965d579a111afdb0ff`
- inspected model revision: `4e3510ce4a8391159f665903544c5150bee7b2cb`
- target: speech enhancement/BWE at 44.1 kHz

Known enhancer files include `enhancer_stage2/hparams.yaml` and a DeepSpeed-style checkpoint at
`enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt` (713,176,232 bytes, SHA256
`f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6`).

Keep it deferred because the dependency surface includes training/runtime packages, the checkpoint
layout likely needs conversion, YAML metadata requires safe parsing, and its 44.1 kHz speech target
does not match the current 48 kHz SR path. If revisited, first isolate an inference-only external
wrapper and declare the target rate explicitly.

## AudioSR

AudioSR remains the optional external high-quality general-audio baseline:

- repositories: <https://github.com/AudioLDM/audiosr> and
  <https://github.com/haoheliu/versatile_audio_super_resolution>
- package: `audiosr==0.0.7`
- models: `basic` and `speech`
- code license: MIT
- behavior: stochastic diffusion controlled by DDIM steps, guidance scale, and seed

The upstream package owns checkpoints and runtime behavior. A package-owned rewrite is not justified
until architecture, scheduler, preprocessing, deterministic validation, and checkpoint conversion
are independently proven.

## FlowHigh

Source:

- repository: <https://github.com/resemble-ai/flowhigh>
- code license: MIT
- inspected source revision: `ce73b7a3701b16e86f69050729c41dfc2b24a35b`
- model repository: `ResembleAI/FlowHigh`
- inspected model revision: `4281fe4119e5f3d209a9a893218fed85af2e5bbe`
- target: single-step general-audio SR to 48 kHz

Observed files:

| File | Size | SHA256 |
| --- | ---: | --- |
| `FLowHigh_basic_400k.json` | 1,264 | Config metadata |
| `FLowHigh_basic_400k.pt` | 481,490,826 | `84688e90d09b6f0788aeabc351ebc5f8d86463adb7ecf7ca3ef9c548e3825b8f` |
| `bigvgan_48khz_256band.json` | 1,012 | Config metadata |
| `bigvgan_48khz_256band.pt` | 56,105,238 | `40c9fbe33e8d9f4090b988733996984e899c2ba69e475435e49704c8378c14bb` |

The current implementation hard-codes CUDA in construction and sampling paths. The Hugging Face
checkpoint repository also lacks an explicit license tag, and `.pt` loading needs a conversion
plan. Keep FlowHigh deferred until CPU/provider-neutral execution, checkpoint licensing, and a
deterministic or statistically stable parity strategy are available.

## Watchlist Admission

A watchlist entry can move to feasibility only when it has:

- an official implementation or reproducible package;
- usable code and checkpoint licenses;
- stable weight revisions, sizes, and hashes;
- documented preprocessing, I/O, and target-rate contracts;
- deterministic or seed-controlled inference;
- an offline validation path and, for package-owned work, a CPU fallback;
- a small license-safe fixture suitable for gated comparison.

Research visibility alone is not an implementation signal.
