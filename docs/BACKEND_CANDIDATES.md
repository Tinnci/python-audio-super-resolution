# Backend Candidates

This document owns implementation-grade evidence for model candidates. Admission rules live in
[MODEL_ADMISSION.md](MODEL_ADMISSION.md); milestone ordering lives in [../ROADMAP.md](../ROADMAP.md).

## Current Decision

The bounded ClearerVoice `MossFormer2_SR_48K` feasibility spike is complete. It clears the checkpoint,
license, offline-loading, CPU-fallback, and basic I/O gates, but backend implementation remains a
separate decision because the model is heavyweight and its output has a fixed alignment offset. Do
not implement MossFormer2, FlowHigh, and Resemble Enhance in parallel.

| Candidate | Domain | Decision |
| --- | --- | --- |
| `lavasr-v2-bwe` | Speech BWE | Implemented experimental baseline; broaden evidence before stable promotion. |
| ClearerVoice `MossFormer2_SR_48K` | Speech SR to 48 kHz | Feasibility passed; eligible for one bounded backend plan with tolerance-based parity. |
| Resemble Enhance | Speech enhancement/BWE to 44.1 kHz | Deferred; consider only as an optional external backend. |
| AudioSR `basic` / `speech` | General audio / speech | Keep the existing external-package backend; do not rewrite now. |
| FlowHigh | General audio SR to 48 kHz | Deferred; CUDA-first runtime and checkpoint-license/provider blockers remain. |
| AP-BWE, NU-Wave2, FastWave, SAGA-SR, FlashSR, UniverSR, Latent Bridge Models | Research/watchlist | Reopen only with official code, usable licenses, weights, and reproducible inference. |

## ClearerVoice MossFormer2_SR_48K

Source:

- repository: <https://github.com/modelscope/ClearerVoice-Studio>
- code license: Apache-2.0
- checkpoint license: Apache-2.0 in the pinned Hugging Face model card and repository tag
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

The Colab spike proved that the pointer plus `m` and `g` checkpoints are sufficient. The 1.74 GB
`do_03925000` file was not downloaded and must not enter a managed manifest.

Recorded Colab CPU evidence:

- `last_best_checkpoint_m.pt` contains a `mossformer` state with 929 tensors and 54,497,666
  parameters; `last_best_checkpoint_g.pt` contains a `generator` state with 311 tensors and
  55,150,402 parameters;
- both files load with `torch.load(..., weights_only=True)` and round-trip exactly through
  safetensors after cloning shared rotary-frequency buffers;
- converted sizes are 218,118,376 and 220,633,048 bytes, with recorded deterministic conversion
  hashes in the evidence bundle;
- a generated 0.25-second 16 kHz mono fixture ran on CPU with Hugging Face offline mode enabled;
  model construction took 14-21 seconds and first inference took 4.1-5.2 seconds on Colab CPU;
- repeated inference in one process was bit-exact, but separate clean sessions produced small
  floating-point differences, so parity must use numeric/audio tolerances rather than a WAV hash;
- 16 kHz mono input produced 48 kHz-domain shape `[1, 11776]`; 16 kHz stereo preserved two
  independently processed channels with shape `[2, 11776]`;
- both a 4000-sample 16 kHz input and a 12000-sample 48 kHz input produced 11776 samples, a fixed
  `-224` sample (`-4.67 ms`) alignment delta for this fixture;
- upstream preprocessing resamples inputs to 48 kHz, processes channels independently, and marks SR
  output as 48 kHz rather than resampling it back to the original input rate.

The package-owned parity fixture should generate the same seeded 440 Hz mono input, compare the
pre-alignment upstream float output, explicitly account for the 224-sample generator boundary loss,
and use bounded waveform/spectral tolerances. Do not use a cross-process file hash as the parity
oracle. The clean-speech limitation must remain visible in catalog metadata.

Implementation is now technically admissible, but it should be accepted only as a bounded,
self-contained backend task with two managed safetensors files, explicit alignment policy, no
implicit downloads, and gated real-weight tests. Its approximately 110 million parameters and slow
CPU startup/inference make it unsuitable as the default lightweight backend.

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
