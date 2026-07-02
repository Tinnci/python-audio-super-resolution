# Speech Backend Candidates

This document records the `v0.3.0` speech super-resolution and bandwidth-extension candidate review. It uses the admission rules in [MODEL_ADMISSION.md](MODEL_ADMISSION.md).

## Decision

The `v0.3.0` speech feasibility pass is complete. No new speech backend should move directly
into implementation yet.

ClearerVoice `MossFormer2_SR_48K` remains the strongest future speech SR target, but it should be
deferred until a conversion/parity plan exists for its PyTorch checkpoints and until a short
upstream CPU smoke can be recorded without implicit provider downloads.

Resemble Enhance should remain a deferred speech enhancement/BWE candidate. It has strong upstream
maintenance and a permissive code license, but its 44.1 kHz target, DeepSpeed-style checkpoint
layout, and dependency surface make it a poor fit for immediate package-owned inference.

Recommended next action:

- Keep LavaSR as the current implemented fast speech BWE baseline.
- Revisit `MossFormer2_SR_48K` only when a checkpoint conversion plan and upstream parity fixture
  are ready.
- Revisit Resemble Enhance as an optional external speech enhancement backend only after
  inference-only dependencies are proven.

Do not start AP-BWE or NU-Wave2 implementation until a maintained source repository, checkpoint license, and reproducible inference path are identified.

## Candidate Summary

| Candidate | Track | Decision |
| --- | --- | --- |
| `lavasr-v2-bwe` | Implemented self-contained speech BWE | Keep as current experimental baseline; broaden validation later. |
| ClearerVoice `MossFormer2_SR_48K` | Speech SR candidate | Deferred. Strong metadata and managed-weight shape, but self-contained admission needs checkpoint conversion/parity work and a recorded CPU upstream smoke. |
| Resemble Enhance | Speech enhancement/BWE candidate | Deferred. Better suited to a future optional external speech-enhancement backend than immediate self-contained SR. |
| AudioSR `speech` | External baseline | Keep external-package backend only; not a self-contained candidate. |
| AP-BWE | Research candidate | Defer until maintained code, weights, and license are verified. |
| NU-Wave2 | Research diffusion candidate | Defer; diffusion path is heavier and less suitable for the next self-contained backend. |

## ClearerVoice MossFormer2_SR_48K

Source facts:

- Repository: <https://github.com/modelscope/ClearerVoice-Studio>
- License: Apache-2.0
- Model repo: <https://huggingface.co/alibabasglab/MossFormer2_SR_48K>
- Task: speech super-resolution to 48 kHz
- Upstream usage: `ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])`
- Public config path: `clearvoice/clearvoice/config/inference/MossFormer2_SR_48K.yaml`
- Public model code path: `clearvoice/clearvoice/models/mossformer2_sr`
- Upstream states that the current SR model is trained for clean speech; noisy speech should use enhancement first.

Feasibility conclusion:

- Decision: defer self-contained implementation.
- Best future track: `clearvoice-mossformer2-sr-compat`, but only after a conversion/parity spike.
- External wrapper is possible, but upstream currently downloads from Hugging Face inside model
  loading when files are missing, which conflicts with this package's explicit provider-download
  boundary.
- Upstream source inspected at `modelscope/ClearerVoice-Studio@6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`.
- Hugging Face model revision inspected: `alibabasglab/MossFormer2_SR_48K@39eb1f25ea84f5e0315ade9ac0070fff216fc690`.
- The `last_best_checkpoint` pointer contains `last_best_checkpoint_m.pt` and
  `last_best_checkpoint_g.pt`; those are the files used by the ClearVoice SR loader. The larger
  `do_03925000` file is present in the Hub repository but was not identified as required by the
  SR inference loader.

Managed weight feasibility:

| File | Size | SHA256 |
| --- | ---: | --- |
| `do_03925000` | 1744458369 | `549035d29a03928a854815b9c1b21c02d825845aa2e5f093af45694a236c1f19` |
| `last_best_checkpoint` | 52 | n/a |
| `last_best_checkpoint_g.pt` | 220712702 | `0bdd13c21466f5963d9d1f86a9d84fc6196868318fe22c6b0a750f041805adda` |
| `last_best_checkpoint_m.pt` | 218471889 | `6cbadb2b6b839e444bb65223c69eea162c8ad08f36e9d0a64144672c4095ab36` |

Admission status:

| Criterion | Status | Notes |
| --- | --- | --- |
| Metadata | Pass | Task, model name, repo, license, and target sample rate are clear. |
| Fixed target sample rate | Pass | 48 kHz. |
| Implementation family | Needs spike | Architecture exists in upstream repo, but self-contained module boundaries and key mapping are not yet reproduced. |
| I/O capability | Likely pass | Upstream has file and NumPy-style interfaces, but exact array shape and sample-rate behavior need local confirmation. |
| CPU fallback | Partial | Upstream `SpeechModel` falls back to CPU when CUDA is unavailable, but a real short CPU smoke was not recorded. |
| Weight metadata | Partial pass | HF file sizes and hashes are available; a package manifest should include only files needed for SR inference. |
| Validation path | Needs spike | Need a gated upstream parity test like LavaSR before implementation is accepted. |

Risks:

- Weight size is much larger than LavaSR, so downloads must stay explicit and gated.
- The upstream task needs at least two checkpoint files and a pointer file; unused files must not be downloaded.
- Generator/vocoder-style components increase parity risk compared with LavaSR.
- Clean-speech limitation should be visible in catalog metadata if added.
- Upstream checkpoints are PyTorch `.pt` files loaded with `torch.load`; self-contained admission
  needs a safe conversion plan before implementation.
- The combined wrapper forward path has a typo, but the public inference path uses `model_m` and
  `model_g` separately. A compatibility backend should mirror the public inference path, not the
  unused combined forward method.

Recommended follow-up:

1. Download only into a temp cache and inspect checkpoint tensor keys without committing weights.
2. Verify CPU inference through upstream ClearVoice on a short fixture.
3. Convert required checkpoints to a safer package-owned format if implementation proceeds.
4. Define `ModelSpec` and managed weight manifest only after required files are finalized.
5. Build a golden parity test before wiring a self-contained backend.

`v0.6.0` spike acceptance:

- Produce a JSON checkpoint-key summary for `last_best_checkpoint_m.pt` and
  `last_best_checkpoint_g.pt` from a temporary cache.
- Record whether upstream CPU inference can run on a sub-second fixture without implicit provider
  downloads.
- Decide whether a safe converted format is practical before any package-owned backend code starts.
- Open implementation only after parity fixture shape, expected sample-rate behavior, and minimal
  required files are known.

## Resemble Enhance

Source facts:

- Repository: <https://github.com/resemble-ai/resemble-enhance>
- License: MIT
- Package: `resemble-enhance`
- Domain: speech denoising and enhancement/bandwidth extension.
- Upstream exposes denoiser and enhancer modules.
- Upstream describes the enhancer as restoring speech distortions and extending bandwidth.
- Training target is high-quality 44.1 kHz speech.
- The enhancer path uses latent conditional flow matching.
- Upstream download metadata references the Hugging Face repository
  `ResembleAI/resemble-enhance`.

Feasibility conclusion:

- Decision: defer implementation.
- Best future track: optional external speech enhancement/BWE backend, not a package-owned
  self-contained SR backend.
- Upstream source inspected at `resemble-ai/resemble-enhance@8e978149bfe8abab3eb77d965d579a111afdb0ff`.
- Hugging Face model revision inspected: `ResembleAI/resemble-enhance@4e3510ce4a8391159f665903544c5150bee7b2cb`.
- Enhancer inference can run on CPU when selected, but the published dependency set includes
  DeepSpeed and training/runtime packages that should not enter the baseline package.
- The model target is 44.1 kHz speech enhancement, so it should not be cataloged as a 48 kHz
  general SR model.

Known enhancer files from upstream metadata:

| File | Size | Notes |
| --- | ---: | --- |
| `enhancer_stage2/hparams.yaml` | 717 | Config metadata. |
| `enhancer_stage2/ds/G/latest` | 7 | DeepSpeed checkpoint pointer. |
| `enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt` | 713176232 | Large checkpoint; SHA256 observed as `f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6`. |

Admission status:

| Criterion | Status | Notes |
| --- | --- | --- |
| Metadata | Pass | Repository, package, domain, and code license are clear. |
| Fixed target sample rate | Partial | Speech target is 44.1 kHz, not the package's current 48 kHz BWE convention. |
| Implementation family | Needs spike | LCFM pipeline and model/vocoder boundaries need mapping. |
| Dependency footprint | Needs spike | Upstream requirements include heavyweight training/runtime packages; inference-only imports must be isolated. |
| Weight metadata | Partial pass | HF source and checkpoint hash are available, but the minimal enhancer-only manifest must be confirmed. |
| Validation path | Needs spike | Need a short fixture and deterministic or statistically stable comparison against upstream. |

Risks:

- This is speech-only and should not be presented as a general-audio SR backend.
- The large checkpoint must stay behind explicit downloads and gated tests.
- DeepSpeed-style checkpoint layout may require conversion before a clean self-contained backend is practical.
- A 44.1 kHz target means catalog metadata and CLI validation must avoid implying 48 kHz output.
- The upstream `hparams.yaml` uses Python-specific YAML tags for paths, so a package-owned loader
  should parse or normalize config without unsafe object construction.

Recommended follow-up:

1. Isolate an inference-only dependency set before any wrapper is added.
2. Confirm exact local file layout and hashable managed-weight manifest.
3. Compare external-wrapper and self-contained compatibility costs before adding a backend.
4. If accepted, expose it as speech enhancement/BWE with explicit 44.1 kHz metadata.

## Current Baselines

`lavasr-v2-bwe` remains the current self-contained speech BWE baseline because it has:

- managed weights
- self-contained torch runtime
- real download validation
- torch smoke validation
- upstream parity validation
- much smaller weight footprint than MossFormer2_SR_48K

AudioSR remains useful for comparison, but it is an external-package backend. Its upstream package owns checkpoint downloads and inference behavior, so it does not satisfy self-contained admission.

## Deferred Research Candidates

AP-BWE and NU-Wave2 are not rejected on quality grounds. They are deferred because this review did not identify enough implementation-grade metadata:

- maintained source repository
- checkpoint URL and license
- exact preprocessing and target sample-rate contract
- reproducible inference command
- CPU/offline validation path

They can be reopened as candidates when those facts are available.
