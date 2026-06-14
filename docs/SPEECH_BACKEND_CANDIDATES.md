# Speech Backend Candidates

This document records the `v0.3.0` speech super-resolution and bandwidth-extension candidate review. It uses the admission rules in [MODEL_ADMISSION.md](MODEL_ADMISSION.md).

## Decision

The next speech SR candidate should remain a ClearerVoice/MossFormer2 feasibility spike, not
immediate backend implementation.

Resemble Enhance should be tracked as a separate speech enhancement/BWE feasibility spike because
it has strong upstream maintenance and a permissive code license, but its enhancer checkpoint size,
LCFM pipeline, and dependency surface need review before any backend is accepted.

Recommended next action:

- Map `MossFormer2_SR_48K` config, checkpoint files, tensor keys, preprocessing, and output alignment.
- Decide whether it can become `clearvoice-mossformer2-sr-compat` with managed weights.
- Map Resemble Enhance enhancer-only inference and decide whether it belongs as an external backend,
  a self-contained compatibility backend, or a deferred recommendation.
- Keep LavaSR as the current implemented fast speech BWE baseline.

Do not start AP-BWE or NU-Wave2 implementation until a maintained source repository, checkpoint license, and reproducible inference path are identified.

## Candidate Summary

| Candidate | Track | Decision |
| --- | --- | --- |
| `lavasr-v2-bwe` | Implemented self-contained speech BWE | Keep as current experimental baseline; broaden validation later. |
| ClearerVoice `MossFormer2_SR_48K` | Speech SR candidate | Best next feasibility target. Apache-2.0 code and weights, explicit 48 kHz SR task, public config, public HF weights. |
| Resemble Enhance | Speech enhancement/BWE candidate | Track in `#33`; verify enhancer-only files, dependency footprint, and 44.1 kHz speech target before deciding wrapper vs compat backend. |
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

Managed weight feasibility:

| File | Size | SHA256 |
| --- | ---: | --- |
| `do_03925000` | 1744458369 | `549035d29a03928a854815b9c1b21c02d825845aa2e5f093af45694a236c1f19` |
| `last_best_checkpoint_g.pt` | 220712702 | `0bdd13c21466f5963d9d1f86a9d84fc6196868318fe22c6b0a750f041805adda` |
| `last_best_checkpoint_m.pt` | 218471889 | `6cbadb2b6b839e444bb65223c69eea162c8ad08f36e9d0a64144672c4095ab36` |

Admission status:

| Criterion | Status | Notes |
| --- | --- | --- |
| Metadata | Pass | Task, model name, repo, license, and target sample rate are clear. |
| Fixed target sample rate | Pass | 48 kHz. |
| Implementation family | Needs spike | Architecture exists in upstream repo, but self-contained module boundaries and key mapping are not yet reproduced. |
| I/O capability | Likely pass | Upstream has file and NumPy-style interfaces, but exact array shape and sample-rate behavior need local confirmation. |
| CPU fallback | Needs spike | Upstream config has `use_cuda: 1`; CPU execution should be tested before admission. |
| Weight metadata | Partial pass | HF file sizes and hashes are available; manifest should include only files needed for SR inference. |
| Validation path | Needs spike | Need a gated upstream parity test like LavaSR before implementation is accepted. |

Risks:

- Weight size is much larger than LavaSR, so downloads must stay explicit and gated.
- The upstream task may need multiple checkpoint files; unused files must not be downloaded.
- Generator/vocoder-style components increase parity risk compared with LavaSR.
- Clean-speech limitation should be visible in catalog metadata if added.

Recommended follow-up:

1. Download only into a temp cache and inspect checkpoint structure without committing weights.
2. Identify required tensor keys and file roles.
3. Verify CPU inference through upstream ClearVoice on a short fixture.
4. Define `ModelSpec` and managed weight manifest only after required files are known.
5. Build a golden parity test before wiring a self-contained backend.

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

Recommended follow-up:

1. Track feasibility in `#33`.
2. Inspect whether enhancer inference can run without installing training-only dependencies.
3. Confirm exact local file layout and hashable managed-weight manifest.
4. Compare external-wrapper and self-contained compatibility costs before adding a backend.
5. If accepted, expose it as speech enhancement/BWE with explicit 44.1 kHz metadata.

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
