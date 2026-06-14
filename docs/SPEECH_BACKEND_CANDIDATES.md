# Speech Backend Candidates

This document records the `v0.3.0` speech super-resolution and bandwidth-extension candidate review. It uses the admission rules in [MODEL_ADMISSION.md](MODEL_ADMISSION.md).

## Decision

The next speech candidate should be a ClearerVoice/MossFormer2 feasibility spike, not immediate backend implementation.

Recommended next action:

- Map `MossFormer2_SR_48K` config, checkpoint files, tensor keys, preprocessing, and output alignment.
- Decide whether it can become `clearvoice-mossformer2-sr-compat` with managed weights.
- Keep LavaSR as the current implemented fast speech BWE baseline.

Do not start AP-BWE or NU-Wave2 implementation until a maintained source repository, checkpoint license, and reproducible inference path are identified.

## Candidate Summary

| Candidate | Track | Decision |
| --- | --- | --- |
| `lavasr-v2-bwe` | Implemented self-contained speech BWE | Keep as current experimental baseline; broaden validation later. |
| ClearerVoice `MossFormer2_SR_48K` | Speech SR candidate | Best next feasibility target. Apache-2.0 code and weights, explicit 48 kHz SR task, public config, public HF weights. |
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
