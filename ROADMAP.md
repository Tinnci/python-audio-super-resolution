# Roadmap

The package has a stable lightweight baseline and is adding model-backed inference behind the same CLI/API surface. The default path must remain offline, deterministic, and small.

## Current Baseline

- `sinc-resample` is the default backend and works without model weights or network access.
- `audiosr` is available as an optional external-package backend; its checkpoint behavior is upstream-controlled.
- `lavasr-compat` is available as an experimental self-contained LavaSR v2 BWE backend with managed weights.
- Managed weight infrastructure is implemented: multi-file manifests, path safety, size/SHA256 verification, explicit Hugging Face downloads, and verified local cache resolution.
- Regression helpers are implemented: run manifests, manifest comparison, quality reports, preprocessing, chunking, and sample JSON artifacts.
- Release automation uses GitHub Actions with PyPI Trusted Publishing / OIDC.

## Completed: v0.1.x

`v0.1.0` shipped the first public alpha baseline. Its PyPI files were yanked after `v0.1.1` replaced it with Python 3.10-compatible metadata/code.

`v0.1.1` is the current published baseline:

- PyPI install and CLI smoke test were validated on Python 3.10.
- The `v0.1.0` milestone is closed.
- First-release dry-run notes remain in [docs/RELEASE_DRY_RUN_0.1.0.md](docs/RELEASE_DRY_RUN_0.1.0.md) as historical release evidence.

## Completed: v0.2.0 Validation Gates

Goal: make the first self-contained compatible model backend useful enough to validate with real weights.

Tracked work:

- `#16` Implement `lavasr-compat` self-contained inference. Completed: self-contained torch inference runs with real LavaSR v2 BWE weights.
- `#17` Add golden-sample validation for compatible backends. Completed: fixture format, metrics, docs, and offline tests are available.
- `#18` Add gated real-weight model validation. Completed: gated LavaSR download verification and torch smoke tests are available.
- `#19` Publish validated Colab and GPU documentation. Completed: repository-based Colab/GPU validation guide passed on a fresh Colab T4 runtime.
- `#25` Add LavaSR upstream golden parity validation. Completed: gated upstream LavaSR/Vocos parity harness passed locally with the same verified weights.
- The `v0.2.0` milestone is closed.

Current `lavasr-compat` status:

- Model spec, managed weight metadata, local bundle validation, and experimental torch runtime are implemented.
- Gated real-weight download, torch smoke, upstream LavaSR/Vocos parity, and Colab T4 CLI inference have passed.
- Golden/parity details live in [docs/GOLDEN.md](docs/GOLDEN.md); gated test commands live in [tests/README.md](tests/README.md).

Remaining before releasing `v0.2.0`:

- Cut the release when changelog, version metadata, and release artifacts are ready.

Remaining before marking `lavasr-compat` stable:

- Broaden fixture coverage beyond the initial parity sample.
- Add stricter mel/STFT exactness tests if future changes touch `lavasr_torch`.
- Keep default installs and default tests CPU/offline.

## Completed: v0.3.0 Planning

Goal: plan model expansion after the first compatible inference path has real validation evidence.

Tracked work:

- `#21` Improve model catalog metadata for backend comparison. Completed: model listings expose task/domain, I/O, accelerator declarations, weight metadata, validation evidence, recommended use, and limitations.
- `#22` Define model admission criteria and candidate scorecard. Completed: documentation and scorecard API are available.
- `#23` Evaluate next speech SR/BWE compatible backend candidates. Completed: ClearerVoice `MossFormer2_SR_48K` is the next feasibility target.
- `#24` Evaluate general-audio SR candidate backends. Completed: keep AudioSR external, track FlowHigh as a feasibility candidate, and defer new general-audio self-contained work until reproducibility is proven.
- `#31` Map ClearerVoice `MossFormer2_SR_48K` compatibility feasibility. Completed: defer self-contained implementation until checkpoint conversion, upstream parity, and CPU smoke evidence exist.
- `#32` Map FlowHigh compatibility feasibility. Completed: defer implementation until CPU/offline execution, checkpoint licensing, and provider-neutral runtime behavior are proven.
- `#33` Map Resemble Enhance compatibility feasibility. Completed: defer as a future optional speech enhancement/BWE backend rather than immediate package-owned SR.

Decision rule:

- `v0.2.0` is for inference framework hardening and validation.
- `v0.3.0` is for choosing what to add next and how to compare candidates.
- `v0.4.0` is for hardware acceleration and runtime-provider work after the model/backend abstractions are clear.
- New candidate backends should not move into implementation until their weight format, license, preprocessing, I/O shape, and validation path are clear.

## Completed: v0.4.0 Runtime Provider Planning

Goal: optimize execution across hardware and external runtimes without making the baseline install heavy.

Tracked work:

- `#26` Define accelerator capability model and fallback policy. Completed: logical devices now cover CPU/CUDA/ROCm/XPU/MPS/DirectML, model/backend listings expose declared accelerator support, `device=auto` is documented, and unsupported explicit devices fail before inference.
- `#27` Add runtime provider abstraction for optimized execution. Completed: import-light runtime provider metadata and resolution are available for `python`, `torch-eager`, `onnxruntime`, and `external-package`, with mocked availability tests.
- `#28` Add gated accelerator validation and benchmark matrix. Completed: default tests remain CPU/offline, hardware validation is opt-in, and enhancement runs can write machine-readable benchmark JSON with timing and quality metrics.
- `#29` Document accelerator install strategy and optional extras. Completed: accelerator install paths and the conservative extras policy live in [docs/ACCELERATORS.md](docs/ACCELERATORS.md).
- `#30` Evaluate LavaSR optimized runtime and export paths. Completed: keep `lavasr-compat` on PyTorch eager until `torch.compile`, ONNX Runtime, TensorRT, or OpenVINO have real-weight benchmark and parity evidence.

Decision rule:

- Accelerator support is a runtime layer, not a model-selection layer.
- Backend code should request capabilities from a runtime/provider abstraction instead of hard-coding CUDA, ROCm, XPU, DirectML, OpenVINO, TensorRT, or ONNX Runtime checks.
- GPU/SDK-specific tests must remain gated and should produce JSON evidence before they become release gates.

## In Progress: v0.5.0 Evaluation And Regression Harness

Goal: build a reproducible backend benchmark and regression workflow that evaluates more than one
audio-quality score.

Tracked work:

- `#38` Add full-reference objective evaluation metrics. Completed: controlled degraders,
  SI-SDR/SDR, LSD, high-band LSD, spectral convergence, optional metric adapter requirements, and
  JSON eval manifests.
- `#39` Add no-reference objective evaluation adapters. Completed: `audio-super-res eval
  no-reference`, builtin CPU/offline `signal-stats` records, stable no-reference JSON shape, and
  gated DNSMOS/NISQA/UTMOS/ViSQOL integration requirements.
- `#40` Add downstream task evaluation workflows. Completed: `audio-super-res eval downstream`,
  builtin transcript WER/CER delta evaluation from precomputed ASR outputs, and gated
  speaker/VAD/KWS adapter schema.
- `#41` Add perceptual listening-test export workflow. Completed: `audio-super-res eval
  listening-export` writes AB/ABX/MUSHRA-ready blind stimuli, public manifest, external answer key,
  and rating dimensions without requiring a browser/survey runtime.
- `#42` Add engineering performance and stability evaluation. Completed: eval manifests include
  elapsed time, RTF, peak RSS strategy/fallbacks, sample-rate checks, duration drift, clipping,
  per-result failure status, lightweight failure-case classification, and backend
  capability/governance facts.
- `#43` Add eval regression manifests and comparison workflow. Completed: `audio-super-res eval run`,
  `audio-super-res eval compare`, threshold-based regression checks, and comparison tables for audio
  quality, downstream, engineering, stability, and governance.

Decision rule:

- Do not reduce backend choice to one aggregate score.
- Keep heavyweight evaluators, ASR models, listening-test tooling, and model weights optional or gated.
- Prefer separate tables for full-reference quality, no-reference quality, downstream impact,
  listening evidence, engineering performance, stability, and governance.

## Later: Release Hardening

- Keep broadening `lavasr-compat` fixture coverage before marking it stable.
- Keep candidate backends deferred until their weight, license, preprocessing, and validation blockers are cleared.
- Use benchmark JSON reports as evidence for future accelerator/provider claims.

## Backend Planning Snapshot

This table is a routing guide only. Detailed model evidence and risks live in the candidate review
documents linked from [docs/README.md](docs/README.md).

| Backend | Role |
| --- | --- |
| `sinc-resample` | Implemented deterministic baseline. |
| `audiosr` | Implemented optional external AudioSR wrapper. |
| `lavasr-compat` | Experimental self-contained speech BWE backend; real-weight download, torch smoke, and initial upstream parity pass. |
| `mossformer-sr-compat` | Deferred feasibility candidate; requires checkpoint conversion/parity and CPU smoke evidence before implementation. |
| `flowhigh-compat` | Deferred feasibility candidate; current upstream path is CUDA-first and checkpoint licensing/provider guarantees need resolution. |
| `resemble-enhance` | Deferred speech enhancement/BWE candidate; better suited to a future optional external backend. |
| `nuwave` | Deferred research candidate. |
| custom backend | User-provided backend implementing the package protocol. |

## Constraints

- Baseline installation must not require GPU libraries, model weights, or network access.
- Normal inference must remain offline unless the user explicitly opts into downloading weights.
- Backend inference code must use verified local weight paths and must not call provider APIs directly.
- Default tests must not require GPU access or large model downloads.
