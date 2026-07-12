# Roadmap

The project has a stable lightweight baseline, an experimental self-contained LavaSR backend, and
a broad evaluation harness. The next phase should convert that implementation breadth into release
evidence and a smaller number of well-supported paths.

## Current Position

| Area | State |
| --- | --- |
| Baseline | `sinc-resample` is stable, deterministic, CPU-only, and offline. |
| External model | `audiosr` remains optional and upstream-owned. |
| Self-contained model | `lavasr-compat` works with verified local weights but remains experimental. |
| Evaluation | Full-reference, no-reference, downstream, listening, matrix, report, and bundle workflows are implemented. |
| Packaging | Source version is `0.6.0`; build, metadata, wheel, lint, type, and default test gates are available through Pixi. |

## Completed: Ship v0.6.0 Cleanly

The immediate goal is a release-quality package, not another model backend.

Release-preparation status:

- [x] Merge the relative matrix run-manifest resolution fix and its regression test.
- [x] Run the complete local release checklist in [docs/RELEASE.md](docs/RELEASE.md).
- [x] Confirm CI on the merge commit and inspect the built wheel/sdist.
- [x] Run a fresh installed-wheel CLI smoke test outside the source tree.
- [x] Tag `v0.6.0`, publish the GitHub release, and verify the trusted-publishing workflow and PyPI
  installation.

Release scope deliberately excludes new heavyweight evaluators, new model weights, and a new
self-contained backend.

## Priority 1: Stabilize LavaSR With Broader Evidence

`lavasr-compat` should not be marked stable from a single parity path. The next validation slice is:

- several short speech fixtures covering silence, low volume, near clipping, stereo/channel
  behavior, different input sample rates, and Chinese/English speech;
- real-weight CPU and CUDA evidence using the same manifest and benchmark schema;
- stricter mel/STFT parity tests before changing `lavasr_torch` internals;
- golden or upstream parity thresholds that distinguish numerical drift from audible regressions;
- documented failure rates, RTF, peak memory, and output quality rather than a single score.

Promotion criteria:

- no implicit downloads during inference;
- verified, license-usable local weights;
- reproducible fixtures and commands;
- passing stability and parity evidence on the supported runtime path;
- known limitations visible in model listings.

## Completed: Produce Real Evaluation Evidence

The harness is implemented; the missing value is a repeatable evidence set.

Next deliverables:

- [x] Define a small licensed speech BWE evaluation set outside the repository.
- [x] Commit a threshold-policy example for release regression use.
- [x] Record baseline matrices for `sinc-resample` and gated `lavasr-compat` runs.
- [x] Add precomputed ASR transcript evidence before integrating any real ASR runtime.
- [x] Export one blind listening bundle and document how results map back to objective and stability
   tables.

First real-speech finding: the eight-speaker LibriSpeech `dev-clean` T4 run had no stability
failures, but `sinc-resample` substantially outperformed `lavasr-compat` on full-reference
wideband-16k fidelity metrics. This prevents stable promotion from objective metrics alone and makes
downstream ASR plus blind listening evidence the next required decision inputs.

Pinned Whisper `tiny.en` downstream evidence on the same eight utterances reported identical mean
WER (`0.1594`) and CER (`0.0757`) for degraded input, sinc output, and LavaSR output. LavaSR neither
improved nor degraded this small ASR slice, so blind listening remains the final P2 evidence gap.

The deterministic MUSHRA export contains eight trials and 32 stimuli: reference, degraded anchor,
sinc output, and LavaSR output for each utterance. Public trial entries expose only blind IDs and
generic paths; role/backend/source mapping remains in the separately hashed answer key. Collecting
human ratings is a follow-up study, not a prerequisite for completing the export workflow.

Do not collapse quality, downstream usefulness, speed, stability, and governance into one ranking.
The evaluation policy lives in [docs/EVALUATION.md](docs/EVALUATION.md).

## Completed: One Candidate Spike, Not Three Implementations

ClearerVoice `MossFormer2_SR_48K` is the first candidate to revisit because it has a clear 48 kHz
speech SR contract and identifiable checkpoint files. The spike must stop before backend
implementation unless all of these are proven:

- minimal required checkpoint files and stable hashes;
- safe loading or a practical conversion format;
- upstream CPU inference on a sub-second fixture without hidden downloads;
- exact input/output sample-rate and channel behavior;
- an upstream parity fixture suitable for a gated test.

FlowHigh remains deferred until CPU/provider-neutral execution and checkpoint licensing are clear.
Resemble Enhance remains a possible external speech-enhancement backend, not an immediate
package-owned 48 kHz SR path. Detailed evidence lives in
[docs/BACKEND_CANDIDATES.md](docs/BACKEND_CANDIDATES.md).

The Colab CPU spike passed with only the pointer plus `m` and `g` checkpoints. Both state dicts can
be loaded with `weights_only=True` and converted exactly to safetensors. Offline 16 kHz mono/stereo
and native 48 kHz inputs ran successfully; stereo channels were preserved. The sub-second fixture
exposed a consistent 224-sample output shortening and cross-process floating-point drift, so any
future backend needs explicit alignment plus tolerance-based parity rather than hash equality.

## Priority 4: Optimize Only After Measurement

Keep `lavasr-compat` on `torch-eager` until another provider has real-weight benchmarks and parity
evidence. Evaluate `torch.compile` first because it changes the packaging surface less than ONNX,
TensorRT, or OpenVINO. An optimized provider must retain deterministic CPU fallback and the same
verified weight-store boundary.

## Deferred

- package-owned FlowHigh or Resemble Enhance implementations;
- ONNX Runtime, TensorRT, OpenVINO, DirectML, ROCm, or XPU extras without an accepted backend;
- heavyweight no-reference or downstream models in default dependencies;
- new research candidates without official code, usable licenses, stable weights, and reproducible
  inference.

## Completed Milestones

| Milestone | Outcome |
| --- | --- |
| v0.1.x | Public lightweight CLI/API baseline and Python 3.10-compatible release. |
| v0.2.0 | Self-contained LavaSR runtime, managed weights, gated real-weight and parity validation. |
| v0.3.0 | Model metadata, admission scorecard, and candidate feasibility decisions. |
| v0.4.0 | Device/runtime-provider model and gated accelerator evidence workflow. |
| v0.5.0 | Multi-dimensional evaluation and regression harness. |

Detailed released and unreleased changes belong in [CHANGELOG.md](CHANGELOG.md), not this roadmap.

## Constraints

- Baseline installation must not require GPU libraries, model weights, or network access.
- Normal inference must remain offline unless weight download is explicitly requested.
- Backend inference must use verified local weights and must not call provider APIs directly.
- Default tests must remain CPU-friendly, offline, and free of large binary fixtures.
