# Accelerators And Runtime Providers

This document owns accelerator selection, hardware validation, and optimized-runtime policy.
Accelerator support must not make the baseline install heavy, non-deterministic, or
network-dependent.

## Current Decision

- Keep `sinc-resample` on the `python` runtime provider and CPU only.
- Keep `audiosr` behind the external package runtime; the upstream package owns its accelerator
  behavior and checkpoint handling.
- Keep `lavasr-compat` on the self-contained `torch-eager` provider. Do not replace it with
  `torch.compile`, ONNX Runtime, TensorRT, or OpenVINO until parity, fallback behavior, dependency
  cost, and packaging risk are proven with real weights.
- Do not add new accelerator extras for vendor SDKs yet. Vendor-specific PyTorch, ONNX Runtime,
  DirectML, OpenVINO, and TensorRT installs are documented commands, not baseline dependencies.

## Public Selection API

The CLI and Python API separate logical device selection from runtime provider selection:

```sh
audio-super-res input.wav output.wav \
  --backend lavasr-compat \
  --device auto \
  --runtime-provider auto
```

```python
from audio_super_resolution import AudioSuperResolver, InferenceConfig

config = InferenceConfig(device="auto", runtime_provider="auto")
resolver = AudioSuperResolver(target_sr=48000, backend="lavasr-compat", config=config)
```

Logical devices are `cpu`, `cuda`, `rocm`, `xpu`, `mps`, `directml`, and `auto`.

Runtime providers are:

| Provider | Purpose | Baseline dependency? |
| --- | --- | --- |
| `python` | NumPy/SciPy CPU execution for lightweight package-owned code. | Yes |
| `torch-eager` | PyTorch eager execution for package-owned torch modules such as `lavasr-compat`. | No |
| `onnxruntime` | Future exported graph runtime family. | No |
| `external-package` | Runtime owned by a selected external backend such as AudioSR. | No |

`device=auto` chooses the first available backend-supported device in this order:
`cuda`, `rocm`, `mps`, `xpu`, `directml`, `cpu`.

If a user requests an unsupported device or runtime provider, the selected backend fails before
inference with an actionable error. If an optional runtime is missing, the error points to the
relevant install path.

## Listing Metadata

Use JSON listing for automation:

```sh
audio-super-res --list-backends --list-format json
audio-super-res --list-models --list-format json
```

Backends and models report declared `accelerators` and `runtime_providers` without importing heavy
runtimes. Listing should remain safe in CPU-only environments.

## Installation Strategy

The package extra policy is intentionally conservative:

| Need | Command or note |
| --- | --- |
| Baseline CPU | `uv pip install audio-super-resolution` |
| LavaSR torch runtime | `uv pip install "audio-super-resolution[lavasr]"` |
| LavaSR managed downloads | `uv pip install "audio-super-resolution[lavasr,download]"` |
| PyTorch CUDA | Install the matching PyTorch wheel from the official PyTorch index for the target CUDA version, then install this package. |
| PyTorch ROCm | Install the matching ROCm PyTorch wheel on supported Linux/AMD systems, then install this package. |
| Intel XPU/IPEX | Use Intel's current PyTorch/IPEX instructions for the target platform; keep it outside default package metadata until validation exists. |
| DirectML | Use `torch-directml` or ONNX Runtime DirectML in a Windows-specific environment; no default extra is declared. |
| ONNX Runtime CPU/CUDA/DirectML/OpenVINO | Install the provider-specific ONNX Runtime package only when an exported graph backend exists. |
| TensorRT/OpenVINO SDK | Treat as external SDK installs and validate on dedicated machines or self-hosted runners. |

No new `pyproject.toml` extras are added for ONNX Runtime, DirectML, OpenVINO, TensorRT, ROCm, or
XPU in this milestone because there is no accepted exported graph backend yet and the vendor wheel
matrix is platform-specific.

## Gated Validation Matrix

Default tests stay CPU/offline:

```sh
pixi run test
```

Hardware validation is opt-in. Use environment variables to make the intent explicit:

```sh
set AUDIO_SUPER_RESOLUTION_RUN_ACCELERATOR_MATRIX=1
set AUDIO_SUPER_RESOLUTION_ACCELERATOR_DEVICE=cuda
pixi run pytest tests/test_accelerator_matrix.py
```

On Unix-like shells:

```sh
export AUDIO_SUPER_RESOLUTION_RUN_ACCELERATOR_MATRIX=1
export AUDIO_SUPER_RESOLUTION_ACCELERATOR_DEVICE=cuda
pixi run pytest tests/test_accelerator_matrix.py
```

For real LavaSR smoke or parity validation, also set the LavaSR-specific environment variables
documented in [../tests/README.md](../tests/README.md).

For runtime benchmarking, write a JSON report during a normal enhancement run:

```sh
audio-super-res input.wav output.wav \
  --backend lavasr-compat \
  --target-sr 48000 \
  --device auto \
  --runtime-provider auto \
  --benchmark-json benchmark.json \
  --quality-report-json quality.json
```

Benchmark reports include backend, requested device, runtime provider, backend load/init time,
enhancement elapsed seconds, total elapsed seconds, total input and output duration, standard RTF
(`elapsed_seconds / audio_duration_seconds`), the legacy `realtime_factor` throughput field, peak
RSS/fallback metadata, per-job metadata, and quality metrics. Throughput should be tracked as
evidence, not a hard release gate, until stable baselines exist per device/provider.

## LavaSR Optimization Recommendation

Current recommendation: keep `lavasr-compat` on PyTorch eager.

Evidence and risk:

- The PyTorch eager path has real-weight download verification, torch smoke coverage, upstream
  parity coverage, and a fresh Colab T4 validation record.
- `torch.compile` may help some shapes, but it needs per-device measurement and output drift checks
  before becoming a selectable provider.
- ONNX export is not yet accepted because the current LavaSR graph includes STFT/ISTFT, mel
  projection, complex tensors, and `FastLRMerge` behavior that need parity fixtures before export.
- TensorRT and OpenVINO should be separate graph-provider experiments after ONNX parity exists.
- Any optimized path must use verified local weights and the same weight-store boundaries as
  PyTorch eager.

Future acceptance for replacing or adding an optimized LavaSR provider requires:

1. A real-weight benchmark JSON report for each target device/provider.
2. A golden or upstream parity report showing output drift is acceptable.
3. A documented dependency and packaging cost.
4. A CPU fallback path that remains deterministic and offline.

## Fresh GPU Validation Workflow

Use a repository checkout when validating unreleased code or gated tests:

```sh
git clone https://github.com/Tinnci/python-audio-super-resolution.git
cd python-audio-super-resolution
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv pip install --system -e ".[lavasr,download]"
```

Inspect the runtime before downloading weights:

```sh
audio-super-res --env-info
audio-super-res --list-backends --list-format json
audio-super-res --list-models --list-format json
```

Prepare weights explicitly, then run a short enhancement with machine-readable evidence:

```sh
audio-super-res --backend lavasr-compat --download-weights --prepare-model-cache
audio-super-res --backend lavasr-compat --verify-weights

audio-super-res input.wav output.wav \
  --backend lavasr-compat \
  --target-sr 48000 \
  --device auto \
  --runtime-provider auto \
  --manifest run.json \
  --quality-report-json quality.json \
  --benchmark-json benchmark.json
```

Record the device, Python/torch versions, environment output, weight revision, inference command,
sample rate, quality result, timing, peak memory strategy, and any parity test result. Keep generated
artifacts in ignored `runs/` or attach them to the corresponding issue/release.

For the maintained v0.6 LavaSR evidence workflow, use the Colab CLI from the repository root. The
script performs all real-weight and GPU computation in the remote session:

```sh
colab new -s asr-v060 --gpu T4
colab exec -s asr-v060 -f examples/colab_lavasr_validation.py --timeout 3600
colab download -s asr-v060 \
  /content/asr-v0.6.0-colab-evidence.tar.gz \
  runs/asr-v0.6.0-colab-evidence.tar.gz
colab stop -s asr-v060
```

Set `ASR_GIT_REF`, `ASR_LAVASR_DEVICE`, or `ASR_COLAB_ARCHIVE` in the remote execution environment
only when validating another immutable ref, device, or evidence destination. Do not run
`examples/colab_lavasr_validation.py` locally: it intentionally refuses to run outside `/content`.

For same-backend release regression evidence across Git refs, use a separate session or a fresh
runtime after the stability workflow:

```sh
colab new -s asr-regression --gpu T4
colab exec -s asr-regression -f examples/colab_eval_regression.py --timeout 3600
colab download -s asr-regression \
  /content/asr-eval-regression.tar.gz \
  runs/asr-eval-regression.tar.gz
colab stop -s asr-regression
```

The defaults compare `v0.6.0` with `main`, share one verified LavaSR cache, and run identical
`wideband_16k`/`lowpass_4k` matrices over the deterministic eight-item smoke evalset. The archive
also contains a `sinc-resample` matrix, the threshold policy, reports, resolved commits, and the
matrix comparison. Override `ASR_BASELINE_REF` and `ASR_CANDIDATE_REF` only with reviewable refs;
recorded evidence always includes the resolved commits.

For the licensed real-speech baseline, run the pinned LibriSpeech workflow in a fresh T4 session:

```sh
colab new -s asr-librispeech --gpu T4
colab exec -s asr-librispeech -f examples/colab_librispeech_eval.py --timeout 3600
colab download -s asr-librispeech \
  /content/asr-librispeech-evidence.tar.gz \
  runs/asr-librispeech-evidence.tar.gz
colab stop -s asr-librispeech
```

The workflow downloads the official `dev-clean` archive remotely, verifies its published MD5,
retains the source license/README/speaker metadata, selects four female and four male speakers, and
converts one utterance per speaker to 48 kHz. It then runs matching sinc/LavaSR matrices. The source
archive, converted audio, and model computations never run or persist in the local repository.

To add downstream ASR evidence without introducing an ASR package dependency, run the pinned
external evaluator immediately after the LibriSpeech workflow in the same session:

```sh
colab exec -s asr-librispeech -f examples/colab_asr_downstream.py --timeout 3600
colab download -s asr-librispeech \
  /content/asr-downstream-evidence.tar.gz \
  runs/asr-downstream-evidence.tar.gz
```

The external model metadata and revision are pinned in
`examples/artifacts/asr-evaluator-whisper-tiny-en.json`. Whisper runs only in Colab and writes
precomputed transcripts. The package's normal `eval downstream` command remains lightweight and
computes WER/CER from JSON rather than importing or downloading an ASR model.

Export the blind listening bundle from the same real-speech matrices before stopping the session:

```sh
colab exec -s asr-librispeech -f examples/colab_listening_export.py --timeout 900
colab download -s asr-librispeech \
  /content/asr-listening-evidence.tar.gz \
  runs/asr-listening-evidence.tar.gz
```

The bundle uses MUSHRA metadata with seed `0` and four stimuli per trial: reference, degraded
anchor, sinc output, and LavaSR output. Public stimuli contain only blind IDs and generic paths;
backend, role, and source mapping remain in the separate answer key.

Run the bounded MossFormer2 feasibility spike on a fresh CPU session. Checkpoint download,
inspection, conversion, and real inference all remain remote:

```sh
colab new -s asr-mossformer2
colab exec -s asr-mossformer2 -f examples/colab_mossformer2_spike.py --timeout 3600
colab download -s asr-mossformer2 \
  /content/asr-mossformer2-spike.tar.gz \
  runs/asr-mossformer2-spike.tar.gz
colab stop -s asr-mossformer2
```

The script pins both upstream revisions, downloads only the pointer and two inference checkpoints,
forces offline mode before model construction, uses an isolated Python worker after dependency
installation, and records safetensors round-trip plus mono/stereo/alignment evidence. Do not run it
locally.

For upstream LavaSR parity, install upstream dependencies deliberately and use the environment gates
in [../tests/README.md](../tests/README.md). Validate AudioSR separately because its external package
owns checkpoint and accelerator behavior.

## Recorded GPU Evidence

A fresh Colab Tesla T4 run was recorded on 2026-07-01 UTC at commit
`a928d899547792fb2588ab3fe28f8a1da0578b8d`:

- Python 3.12.13 and torch 2.11.0+cu128 detected CUDA successfully;
- the default test suite passed;
- LavaSR v2 BWE weights downloaded and verified explicitly;
- the gated real-weight CUDA smoke passed;
- CLI CUDA inference wrote a 48 kHz output plus passing run and quality records.

This is historical evidence for the eager CUDA path, not a permanent performance baseline. Repeat
the workflow on the release commit before making new device/provider claims.

A v0.6.0 Colab CLI run on 2026-07-12 validated the published tag at commit
`5e4241c5ed7399b29d04bbfe39f04fdfd9f100dc` on the same Tesla T4 environment:

- real-weight download/verification and CUDA torch smoke passed;
- explicit CUDA device detection passed;
- four of five generated failure cases passed;
- exact digital silence exposed deterministic model output around `-75 dBFS` RMS and was correctly
  reported as `silence_hallucination` by the eval harness;
- the follow-up exact-silence preservation fix passed all five cases on the same remote session,
  with zero duration drift for the silence fixture.

The downloaded evidence archive belongs under ignored `runs/` or the associated GitHub issue/PR,
not in the source distribution.
