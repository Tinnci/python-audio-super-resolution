# Accelerators And Runtime Providers

This document owns the `v0.4.0` accelerator and optimized-runtime policy. Accelerator support must
not make the baseline install heavy, non-deterministic, or network-dependent.

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

Benchmark reports include backend, requested device, runtime provider, elapsed seconds, total input
and output duration, standard RTF (`elapsed_seconds / audio_duration_seconds`), the legacy
`realtime_factor` throughput field, peak RSS/fallback metadata, per-job metadata, and quality
metrics. Throughput should be tracked as evidence, not a hard release gate, until stable baselines
exist per device/provider.

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
