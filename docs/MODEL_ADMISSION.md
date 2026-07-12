# Model Admission

This document defines the gate for adding model-backed inference paths. It keeps model selection separate from runtime acceleration work.

## Tracks

Use two admission targets:

| Target | Meaning |
| --- | --- |
| `catalog` | The model can be listed or tracked as a candidate, but may still depend on external code or incomplete validation. |
| `self-contained` | The model is suitable for implementation in this package without importing upstream project code at inference time. |

Use this gate before opening backend implementation. CUDA, ROCm, XPU, DirectML, OpenVINO,
TensorRT, and ONNX Runtime provider work remains separate from model admission.

## Required Metadata

Every candidate needs:

- domain and task labels
- architecture family
- implementation family
- target sample-rate metadata
- I/O capability metadata
- license metadata for code and weights
- weight source/provider metadata when managed weights are required
- known limitations
- validation evidence or a concrete validation plan

Self-contained candidates additionally need:

- one fixed target sample rate
- implementation family `self_torch`, `self_onnx`, or another package-owned implementation
- CPU fallback for default validation
- verifiable managed weights, including file sizes and SHA256 hashes
- a gated real-weight or golden/parity validation path

## Hard Rejects

Do not start implementation for a self-contained backend when any of these are true:

- architecture is not declared or cannot be reproduced from public config/code
- target sample rate is ambiguous
- weight license is unknown, research-only, or non-commercial
- weights require unsafe whole-object pickle loading without a conversion plan
- preprocessing, STFT/mel, chunking, or output alignment is undocumented
- no CPU fallback or offline validation path exists
- backend inference would need to call a provider API directly

External-package backends can still be cataloged, but they should be clearly marked `external_package` and should not be treated as self-contained candidates.

## Scorecard API

Use the public admission helper to score a registered model spec:

```python
from audio_super_resolution import evaluate_model_admission, get_model_spec

spec = get_model_spec("lavasr-v2-bwe")
report = evaluate_model_admission(spec, target="self-contained")

print(report.passed)
print(report.score, report.max_score)
print(report.blockers)
```

The report checks:

- core metadata
- fixed target sample rate
- implementation family
- I/O capability
- CPU fallback
- managed weight metadata
- validation evidence

The score is a prioritization aid. Required blockers decide whether implementation should proceed.

## Current Built-ins

| Model | Self-contained admission |
| --- | --- |
| `lavasr-v2-bwe` | Passes current self-contained gate; still experimental until broader validation and release docs mature. |
| `audiosr-basic` / `audiosr-speech` | Fails self-contained gate by design because inference is owned by the external `audiosr` package. |
| `sinc-resample` | Stable baseline, not a model-backed self-contained SR candidate because it has no fixed target sample rate and does not reconstruct missing bandwidth. |

## Candidate Review Flow

1. Add candidate metadata to a scorecard or issue comment before implementation.
2. Run or manually apply the admission criteria.
3. Reject, defer, or accept the candidate for a specific track.
4. Only then add weight manifests, download support, backend code, and gated validation.
5. Keep hardware provider optimization separate unless the model cannot run correctly without a specific provider.
