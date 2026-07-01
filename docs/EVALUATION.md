# Evaluation And Regression Harness

This document owns the `v0.5.0` backend evaluation plan. Evaluation is not a single audio-quality
score. It is a reproducible workflow that compares backends across quality, downstream usefulness,
engineering cost, stability, and governance.

## Core Flow

```text
clean_48k_reference.wav
  -> degrade: lowpass / downsample / codec / noise / reverb
  -> low_quality_input.wav
  -> model/backend enhance
  -> enhanced_48k.wav
  -> compare enhanced_48k.wav against clean_48k_reference.wav
```

## First Supported Slice

The first implementation is intentionally lightweight and CPU/offline friendly:

```sh
audio-super-res eval run \
  --dataset evalsets/speech_clean_48k \
  --backend sinc-resample \
  --degrader wideband_16k \
  --work-dir runs/sinc-work \
  --output runs/sinc.json

audio-super-res eval compare runs/sinc.json runs/lavasr.json --output runs/comparison.json
```

`eval run` accepts a directory of clean `.wav` reference files. It writes degraded inputs, backend
outputs, full-reference metrics, quality/stability checks, and simple runtime data into a JSON
manifest.

Implemented degraders:

| Degrader | Behavior |
| --- | --- |
| `lowpass_4k` | Keeps the original sample rate and low-passes around 4 kHz. |
| `narrowband_8k` | Downsamples to 8 kHz. |
| `wideband_16k` | Downsamples to 16 kHz. |
| `noisy_16k` | Downsamples to 16 kHz and adds deterministic light noise. |

Implemented lightweight full-reference metrics:

| Metric | Purpose |
| --- | --- |
| `SI-SDR` | Scale-invariant signal recovery. |
| `SDR` | Direct signal reconstruction error. |
| `LSD` | Log-spectral distance over the whole spectrum. |
| `High-band LSD 4-8 kHz` | Speech bandwidth-extension high-frequency recovery. |
| `High-band LSD 8-16 kHz` | Higher-band reconstruction, especially important for 48 kHz SR. |
| `Spectral convergence` | Overall magnitude-spectrum closeness. |

PESQ, STOI/ESTOI, MCD, DNSMOS, NISQA, UTMOS, ViSQOL, ASR WER/CER, speaker similarity, and listening
test exports are planned as optional or gated adapters. They must not become default CI
dependencies.

## Manifest Shape

Each result contains:

- `status`: `passed`, `stability_failed`, or `failed`
- `failure`: exception stage/type/message when a backend or inspection step fails
- `reference_path`, `degraded_path`, and `enhanced_path`
- degrader recipe metadata
- full-reference metrics
- quality checks: sample rate, duration drift, peak/clipping, pass/fail
- stability checks: sample-rate correctness, duration drift, clipping, and classified failure cases
- performance: backend init time, elapsed seconds, audio duration, RTF, and peak RSS where the
  platform exposes `resource.getrusage(RUSAGE_SELF).ru_maxrss`

`eval compare` compares manifests by item id and reports separate metric deltas. It does not hide
raw metrics behind a single aggregate score. Candidate results with failed status, failed stability
checks, or failed quality checks are reported as regressions.

The manifest also includes a `backend_profile` with capability and governance facts from the model
catalog:

- batch mode is currently represented as file-loop batch processing, not true model batch inference
- stream support is reported separately from chunked offline processing
- CPU/CUDA/MPS support, runtime providers, precision modes, and chunking support
- offline capability and reproducibility based on deterministic execution and explicit weight hashes
- code license, weight license, weight source/provider, weight size, and whether weights are explicit
- dependency footprint based on optional extras, external packages, and managed weight size

Peak RSS is a lightweight engineering signal, not a full profiler. On Darwin `ru_maxrss` is reported
as bytes; on Linux and most Unix platforms it is reported as kilobytes. When the platform does not
provide `resource`, the manifest records a documented fallback with `peak_rss_mb: null`.

## Engineering And Stability Tables

Keep engineering and stability evidence separate from audio-quality metrics:

| Table | Fields |
| --- | --- |
| Performance | `backend_init_seconds`, `elapsed_seconds`, `rtf`, `peak_rss_mb`, `peak_rss_delta_mb` |
| Stability | `status`, `sample_rate_correct`, `duration_drift_seconds`, `clipped_fraction`, `failure_cases` |
| Governance | `offline`, `reproducible`, `license_usable`, `explicit_weights`, dependency footprint |

Current failure-case classification covers checks that are cheap and generated-fixture friendly:

- backend or inspection failure
- missing output file
- sample-rate mismatch
- duration drift
- clipping
- silence hallucination
- low-volume over-amplification
- channel-count changes

Long-audio memory pressure, stereo drift, non-speech speechification, noise hallucinated as high-band
detail, speaker timbre changes, and ASR degradation remain future optional/gated slices because they
need larger fixtures or downstream models.

## Scenario-Based Selection

Do not pick a backend by sorting one score. Choose by the deployment scenario:

| Scenario | Prefer |
| --- | --- |
| Offline restoration | stronger full-reference quality and high-band LSD, acceptable RTF, offline weights |
| ASR preprocessing | WER/CER improvement, STOI/SI-SDR, low artifact rate, no speaker or timing damage |
| Batch augmentation | reproducibility, failure rate, RTF, dependency footprint, deterministic seeds |
| Low-resource devices | CPU-only support, low peak RSS, small/no weights, low dependency tier |
| PyPI release readiness | light default dependencies, explicit weight provenance, usable licenses, offline CI |

## Minimal Dataset Recommendation

Start small:

```text
evalsets/
  speech_clean_48k/
    20 clean WAV references
    mixed gender
    mixed speaking rates
    quiet and mildly noisy sources
    Chinese and English coverage when available
```

Keep this dataset outside the repository unless licensing allows redistribution. Default tests use
generated audio fixtures only.

## Future Work

The remaining `v0.5.0` issues add:

- no-reference objective adapters
- downstream ASR/speaker/VAD/KWS evaluation
- AB/ABX/MUSHRA listening-test exports
- richer memory/load-time profiling, dependency-footprint, and governance tables
- threshold-based regression policies for release gates
