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

- `reference_path`, `degraded_path`, and `enhanced_path`
- degrader recipe metadata
- full-reference metrics
- quality checks: sample rate, duration drift, peak/clipping, pass/fail
- performance: elapsed seconds, audio duration, and RTF

`eval compare` compares manifests by item id and reports separate metric deltas. It does not hide
raw metrics behind a single aggregate score.

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
- peak memory, load-time, dependency-footprint, and governance tables
- threshold-based regression policies for release gates
