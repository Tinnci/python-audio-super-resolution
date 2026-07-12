# Evaluation And Regression Harness

This document owns backend evaluation and regression policy. Evaluation is not a single
audio-quality score. It is a reproducible workflow that compares backends across quality,
downstream usefulness, engineering cost, stability, and governance.

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
audio-super-res eval init-speech-bwe \
  --output-dir evalsets/speech_bwe_v1 \
  --count 20

audio-super-res eval init-failure-cases \
  --output-dir evalsets/speech_bwe_failure_cases_v1

audio-super-res eval run \
  --dataset evalsets/speech_bwe_v1/speech_clean_48k \
  --backend sinc-resample \
  --degrader wideband_16k \
  --optional-metric mcd \
  --work-dir runs/sinc-work \
  --output runs/sinc.json

audio-super-res eval matrix \
  --dataset evalsets/speech_bwe_v1/speech_clean_48k \
  --backend sinc-resample \
  --degrader lowpass_4k \
  --degrader wideband_16k \
  --degrader mp3_32kbps \
  --optional-metric mcd \
  --output-dir runs/matrix-smoke

audio-super-res eval compare runs/sinc.json runs/lavasr.json --output runs/comparison.json

audio-super-res eval matrix-compare runs/baseline-matrix/matrix.json runs/candidate-matrix/matrix.json \
  --threshold-policy eval-thresholds.json \
  --output runs/matrix-comparison.json

audio-super-res eval report --manifest runs/matrix-comparison.json --output runs/matrix-report.md
audio-super-res eval bundle --manifest runs/baseline-matrix/matrix.json --output-dir runs/evidence --archive runs/evidence.tar.gz

audio-super-res eval validate-dataset --manifest evalsets/speech_bwe_v1/manifest.json
audio-super-res eval inspect-checkpoint --checkpoint /path/to/model.pt --output runs/checkpoint-keys.json
```

For real recordings without a clean reference, run lightweight no-reference screening:

```sh
audio-super-res eval no-reference \
  --input recordings/real_world \
  --output runs/no-reference.json \
  --recursive
```

For downstream ASR transcript analysis from precomputed recognizer outputs:

```sh
audio-super-res eval downstream \
  --dataset evalsets/asr_transcripts_tiny.json \
  --output runs/downstream-asr.json
```

For human listening tests, export a runtime-neutral blind bundle:

```sh
audio-super-res eval listening-export \
  --manifest runs/sinc.json \
  --manifest runs/lavasr.json \
  --output-dir runs/listening-mushra \
  --protocol mushra \
  --seed 0
```

Regression thresholds can be repeated. The comparator infers metric direction: SI-SDR/PESQ/STOI
drops fail, while LSD/high-band LSD/RTF/peak RSS increases fail.

```sh
audio-super-res eval compare runs/sinc.json runs/lavasr.json \
  --threshold si_sdr_db=0.5 \
  --threshold highband_lsd_8_16k=0.25 \
  --threshold rtf=0.2
```

`eval run` accepts a directory of clean `.wav` reference files. It writes degraded inputs, backend
outputs, full-reference metrics, quality/stability checks, and simple runtime data into a JSON
manifest.

`eval matrix` runs multiple backend/degrader combinations and writes each run manifest plus a
`matrix.json` index. Use it for smoke grids, Colab evidence bundles, and release regression
artifacts before comparing specific candidate manifests.

`eval matrix-compare` compares two `matrix.json` indexes by matching backend/degrader runs and
delegating each pair to the normal manifest comparator. `--threshold-policy` accepts a JSON object
or `{"thresholds": {...}}`, and inline `--threshold FIELD=MAX_REGRESSION` values override policy
entries. `eval report` writes a compact Markdown summary. `eval bundle` copies matrix manifests and
referenced run manifests into a portable evidence directory, optionally archived as `.tar.gz`.
Use `eval matrix --reuse-existing` to keep completed run manifests, and `--fail-fast` when a
combination-level failure should stop the whole matrix instead of being recorded as a failed run.

The maintained example policy is
[`examples/artifacts/eval-threshold-policy.json`](../examples/artifacts/eval-threshold-policy.json).
It covers same-backend quality and stability regressions. It deliberately excludes RTF, elapsed
time, and memory thresholds until repeated runs establish variance for a specific device/runtime
pair. Never compare performance numbers across unlike hardware with this policy.

`eval init-speech-bwe` creates a deterministic synthetic tiny evalset for smoke and regression
testing. It is useful for CI and example commands, but it is not a substitute for a licensed real
speech dataset when making backend quality claims.

`eval init-failure-cases` creates deterministic synthetic stability fixtures for silence,
low-volume, near-clipping, stereo/channel, and long-smoke paths. These are intended for backend
hardening, including gated LavaSR validation, not for perceptual quality claims.

`eval inspect-checkpoint` is a gated local-only helper for checkpoint feasibility spikes such as
MossFormer2. It requires an explicit local checkpoint path and an explicitly installed torch
runtime. It does not download weights.

Implemented degraders:

| Degrader | Behavior |
| --- | --- |
| `lowpass_4k` | Keeps the original sample rate and low-passes around 4 kHz. |
| `narrowband_8k` | Downsamples to 8 kHz. |
| `wideband_16k` | Downsamples to 16 kHz. |
| `opus_16k_24kbps` | Downsamples to 16 kHz and applies deterministic codec-like quantization. |
| `mp3_32kbps` | Applies deterministic low-pass and codec-like quantization at the original sample rate. |
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
| `MCD` | Lightweight mel-cepstral distortion, available through `--optional-metric mcd`. |

PESQ, STOI/ESTOI, MCD, DNSMOS, NISQA, UTMOS, ViSQOL, real ASR runners, speaker similarity, VAD, and
keyword spotting are planned as optional or gated adapters. They must not become default CI
dependencies. Listening-test export and transcript WER/CER from precomputed ASR outputs are already
implemented in the default package.

## Optional Full-Reference Adapters

The default package intentionally implements only lightweight CPU/offline metrics. Optional
full-reference adapters can be added behind extras or environment gates when they meet these
requirements:

| Metric | Requirement |
| --- | --- |
| PESQ | Optional dependency only; document narrowband/wideband sample-rate constraints and license status. |
| STOI/ESTOI | Optional dependency only; resample explicitly and record the effective sample rate in the manifest. |
| MCD | Use a maintained cepstral feature implementation or a small local implementation with tests; record feature settings. |
| Codec degraders | Keep `ffmpeg`/codec tools optional; record codec, bitrate, sample rate, and command/tool version. |

Optional metrics must write `null` or an explicit skipped reason when unavailable. They must not
change the meaning of the lightweight schema, and they must not be folded into a single aggregate
score.

`eval run --optional-metric FIELD` currently supports explicit adapter attempts for `pesq`, `stoi`,
`estoi`, and `mcd`. MCD uses the package's lightweight NumPy/SciPy mel-cepstral implementation.
PESQ/STOI/ESTOI use dynamic imports when their optional packages are installed. Unavailable metrics
write `optional_metric_records` with `status: "skipped"` and install guidance. Install
`audio-super-resolution[eval-fullref]` to request PESQ/STOI adapters in an environment that supports
their upstream packages.

## No-Reference Screening

No-reference scores are screening signals for recordings without a clean reference. They are not
absolute truth and should not override full-reference, downstream, or listening evidence.

The default implemented evaluator is `signal-stats`. It is CPU/offline and uses only default
dependencies:

```json
{
  "evaluation_type": "no_reference",
  "evaluator": {
    "name": "signal-stats",
    "version": "builtin",
    "score_fields": ["rms_dbfs", "peak_level", "clipped_fraction", "silence_fraction", "dc_offset"],
    "absolute_truth": false
  },
  "records": [
    {
      "id": "sample",
      "input_path": "recordings/sample.wav",
      "status": "passed",
      "scores": {
        "rms_dbfs": -18.2,
        "peak_level": 0.42,
        "clipped_fraction": 0.0,
        "silence_fraction": 0.03,
        "dc_offset": 0.0001
      },
      "metadata": {
        "sample_rate": 48000,
        "duration_seconds": 4.2,
        "channels": 1
      },
      "error": null,
      "install_guidance": null
    }
  ]
}
```

DNSMOS, NISQA, UTMOS, and ViSQOL are documented as planned optional adapters. They must remain
explicit opt-ins because they introduce heavyweight dependencies, model downloads or external
binaries, and license/runtime obligations. Invoking one from the default install returns actionable
guidance instead of silently downloading assets.

## Downstream Evaluation

Downstream evaluation asks whether enhancement helps the task that consumes the audio. For ASR
preprocessing, an SR model that improves high-band LSD but worsens WER/CER is a regression for that
scenario.

The default implemented downstream evaluator is `transcript-error-rate`. It does not run an ASR
model; it compares precomputed ASR transcripts for the low-quality input and enhanced output against
a reference transcript. This keeps default CI CPU/offline and makes ASR model choice explicit.

Input dataset shape:

```json
{
  "dataset_id": "speech_bwe_tiny",
  "records": [
    {
      "id": "sample_001",
      "reference_transcript": "hello world",
      "baseline_transcript": "hello word",
      "enhanced_transcript": "hello world"
    }
  ]
}
```

Output records include baseline input score, enhanced score, delta, evaluator version, and dataset
identity:

```json
{
  "evaluation_type": "downstream",
  "dataset_id": "speech_bwe_tiny",
  "evaluator": {
    "name": "transcript-error-rate",
    "version": "builtin",
    "task": "asr",
    "score_fields": ["wer", "cer"]
  },
  "records": [
    {
      "id": "sample_001",
      "dataset_id": "speech_bwe_tiny",
      "task": "asr",
      "evaluator_version": "builtin",
      "baseline_input_score": {"wer": 0.5, "cer": 0.1},
      "enhanced_score": {"wer": 0.0, "cer": 0.0},
      "delta": {"wer": -0.5, "cer": -0.1}
    }
  ]
}
```

Speaker similarity, VAD/endpoint accuracy, and keyword spotting are represented as planned optional
adapters in the manifest. They must remain gated until their models, labels, licenses, and runtime
requirements are explicit.

## Listening-Test Export

Listening export creates files for AB, ABX, or MUSHRA-style studies without requiring a browser,
survey platform, or web runtime in the default package.

Bundle layout:

```text
runs/listening-mushra/
  listening_manifest.json   # public blind manifest for the listening tool
  answer_key.json           # machine-readable mapping; keep away from listeners
  stimuli/
    t001_s01.wav
    t001_s02.wav
    t001_s03.wav
```

The public manifest contains protocol, source manifest paths, deterministic seed, blind stimuli,
and rating dimensions. It intentionally omits backend names and source roles from each stimulus.
The answer key maps each blind id back to source manifest, item id, role (`reference`, `anchor`, or
`system`), backend, and original path.

Rating dimensions are explicit:

- clarity
- naturalness
- high-frequency harshness
- metallic artifacts
- noise
- intelligibility
- speaker fidelity
- music/environment artifacts
- latency
- stability

Do not ask only "which sample sounds better". Listening evidence should be reviewed with objective
quality, no-reference screening, downstream task impact, engineering performance, stability, and
governance tables. A model that sounds impressive but increases WER, changes speaker identity, clips,
or has unclear weight licensing is not automatically a better backend.

## Manifest Shape

Each result contains:

- `status`: `passed`, `stability_failed`, or `failed`
- `failure`: exception stage/type/message when a backend or inspection step fails
- `reference_path`, `degraded_path`, and `enhanced_path`
- degrader recipe metadata
- full-reference metrics
- quality checks: sample rate, duration drift, peak/clipping, pass/fail
- stability checks: sample-rate correctness, duration drift, clipping, and classified failure cases
- performance: backend init/load time, enhancement elapsed seconds, total elapsed seconds, audio
  duration, RTF, and peak RSS where the platform exposes `resource.getrusage(RUSAGE_SELF).ru_maxrss`

`eval compare` compares manifests by item id and reports separate metric deltas. It does not hide
raw metrics behind a single aggregate score. Candidate results with failed status, failed stability
checks, failed quality checks, or threshold violations are reported as regressions.

The comparison JSON keeps raw `metric_summary` data and also groups it into `tables`:

- `audio_quality`: full-reference and optional objective metrics
- `no_reference`: no-reference screening scores
- `downstream`: WER/CER/speaker/VAD/KWS metrics when optional adapters add them
- `engineering`: RTF, elapsed time, init time, and peak RSS
- `stability`: drift/clipping metrics plus candidate status and failure-case counts
- `governance`: backend/profile facts for offline use, reproducibility, license usability, explicit weights, and dependency footprint

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

The first maintained real-speech baseline is specified by
[`examples/artifacts/librispeech-dev-clean-tiny-v1.json`](../examples/artifacts/librispeech-dev-clean-tiny-v1.json).
It pins the official LibriSpeech `dev-clean` archive URL and MD5, records the CC BY 4.0 license, and
selects one utterance from four female and four male speakers deterministically. The Colab workflow
downloads and converts the audio remotely; neither the source archive nor converted WAV files are
committed.

The first T4 run covered eight utterances (about 61 seconds total) with `wideband_16k` and
`lowpass_4k`. Both backends completed all 16 items without stability failures. Mean results included:

| Degrader | Backend | SI-SDR dB | LSD dB | High-band LSD 8–16 kHz | MCD | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `wideband_16k` | `sinc-resample` | 43.88 | 5.57 | 6.45 | 2.68 | 0.098 |
| `wideband_16k` | `lavasr-compat` | 28.62 | 26.94 | 31.78 | 46.71 | 0.175 |
| `lowpass_4k` | `sinc-resample` | 20.81 | 19.68 | 14.19 | 52.45 | 0.003 |
| `lowpass_4k` | `lavasr-compat` | 20.81 | 19.82 | 14.17 | 52.45 | 0.074 |

These figures are evidence, not an aggregate ranking. The wideband result shows that LavaSR's
generated high-frequency content is penalized heavily by reference-fidelity metrics. Stable
promotion therefore requires downstream ASR and blind listening evidence rather than interpreting
matrix completion or one spectral score as proof of improvement.

## Golden Compatibility Validation

Golden validation compares a package-owned compatible backend with an upstream output or a
previously accepted artifact. It is intended for paths such as `lavasr-compat`; real checkpoints
must remain outside the default suite.

A fixture records the backend/model identity, relative input and reference paths, sample rates,
upstream source revision, and comparison thresholds. Example:

```json
{
  "schema_version": 1,
  "id": "lavasr-v2-bwe-sine-16k",
  "backend": "lavasr-compat",
  "model_id": "lavasr-v2-bwe",
  "input": {"path": "input.wav", "sample_rate": 16000},
  "reference": {
    "path": "reference.wav",
    "source": "upstream LavaSR commit <sha>",
    "sample_rate": 48000
  },
  "thresholds": {
    "max_duration_drift_seconds": 0.05,
    "max_peak_delta": 0.02,
    "max_rms_delta": 0.02,
    "max_log_mel_l1": 0.5,
    "max_hf_energy_ratio_delta": 0.05,
    "high_frequency_start_hz": 8000
  }
}
```

`compare_golden_outputs()` and `compare_golden_fixture()` check sample rate, duration drift, peak
and RMS deltas, mean absolute log-mel difference, and high-frequency energy-ratio drift. Use
spectral statistics and relaxed thresholds for stochastic models instead of strict waveform
identity.

Regeneration procedure:

1. Pin upstream code, checkpoint revision, device, precision, and seed.
2. Use a short license-safe input.
3. Generate the upstream reference and local output from the same verified weights.
4. Tune thresholds only for expected numerical drift.
5. Keep small legal fixtures in the repository; keep large or third-party artifacts in gated
   external storage.

The gated LavaSR parity test compares `lavasr-compat` with upstream `LavaSR.enhancer.LavaBWE` using
the same local weights. Installation and environment gates are documented in
[../tests/README.md](../tests/README.md).

## Future Work

Prioritize evidence over additional adapters:

- define a small licensed evaluation set outside the repository;
- publish an example threshold policy for release regression runs;
- record repeatable `sinc-resample` and gated `lavasr-compat` matrix evidence;
- broaden real-weight stability and golden fixtures before changing model internals;
- add heavyweight no-reference, ASR, speaker, VAD, or KWS adapters only for a concrete deployment
  question with explicit model, license, label, and runtime requirements.
