# Golden Validation

Golden validation compares a backend output against a reference output produced by an upstream implementation or a previously accepted artifact. It is for compatible model backends such as `lavasr-compat`; it should not run real model checkpoints in the default test suite.

## Fixture Format

Use one JSON file per golden sample:

```json
{
  "schema_version": 1,
  "id": "lavasr-v2-bwe-sine-16k",
  "backend": "lavasr-compat",
  "model_id": "lavasr-v2-bwe",
  "input": {
    "path": "input.wav",
    "sample_rate": 16000
  },
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

Paths are relative to the fixture file unless a test explicitly resolves them another way. Reference outputs should stay small and license-safe; large or third-party generated artifacts should remain external and be downloaded only by gated tests.

## Compared Metrics

`audio_super_resolution.golden.compare_golden_outputs()` and `compare_golden_fixture()` check:

- sample rate equality
- duration drift
- peak level delta
- RMS level delta
- mean absolute log-mel difference
- high-frequency energy ratio delta

For stochastic models, use spectral statistics and relaxed thresholds instead of strict waveform identity.

## Regeneration Flow

1. Pin the upstream implementation commit, checkpoint revision, device, precision, and random seed.
2. Generate or choose a short license-safe input WAV.
3. Run the upstream implementation and save the reference WAV.
4. Run the local backend with the same input and managed verified weights.
5. Compare with `compare_golden_fixture()` and tune thresholds only enough to cover expected numerical drift.
6. Store the fixture JSON and small artifacts, or document the external cache location if artifacts are too large.

Real checkpoint generation must remain gated. For LavaSR, use the environment variables documented in [tests/README.md](../tests/README.md) before regenerating or validating reference artifacts.
