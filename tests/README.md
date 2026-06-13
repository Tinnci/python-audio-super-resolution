# Tests

The default test suite is intentionally lightweight:

```sh
pixi run test
```

It verifies:

- CLI argument handling and informational commands.
- Model catalog and backend discovery output.
- JSON manifests for planned and completed jobs.
- Manifest regression comparison.
- JSON quality report artifacts.
- Chunked enhancement and crossfaded chunk writing.
- Single-file and batch path planning.
- Baseline `sinc-resample` enhancement.
- Optional AudioSR backend wiring with a fake in-memory module.
- Inference configuration validation.
- Optional low-pass preprocessing.
- Audio quality reports.
- Release example artifacts.

The test suite must not download model weights or require GPU access. Model-backed integration tests should be added separately and skipped by default unless the required dependency and weights are present.

Run the optional real AudioSR integration test only when model inference and possible weight downloads are intended:

```sh
set AUDIO_SUPER_RESOLUTION_RUN_AUDIOSR_INTEGRATION=1
pixi run pytest tests/test_audiosr_integration.py
```

Use generated audio fixtures inside temporary directories instead of committing large binary audio files unless a regression test genuinely needs a fixed reference artifact.
