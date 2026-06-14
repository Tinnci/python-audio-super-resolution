# Tests

The default test suite is intentionally lightweight and offline:

```sh
pixi run test
```

It verifies:

- CLI argument handling and informational commands.
- Model catalog and backend discovery output.
- Backend registry and model metadata specs.
- Weight manifest paths, multi-file SHA256 verification, fake-provider downloads, and local weight store helpers.
- Lightweight device discovery helpers.
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

The test suite must not download model weights or require GPU access. Model-backed integration tests should stay separate and skipped by default unless the required dependency and weights are present.

Default download tests use fake providers only. Real provider tests must be gated by an environment variable such as `AUDIO_SUPER_RESOLUTION_RUN_WEIGHT_DOWNLOAD=1`.

Run the optional real AudioSR integration test only when model inference and upstream checkpoint handling are intended:

```sh
set AUDIO_SUPER_RESOLUTION_RUN_AUDIOSR_INTEGRATION=1
pixi run pytest tests/test_audiosr_integration.py
```

Use generated audio fixtures inside temporary directories instead of committing large binary audio files unless a regression test genuinely needs a fixed reference artifact.
