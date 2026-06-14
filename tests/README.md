# Tests

The default suite is lightweight, CPU-friendly, and offline:

```sh
pixi run test
```

It covers the baseline CLI/API path, backend and model discovery, manifests, quality reports, preprocessing, chunking, weight manifest validation, fake-provider downloads, LavaSR metadata validation, device discovery, and release example artifacts.

## Policy

- Do not download real model weights in default tests.
- Do not require GPU access in default tests.
- Use generated audio fixtures in temporary directories instead of committing large binary files.
- Keep provider tests mocked unless they are explicitly gated by an environment variable.
- Keep optional model inference tests separate from the default suite.
- Keep torch-dependent LavaSR runtime tests skipped unless torch is installed.

## Optional Integrations

Run real AudioSR integration only when upstream checkpoint handling is expected:

```sh
set AUDIO_SUPER_RESOLUTION_RUN_AUDIOSR_INTEGRATION=1
pixi run pytest tests/test_audiosr_integration.py
```

Real weight download tests should use `AUDIO_SUPER_RESOLUTION_RUN_WEIGHT_DOWNLOAD=1` and remain skipped by default.
