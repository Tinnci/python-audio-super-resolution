# Documentation Index

This directory contains maintainer-facing design, validation, release, and model-candidate notes.
The top-level [README.md](../README.md) is the user entry point; detailed records live here.

## Start Here

| Document | Purpose |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Package layers, backend contract, and weight-management boundaries. |
| [MODEL_ADMISSION.md](MODEL_ADMISSION.md) | Gates for adding model-backed inference paths and the scorecard API. |
| [ACCELERATORS.md](ACCELERATORS.md) | Accelerator/runtime-provider policy, install strategy, gated validation matrix, and LavaSR optimization recommendation. |
| [GOLDEN.md](GOLDEN.md) | Golden-sample fixture format, metrics, and gated parity strategy. |
| [COLAB.md](COLAB.md) | Repository-based LavaSR/GPU validation guide and evidence checklist. |
| [RELEASE.md](RELEASE.md) | Release checklist and PyPI Trusted Publishing notes. |

## Model Planning

| Document | Scope |
| --- | --- |
| [SPEECH_BACKEND_CANDIDATES.md](SPEECH_BACKEND_CANDIDATES.md) | Speech SR/BWE candidate review, currently ClearerVoice and Resemble Enhance feasibility. |
| [GENERAL_AUDIO_CANDIDATES.md](GENERAL_AUDIO_CANDIDATES.md) | General-audio SR candidate review, currently AudioSR external baseline and FlowHigh feasibility. |
| [../ROADMAP.md](../ROADMAP.md) | Milestone state, open implementation tracks, and release gates. |

## Release Records

| Document | Scope |
| --- | --- |
| [../CHANGELOG.md](../CHANGELOG.md) | User-facing release history and unreleased changes. |
| [RELEASE_DRY_RUN_0.1.0.md](RELEASE_DRY_RUN_0.1.0.md) | Historical first-release dry-run notes. |

## Ownership Rules

- `README.md` should stay short and user-facing: install, basic commands, current support, and links.
- `ARCHITECTURE.md` owns layering and API boundaries, not milestone status.
- `ROADMAP.md` owns milestone status, not detailed candidate evidence.
- Candidate reviews own model-specific evidence, risks, and follow-up issue links.
- `CHANGELOG.md` records release-facing changes, not full design rationale.
