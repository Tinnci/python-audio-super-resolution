# Documentation Index

This directory contains maintainer-facing design, validation, release, and model-candidate notes.
The top-level [README.md](../README.md) is the user entry point; detailed records live here.

## Start Here

| Document | Purpose |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Package layers, backend contract, and weight-management boundaries. |
| [MODEL_ADMISSION.md](MODEL_ADMISSION.md) | Gates for adding model-backed inference paths and the scorecard API. |
| [ACCELERATORS.md](ACCELERATORS.md) | Runtime-provider policy, installation, hardware validation, Colab/GPU evidence, and optimization gates. |
| [EVALUATION.md](EVALUATION.md) | Eval/regression workflows, metrics, golden compatibility, manifests, and selection policy. |
| [RELEASE.md](RELEASE.md) | Merge/release checklist, trusted publishing, and evidence-retention policy. |

## Model Planning

| Document | Scope |
| --- | --- |
| [BACKEND_CANDIDATES.md](BACKEND_CANDIDATES.md) | Consolidated speech/general-audio candidate evidence and blockers. |
| [../ROADMAP.md](../ROADMAP.md) | Prioritized post-v0.6 direction and completed milestone summary. |

## Release Records

| Document | Scope |
| --- | --- |
| [../CHANGELOG.md](../CHANGELOG.md) | User-facing release history and unreleased changes. |
| [RELEASE.md](RELEASE.md) | Stable merge/release procedure plus the durable first-release baseline. |

## Ownership Rules

- `README.md` should stay short and user-facing: install, basic commands, current support, and links.
- `ARCHITECTURE.md` owns layering and API boundaries, not milestone status.
- `ROADMAP.md` owns priorities and sequencing, not detailed candidate evidence or full history.
- `BACKEND_CANDIDATES.md` owns model-specific evidence and blockers.
- `CHANGELOG.md` records release-facing changes, not full design rationale.
- Avoid permanent pre-merge snapshots when Git history, the changelog, and generated evidence can
  reconstruct the same facts.
