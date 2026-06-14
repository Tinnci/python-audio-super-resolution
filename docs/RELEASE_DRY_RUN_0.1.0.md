# Release Dry Run: 0.1.0

Date: 2026-06-14

Status: historical first-release record. `0.1.0` was published, then both PyPI files were yanked after `0.1.1` superseded it with Python 3.10-compatible code.

## Local Checks

- `pixi run lint`: passed
- `pixi run test`: passed
- `pixi run build`: passed
- `pixi run python -m pip check`: passed

## GitHub Checks

- CI workflow on `main`: passed
- Security workflow on `main`: passed
- Latest checked remote workflow commit before this release-prep update: `a2e1350`
- Latest checked remote workflows:
  - CI: https://github.com/Tinnci/python-audio-super-resolution/actions/runs/27491044239
  - Security: https://github.com/Tinnci/python-audio-super-resolution/actions/runs/27491044218

## Artifacts

- Source distribution: `dist/audio_super_resolution-0.1.0.tar.gz`
- Wheel: `dist/audio_super_resolution-0.1.0-py3-none-any.whl`
- Example dry-run manifest: `examples/artifacts/sample-plan-manifest.json`
- Example completed manifest: `examples/artifacts/sample-completed-manifest.json`
- Example quality report: `examples/artifacts/sample-quality-report.json`

## External Release Setup

PyPI pending trusted publisher was confirmed by the project owner before the first public publish:

- Project name: `audio-super-resolution`
- Owner: `Tinnci`
- Repository: `python-audio-super-resolution`
- Workflow: `release.yml`
- Environment: `pypi`

The GitHub `pypi` environment existed and was intentionally unprotected for the first alpha publish. The release workflow uses GitHub OIDC and does not require a PyPI API token. After the first successful publish, the pending publisher became the active PyPI trusted publisher for the project.
