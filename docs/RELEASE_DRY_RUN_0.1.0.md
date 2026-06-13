# Release Dry Run: 0.1.0

Date: 2026-06-14

## Local Checks

- `pixi run lint`: passed
- `pixi run test`: passed
- `pixi run build`: passed
- `pixi run python -m pip check`: passed

## GitHub Checks

- CI workflow on `main`: passed
- Security workflow on `main`: passed

## Artifacts

- Source distribution: `dist/audio_super_resolution-0.1.0.tar.gz`
- Wheel: `dist/audio_super_resolution-0.1.0-py3-none-any.whl`
- Example dry-run manifest: `examples/artifacts/sample-plan-manifest.json`
- Example completed manifest: `examples/artifacts/sample-completed-manifest.json`
- Example quality report: `examples/artifacts/sample-quality-report.json`

## External Release Setup

PyPI trusted publishing must be confirmed in PyPI by a project owner before the first public publish:

- Project name: `audio-super-resolution`
- Owner: `Tinnci`
- Repository: `python-audio-super-resolution`
- Workflow: `release.yml`
- Environment: `pypi`

This cannot be verified from the repository alone.
