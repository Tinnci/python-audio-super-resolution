# Release Checklist

Use this checklist when publishing a package release.

## Local Verification

```sh
pixi run lint
pixi run test
pixi run build
pixi run python -m pip check
```

Also confirm:

- `CHANGELOG.md` describes the release.
- `pyproject.toml` and `src/audio_super_resolution/__init__.py` use the same version.
- README commands still match the CLI.
- `examples/artifacts/` still contains current sample JSON artifacts.
- `docs/COLAB.md` still reflects the current model status.

## Publish Flow

1. Confirm PyPI trusted publishing is configured for this repository.
2. Create a version tag such as `v0.1.0`.
3. Push the tag to GitHub.
4. Create a GitHub release from the tag.
5. The release workflow builds with Pixi and publishes through `pypa/gh-action-pypi-publish`.

## PyPI Trusted Publishing

Repository setup required in PyPI:

- Project name: `audio-super-resolution`
- Publisher: GitHub Actions
- Owner: `Tinnci`
- Repository: `python-audio-super-resolution`
- Workflow: `release.yml`
- Environment: `pypi`

This repository cannot verify the PyPI-side configuration. Confirm it in PyPI before tagging the first public release.

## Records And Artifacts

- Current dry-run record: [RELEASE_DRY_RUN_0.1.0.md](RELEASE_DRY_RUN_0.1.0.md)
- Sample release artifacts: [examples/artifacts/](../examples/artifacts/)
- Docker usage: [README.md](../README.md#docker)
