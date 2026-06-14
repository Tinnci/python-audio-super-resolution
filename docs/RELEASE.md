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

Trigger a GitHub Release only after the matching milestone gates in [ROADMAP.md](../ROADMAP.md) are satisfied.

1. Confirm PyPI trusted publishing is configured for this repository.
2. Create a version tag such as `v0.1.0`.
3. Push the tag to GitHub.
4. Create a GitHub release from the tag.
5. The release workflow builds with Pixi and publishes through `pypa/gh-action-pypi-publish`.

## PyPI Trusted Publishing

The release workflow already uses PyPI Trusted Publishing / GitHub OIDC:

- `.github/workflows/release.yml` grants `id-token: write`.
- The publish job uses the `pypi` GitHub environment.
- `pypa/gh-action-pypi-publish` is called without a password or API token.

Because the package has not been created on PyPI yet, configure a pending trusted publisher from the PyPI account publishing page before the first release. Do not create a long-lived API token for this repository.

Pending publisher settings:

- Project name: `audio-super-resolution`
- Publisher: GitHub Actions
- Owner: `Tinnci`
- Repository: `python-audio-super-resolution`
- Workflow: `release.yml`
- Environment: `pypi`

When the first GitHub release runs successfully, PyPI will create the project from that pending publisher and bind future releases to the trusted publisher. This repository cannot verify the PyPI-side configuration; confirm it in PyPI before tagging the first public release.

## Records And Artifacts

- Current dry-run record: [RELEASE_DRY_RUN_0.1.0.md](RELEASE_DRY_RUN_0.1.0.md)
- Sample release artifacts: [examples/artifacts/](../examples/artifacts/)
- Docker usage: [README.md](../README.md#docker)
