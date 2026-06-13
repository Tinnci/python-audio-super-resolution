# Release Checklist

Use this checklist when publishing a new release.

## Before Release

- Confirm `pixi run lint` passes.
- Confirm `pixi run test` passes.
- Confirm `pixi run build` produces both sdist and wheel artifacts.
- Confirm `pixi run python -m pip check` passes.
- Update `CHANGELOG.md`.
- Update the version in `pyproject.toml`.
- Update `src/audio_super_resolution/__init__.py`.
- Confirm the README examples still match the CLI.
- Confirm `examples/artifacts/` has current sample JSON artifacts.
- Confirm the Colab plan in `docs/COLAB.md` matches the current CLI.

## GitHub Release

1. Create a tag using the package version, for example `v0.1.0`.
2. Push the tag to GitHub.
3. Create a GitHub release from the tag.
4. The release workflow builds the package with Pixi and publishes to PyPI.

## PyPI Publishing

The release workflow uses PyPI trusted publishing through `pypa/gh-action-pypi-publish`.

Repository setup required in PyPI:

- Project name: `audio-super-resolution`
- Publisher: GitHub Actions
- Owner: `Tinnci`
- Repository: `python-audio-super-resolution`
- Workflow: `release.yml`
- Environment: `pypi`

The repository cannot prove that PyPI trusted publishing has been configured. Confirm this in PyPI before tagging the first public release.

## Dry Run

The current `0.1.0` dry-run record is available at [RELEASE_DRY_RUN_0.1.0.md](RELEASE_DRY_RUN_0.1.0.md).

## Example Artifacts

Release notes can reference the sample artifacts under [examples/artifacts/](../examples/artifacts/):

- `sample-plan-manifest.json`
- `sample-completed-manifest.json`
- `sample-quality-report.json`

## Docker

The Dockerfile builds a baseline CPU image with the package installed from source. It is intended for CLI workflows and future model-backed images.

Build locally:

```sh
docker build -t audio-super-resolution .
```

Run against the current directory:

```sh
docker run --rm -v "%cd%":/workdir audio-super-resolution input.wav output.wav --target-sr 48000
```

On Unix-like shells:

```sh
docker run --rm -v "$PWD":/workdir audio-super-resolution input.wav output.wav --target-sr 48000
```
