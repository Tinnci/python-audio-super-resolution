# Release And Merge Checklist

This document owns repeatable merge and release verification. Milestone direction belongs in
[../ROADMAP.md](../ROADMAP.md); user-facing changes belong in [../CHANGELOG.md](../CHANGELOG.md).

## Before Merge

1. Confirm the branch base and inspect the complete diff.
2. Keep unrelated working-tree changes out of the merge.
3. Add regression coverage for behavior changes.
4. Update `CHANGELOG.md` when users, manifests, CLI behavior, packaging, or compatibility changes.
5. Run:

```sh
pixi run lint
pixi run format-check
pixi run typecheck
pixi run test
```

6. Run `git diff --check` and review generated/untracked files.
7. Treat real weights, external packages, and accelerator tests as explicit gates; record which were
   run instead of implying they are covered by the default suite.

## Before Release

Run the merge checks, then build and inspect the distributions:

```sh
pixi run build
pixi run metadata-check
pixi run wheel-check
```

Also verify:

- `pyproject.toml`, `audio_super_resolution.__version__`, tag, and changelog version agree;
- the wheel contains `py.typed` and the expected package modules;
- README installation and CLI examples match the release;
- a clean environment can install the wheel and run `audio-super-res --version`,
  `--list-backends`, and a short `sinc-resample` enhancement;
- optional model status and limitations match `--list-models --list-format json`;
- CI passes on the exact release commit.

## Publish Flow

1. Merge the prepared release changes.
2. Create a signed or annotated version tag matching the source version.
3. Push the tag and use the GitHub release workflow.
4. Verify the published files and metadata on PyPI.
5. Install the published wheel in a clean environment and repeat the CLI smoke test.
6. Record any release-specific exception in the GitHub release notes or issue tracker rather than
   creating another permanent snapshot document.

## Trusted Publishing

Publishing uses GitHub Actions OIDC with the `pypi` environment. The configured PyPI project is
`audio-super-resolution`, owned by `Tinnci`, and the release workflow is `release.yml`. No PyPI API
token should be stored in the repository.

Before publishing, confirm the repository, workflow, environment, owner, and active GitHub account
still match the trusted-publisher configuration.

## Historical Baseline

The first `0.1.0` release dry run established the reusable process now captured above:

- local test and build gates passed;
- CI and security workflows passed on `main`;
- sdist, wheel, and example JSON artifacts were inspected;
- the GitHub `pypi` environment and PyPI pending trusted publisher were confirmed;
- publishing used OIDC without an API token.

`0.1.0` was later superseded by `0.1.1` for Python 3.10 compatibility. Exact historical commits,
workflow runs, and release artifacts remain available in Git and GitHub history; they do not need a
separate living document.

## Evidence Retention

- Keep stable commands and policies in this document.
- Keep milestone priorities in `ROADMAP.md`.
- Keep release-facing changes in `CHANGELOG.md`.
- Keep generated matrix, benchmark, quality, and listening evidence under ignored `runs/` or attach
  it to the relevant release/issue.
- Add a permanent record only when it contains a decision or procedure that cannot be reconstructed
  from Git history and generated artifacts.
