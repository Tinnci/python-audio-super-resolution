# Example Artifacts

These files are small, static examples of the JSON artifacts produced by the CLI.

- `sample-plan-manifest.json`: output from a dry-run batch plan.
- `sample-completed-manifest.json`: output from a completed enhancement run.
- `sample-quality-report.json`: output from `--quality-report-json`.
- `eval-threshold-policy.json`: conservative same-backend release regression limits for matrix
  comparison. It intentionally excludes performance thresholds until repeated device-specific
  variance is available.
- `librispeech-dev-clean-tiny-v1.json`: source, checksum, license, deterministic selection, and
  storage policy for the remote-only real-speech evaluation baseline. No dataset audio is committed.

They are intended for release notes, documentation, and downstream CI examples.
