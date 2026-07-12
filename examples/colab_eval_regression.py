"""Collect same-backend baseline/candidate eval evidence in a Colab runtime.

The workflow runs all model inference remotely, compares immutable Git refs with the
maintained threshold policy, and writes one downloadable archive. It intentionally
refuses to execute outside Colab's ``/content`` workspace.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = os.environ.get("ASR_REPO_URL", "https://github.com/Tinnci/python-audio-super-resolution.git")
BASELINE_REF = os.environ.get("ASR_BASELINE_REF", "v0.6.0")
CANDIDATE_REF = os.environ.get("ASR_CANDIDATE_REF", "main")
DEVICE = os.environ.get("ASR_EVAL_DEVICE", "cuda")
WORK_ROOT = Path(os.environ.get("ASR_COLAB_EVAL_ROOT", "/content/audio-super-resolution-regression"))
BASELINE_DIR = WORK_ROOT / "baseline-repo"
CANDIDATE_DIR = WORK_ROOT / "candidate-repo"
CACHE_DIR = WORK_ROOT / "models"
EVIDENCE_DIR = WORK_ROOT / "evidence"
DATASET_DIR = EVIDENCE_DIR / "evalset"
ARCHIVE_PATH = Path(os.environ.get("ASR_COLAB_EVAL_ARCHIVE", "/content/asr-eval-regression.tar.gz"))


def _run(name: str, command: list[str], *, cwd: Path | None = None) -> None:
    log_path = EVIDENCE_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[colab-regression] {name}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _clone(name: str, ref: str, destination: Path) -> str:
    _run(f"clone-{name}", ["git", "clone", "--filter=blob:none", REPO_URL, str(destination)])
    _run(f"checkout-{name}", ["git", "checkout", ref], cwd=destination)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _install_project(name: str, repository: Path) -> None:
    _run(
        f"install-{name}",
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--reinstall-package",
            "audio-super-resolution",
            "-e",
            f"{repository}[lavasr,download]",
        ],
    )


def _matrix_command(*, backend: str, output_dir: Path) -> list[str]:
    command = [
        "audio-super-res",
        "eval",
        "matrix",
        "--dataset",
        str(DATASET_DIR / "speech_clean_48k"),
        "--backend",
        backend,
        "--degrader",
        "wideband_16k",
        "--degrader",
        "lowpass_4k",
        "--optional-metric",
        "mcd",
        "--output-dir",
        str(output_dir),
        "--device",
        DEVICE if backend == "lavasr-compat" else "cpu",
    ]
    if backend == "lavasr-compat":
        command.extend(
            [
                "--runtime-provider",
                "torch-eager",
                "--model-cache-dir",
                str(CACHE_DIR),
            ]
        )
    return command


def _write_report(name: str, manifest: Path, output: Path, *, cwd: Path) -> None:
    _run(
        name,
        ["audio-super-res", "eval", "report", "--manifest", str(manifest), "--output", str(output)],
        cwd=cwd,
    )


def _matrix_summary(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        "passed": manifest.get("passed"),
        "run_count": manifest.get("run_count"),
        "runs": [
            {
                "backend": run.get("backend"),
                "degrader": run.get("degrader"),
                "passed": run.get("passed"),
                "result_count": run.get("result_count"),
                "failure_count": run.get("failure_count"),
            }
            for run in manifest.get("runs", [])
            if isinstance(run, dict)
        ],
    }


def _write_summary(
    *,
    status: str,
    baseline_commit: str | None,
    candidate_commit: str | None,
    error: str | None = None,
) -> None:
    summary: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "repository": REPO_URL,
        "baseline_ref": BASELINE_REF,
        "baseline_commit": baseline_commit,
        "candidate_ref": CANDIDATE_REF,
        "candidate_commit": candidate_commit,
        "device": DEVICE,
        "error": error,
    }
    matrix_paths = {
        "sinc_baseline": EVIDENCE_DIR / "sinc-baseline" / "matrix.json",
        "lavasr_baseline": EVIDENCE_DIR / "lavasr-baseline" / "matrix.json",
        "lavasr_candidate": EVIDENCE_DIR / "lavasr-candidate" / "matrix.json",
    }
    summary["matrices"] = {name: _matrix_summary(path) for name, path in matrix_paths.items() if path.is_file()}
    comparison_path = EVIDENCE_DIR / "lavasr-comparison.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        summary["comparison"] = {
            "passed": comparison.get("passed"),
            "run_count": comparison.get("run_count"),
            "regression_count": len(comparison.get("regressions", [])),
            "regressions": comparison.get("regressions", []),
        }
    (EVIDENCE_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _archive() -> None:
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
        archive.add(EVIDENCE_DIR, arcname="evidence")
    print(f"[colab-regression] evidence archive: {ARCHIVE_PATH}", flush=True)


def main() -> int:
    if Path.cwd() != Path("/content"):
        raise RuntimeError("This regression workflow must run inside a Colab runtime rooted at /content")

    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    EVIDENCE_DIR.mkdir(parents=True)

    baseline_commit: str | None = None
    candidate_commit: str | None = None
    error: str | None = None
    try:
        _run("install-uv", [sys.executable, "-m", "pip", "install", "uv"])
        baseline_commit = _clone("baseline", BASELINE_REF, BASELINE_DIR)
        candidate_commit = _clone("candidate", CANDIDATE_REF, CANDIDATE_DIR)

        _install_project("candidate-bootstrap", CANDIDATE_DIR)
        _run(
            "init-evalset",
            [
                "audio-super-res",
                "eval",
                "init-speech-bwe",
                "--output-dir",
                str(DATASET_DIR),
                "--count",
                "8",
                "--duration-seconds",
                "0.2",
            ],
            cwd=CANDIDATE_DIR,
        )

        _install_project("baseline", BASELINE_DIR)
        _run(
            "prepare-lavasr-weights",
            [
                "audio-super-res",
                "--backend",
                "lavasr-compat",
                "--model-cache-dir",
                str(CACHE_DIR),
                "--download-weights",
                "--prepare-model-cache",
            ],
            cwd=BASELINE_DIR,
        )
        _run(
            "lavasr-baseline-matrix",
            _matrix_command(backend="lavasr-compat", output_dir=EVIDENCE_DIR / "lavasr-baseline"),
            cwd=BASELINE_DIR,
        )

        _install_project("candidate", CANDIDATE_DIR)
        _run(
            "sinc-baseline-matrix",
            _matrix_command(backend="sinc-resample", output_dir=EVIDENCE_DIR / "sinc-baseline"),
            cwd=CANDIDATE_DIR,
        )
        _run(
            "lavasr-candidate-matrix",
            _matrix_command(backend="lavasr-compat", output_dir=EVIDENCE_DIR / "lavasr-candidate"),
            cwd=CANDIDATE_DIR,
        )

        policy_source = CANDIDATE_DIR / "examples" / "artifacts" / "eval-threshold-policy.json"
        policy_copy = EVIDENCE_DIR / "eval-threshold-policy.json"
        shutil.copy2(policy_source, policy_copy)
        comparison_path = EVIDENCE_DIR / "lavasr-comparison.json"
        _run(
            "lavasr-matrix-compare",
            [
                "audio-super-res",
                "eval",
                "matrix-compare",
                str(EVIDENCE_DIR / "lavasr-baseline" / "matrix.json"),
                str(EVIDENCE_DIR / "lavasr-candidate" / "matrix.json"),
                "--threshold-policy",
                str(policy_copy),
                "--output",
                str(comparison_path),
            ],
            cwd=CANDIDATE_DIR,
        )

        _write_report(
            "sinc-baseline-report",
            EVIDENCE_DIR / "sinc-baseline" / "matrix.json",
            EVIDENCE_DIR / "sinc-baseline-report.md",
            cwd=CANDIDATE_DIR,
        )
        _write_report(
            "lavasr-baseline-report",
            EVIDENCE_DIR / "lavasr-baseline" / "matrix.json",
            EVIDENCE_DIR / "lavasr-baseline-report.md",
            cwd=CANDIDATE_DIR,
        )
        _write_report(
            "lavasr-candidate-report",
            EVIDENCE_DIR / "lavasr-candidate" / "matrix.json",
            EVIDENCE_DIR / "lavasr-candidate-report.md",
            cwd=CANDIDATE_DIR,
        )
        _write_report(
            "lavasr-comparison-report",
            comparison_path,
            EVIDENCE_DIR / "lavasr-comparison-report.md",
            cwd=CANDIDATE_DIR,
        )
        _write_summary(
            status="passed",
            baseline_commit=baseline_commit,
            candidate_commit=candidate_commit,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_summary(
            status="failed",
            baseline_commit=baseline_commit,
            candidate_commit=candidate_commit,
            error=error,
        )
    finally:
        _archive()

    if error:
        print(f"[colab-regression] failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
