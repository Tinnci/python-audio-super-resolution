"""Run v0.6 LavaSR GPU evidence collection inside a Colab CLI session.

This script is intended for ``colab exec -f``. It installs and runs real-weight/GPU
validation in the remote Colab runtime and writes one downloadable evidence archive.
Do not execute it locally as part of the default development workflow.
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
GIT_REF = os.environ.get("ASR_GIT_REF", "v0.6.0")
DEVICE = os.environ.get("ASR_LAVASR_DEVICE", "cuda")
WORK_ROOT = Path(os.environ.get("ASR_COLAB_WORK_ROOT", "/content/audio-super-resolution-validation"))
REPO_DIR = WORK_ROOT / "repo"
CACHE_DIR = WORK_ROOT / "models"
EVIDENCE_DIR = WORK_ROOT / "evidence"
ARCHIVE_PATH = Path(os.environ.get("ASR_COLAB_ARCHIVE", "/content/asr-v0.6.0-colab-evidence.tar.gz"))


def _run(name: str, command: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    log_path = EVIDENCE_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[colab-validation] {name}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _capture_environment() -> None:
    script = (
        "import json, platform; "
        "import torch; "
        "print(json.dumps({"
        "'python': platform.python_version(), "
        "'platform': platform.platform(), "
        "'torch': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'cuda_version': torch.version.cuda, "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}, indent=2))"
    )
    _run("python-torch-environment", [sys.executable, "-c", script], cwd=REPO_DIR)
    if shutil.which("nvidia-smi"):
        _run("nvidia-smi", ["nvidia-smi"], cwd=REPO_DIR)
    _run("package-environment", ["audio-super-res", "--env-info"], cwd=REPO_DIR)
    _run(
        "model-list",
        ["audio-super-res", "--list-models", "--list-format", "json"],
        cwd=REPO_DIR,
    )


def _validation_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "AUDIO_SUPER_RESOLUTION_RUN_WEIGHT_DOWNLOAD": "1",
            "AUDIO_SUPER_RESOLUTION_RUN_LAVASR_SMOKE": "1",
            "AUDIO_SUPER_RESOLUTION_LAVASR_CACHE": str(CACHE_DIR),
            "AUDIO_SUPER_RESOLUTION_LAVASR_DEVICE": DEVICE,
            "AUDIO_SUPER_RESOLUTION_LAVASR_SMOKE_SECONDS": "0.05",
            "AUDIO_SUPER_RESOLUTION_LAVASR_SMOKE_SR": "16000",
            "AUDIO_SUPER_RESOLUTION_RUN_ACCELERATOR_MATRIX": "1",
            "AUDIO_SUPER_RESOLUTION_ACCELERATOR_DEVICE": DEVICE,
        }
    )
    return env


def _write_summary(*, status: str, error: str | None = None) -> None:
    commit = None
    if REPO_DIR.is_dir():
        resolved_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        commit = resolved_commit or None
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "repository": REPO_URL,
        "requested_ref": GIT_REF,
        "commit": commit,
        "device": DEVICE,
        "cache_dir": str(CACHE_DIR),
        "evidence_dir": str(EVIDENCE_DIR),
        "archive_path": str(ARCHIVE_PATH),
        "error": error,
    }
    (EVIDENCE_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _archive_evidence() -> None:
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
        archive.add(EVIDENCE_DIR, arcname=EVIDENCE_DIR.name)
    print(f"[colab-validation] evidence archive: {ARCHIVE_PATH}", flush=True)


def main() -> int:
    if Path.cwd() != Path("/content"):
        raise RuntimeError("This validation script must run inside a Colab runtime rooted at /content")

    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    EVIDENCE_DIR.mkdir(parents=True)

    error: str | None = None
    try:
        _run("clone", ["git", "clone", "--filter=blob:none", REPO_URL, str(REPO_DIR)])
        _run("checkout", ["git", "checkout", GIT_REF], cwd=REPO_DIR)
        _run("install-uv", [sys.executable, "-m", "pip", "install", "uv"])
        _run(
            "install-project",
            ["uv", "pip", "install", "--system", "-e", ".[lavasr,download]", "pytest"],
            cwd=REPO_DIR,
        )
        _capture_environment()

        validation_env = _validation_environment()
        _run(
            "real-weight-and-smoke-tests",
            [sys.executable, "-m", "pytest", "tests/test_lavasr_real_weights.py", "-q", "-rA"],
            cwd=REPO_DIR,
            env=validation_env,
        )
        _run(
            "accelerator-device-test",
            [sys.executable, "-m", "pytest", "tests/test_accelerator_matrix.py", "-q", "-rA"],
            cwd=REPO_DIR,
            env=validation_env,
        )

        failure_cases = EVIDENCE_DIR / "failure-cases"
        matrix_dir = EVIDENCE_DIR / "lavasr-failure-matrix"
        _run(
            "init-failure-cases",
            ["audio-super-res", "eval", "init-failure-cases", "--output-dir", str(failure_cases)],
            cwd=REPO_DIR,
        )
        _run(
            "lavasr-failure-matrix",
            [
                "audio-super-res",
                "eval",
                "matrix",
                "--dataset",
                str(failure_cases / "speech_clean_48k"),
                "--backend",
                "lavasr-compat",
                "--degrader",
                "wideband_16k",
                "--output-dir",
                str(matrix_dir),
                "--device",
                DEVICE,
                "--runtime-provider",
                "torch-eager",
                "--model-cache-dir",
                str(CACHE_DIR),
            ],
            cwd=REPO_DIR,
        )
        _run(
            "lavasr-failure-report",
            [
                "audio-super-res",
                "eval",
                "report",
                "--manifest",
                str(matrix_dir / "matrix.json"),
                "--output",
                str(EVIDENCE_DIR / "lavasr-failure-report.md"),
            ],
            cwd=REPO_DIR,
        )
        _write_summary(status="passed")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_summary(status="failed", error=error)
    finally:
        _archive_evidence()

    if error:
        print(f"[colab-validation] failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
