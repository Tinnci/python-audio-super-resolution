"""Export and verify the LibriSpeech blind listening bundle in Colab.

Run this after ``colab_librispeech_eval.py`` in the same remote session. The
script copies the reference, degraded anchor, sinc output, and LavaSR output into
deterministically blinded stimuli and writes a compact downloadable archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

WORK_ROOT = Path(os.environ.get("ASR_LIBRISPEECH_ROOT", "/content/audio-super-resolution-librispeech"))
REPO_DIR = WORK_ROOT / "repo"
EVIDENCE_DIR = WORK_ROOT / "evidence"
LISTENING_DIR = EVIDENCE_DIR / "listening-mushra"
ARCHIVE_PATH = Path(os.environ.get("ASR_LISTENING_ARCHIVE", "/content/asr-listening-evidence.tar.gz"))


def _run(command: list[str]) -> None:
    print(f"[colab-listening] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_DIR, text=True, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bundle() -> dict[str, object]:
    public_path = LISTENING_DIR / "listening_manifest.json"
    answer_path = LISTENING_DIR / "answer_key.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    if public["trial_count"] != len(public["trials"]):
        raise ValueError("public listening trial count does not match its trial list")
    if len(public["trials"]) != len(answer["trials"]):
        raise ValueError("public and answer-key trial counts differ")

    public_stimuli = [stimulus for trial in public["trials"] for stimulus in trial["stimuli"]]
    answer_stimuli = [stimulus for trial in answer["trials"] for stimulus in trial["stimuli"]]
    if any(set(stimulus) != {"blind_id", "path"} for stimulus in public_stimuli):
        raise ValueError("public stimuli leaked fields beyond blind_id and path")
    if {stimulus["blind_id"] for stimulus in public_stimuli} != {stimulus["blind_id"] for stimulus in answer_stimuli}:
        raise ValueError("public and answer-key blind IDs differ")

    role_counts = Counter(str(stimulus["role"]) for stimulus in answer_stimuli)
    backend_counts = Counter(str(stimulus["backend"]) for stimulus in answer_stimuli)
    expected_roles = {"reference": len(public["trials"]), "anchor": len(public["trials"]), "system": 16}
    if dict(role_counts) != expected_roles:
        raise ValueError(f"unexpected listening roles: {dict(role_counts)}")
    if backend_counts["sinc-resample"] != 8 or backend_counts["lavasr-compat"] != 8:
        raise ValueError(f"unexpected backend counts: {dict(backend_counts)}")

    stimuli = sorted((LISTENING_DIR / "stimuli").glob("*.wav"))
    if len(stimuli) != len(public_stimuli):
        raise ValueError("stimulus file count does not match the public manifest")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "protocol": public["protocol"],
        "seed": public["seed"],
        "trial_count": len(public["trials"]),
        "stimulus_count": len(public_stimuli),
        "stimuli_per_trial": sorted({len(trial["stimuli"]) for trial in public["trials"]}),
        "role_counts": dict(role_counts),
        "backend_counts": dict(backend_counts),
        "public_stimulus_fields": ["blind_id", "path"],
        "answer_key_external": public["answer_key_external"],
        "rating_dimensions": public["rating_dimensions"],
        "manifest_sha256": _sha256(public_path),
        "answer_key_sha256": _sha256(answer_path),
        "error": None,
    }


def _archive() -> None:
    with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
        archive.add(LISTENING_DIR, arcname="listening-mushra")
        archive.add(EVIDENCE_DIR / "listening-summary.json", arcname="listening-summary.json")
    print(f"[colab-listening] evidence archive: {ARCHIVE_PATH}", flush=True)


def main() -> int:
    if Path.cwd() != Path("/content"):
        raise RuntimeError("This listening workflow must run inside a Colab runtime rooted at /content")
    sinc_manifest = EVIDENCE_DIR / "sinc-matrix" / "runs" / "sinc-resample__wideband_16k.json"
    lavasr_manifest = EVIDENCE_DIR / "lavasr-matrix" / "runs" / "lavasr-compat__wideband_16k.json"
    if not sinc_manifest.is_file() or not lavasr_manifest.is_file():
        raise FileNotFoundError("Run examples/colab_librispeech_eval.py before listening export")

    error: str | None = None
    try:
        _run(
            [
                "audio-super-res",
                "eval",
                "listening-export",
                "--manifest",
                str(sinc_manifest),
                "--manifest",
                str(lavasr_manifest),
                "--output-dir",
                str(LISTENING_DIR),
                "--protocol",
                "mushra",
                "--seed",
                "0",
            ]
        )
        summary = _verify_bundle()
        (EVIDENCE_DIR / "listening-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (EVIDENCE_DIR / "listening-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "error": error,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        if LISTENING_DIR.is_dir():
            _archive()

    if error:
        print(f"[colab-listening] failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
