"""Build and evaluate the licensed LibriSpeech tiny baseline in Colab.

The source archive, conversion, managed weights, and model inference remain in the
remote runtime. The script writes a compact evidence archive for download into the
ignored local ``runs/`` directory and refuses to run outside ``/content``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = os.environ.get("ASR_REPO_URL", "https://github.com/Tinnci/python-audio-super-resolution.git")
GIT_REF = os.environ.get("ASR_GIT_REF", "main")
DEVICE = os.environ.get("ASR_LIBRISPEECH_DEVICE", "cuda")
WORK_ROOT = Path(os.environ.get("ASR_LIBRISPEECH_ROOT", "/content/audio-super-resolution-librispeech"))
REPO_DIR = WORK_ROOT / "repo"
DOWNLOAD_DIR = WORK_ROOT / "downloads"
EXTRACT_DIR = WORK_ROOT / "source"
CACHE_DIR = WORK_ROOT / "models"
EVIDENCE_DIR = WORK_ROOT / "evidence"
EVALSET_DIR = EVIDENCE_DIR / "librispeech-dev-clean-tiny-v1"
ARCHIVE_PATH = Path(os.environ.get("ASR_LIBRISPEECH_ARCHIVE", "/content/asr-librispeech-evidence.tar.gz"))


def _run(name: str, command: list[str], *, cwd: Path | None = None) -> None:
    log_path = EVIDENCE_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[colab-librispeech] {name}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _load_spec() -> dict[str, object]:
    path = REPO_DIR / "examples" / "artifacts" / "librispeech-dev-clean-tiny-v1.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("LibriSpeech evalset specification must be a JSON object")
    return loaded


def _verify_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5()  # noqa: S324 - MD5 is the publisher-provided archive identity.
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"LibriSpeech archive MD5 mismatch: expected {expected}, got {actual}")


def _extract_archive(archive_path: Path) -> Path:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(EXTRACT_DIR, filter="data")
    source_root = EXTRACT_DIR / "LibriSpeech"
    if not source_root.is_dir():
        raise FileNotFoundError("LibriSpeech root was not found after extraction")
    return source_root


def _speaker_ids(source_root: Path, *, female: int, male: int) -> list[tuple[str, str]]:
    by_sex: dict[str, list[str]] = {"F": [], "M": []}
    for raw_line in (source_root / "SPEAKERS.TXT").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 5 or fields[2].lower() != "dev-clean":
            continue
        speaker_id, sex = fields[0], fields[1].upper()
        if sex in by_sex:
            by_sex[sex].append(speaker_id)
    selected = [(speaker_id, "female") for speaker_id in sorted(by_sex["F"])[:female]]
    selected.extend((speaker_id, "male") for speaker_id in sorted(by_sex["M"])[:male])
    if len(selected) != female + male:
        raise ValueError("LibriSpeech speaker metadata did not contain the requested balanced sample")
    return selected


def _transcript_for(source_path: Path) -> str:
    utterance_id = source_path.stem
    transcript_path = next(source_path.parent.glob("*.trans.txt"))
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        item_id, transcript = line.split(" ", 1)
        if item_id == utterance_id:
            return transcript
    raise ValueError(f"Transcript not found for {utterance_id}")


def _build_evalset(spec: dict[str, object], source_root: Path) -> dict[str, object]:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    source = spec["source"]
    selection = spec["selection"]
    if not isinstance(source, dict) or not isinstance(selection, dict):
        raise ValueError("LibriSpeech spec source and selection must be objects")
    target_sample_rate = int(selection["target_sample_rate"])
    selected_speakers = _speaker_ids(
        source_root,
        female=int(selection["female_speakers"]),
        male=int(selection["male_speakers"]),
    )
    clean_dir = EVALSET_DIR / "speech_clean_48k"
    metadata_dir = EVALSET_DIR / "source-metadata"
    clean_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for speaker_id, speaker_gender in selected_speakers:
        source_path = sorted((source_root / "dev-clean" / speaker_id).glob("*/*.flac"))[0]
        audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=False)
        divisor = np.gcd(sample_rate, target_sample_rate)
        converted = resample_poly(audio, target_sample_rate // divisor, sample_rate // divisor).astype(np.float32)
        output_path = clean_dir / f"{source_path.stem}.wav"
        sf.write(output_path, converted, target_sample_rate, subtype="PCM_16")
        records.append(
            {
                "id": source_path.stem,
                "path": str(output_path.relative_to(EVALSET_DIR)),
                "source_path": str(source_path.relative_to(source_root)),
                "speaker_id": speaker_id,
                "speaker_gender": speaker_gender,
                "language": selection["language"],
                "transcript": _transcript_for(source_path),
                "source_sample_rate": sample_rate,
                "sample_rate": target_sample_rate,
                "duration_seconds": converted.shape[0] / target_sample_rate,
                "synthetic": False,
            }
        )

    for name in ("LICENSE.TXT", "README.TXT", "SPEAKERS.TXT"):
        shutil.copy2(source_root / name, metadata_dir / name)
    shutil.copy2(
        REPO_DIR / "examples" / "artifacts" / "librispeech-dev-clean-tiny-v1.json",
        metadata_dir / "dataset-spec.json",
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(EVALSET_DIR),
        "reference_dir": "speech_clean_48k",
        "sample_rate": target_sample_rate,
        "record_count": len(records),
        "source": source,
        "selection": selection,
        "records": records,
        "notes": [
            "Real licensed speech; generated remotely from the pinned official archive.",
            "Converted WAV files and source audio are evidence artifacts and are not committed.",
        ],
    }
    (EVALSET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _matrix_command(*, backend: str, output_dir: Path) -> list[str]:
    command = [
        "audio-super-res",
        "eval",
        "matrix",
        "--dataset",
        str(EVALSET_DIR / "speech_clean_48k"),
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


def _metric_means(matrix_dir: Path) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    matrix = json.loads((matrix_dir / "matrix.json").read_text(encoding="utf-8"))
    for run in matrix["runs"]:
        manifest = json.loads(Path(run["manifest_path"]).read_text(encoding="utf-8"))
        metric_values: dict[str, list[float]] = {}
        for result in manifest["results"]:
            for name, value in result.get("metrics", {}).items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metric_values.setdefault(name, []).append(float(value))
            for name in ("rtf", "peak_rss_mb"):
                value = result.get("performance", {}).get(name)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metric_values.setdefault(name, []).append(float(value))
        summary[str(run["degrader"])] = {
            name: sum(values) / len(values) for name, values in sorted(metric_values.items()) if values
        }
    return summary


def _write_summary(*, status: str, commit: str | None, manifest: dict[str, object] | None, error: str | None) -> None:
    summary: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "repository": REPO_URL,
        "requested_ref": GIT_REF,
        "commit": commit,
        "device": DEVICE,
        "dataset_id": manifest.get("dataset_id") if manifest else None,
        "record_count": manifest.get("record_count") if manifest else None,
        "error": error,
    }
    matrix_dirs = {
        "sinc_resample": EVIDENCE_DIR / "sinc-matrix",
        "lavasr_compat": EVIDENCE_DIR / "lavasr-matrix",
    }
    summary["matrices"] = {
        name: {
            "manifest": str(matrix_dir / "matrix.json"),
            "passed": json.loads((matrix_dir / "matrix.json").read_text(encoding="utf-8")).get("passed"),
            "metric_means": _metric_means(matrix_dir),
        }
        for name, matrix_dir in matrix_dirs.items()
        if (matrix_dir / "matrix.json").is_file()
    }
    (EVIDENCE_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _archive() -> None:
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
        archive.add(EVIDENCE_DIR, arcname="evidence")
    print(f"[colab-librispeech] evidence archive: {ARCHIVE_PATH}", flush=True)


def main() -> int:
    if Path.cwd() != Path("/content"):
        raise RuntimeError("This LibriSpeech workflow must run inside a Colab runtime rooted at /content")

    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    EVIDENCE_DIR.mkdir(parents=True)
    DOWNLOAD_DIR.mkdir(parents=True)

    commit: str | None = None
    manifest: dict[str, object] | None = None
    error: str | None = None
    try:
        _run("clone", ["git", "clone", "--filter=blob:none", REPO_URL, str(REPO_DIR)])
        _run("checkout", ["git", "checkout", GIT_REF], cwd=REPO_DIR)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        _run("install-uv", [sys.executable, "-m", "pip", "install", "uv"])
        _run(
            "install-project",
            ["uv", "pip", "install", "--system", "-e", f"{REPO_DIR}[lavasr,download]"],
        )

        spec = _load_spec()
        source = spec["source"]
        if not isinstance(source, dict):
            raise ValueError("LibriSpeech source specification must be an object")
        source_archive = DOWNLOAD_DIR / "dev-clean.tar.gz"
        _run(
            "download-librispeech",
            ["curl", "-L", "--fail", "--retry", "3", "-o", str(source_archive), str(source["url"])],
        )
        _verify_md5(source_archive, str(source["archive_md5"]))
        source_root = _extract_archive(source_archive)
        manifest = _build_evalset(spec, source_root)
        _run(
            "validate-evalset",
            ["audio-super-res", "eval", "validate-dataset", "--manifest", str(EVALSET_DIR / "manifest.json")],
            cwd=REPO_DIR,
        )
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
            cwd=REPO_DIR,
        )
        _run(
            "sinc-matrix",
            _matrix_command(backend="sinc-resample", output_dir=EVIDENCE_DIR / "sinc-matrix"),
            cwd=REPO_DIR,
        )
        _run(
            "lavasr-matrix",
            _matrix_command(backend="lavasr-compat", output_dir=EVIDENCE_DIR / "lavasr-matrix"),
            cwd=REPO_DIR,
        )
        for name in ("sinc", "lavasr"):
            _run(
                f"{name}-report",
                [
                    "audio-super-res",
                    "eval",
                    "report",
                    "--manifest",
                    str(EVIDENCE_DIR / f"{name}-matrix" / "matrix.json"),
                    "--output",
                    str(EVIDENCE_DIR / f"{name}-report.md"),
                ],
                cwd=REPO_DIR,
            )
        _write_summary(status="passed", commit=commit, manifest=manifest, error=None)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_summary(status="failed", commit=commit, manifest=manifest, error=error)
    finally:
        _archive()

    if error:
        print(f"[colab-librispeech] failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
