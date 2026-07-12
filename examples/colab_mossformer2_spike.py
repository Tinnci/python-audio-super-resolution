"""Run the bounded MossFormer2_SR_48K feasibility spike through Colab CLI.

The script pins upstream source/model revisions, downloads only the three files
believed necessary for inference, then forces Hugging Face offline mode before
constructing or running the model. It writes a downloadable evidence archive.
Never run this script locally; it intentionally requires ``/content``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

SOURCE_REPO = "https://github.com/modelscope/ClearerVoice-Studio.git"
SOURCE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
MODEL_REPO = "alibabasglab/MossFormer2_SR_48K"
MODEL_REVISION = "39eb1f25ea84f5e0315ade9ac0070fff216fc690"
EXPECTED_FILES = {
    "last_best_checkpoint": (52, None),
    "last_best_checkpoint_g.pt": (
        220_712_702,
        "0bdd13c21466f5963d9d1f86a9d84fc6196868318fe22c6b0a750f041805adda",
    ),
    "last_best_checkpoint_m.pt": (
        218_471_889,
        "6cbadb2b6b839e444bb65223c69eea162c8ad08f36e9d0a64144672c4095ab36",
    ),
}
WORK_ROOT = Path(os.environ.get("ASR_COLAB_WORK_ROOT", "/content/mossformer2-spike"))
SOURCE_DIR = WORK_ROOT / "ClearerVoice-Studio"
CLEARVOICE_ROOT = SOURCE_DIR / "clearvoice"
CHECKPOINT_DIR = CLEARVOICE_ROOT / "checkpoints" / "MossFormer2_SR_48K"
EVIDENCE_DIR = WORK_ROOT / "evidence"
ARCHIVE_PATH = Path(os.environ.get("ASR_COLAB_ARCHIVE", "/content/asr-mossformer2-spike.tar.gz"))


def run(name: str, command: list[str], *, cwd: Path | None = None) -> None:
    log = EVIDENCE_DIR / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mossformer2-spike] {name}: {' '.join(command)}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, text=True, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoints() -> list[dict[str, object]]:
    from huggingface_hub import hf_hub_download

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, (expected_size, expected_hash) in EXPECTED_FILES.items():
        cached = Path(hf_hub_download(repo_id=MODEL_REPO, filename=filename, revision=MODEL_REVISION))
        destination = CHECKPOINT_DIR / filename
        shutil.copy2(cached, destination)
        actual_hash = sha256(destination)
        if destination.stat().st_size != expected_size:
            raise RuntimeError(f"unexpected size for {filename}")
        if expected_hash is not None and actual_hash != expected_hash:
            raise RuntimeError(f"unexpected SHA256 for {filename}")
        records.append(
            {
                "filename": filename,
                "size": destination.stat().st_size,
                "sha256": actual_hash,
            }
        )
    (EVIDENCE_DIR / "checkpoints.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def inspect_and_convert() -> dict[str, object]:
    import torch
    from safetensors.torch import load_file, save_file

    result: dict[str, object] = {}
    conversion_dir = EVIDENCE_DIR / "converted"
    conversion_dir.mkdir()
    state_keys = {"m": "mossformer", "g": "generator"}
    for suffix, state_key in state_keys.items():
        source = CHECKPOINT_DIR / f"last_best_checkpoint_{suffix}.pt"
        loaded = torch.load(source, map_location="cpu", weights_only=True)
        state = loaded.get(state_key, loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
            raise RuntimeError(f"checkpoint {suffix} is not a state dict")
        # Break shared storage explicitly. MossFormer repeats one rotary-frequency
        # buffer across layers, which safetensors intentionally refuses to encode
        # through its generic state-dict writer.
        tensors = {key: value.detach().clone().contiguous() for key, value in state.items() if torch.is_tensor(value)}
        if len(tensors) != len(state):
            raise RuntimeError(f"checkpoint {suffix} contains non-tensor state")
        target = conversion_dir / f"mossformer2_{suffix}.safetensors"
        save_file(tensors, target)
        converted = load_file(target, device="cpu")
        if converted.keys() != tensors.keys() or any(not torch.equal(converted[key], tensors[key]) for key in tensors):
            raise RuntimeError(f"safetensors round trip failed for checkpoint {suffix}")
        result[suffix] = {
            "root_keys": sorted(loaded) if isinstance(loaded, dict) else None,
            "selected_state_key": state_key,
            "tensor_count": len(tensors),
            "parameter_count": sum(value.numel() for value in tensors.values()),
            "first_keys": sorted(tensors)[:25],
            "safetensors_size": target.stat().st_size,
            "safetensors_sha256": sha256(target),
            "safetensors_round_trip_exact": True,
        }
        target.unlink()
    (EVIDENCE_DIR / "checkpoint-inspection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_offline_inference() -> dict[str, object]:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    import numpy as np
    import soundfile as sf
    import torch

    sys.path.insert(0, str(SOURCE_DIR / "clearvoice"))
    from clearvoice import ClearVoice

    torch.manual_seed(0)
    source_rate = 16_000
    sample_count = 4_000
    timeline = np.arange(sample_count, dtype=np.float32) / source_rate
    fixture = (0.08 * np.sin(2 * np.pi * 440 * timeline)).astype(np.float32)
    fixture_path = WORK_ROOT / "fixture-16k.wav"
    sf.write(fixture_path, fixture, source_rate, subtype="PCM_16")
    stereo_path = WORK_ROOT / "fixture-16k-stereo.wav"
    sf.write(stereo_path, np.column_stack((fixture, fixture * 0.5)), source_rate, subtype="PCM_16")
    fixture_48k = np.repeat(fixture, 3)
    fixture_48k_path = WORK_ROOT / "fixture-48k.wav"
    sf.write(fixture_48k_path, fixture_48k, 48_000, subtype="PCM_16")

    original_cwd = Path.cwd()
    os.chdir(CLEARVOICE_ROOT)
    try:
        started = time.perf_counter()
        model = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        first = np.asarray(model(input_path=str(fixture_path), online_write=False))
        inference_seconds = time.perf_counter() - started
        second = np.asarray(model(input_path=str(fixture_path), online_write=False))
        stereo = np.asarray(model(input_path=str(stereo_path), online_write=False))
        native_48k = np.asarray(model(input_path=str(fixture_48k_path), online_write=False))
    finally:
        os.chdir(original_cwd)
    output_path = EVIDENCE_DIR / "upstream-output-48k.wav"
    flat = np.squeeze(first)
    sf.write(output_path, flat, 48_000, subtype="FLOAT")
    expected_length = round(sample_count * 48_000 / source_rate)
    record = {
        "device": str(model.models[0].device),
        "input_sample_rate": source_rate,
        "input_channels": 1,
        "input_samples": sample_count,
        "output_contract_sample_rate": 48_000,
        "output_shape": list(first.shape),
        "output_samples": int(flat.size),
        "expected_output_samples": expected_length,
        "alignment_delta_samples": int(flat.size - expected_length),
        "finite": bool(np.isfinite(flat).all()),
        "peak": float(np.max(np.abs(flat))),
        "deterministic_exact": bool(np.array_equal(first, second)),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "output_sha256": sha256(output_path),
        "stereo_16k_output_shape": list(stereo.shape),
        "stereo_channels_preserved": bool(stereo.shape[0] == 2),
        "native_48k_input_samples": int(fixture_48k.size),
        "native_48k_output_shape": list(native_48k.shape),
        "native_48k_alignment_delta_samples": int(np.squeeze(native_48k).size - fixture_48k.size),
    }
    if (
        not record["finite"]
        or abs(record["alignment_delta_samples"]) > 512
        or not record["stereo_channels_preserved"]
        or abs(record["native_48k_alignment_delta_samples"]) > 512
    ):
        raise RuntimeError("upstream inference produced an invalid or materially misaligned output")
    (EVIDENCE_DIR / "inference.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_summary(status: str, error: str | None, **evidence: object) -> None:
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_repository": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "model_repository": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "offline_inference": True,
        "error": error,
        **evidence,
    }
    (EVIDENCE_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def archive() -> None:
    with tarfile.open(ARCHIVE_PATH, "w:gz") as bundle:
        bundle.add(EVIDENCE_DIR, arcname="evidence")
    print(f"[mossformer2-spike] evidence archive: {ARCHIVE_PATH}", flush=True)


def worker_main() -> int:
    error = None
    evidence: dict[str, object] = {}
    try:
        evidence["checkpoints"] = download_checkpoints()
        evidence["inspection"] = inspect_and_convert()
        evidence["inference"] = run_offline_inference()
        write_summary("passed", None, **evidence)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_summary("failed", error, **evidence)
    if error:
        print(f"[mossformer2-spike] failed: {error}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if "--worker" in sys.argv:
        return worker_main()
    if Path.cwd() != Path("/content"):
        raise RuntimeError("This spike must run inside a Colab runtime rooted at /content")
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    EVIDENCE_DIR.mkdir(parents=True)
    try:
        runner_source = Path(__file__).read_text(encoding="utf-8")
    except NameError:
        runner_source = get_ipython().history_manager.input_hist_raw[-1]  # noqa: F821
    runner_path = WORK_ROOT / "colab_mossformer2_spike.py"
    runner_path.write_text(runner_source, encoding="utf-8")
    error = None
    try:
        run("clone", ["git", "clone", "--filter=blob:none", SOURCE_REPO, str(SOURCE_DIR)])
        run("checkout", ["git", "checkout", SOURCE_REVISION], cwd=SOURCE_DIR)
        run("install-uv", [sys.executable, "-m", "pip", "install", "uv"])
        run(
            "install-dependencies",
            [
                "uv",
                "pip",
                "install",
                "--system",
                "-e",
                str(SOURCE_DIR / "clearvoice"),
                "safetensors",
            ],
        )
        run(
            "environment",
            [
                sys.executable,
                "-c",
                "import platform,numpy,torch; "
                "print(platform.python_version(), numpy.__version__, torch.__version__, torch.cuda.is_available())",
            ],
        )
        run("isolated-worker", [sys.executable, str(runner_path), "--worker"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if not (EVIDENCE_DIR / "summary.json").exists():
            write_summary("failed", error)
    finally:
        archive()
    if error:
        print(f"[mossformer2-spike] failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
