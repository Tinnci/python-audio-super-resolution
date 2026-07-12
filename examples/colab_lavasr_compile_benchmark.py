"""Benchmark LavaSR torch eager versus torch.compile in a Colab T4 runtime."""

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

REPO_URL = os.environ.get("ASR_REPO_URL", "https://github.com/Tinnci/python-audio-super-resolution.git")
GIT_REF = os.environ.get("ASR_GIT_REF", "agent/lavasr-torch-compile-evidence")
WORK_ROOT = Path(os.environ.get("ASR_COLAB_WORK_ROOT", "/content/lavasr-compile-benchmark"))
REPO_DIR = WORK_ROOT / "repo"
CACHE_DIR = WORK_ROOT / "models"
EVIDENCE_DIR = WORK_ROOT / "evidence"
ARCHIVE_PATH = Path(os.environ.get("ASR_COLAB_ARCHIVE", "/content/asr-lavasr-compile-benchmark.tar.gz"))


def run(name: str, command: list[str], *, cwd: Path | None = None) -> None:
    log_path = EVIDENCE_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[compile-benchmark] {name}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True, check=True)


def sha256_array(array) -> str:
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest()


def synchronize(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_provider(provider: str, fixture, *, repeats: int = 3) -> tuple[dict[str, object], object]:
    import numpy as np
    import torch

    from audio_super_resolution import InferenceConfig
    from audio_super_resolution.backends.lavasr_compat import LavaSRCompatBackend

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    config = InferenceConfig(
        device="cuda",
        runtime_provider="torch-eager",
        model_cache_dir=CACHE_DIR,
        weights_manifest=CACHE_DIR / "lavasr-v2-bwe" / "manifest.json",
    )
    started = time.perf_counter()
    backend = LavaSRCompatBackend(config)
    if provider == "torch-compile":
        eager_loader = backend._load_model
        compiled = False

        def load_compiled(*args, **kwargs):
            nonlocal compiled
            model = eager_loader(*args, **kwargs)
            if not compiled:
                model = torch.compile(model, fullgraph=False, dynamic=True)
                backend._model = model
                compiled = True
            return model

        backend._load_model = load_compiled  # type: ignore[method-assign]
    synchronize(torch)
    construction_seconds = time.perf_counter() - started

    started = time.perf_counter()
    first = backend.enhance(fixture, 16_000, 48_000)
    synchronize(torch)
    first_seconds = time.perf_counter() - started

    warm_times = []
    warm_outputs = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = backend.enhance(fixture, 16_000, 48_000)
        synchronize(torch)
        warm_times.append(time.perf_counter() - started)
        warm_outputs.append(output)

    first_array = np.asarray(first, dtype=np.float32)
    exact_repeat = all(np.array_equal(first_array, np.asarray(output)) for output in warm_outputs)
    record = {
        "provider": provider,
        "construction_seconds": construction_seconds,
        "first_seconds": first_seconds,
        "warm_seconds": warm_times,
        "warm_mean_seconds": float(np.mean(warm_times)),
        "warm_median_seconds": float(np.median(warm_times)),
        "output_shape": list(first_array.shape),
        "output_sha256": sha256_array(first_array),
        "same_provider_exact_repeat": exact_repeat,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    return record, first_array


def worker_main() -> int:
    import numpy as np
    import torch

    timeline = np.arange(16_000, dtype=np.float32) / 16_000
    fixture = (0.08 * np.sin(2 * np.pi * 440 * timeline) + 0.02 * np.sin(2 * np.pi * 3000 * timeline)).astype(
        np.float32
    )
    providers = {}
    outputs = {}
    errors = {}
    for provider in ("torch-eager", "torch-compile"):
        try:
            providers[provider], outputs[provider] = benchmark_provider(provider, fixture)
        except Exception as exc:
            errors[provider] = f"{type(exc).__name__}: {exc}"

    comparison = None
    if len(outputs) == 2:
        eager = outputs["torch-eager"]
        compiled = outputs["torch-compile"]
        delta = compiled.astype(np.float64) - eager.astype(np.float64)
        comparison = {
            "shape_equal": eager.shape == compiled.shape,
            "exact_equal": bool(np.array_equal(eager, compiled)),
            "max_abs_error": float(np.max(np.abs(delta))),
            "rms_error": float(np.sqrt(np.mean(delta**2))),
            "eager_rms": float(np.sqrt(np.mean(eager.astype(np.float64) ** 2))),
            "warm_speedup": providers["torch-eager"]["warm_median_seconds"]
            / providers["torch-compile"]["warm_median_seconds"],
        }
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fixture_sample_rate": 16_000,
        "fixture_samples": fixture.size,
        "fixture_sha256": sha256_array(fixture),
        "providers": providers,
        "comparison": comparison,
        "errors": errors,
        "passed": len(outputs) == 2,
    }
    (EVIDENCE_DIR / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


def write_summary(status: str, error: str | None) -> None:
    commit = None
    if REPO_DIR.is_dir():
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True, check=False
        ).stdout.strip()
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "repository": REPO_URL,
        "requested_ref": GIT_REF,
        "commit": commit or None,
        "error": error,
    }
    (EVIDENCE_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def archive() -> None:
    with tarfile.open(ARCHIVE_PATH, "w:gz") as bundle:
        bundle.add(EVIDENCE_DIR, arcname="evidence")
    print(f"[compile-benchmark] evidence archive: {ARCHIVE_PATH}", flush=True)


def main() -> int:
    if "--worker" in sys.argv:
        return worker_main()
    if Path.cwd() != Path("/content"):
        raise RuntimeError("This benchmark must run inside a Colab runtime rooted at /content")
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    EVIDENCE_DIR.mkdir(parents=True)
    try:
        runner_source = Path(__file__).read_text(encoding="utf-8")
    except NameError:
        runner_source = get_ipython().history_manager.input_hist_raw[-1]  # noqa: F821
    runner_path = WORK_ROOT / "colab_lavasr_compile_benchmark.py"
    runner_path.write_text(runner_source, encoding="utf-8")

    error = None
    try:
        run("clone", ["git", "clone", "--filter=blob:none", REPO_URL, str(REPO_DIR)])
        run("checkout", ["git", "checkout", GIT_REF], cwd=REPO_DIR)
        run("install-uv", [sys.executable, "-m", "pip", "install", "uv"])
        run("install-project", ["uv", "pip", "install", "--system", "-e", ".[lavasr,download]"], cwd=REPO_DIR)
        run(
            "prepare-model-cache",
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
        run("benchmark", [sys.executable, str(runner_path), "--worker"], cwd=REPO_DIR)
        write_summary("passed", None)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_summary("failed", error)
    finally:
        archive()
    if error:
        print(f"[compile-benchmark] failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
