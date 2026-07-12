"""Generate precomputed ASR downstream evidence in a Colab GPU runtime.

Run this after ``colab_librispeech_eval.py`` in the same session. The external
Whisper model is used only to create transcript JSON; the package itself keeps
its lightweight transcript-only downstream evaluator and gains no ASR dependency.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

WORK_ROOT = Path(os.environ.get("ASR_LIBRISPEECH_ROOT", "/content/audio-super-resolution-librispeech"))
REPO_DIR = WORK_ROOT / "repo"
EVIDENCE_DIR = WORK_ROOT / "evidence"
EVALSET_DIR = EVIDENCE_DIR / "librispeech-dev-clean-tiny-v1"
MODEL_SPEC_PATH = REPO_DIR / "examples" / "artifacts" / "asr-evaluator-whisper-tiny-en.json"
ARCHIVE_PATH = Path(os.environ.get("ASR_DOWNSTREAM_ARCHIVE", "/content/asr-downstream-evidence.tar.gz"))


def _run(name: str, command: list[str]) -> None:
    log_path = EVIDENCE_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[colab-asr] {name}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=REPO_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _normalize(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9' ]+", " ", text.casefold())
    return " ".join(normalized.split())


def _load_run(backend: str) -> dict[str, dict[str, object]]:
    run_id = f"{backend}__wideband_16k"
    path = EVIDENCE_DIR / f"{'sinc' if backend == 'sinc-resample' else 'lavasr'}-matrix" / "runs" / f"{run_id}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["id"]): record for record in manifest["results"]}


def _load_audio(path: str, *, target_sample_rate: int):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sample_rate != target_sample_rate:
        divisor = np.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(audio, target_sample_rate // divisor, sample_rate // divisor).astype(np.float32)
    return audio


def _transcribe_paths(paths: list[str], model_spec: dict[str, object]) -> dict[str, str]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_info = model_spec["model"]
    runtime = model_spec["runtime"]
    if not isinstance(model_info, dict) or not isinstance(runtime, dict):
        raise ValueError("ASR evaluator model and runtime specifications must be objects")
    model_id = str(model_info["id"])
    revision = str(model_info["revision"])
    sample_rate = int(runtime["input_sample_rate"])

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to("cuda")
    model.eval()

    transcripts: dict[str, str] = {}
    for index, path in enumerate(paths, start=1):
        print(f"[colab-asr] transcribing {index}/{len(paths)}: {path}", flush=True)
        audio = _load_audio(path, target_sample_rate=sample_rate)
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            predicted_ids = model.generate(input_features)
        transcripts[path] = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return transcripts


def _downstream_records(
    *,
    references: dict[str, dict[str, object]],
    baseline_results: dict[str, dict[str, object]],
    enhanced_results: dict[str, dict[str, object]],
    transcripts: dict[str, str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item_id, reference in references.items():
        baseline_path = str(baseline_results[item_id]["degraded_path"])
        enhanced_path = str(enhanced_results[item_id]["enhanced_path"])
        records.append(
            {
                "id": item_id,
                "reference_transcript": _normalize(str(reference["transcript"])),
                "baseline_transcript": _normalize(transcripts[baseline_path]),
                "enhanced_transcript": _normalize(transcripts[enhanced_path]),
                "input_path": baseline_path,
                "enhanced_path": enhanced_path,
                "raw_reference_transcript": reference["transcript"],
                "raw_baseline_transcript": transcripts[baseline_path],
                "raw_enhanced_transcript": transcripts[enhanced_path],
            }
        )
    return records


def _aggregate(path: Path) -> dict[str, float]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    fields = ("baseline_wer", "wer", "wer_delta", "baseline_cer", "cer", "cer_delta")
    return {
        field: sum(float(record["scores"][field]) for record in manifest["records"]) / len(manifest["records"])
        for field in fields
    }


def _archive() -> None:
    with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
        archive.add(EVIDENCE_DIR, arcname="evidence")
    print(f"[colab-asr] evidence archive: {ARCHIVE_PATH}", flush=True)


def main() -> int:
    if Path.cwd() != Path("/content"):
        raise RuntimeError("This ASR workflow must run inside a Colab runtime rooted at /content")
    if not (EVALSET_DIR / "manifest.json").is_file():
        raise FileNotFoundError("Run examples/colab_librispeech_eval.py in this session before downstream ASR")

    error: str | None = None
    try:
        model_spec = json.loads(MODEL_SPEC_PATH.read_text(encoding="utf-8"))
        runtime = model_spec["runtime"]
        if not isinstance(runtime, dict):
            raise ValueError("ASR evaluator runtime specification must be an object")
        _run(
            "install-asr-evaluator",
            [
                "uv",
                "pip",
                "install",
                "--system",
                f"transformers=={runtime['transformers_version']}",
                "accelerate",
            ],
        )

        dataset_manifest = json.loads((EVALSET_DIR / "manifest.json").read_text(encoding="utf-8"))
        references = {str(record["id"]): record for record in dataset_manifest["records"]}
        sinc_results = _load_run("sinc-resample")
        lavasr_results = _load_run("lavasr-compat")
        paths = sorted(
            {
                *(str(record["degraded_path"]) for record in sinc_results.values()),
                *(str(record["enhanced_path"]) for record in sinc_results.values()),
                *(str(record["enhanced_path"]) for record in lavasr_results.values()),
            }
        )
        transcripts = _transcribe_paths(paths, model_spec)
        precomputed_path = EVIDENCE_DIR / "asr-precomputed-transcripts.json"
        precomputed_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "dataset_id": dataset_manifest["dataset_id"],
                    "evaluator": model_spec,
                    "transcripts": transcripts,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        downstream_paths: dict[str, Path] = {}
        for name, enhanced_results in (("sinc", sinc_results), ("lavasr", lavasr_results)):
            dataset_path = EVIDENCE_DIR / f"asr-{name}-dataset.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_id": f"{dataset_manifest['dataset_id']}-{name}-asr",
                        "records": _downstream_records(
                            references=references,
                            baseline_results=sinc_results,
                            enhanced_results=enhanced_results,
                            transcripts=transcripts,
                        ),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = EVIDENCE_DIR / f"asr-{name}-downstream.json"
            _run(
                f"evaluate-asr-{name}",
                [
                    "audio-super-res",
                    "eval",
                    "downstream",
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(output_path),
                ],
            )
            downstream_paths[name] = output_path

        summary = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "dataset_id": dataset_manifest["dataset_id"],
            "record_count": len(references),
            "evaluator": model_spec,
            "conditions": {name: _aggregate(path) for name, path in downstream_paths.items()},
            "error": None,
        }
        (EVIDENCE_DIR / "asr-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (EVIDENCE_DIR / "asr-summary.json").write_text(
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
        _archive()

    if error:
        print(f"[colab-asr] failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
