import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_super_resolution.cli import main
from audio_super_resolution.evaluation import compare_eval_manifests, run_eval_dataset


def test_run_eval_dataset_writes_full_reference_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    output_path = tmp_path / "runs" / "sinc.json"

    manifest = run_eval_dataset(
        dataset_dir=dataset,
        backend="sinc-resample",
        output_path=output_path,
        work_dir=tmp_path / "work",
        target_sample_rate=48000,
        degrader="wideband_16k",
    )

    assert output_path.is_file()
    assert manifest["schema_version"] == 1
    assert manifest["backend"] == "sinc-resample"
    assert manifest["degrader"]["name"] == "wideband_16k"
    assert manifest["results"][0]["reference_path"] == str(dataset / "sample.wav")
    assert Path(manifest["results"][0]["degraded_path"]).is_file()
    assert Path(manifest["results"][0]["enhanced_path"]).is_file()
    metrics = manifest["results"][0]["metrics"]
    assert {"si_sdr_db", "sdr_db", "lsd_db", "spectral_convergence", "highband_lsd_8_16k"} <= set(metrics)
    assert manifest["results"][0]["quality"]["passed"] is True
    assert manifest["results"][0]["performance"]["rtf"] is not None


def test_compare_eval_manifests_reports_metric_deltas() -> None:
    baseline = _eval_manifest("sinc-resample", si_sdr=10.0, lsd=3.0)
    candidate = _eval_manifest("lavasr-compat", si_sdr=12.5, lsd=2.2)

    comparison = compare_eval_manifests(baseline, candidate)

    assert comparison["schema_version"] == 1
    assert comparison["passed"] is True
    assert comparison["metric_summary"]["si_sdr_db"]["delta"] == 2.5
    assert comparison["metric_summary"]["lsd_db"]["delta"] == pytest.approx(-0.8)
    assert comparison["regressions"] == []


def test_cli_eval_run_and_compare(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    baseline_path = tmp_path / "sinc.json"
    candidate_path = tmp_path / "sinc2.json"
    comparison_path = tmp_path / "comparison.json"

    assert (
        main(
            [
                "eval",
                "run",
                "--dataset",
                str(dataset),
                "--backend",
                "sinc-resample",
                "--output",
                str(baseline_path),
                "--work-dir",
                str(tmp_path / "work-a"),
                "--degrader",
                "lowpass_4k",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "eval",
                "run",
                "--dataset",
                str(dataset),
                "--backend",
                "sinc-resample",
                "--output",
                str(candidate_path),
                "--work-dir",
                str(tmp_path / "work-b"),
                "--degrader",
                "wideband_16k",
            ]
        )
        == 0
    )
    assert main(["eval", "compare", str(baseline_path), str(candidate_path), "--output", str(comparison_path)]) == 0

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["baseline_backend"] == "sinc-resample"
    assert comparison["candidate_backend"] == "sinc-resample"
    assert "highband_lsd_8_16k" in comparison["metric_summary"]


def _write_reference(path: Path, *, sample_rate: int = 48000) -> None:
    time = np.arange(int(sample_rate * 0.06)) / sample_rate
    audio = 0.15 * np.sin(2 * np.pi * 440 * time) + 0.04 * np.sin(2 * np.pi * 9000 * time)
    sf.write(path, audio.astype("float32"), sample_rate)


def _eval_manifest(backend: str, *, si_sdr: float, lsd: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": backend,
        "results": [
            {
                "id": "sample",
                "metrics": {
                    "si_sdr_db": si_sdr,
                    "lsd_db": lsd,
                },
                "quality": {"passed": True},
            }
        ],
    }
