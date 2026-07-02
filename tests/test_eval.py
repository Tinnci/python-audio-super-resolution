import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_super_resolution.cli import main
from audio_super_resolution.evaluation import (
    compare_eval_manifests,
    compare_eval_matrices,
    degrade_audio,
    init_failure_case_evalset,
    init_speech_bwe_evalset,
    inspect_torch_checkpoint,
    load_threshold_policy,
    run_downstream_eval,
    run_eval_dataset,
    run_eval_matrix,
    run_listening_export,
    run_no_reference_eval,
    transcript_error_rates,
    validate_eval_dataset_manifest,
    write_eval_report,
)
from audio_super_resolution.resolver import EnhancementResult


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
    assert manifest["passed"] is True
    assert manifest["backend"] == "sinc-resample"
    assert manifest["backend_profile"]["capabilities"]["offline"] is True
    assert manifest["backend_profile"]["capabilities"]["cpu_only"] is True
    assert manifest["backend_profile"]["governance"]["license_usable"] is True
    assert manifest["backend_profile"]["governance"]["explicit_weights"] is True
    assert manifest["backend_profile"]["dependency_footprint"]["dependency_tier"] == "baseline-no-weights"
    assert manifest["degrader"]["name"] == "wideband_16k"
    assert manifest["status_counts"] == {"passed": 1}
    assert manifest["results"][0]["status"] == "passed"
    assert manifest["results"][0]["reference_path"] == str(dataset / "sample.wav")
    assert Path(manifest["results"][0]["degraded_path"]).is_file()
    assert Path(manifest["results"][0]["enhanced_path"]).is_file()
    metrics = manifest["results"][0]["metrics"]
    assert {"si_sdr_db", "sdr_db", "lsd_db", "spectral_convergence", "highband_lsd_8_16k"} <= set(metrics)
    assert manifest["results"][0]["quality"]["passed"] is True
    assert manifest["results"][0]["stability"]["sample_rate_correct"] is True
    assert manifest["results"][0]["stability"]["clipped"] is False
    assert manifest["results"][0]["failure_cases"] == []
    assert manifest["results"][0]["performance"]["rtf"] is not None
    assert manifest["results"][0]["performance"]["load_time_seconds"] >= 0
    assert (
        manifest["results"][0]["performance"]["total_elapsed_seconds"]
        >= manifest["results"][0]["performance"]["elapsed_seconds"]
    )
    assert manifest["results"][0]["performance"]["memory"]["strategy"] == "resource.getrusage(RUSAGE_SELF).ru_maxrss"
    assert "peak_rss_mb" in manifest["results"][0]["performance"]
    assert manifest["planned_adapters"]["full_reference"][0]["name"] == "pesq"


def test_compare_eval_manifests_reports_metric_deltas() -> None:
    baseline = _eval_manifest("sinc-resample", si_sdr=10.0, lsd=3.0, rtf=1.0)
    candidate = _eval_manifest("lavasr-compat", si_sdr=12.5, lsd=2.2, rtf=0.8)

    comparison = compare_eval_manifests(baseline, candidate)

    assert comparison["schema_version"] == 1
    assert comparison["passed"] is True
    assert comparison["metric_summary"]["si_sdr_db"]["delta"] == 2.5
    assert comparison["metric_summary"]["lsd_db"]["delta"] == pytest.approx(-0.8)
    assert comparison["metric_summary"]["lsd_db"]["direction"] == "lower_is_better"
    assert "lsd_db" in comparison["tables"]["audio_quality"]
    assert "rtf" in comparison["tables"]["engineering"]
    assert comparison["tables"]["stability"]["candidate_status_counts"] == {"passed": 1}
    assert comparison["regressions"] == []


def test_compare_eval_manifests_applies_threshold_directions() -> None:
    baseline = _eval_manifest("sinc-resample", si_sdr=10.0, lsd=3.0, rtf=1.0)
    candidate = _eval_manifest("lavasr-compat", si_sdr=9.4, lsd=3.3, rtf=1.4)

    comparison = compare_eval_manifests(
        baseline,
        candidate,
        thresholds={"si_sdr_db": 0.5, "lsd_db": 0.2, "rtf": 0.2},
    )

    assert comparison["passed"] is False
    regressions = {(regression["field"], regression["direction"]) for regression in comparison["regressions"]}
    assert ("si_sdr_db", "higher_is_better") in regressions
    assert ("lsd_db", "lower_is_better") in regressions
    assert ("rtf", "lower_is_better") in regressions


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
    assert "audio_quality" in comparison["tables"]


def test_eval_run_records_lightweight_mcd_optional_metric(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    output_path = tmp_path / "runs" / "sinc.json"

    manifest = run_eval_dataset(
        dataset_dir=dataset,
        backend="sinc-resample",
        output_path=output_path,
        work_dir=tmp_path / "work",
        optional_metrics=("mcd",),
    )

    assert manifest["optional_metrics_requested"] == ["mcd"]
    optional_records = manifest["results"][0]["optional_metric_records"]
    assert optional_records[0]["name"] == "mcd"
    assert optional_records[0]["status"] == "passed"
    assert isinstance(optional_records[0]["score"], float)
    assert "mcd" in manifest["results"][0]["metrics"]


def test_cli_eval_run_accepts_optional_metric(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    output_path = tmp_path / "runs" / "sinc.json"

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
                str(output_path),
                "--work-dir",
                str(tmp_path / "work"),
                "--optional-metric",
                "mcd",
            ]
        )
        == 0
    )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["optional_metrics_requested"] == ["mcd"]
    assert manifest["results"][0]["optional_metric_records"][0]["status"] == "passed"


def test_run_eval_matrix_writes_index_and_run_manifests(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")

    matrix = run_eval_matrix(
        dataset_dir=dataset,
        output_dir=tmp_path / "matrix",
        backends=("sinc-resample",),
        degraders=("lowpass_4k", "wideband_16k"),
        optional_metrics=("mcd",),
    )

    assert matrix["evaluation_type"] == "matrix"
    assert matrix["passed"] is True
    assert matrix["run_count"] == 2
    assert matrix["backends"] == ["sinc-resample"]
    assert matrix["degraders"] == ["lowpass_4k", "wideband_16k"]
    assert matrix["optional_metrics_requested"] == ["mcd"]
    assert (tmp_path / "matrix" / "matrix.json").is_file()
    for run in matrix["runs"]:
        run_manifest_path = Path(run["manifest_path"])
        assert run_manifest_path.is_file()
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        assert run_manifest["results"][0]["optional_metric_records"][0]["status"] == "passed"


def test_run_eval_matrix_can_reuse_existing_manifests(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    output_dir = tmp_path / "matrix"

    first = run_eval_matrix(
        dataset_dir=dataset,
        output_dir=output_dir,
        backends=("sinc-resample",),
        degraders=("lowpass_4k",),
    )
    second = run_eval_matrix(
        dataset_dir=dataset,
        output_dir=output_dir,
        backends=("sinc-resample",),
        degraders=("lowpass_4k",),
        reuse_existing=True,
    )

    assert first["run_count"] == second["run_count"] == 1
    assert second["runs"][0]["reused"] is True


def test_init_failure_case_evalset_and_checkpoint_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evalset = init_failure_case_evalset(output_dir=tmp_path / "failure-cases")
    assert evalset["dataset_id"] == "speech_bwe_failure_cases_v1"
    assert evalset["record_count"] == 5
    assert (tmp_path / "failure-cases" / "speech_clean_48k" / "silence.wav").is_file()

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a checkpoint")

    def missing_torch(name: str):
        if name == "torch":
            raise ImportError("missing torch")
        raise AssertionError(name)

    monkeypatch.setattr("audio_super_resolution.evaluation.import_module", missing_torch)
    with pytest.raises(RuntimeError, match="torch is required"):
        inspect_torch_checkpoint(checkpoint)


def test_compare_eval_matrices_writes_run_comparisons(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    baseline = run_eval_matrix(
        dataset_dir=dataset,
        output_dir=tmp_path / "baseline",
        backends=("sinc-resample",),
        degraders=("lowpass_4k",),
    )
    candidate = run_eval_matrix(
        dataset_dir=dataset,
        output_dir=tmp_path / "candidate",
        backends=("sinc-resample",),
        degraders=("lowpass_4k",),
    )

    comparison = compare_eval_matrices(
        tmp_path / "baseline" / "matrix.json",
        tmp_path / "candidate" / "matrix.json",
        thresholds={"rtf": 100.0},
    )

    assert baseline["run_count"] == candidate["run_count"] == 1
    assert comparison["evaluation_type"] == "matrix_compare"
    assert comparison["passed"] is True
    assert comparison["run_count"] == 1
    assert comparison["comparisons"][0]["matrix_key"] == "sinc-resample::lowpass_4k"


def test_threshold_policy_report_bundle_and_dataset_validation(tmp_path: Path) -> None:
    dataset = tmp_path / "evalset"
    manifest = init_speech_bwe_evalset(output_dir=dataset, count=2, duration_seconds=0.03)
    validation = validate_eval_dataset_manifest(dataset / "manifest.json")
    assert validation["passed"] is True
    assert validation["record_count"] == 2

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"thresholds": {"rtf": 0.25, "mcd": 1.0}}), encoding="utf-8")
    assert load_threshold_policy(policy_path) == {"rtf": 0.25, "mcd": 1.0}

    matrix = run_eval_matrix(
        dataset_dir=dataset / manifest["reference_dir"],
        output_dir=tmp_path / "matrix",
        backends=("sinc-resample",),
        degraders=("lowpass_4k",),
    )
    report_path = tmp_path / "report.md"
    report = write_eval_report(tmp_path / "matrix" / "matrix.json", report_path)
    assert "# Audio Super Resolution Eval Report" in report
    assert report_path.is_file()

    from audio_super_resolution.evaluation import bundle_eval_artifacts

    bundle = bundle_eval_artifacts(
        manifest_paths=[tmp_path / "matrix" / "matrix.json"],
        output_dir=tmp_path / "bundle",
        archive_path=tmp_path / "bundle.tar.gz",
    )
    assert matrix["run_count"] == 1
    assert bundle["artifact_count"] == 2
    assert (tmp_path / "bundle.tar.gz").is_file()


def test_init_speech_bwe_evalset_writes_tiny_reference_set(tmp_path: Path) -> None:
    evalset_dir = tmp_path / "evalsets" / "speech_bwe_v1"

    manifest = init_speech_bwe_evalset(output_dir=evalset_dir, count=6, duration_seconds=0.04)

    assert manifest["dataset_id"] == "speech_bwe_v1_tiny"
    assert manifest["record_count"] == 6
    assert (evalset_dir / "manifest.json").is_file()
    assert len(list((evalset_dir / "speech_clean_48k").glob("*.wav"))) == 6
    assert {record["language"] for record in manifest["records"]} == {"zh", "en"}
    first = sf.info(evalset_dir / manifest["records"][0]["path"])
    assert first.samplerate == 48000


def test_new_eval_degraders_are_deterministic_and_lightweight() -> None:
    sample_rate = 48000
    audio = np.zeros((4800, 1), dtype=np.float32)
    audio[:, 0] = 0.25 * np.sin(2 * np.pi * 440 * np.arange(4800) / sample_rate)

    opus_like = degrade_audio(audio, sample_rate, "opus_16k_24kbps")
    mp3_like = degrade_audio(audio, sample_rate, "mp3_32kbps")

    assert opus_like.sample_rate == 16000
    assert opus_like.recipe["codec"] == "opus-like"
    assert mp3_like.sample_rate == 48000
    assert mp3_like.recipe["codec"] == "mp3-like"


def test_cli_eval_compare_thresholds(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    comparison_path = tmp_path / "comparison.json"
    baseline_path.write_text(json.dumps(_eval_manifest("sinc-resample", si_sdr=10.0, lsd=3.0)), encoding="utf-8")
    candidate_path.write_text(json.dumps(_eval_manifest("sinc-resample", si_sdr=9.0, lsd=3.0)), encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "compare",
                str(baseline_path),
                str(candidate_path),
                "--threshold",
                "si_sdr_db=0.5",
                "--output",
                str(comparison_path),
            ]
        )
        == 1
    )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["passed"] is False
    assert comparison["thresholds"] == {"si_sdr_db": 0.5}
    assert comparison["regressions"][0]["field"] == "si_sdr_db"


def test_cli_eval_run_from_console_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    output_path = tmp_path / "sinc.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audio-super-res",
            "eval",
            "run",
            "--dataset",
            str(dataset),
            "--backend",
            "sinc-resample",
            "--output",
            str(output_path),
            "--work-dir",
            str(tmp_path / "work"),
        ],
    )

    assert main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["results"]


def test_run_no_reference_eval_writes_signal_stats_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    output_path = tmp_path / "runs" / "no-reference.json"

    manifest = run_no_reference_eval(input_path=dataset, output_path=output_path)

    assert output_path.is_file()
    assert manifest["evaluation_type"] == "no_reference"
    assert manifest["passed"] is True
    assert manifest["evaluator"]["name"] == "signal-stats"
    assert manifest["evaluator"]["absolute_truth"] is False
    assert manifest["planned_adapters"][0]["name"] == "dnsmos"
    record = manifest["records"][0]
    assert record["status"] == "passed"
    assert record["scores"]["rms_dbfs"] < 0
    assert 0 <= record["scores"]["clipped_fraction"] <= 1
    assert record["metadata"]["sample_rate"] == 48000

    comparison = compare_eval_manifests(manifest, manifest)
    assert comparison["passed"] is True
    assert "rms_dbfs" in comparison["tables"]["no_reference"]


def test_run_no_reference_eval_rejects_gated_heavy_evaluator(tmp_path: Path) -> None:
    sample = tmp_path / "sample.wav"
    _write_reference(sample)

    with pytest.raises(ValueError, match="Heavy no-reference evaluators must remain opt-in"):
        run_no_reference_eval(input_path=sample, output_path=tmp_path / "dnsmos.json", evaluator="dnsmos")


def test_cli_eval_no_reference(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.wav"
    output_path = tmp_path / "no-reference.json"
    _write_reference(input_path)

    assert main(["eval", "no-reference", "--input", str(input_path), "--output", str(output_path)]) == 0

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["evaluation_type"] == "no_reference"
    assert manifest["results"][0]["evaluator"]["name"] == "signal-stats"
    assert "silence_fraction" in manifest["results"][0]["scores"]


def test_transcript_error_rates_compute_wer_and_cer() -> None:
    rates = transcript_error_rates("hello world", "hello word")

    assert rates["wer"] == 0.5
    assert rates["cer"] == pytest.approx(1 / 10)


def test_run_downstream_eval_writes_transcript_manifest(tmp_path: Path) -> None:
    dataset_path = tmp_path / "downstream.json"
    output_path = tmp_path / "runs" / "downstream.json"
    dataset_path.write_text(
        json.dumps(
            {
                "dataset_id": "speech_bwe_tiny",
                "records": [
                    {
                        "id": "sample",
                        "reference_transcript": "hello world",
                        "baseline_transcript": "hello word",
                        "enhanced_transcript": "hello world",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = run_downstream_eval(dataset_path=dataset_path, output_path=output_path)

    assert output_path.is_file()
    assert manifest["evaluation_type"] == "downstream"
    assert manifest["dataset_id"] == "speech_bwe_tiny"
    assert manifest["evaluator"]["name"] == "transcript-error-rate"
    assert manifest["planned_adapters"][0]["name"] == "speaker-similarity"
    record = manifest["records"][0]
    assert record["baseline_input_score"]["wer"] == 0.5
    assert record["enhanced_score"]["wer"] == 0.0
    assert record["delta"]["wer"] == -0.5
    assert record["scores"]["wer"] == 0.0

    comparison = compare_eval_manifests(manifest, manifest)
    assert comparison["passed"] is True
    assert "wer" in comparison["tables"]["downstream"]


def test_run_downstream_eval_rejects_gated_heavy_evaluator(tmp_path: Path) -> None:
    dataset_path = tmp_path / "downstream.json"
    dataset_path.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="Downstream models must remain opt-in"):
        run_downstream_eval(
            dataset_path=dataset_path,
            output_path=tmp_path / "speaker.json",
            evaluator="speaker-similarity",
        )


def test_cli_eval_downstream(tmp_path: Path) -> None:
    dataset_path = tmp_path / "downstream.json"
    output_path = tmp_path / "downstream-output.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "reference_transcript": "hello world",
                    "baseline_transcript": "hello word",
                    "enhanced_transcript": "hello world",
                }
            ]
        ),
        encoding="utf-8",
    )

    assert main(["eval", "downstream", "--dataset", str(dataset_path), "--output", str(output_path)]) == 0

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["evaluation_type"] == "downstream"
    assert manifest["results"][0]["delta"]["wer"] == -0.5


def test_run_listening_export_writes_blind_bundle(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    eval_manifest_path = tmp_path / "runs" / "sinc.json"
    run_eval_dataset(
        dataset_dir=dataset,
        backend="sinc-resample",
        output_path=eval_manifest_path,
        work_dir=tmp_path / "work",
    )

    bundle = run_listening_export(
        manifest_paths=[eval_manifest_path],
        output_dir=tmp_path / "listening",
        protocol="mushra",
        seed=123,
    )

    public_manifest = bundle["manifest"]
    answer_key = bundle["answer_key"]
    assert Path(bundle["manifest_path"]).is_file()
    assert Path(bundle["answer_key_path"]).is_file()
    assert public_manifest["evaluation_type"] == "listening_export"
    assert public_manifest["answer_key_external"] is True
    assert "clarity" in public_manifest["rating_dimensions"]
    assert len(public_manifest["trials"][0]["stimuli"]) == 3
    assert "backend" not in public_manifest["trials"][0]["stimuli"][0]
    assert {stimulus["role"] for stimulus in answer_key["trials"][0]["stimuli"]} == {
        "anchor",
        "reference",
        "system",
    }
    for stimulus in public_manifest["trials"][0]["stimuli"]:
        assert Path(stimulus["path"]).is_file()


def test_cli_eval_listening_export(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")
    eval_manifest_path = tmp_path / "runs" / "sinc.json"
    output_dir = tmp_path / "listening"
    run_eval_dataset(
        dataset_dir=dataset,
        backend="sinc-resample",
        output_path=eval_manifest_path,
        work_dir=tmp_path / "work",
    )

    assert (
        main(
            [
                "eval",
                "listening-export",
                "--manifest",
                str(eval_manifest_path),
                "--output-dir",
                str(output_dir),
                "--protocol",
                "abx",
                "--seed",
                "7",
            ]
        )
        == 0
    )

    manifest = json.loads((output_dir / "listening_manifest.json").read_text(encoding="utf-8"))
    answer_key = json.loads((output_dir / "answer_key.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "abx"
    assert manifest["seed"] == 7
    assert answer_key["trials"][0]["stimuli"][0]["blind_id"].startswith("t001_s")


def test_run_eval_dataset_records_backend_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_reference(dataset / "sample.wav")

    class FailingResolver:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def enhance(self, input_path: Path, output_path: Path, target_sr: int) -> EnhancementResult:
            raise RuntimeError("backend exploded")

    monkeypatch.setattr("audio_super_resolution.evaluation.AudioSuperResolver", FailingResolver)

    manifest = run_eval_dataset(
        dataset_dir=dataset,
        backend="sinc-resample",
        output_path=tmp_path / "runs" / "failed.json",
        work_dir=tmp_path / "work",
    )

    result = manifest["results"][0]
    assert manifest["passed"] is False
    assert manifest["status_counts"] == {"failed": 1}
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "enhance"
    assert result["failure"]["type"] == "RuntimeError"
    assert result["stability"]["failure_status"] == "enhance_failed"
    assert result["stability"]["failure_case_classification"]["backend_failure"] is True
    assert "output_missing" in result["failure_cases"]

    comparison = compare_eval_manifests(_eval_manifest("sinc-resample", si_sdr=10.0, lsd=3.0), manifest)
    assert comparison["passed"] is False
    assert any(regression["field"] == "status" for regression in comparison["regressions"])


def test_run_eval_dataset_classifies_silence_hallucination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_silence(dataset / "sample.wav")

    class LoudResolver:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def enhance(self, input_path: Path, output_path: Path, target_sr: int) -> EnhancementResult:
            input_info = sf.info(input_path)
            duration_seconds = input_info.frames / input_info.samplerate
            output_audio = np.full(int(target_sr * duration_seconds), 0.2, dtype=np.float32)
            sf.write(output_path, output_audio, target_sr)
            return EnhancementResult(
                input_path=Path(input_path),
                output_path=Path(output_path),
                input_sample_rate=input_info.samplerate,
                sample_rate=target_sr,
                input_duration_seconds=duration_seconds,
                duration_seconds=duration_seconds,
                channels=1,
                backend="sinc-resample",
            )

    monkeypatch.setattr("audio_super_resolution.evaluation.AudioSuperResolver", LoudResolver)

    manifest = run_eval_dataset(
        dataset_dir=dataset,
        backend="sinc-resample",
        output_path=tmp_path / "runs" / "hallucination.json",
        work_dir=tmp_path / "work",
    )

    result = manifest["results"][0]
    assert manifest["passed"] is False
    assert result["status"] == "stability_failed"
    assert result["stability"]["failure_case_classification"]["silence_hallucination"] is True
    assert "silence_hallucination" in result["failure_cases"]


def _write_reference(path: Path, *, sample_rate: int = 48000) -> None:
    time = np.arange(int(sample_rate * 0.06)) / sample_rate
    audio = 0.15 * np.sin(2 * np.pi * 440 * time) + 0.04 * np.sin(2 * np.pi * 9000 * time)
    sf.write(path, audio.astype("float32"), sample_rate)


def _write_silence(path: Path, *, sample_rate: int = 48000) -> None:
    sf.write(path, np.zeros(int(sample_rate * 0.06), dtype=np.float32), sample_rate)


def _eval_manifest(backend: str, *, si_sdr: float, lsd: float, rtf: float = 1.0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": backend,
        "backend_profile": {
            "backend": backend,
            "model_id": backend,
            "capabilities": {
                "offline": True,
                "reproducible": True,
            },
            "governance": {
                "license_usable": True,
                "explicit_weights": True,
            },
            "dependency_footprint": {
                "dependency_tier": "baseline-no-weights",
            },
        },
        "results": [
            {
                "id": "sample",
                "metrics": {
                    "si_sdr_db": si_sdr,
                    "lsd_db": lsd,
                },
                "performance": {
                    "rtf": rtf,
                },
                "quality": {"passed": True},
                "stability": {"passed": True},
                "status": "passed",
            }
        ],
    }
