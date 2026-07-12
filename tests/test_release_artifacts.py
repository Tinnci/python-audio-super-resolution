import json
from pathlib import Path

from audio_super_resolution.evaluation import load_threshold_policy

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "examples" / "artifacts"


def test_sample_manifests_are_valid_json_artifacts() -> None:
    plan = _load_json("sample-plan-manifest.json")
    completed = _load_json("sample-completed-manifest.json")

    assert plan["schema_version"] == 1
    assert plan["mode"] == "dry-run"
    assert plan["results"] == []
    assert plan["config"]["chunked"] is False

    assert completed["schema_version"] == 1
    assert completed["mode"] == "completed"
    assert completed["results"][0]["sample_rate"] == completed["target_sample_rate"]
    assert completed["quality_reports"][0]["passed"] is True
    assert completed["config"]["chunked"] is True


def test_sample_quality_report_is_valid_json_artifact() -> None:
    report = _load_json("sample-quality-report.json")

    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert report["report_count"] == 1
    assert report["issue_count"] == 0
    assert report["reports"][0]["sample_rate"] == 48000


def test_eval_threshold_policy_is_valid_release_artifact() -> None:
    policy_path = ARTIFACTS / "eval-threshold-policy.json"
    policy = _load_json("eval-threshold-policy.json")

    assert policy["schema_version"] == 1
    assert policy["name"] == "release-regression-v1"
    assert load_threshold_policy(policy_path) == {
        "si_sdr_db": 0.5,
        "sdr_db": 0.5,
        "lsd_db": 0.25,
        "spectral_convergence": 0.05,
        "highband_lsd_4_8k": 0.25,
        "highband_lsd_8_16k": 0.25,
        "mcd": 0.5,
        "duration_drift_seconds": 0.01,
        "clipped_fraction": 0.001,
    }


def test_librispeech_evalset_spec_is_pinned_and_remote_only() -> None:
    spec = _load_json("librispeech-dev-clean-tiny-v1.json")
    source = spec["source"]
    selection = spec["selection"]
    storage_policy = spec["storage_policy"]

    assert spec["dataset_id"] == "librispeech-dev-clean-tiny-v1"
    assert source["subset"] == "dev-clean"
    assert source["archive_md5"] == "42e2234ba48799c1f50f24a7926300a1"
    assert source["license"] == "CC BY 4.0"
    assert selection["speaker_count"] == selection["female_speakers"] + selection["male_speakers"] == 8
    assert selection["target_sample_rate"] == 48000
    assert storage_policy["commit_audio"] is False
    assert storage_policy["run_location"] == "Colab CLI remote runtime"


def test_asr_evaluator_spec_is_pinned_and_not_a_package_dependency() -> None:
    spec = _load_json("asr-evaluator-whisper-tiny-en.json")
    model = spec["model"]
    runtime = spec["runtime"]
    integration_policy = spec["integration_policy"]

    assert spec["evaluator_id"] == "whisper-tiny-en-pinned-v1"
    assert model["id"] == "openai/whisper-tiny.en"
    assert model["revision"] == "87c7102498dcde7456f24cfd30239ca606ed9063"
    assert model["license"] == "Apache-2.0"
    assert runtime["transformers_version"] == "5.13.1"
    assert runtime["device"] == "cuda"
    assert integration_policy["package_dependency"] is False
    assert integration_policy["package_input"] == "Precomputed reference, baseline, and enhanced transcript JSON only"


def _load_json(name: str) -> dict[str, object]:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
