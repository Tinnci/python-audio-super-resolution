import json
from pathlib import Path

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


def _load_json(name: str) -> dict[str, object]:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
