from __future__ import annotations

import pytest

from audio_super_resolution import (
    evaluate_model_admission,
    get_model_spec,
    model_admission_report_to_dict,
)


def test_lavasr_passes_self_contained_admission() -> None:
    report = evaluate_model_admission(get_model_spec("lavasr-v2-bwe"), target="self-contained")

    assert report.passed
    assert report.score == report.max_score
    assert report.blockers == ()
    assert model_admission_report_to_dict(report)["passed"] is True


def test_external_audiosr_fails_self_contained_admission() -> None:
    report = evaluate_model_admission(get_model_spec("audiosr-basic"), target="self-contained")

    assert not report.passed
    assert "implementation" in report.blockers
    assert "weights" in report.blockers


def test_external_audiosr_can_be_cataloged() -> None:
    report = evaluate_model_admission(get_model_spec("audiosr-basic"), target="catalog")

    assert report.passed
    assert report.score < report.max_score


def test_unknown_admission_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="admission target"):
        evaluate_model_admission(get_model_spec("lavasr-v2-bwe"), target="gpu")  # type: ignore[arg-type]
