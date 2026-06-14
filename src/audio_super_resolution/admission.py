from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .specs import ModelSpec, WeightFileSpec

AdmissionTarget = Literal["catalog", "self-contained"]

UNKNOWN_WEIGHT_LICENSES = {None, "", "unknown", "research-only", "non-commercial"}
SELF_CONTAINED_IMPLEMENTATIONS = {"baseline", "self_torch", "self_onnx"}


@dataclass(frozen=True)
class ModelAdmissionCriterion:
    """One scored model admission criterion."""

    id: str
    title: str
    passed: bool
    required: bool
    points: int
    max_points: int
    detail: str


@dataclass(frozen=True)
class ModelAdmissionReport:
    """Admission report for a model spec."""

    model_id: str
    target: AdmissionTarget
    criteria: tuple[ModelAdmissionCriterion, ...]

    @property
    def passed(self) -> bool:
        return all(criterion.passed or not criterion.required for criterion in self.criteria)

    @property
    def score(self) -> int:
        return sum(criterion.points for criterion in self.criteria)

    @property
    def max_score(self) -> int:
        return sum(criterion.max_points for criterion in self.criteria)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(criterion.id for criterion in self.criteria if criterion.required and not criterion.passed)


def evaluate_model_admission(
    spec: ModelSpec,
    *,
    target: AdmissionTarget = "self-contained",
) -> ModelAdmissionReport:
    """Evaluate whether a model spec is ready for a target admission track."""

    if target not in {"catalog", "self-contained"}:
        raise ValueError("admission target must be 'catalog' or 'self-contained'")

    criteria = (
        _criterion(
            "metadata",
            "Core metadata",
            bool(spec.domain and spec.tasks and spec.architecture),
            True,
            "domain, tasks, and architecture are declared",
            "domain, tasks, and architecture must be declared",
        ),
        _criterion(
            "target-sample-rate",
            "Target sample rate",
            spec.fixed_target_sr,
            target == "self-contained",
            "fixed target sample rate is declared",
            "self-contained candidates need one fixed target sample rate",
        ),
        _criterion(
            "implementation",
            "Implementation family",
            target == "catalog" or spec.implementation in SELF_CONTAINED_IMPLEMENTATIONS,
            target == "self-contained",
            f"implementation {spec.implementation!r} is acceptable for {target}",
            f"implementation {spec.implementation!r} is not self-contained",
        ),
        _criterion(
            "io-capability",
            "I/O capability",
            spec.capability is not None and (spec.capability.supports_array_io or spec.capability.supports_file_io),
            True,
            "array or file I/O capability is declared",
            "array or file I/O capability must be declared",
        ),
        _criterion(
            "cpu-fallback",
            "CPU fallback",
            spec.capability is not None and spec.capability.supports_cpu and not spec.capability.requires_gpu,
            target == "self-contained",
            "CPU fallback is declared",
            "self-contained candidates need a CPU fallback for default validation",
        ),
        _criterion(
            "weights",
            "Weight metadata",
            _has_acceptable_weight_metadata(spec),
            target == "self-contained",
            "weight metadata is complete or no managed weights are required",
            "managed weights need provider/source/license/files with size and SHA256",
        ),
        _criterion(
            "validation",
            "Validation evidence",
            bool(spec.validation),
            target == "self-contained",
            "validation evidence is declared",
            "self-contained candidates need a planned or completed validation path",
        ),
    )
    return ModelAdmissionReport(model_id=spec.id, target=target, criteria=criteria)


def model_admission_report_to_dict(report: ModelAdmissionReport) -> dict[str, object]:
    """Return a JSON-friendly admission report."""

    return {
        "schema_version": 1,
        "model_id": report.model_id,
        "target": report.target,
        "passed": report.passed,
        "score": report.score,
        "max_score": report.max_score,
        "blockers": list(report.blockers),
        "criteria": [
            {
                "id": criterion.id,
                "title": criterion.title,
                "passed": criterion.passed,
                "required": criterion.required,
                "points": criterion.points,
                "max_points": criterion.max_points,
                "detail": criterion.detail,
            }
            for criterion in report.criteria
        ],
    }


def _criterion(
    criterion_id: str,
    title: str,
    passed: bool,
    required: bool,
    passed_detail: str,
    failed_detail: str,
) -> ModelAdmissionCriterion:
    return ModelAdmissionCriterion(
        id=criterion_id,
        title=title,
        passed=passed,
        required=required,
        points=1 if passed else 0,
        max_points=1,
        detail=passed_detail if passed else failed_detail,
    )


def _has_acceptable_weight_metadata(spec: ModelSpec) -> bool:
    if not spec.requires_weights:
        return spec.weights_source is None

    if spec.weight_provider is None or spec.weights_source is None:
        return False
    if spec.weights_license in UNKNOWN_WEIGHT_LICENSES:
        return False
    if not spec.weight_files:
        return False
    return all(_has_verifiable_file_metadata(file_spec) for file_spec in spec.weight_files)


def _has_verifiable_file_metadata(file_spec: WeightFileSpec) -> bool:
    return file_spec.sha256 is not None and file_spec.size is not None
