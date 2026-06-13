from __future__ import annotations

from dataclasses import dataclass

from .audiosr_backend import AUDIOSR_MODEL_NAMES, AUDIOSR_SAMPLE_RATE
from .resolver import available_backends


@dataclass(frozen=True)
class ModelInfo:
    """User-facing metadata for a selectable enhancement model."""

    id: str
    backend: str
    name: str
    description: str
    target_sample_rate: int | None
    installed: bool
    package_extra: str | None = None
    model_name: str | None = None


def list_models(filter_text: str | None = None) -> list[ModelInfo]:
    """Return known enhancement models, optionally filtered by text."""

    backend_map = {backend.name: backend for backend in available_backends()}
    models = [
        ModelInfo(
            id="sinc-resample",
            backend="sinc-resample",
            name="Sinc Resample Baseline",
            description="Deterministic polyphase sinc resampling baseline.",
            target_sample_rate=None,
            installed=backend_map["sinc-resample"].installed,
            package_extra=backend_map["sinc-resample"].package_extra,
        )
    ]

    audiosr_backend = backend_map["audiosr"]
    models.extend(
        ModelInfo(
            id=f"audiosr-{model_name}",
            backend="audiosr",
            model_name=model_name,
            name=f"AudioSR {model_name.title()}",
            description=f"AudioSR latent diffusion {model_name} model.",
            target_sample_rate=AUDIOSR_SAMPLE_RATE,
            installed=audiosr_backend.installed,
            package_extra=audiosr_backend.package_extra,
        )
        for model_name in AUDIOSR_MODEL_NAMES
    )

    if filter_text is None:
        return models

    needle = filter_text.lower()
    return [
        model
        for model in models
        if needle in model.id.lower()
        or needle in model.backend.lower()
        or needle in model.name.lower()
        or needle in model.description.lower()
    ]
