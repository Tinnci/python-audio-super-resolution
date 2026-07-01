from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightFileSpec:
    """Static metadata for one expected model weight file."""

    path: str
    sha256: str | None = None
    size: int | None = None


@dataclass(frozen=True)
class BackendCapability:
    """Capabilities that affect backend selection and CLI reporting."""

    supports_array_io: bool
    supports_file_io: bool
    supports_chunking: bool
    deterministic: bool
    supports_cpu: bool
    supports_cuda: bool = False
    supports_mps: bool = False
    requires_gpu: bool = False
    precision_modes: tuple[str, ...] = ("float32",)


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for a model exposed by a backend."""

    id: str
    backend: str
    name: str
    description: str
    implementation: str
    domain: tuple[str, ...]
    target_sample_rates: tuple[int, ...] | None
    tasks: tuple[str, ...] = ()
    input_sample_rates: tuple[int, ...] | None = None
    input_sample_rate_range: tuple[int, int] | None = None
    architecture: str | None = None
    model_name: str | None = None
    code_license: str | None = None
    weights_license: str | None = None
    weights_source: str | None = None
    weights_hash: str | None = None
    weight_provider: str | None = None
    weight_files: tuple[WeightFileSpec, ...] = ()
    weight_manifest_url: str | None = None
    default_weight_revision: str | None = None
    requires_weights: bool = False
    upstream_url: str | None = None
    recommended_for: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    attribution_required: bool = False
    maturity: str = "stable"
    capability: BackendCapability | None = None

    @property
    def fixed_target_sr(self) -> bool:
        """Return whether the model supports exactly one target sample rate."""

        return self.target_sample_rates is not None and len(self.target_sample_rates) == 1

    @property
    def target_sample_rate(self) -> int | None:
        """Return the single target sample rate for legacy callers, if any."""

        target_sample_rates = self.target_sample_rates
        if target_sample_rates is None or len(target_sample_rates) != 1:
            return None
        return target_sample_rates[0]
