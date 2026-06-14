from __future__ import annotations

import pickletools
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..weight_store import ResolvedWeights

LAVASR_CONFIG_PATH = "enhancer_v2/config.yaml"
LAVASR_WEIGHTS_PATH = "enhancer_v2/pytorch_model.bin"

FEATURE_EXTRACTOR_CLASS = "vocos.feature_extractors.MelSpectrogramFeatures"
BACKBONE_CLASS = "vocos.models.VocosBackbone"
HEAD_CLASS = "vocos.heads.ISTFTHead"

STATE_KEY_SUFFIXES = (".weight", ".bias", ".gamma", ".window", ".fb")
INTEGER_PATTERN = re.compile(r"-?\d+")


@dataclass(frozen=True)
class LavaSRFeatureConfig:
    class_path: str
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    padding: str
    f_min: int
    f_max: int
    norm: str
    mel_scale: str


@dataclass(frozen=True)
class LavaSRBackboneConfig:
    class_path: str
    input_channels: int
    dim: int
    intermediate_dim: int
    num_layers: int


@dataclass(frozen=True)
class LavaSRHeadConfig:
    class_path: str
    dim: int
    n_fft: int
    hop_length: int
    padding: str


@dataclass(frozen=True)
class LavaSRConfig:
    feature_extractor: LavaSRFeatureConfig
    backbone: LavaSRBackboneConfig
    head: LavaSRHeadConfig


@dataclass(frozen=True)
class LavaSRWeightBundleInfo:
    config: LavaSRConfig
    state_key_count: int


def validate_lavasr_v2_weight_bundle(resolved_weights: ResolvedWeights) -> LavaSRWeightBundleInfo:
    """Validate LavaSR v2 config metadata and PyTorch state-dict key layout."""

    config = read_lavasr_config(resolved_weights.path_for(LAVASR_CONFIG_PATH))
    validate_lavasr_v2_config(config)

    state_keys = extract_torch_state_dict_keys(resolved_weights.path_for(LAVASR_WEIGHTS_PATH))
    validate_lavasr_v2_state_keys(state_keys, num_layers=config.backbone.num_layers)

    return LavaSRWeightBundleInfo(config=config, state_key_count=len(state_keys))


def read_lavasr_config(path: str | Path) -> LavaSRConfig:
    """Read and normalize a LavaSR/Vocos YAML config file."""

    loaded = _load_lavasr_yaml_mapping(Path(path))
    return LavaSRConfig(
        feature_extractor=_read_feature_config(_section(loaded, "feature_extractor")),
        backbone=_read_backbone_config(_section(loaded, "backbone")),
        head=_read_head_config(_section(loaded, "head")),
    )


def _load_lavasr_yaml_mapping(path: Path) -> dict[str, object]:
    content = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        loaded = _parse_simple_lavasr_yaml(content)
    else:
        loaded = yaml.safe_load(content)

    if not isinstance(loaded, dict):
        raise ValueError("LavaSR config root must be a mapping")
    return loaded


def _parse_simple_lavasr_yaml(content: str) -> dict[str, object]:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]

    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if "\t" in raw_line:
            raise ValueError(f"LavaSR config line {line_number} uses tabs, which are unsupported")

        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        while indent <= stack[-1][0]:
            stack.pop()

        if ":" not in stripped:
            raise ValueError(f"LavaSR config line {line_number} must contain a key-value separator")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"LavaSR config line {line_number} has an empty key")

        parent = stack[-1][1]
        value = raw_value.strip()
        if not value:
            child: dict[str, object] = {}
            parent[key] = child
            stack.append((indent, child))
            continue

        parent[key] = _parse_simple_lavasr_scalar(value)

    return root


def _parse_simple_lavasr_scalar(value: str) -> int | str:
    if INTEGER_PATTERN.fullmatch(value):
        return int(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.startswith(("[", "{")):
        raise ValueError("LavaSR fallback config parser only supports scalar values and mappings")
    return value


def validate_lavasr_v2_config(config: LavaSRConfig) -> None:
    """Validate the LavaSR v2 enhancer config shape expected by the compatible backend."""

    _require_equal("feature_extractor.class_path", config.feature_extractor.class_path, FEATURE_EXTRACTOR_CLASS)
    _require_equal("feature_extractor.sample_rate", config.feature_extractor.sample_rate, 44100)
    _require_equal("feature_extractor.n_fft", config.feature_extractor.n_fft, 2048)
    _require_equal("feature_extractor.hop_length", config.feature_extractor.hop_length, 512)
    _require_equal("feature_extractor.n_mels", config.feature_extractor.n_mels, 80)
    _require_equal("feature_extractor.padding", config.feature_extractor.padding, "same")
    _require_equal("feature_extractor.f_min", config.feature_extractor.f_min, 0)
    _require_equal("feature_extractor.f_max", config.feature_extractor.f_max, 8000)
    _require_equal("feature_extractor.norm", config.feature_extractor.norm, "slaney")
    _require_equal("feature_extractor.mel_scale", config.feature_extractor.mel_scale, "slaney")

    _require_equal("backbone.class_path", config.backbone.class_path, BACKBONE_CLASS)
    _require_equal("backbone.input_channels", config.backbone.input_channels, 80)
    _require_equal("backbone.dim", config.backbone.dim, 512)
    _require_equal("backbone.intermediate_dim", config.backbone.intermediate_dim, 1536)
    _require_equal("backbone.num_layers", config.backbone.num_layers, 8)

    _require_equal("head.class_path", config.head.class_path, HEAD_CLASS)
    _require_equal("head.dim", config.head.dim, 512)
    _require_equal("head.n_fft", config.head.n_fft, 2048)
    _require_equal("head.hop_length", config.head.hop_length, 512)
    _require_equal("head.padding", config.head.padding, "same")


def extract_torch_state_dict_keys(path: str | Path) -> set[str]:
    """Extract state-dict-like keys from a PyTorch zip checkpoint without importing torch."""

    checkpoint_path = Path(path)
    try:
        with zipfile.ZipFile(checkpoint_path) as archive:
            pickle_name = next((name for name in archive.namelist() if name.endswith("data.pkl")), None)
            if pickle_name is None:
                raise ValueError(f"PyTorch checkpoint {checkpoint_path} does not contain data.pkl")
            data = archive.read(pickle_name)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"PyTorch checkpoint {checkpoint_path} is not a readable zip archive") from exc

    strings = _pickle_strings(data)
    return {value for value in strings if _is_lavasr_state_key(value)}


def validate_lavasr_v2_state_keys(state_keys: set[str], *, num_layers: int = 8) -> None:
    """Validate required LavaSR v2 state-dict keys are present."""

    expected = expected_lavasr_v2_state_keys(num_layers=num_layers)
    missing = sorted(expected - state_keys)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing)"
        raise ValueError(f"LavaSR v2 checkpoint is missing required state keys: {preview}{suffix}")


def expected_lavasr_v2_state_keys(*, num_layers: int = 8) -> set[str]:
    """Return the required state-dict keys for the LavaSR v2 Vocos-style architecture."""

    keys = {
        "feature_extractor.mel_spec.spectrogram.window",
        "feature_extractor.mel_spec.mel_scale.fb",
        "backbone.embed.weight",
        "backbone.embed.bias",
        "backbone.norm.weight",
        "backbone.norm.bias",
        "backbone.final_layer_norm.weight",
        "backbone.final_layer_norm.bias",
        "head.out.weight",
        "head.out.bias",
        "head.istft.window",
    }
    convnext_suffixes = (
        "gamma",
        "dwconv.weight",
        "dwconv.bias",
        "norm.weight",
        "norm.bias",
        "pwconv1.weight",
        "pwconv1.bias",
        "pwconv2.weight",
        "pwconv2.bias",
    )
    for layer_index in range(num_layers):
        keys.update(f"backbone.convnext.{layer_index}.{suffix}" for suffix in convnext_suffixes)
    return keys


def _read_feature_config(section: dict[str, object]) -> LavaSRFeatureConfig:
    init_args = _section(section, "init_args")
    return LavaSRFeatureConfig(
        class_path=_string(section, "class_path"),
        sample_rate=_integer(init_args, "sample_rate"),
        n_fft=_integer(init_args, "n_fft"),
        hop_length=_integer(init_args, "hop_length"),
        n_mels=_integer(init_args, "n_mels"),
        padding=_string(init_args, "padding"),
        f_min=_integer(init_args, "f_min"),
        f_max=_integer(init_args, "f_max"),
        norm=_string(init_args, "norm"),
        mel_scale=_string(init_args, "mel_scale"),
    )


def _read_backbone_config(section: dict[str, object]) -> LavaSRBackboneConfig:
    init_args = _section(section, "init_args")
    return LavaSRBackboneConfig(
        class_path=_string(section, "class_path"),
        input_channels=_integer(init_args, "input_channels"),
        dim=_integer(init_args, "dim"),
        intermediate_dim=_integer(init_args, "intermediate_dim"),
        num_layers=_integer(init_args, "num_layers"),
    )


def _read_head_config(section: dict[str, object]) -> LavaSRHeadConfig:
    init_args = _section(section, "init_args")
    return LavaSRHeadConfig(
        class_path=_string(section, "class_path"),
        dim=_integer(init_args, "dim"),
        n_fft=_integer(init_args, "n_fft"),
        hop_length=_integer(init_args, "hop_length"),
        padding=_string(init_args, "padding"),
    )


def _section(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"LavaSR config field {key!r} must be a mapping")
    return value


def _string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"LavaSR config field {key!r} must be a string")
    return value


def _integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"LavaSR config field {key!r} must be an integer")
    return value


def _require_equal(field: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"LavaSR v2 config {field} mismatch: expected {expected!r}, got {actual!r}")


def _pickle_strings(data: bytes) -> list[str]:
    strings: list[str] = []
    for opcode, argument, _position in pickletools.genops(data):
        if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"} and isinstance(argument, str):
            strings.append(argument)
    return strings


def _is_lavasr_state_key(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(("feature_extractor.", "backbone.", "head."))
        and value.endswith(STATE_KEY_SUFFIXES)
    )
