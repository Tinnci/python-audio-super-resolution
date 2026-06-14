from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import soundfile as sf

from ..config import InferenceConfig
from ..specs import ModelSpec
from .base import DEFAULT_FILE_BACKEND_CAPABILITY

AUDIOSR_SAMPLE_RATE = 48000
AUDIOSR_MODEL_NAMES = ("basic", "speech")


class AudiosrBackend:
    """AudioSR latent diffusion backend.

    The heavy `audiosr` dependency is imported only when this backend is selected.
    """

    name = "audiosr"
    description = "AudioSR latent diffusion backend. Requires the optional audiosr dependency."
    optional_dependency = "audiosr"
    package_extra = "audiosr"

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            return importlib.util.find_spec("audiosr") is not None
        except (ImportError, ValueError):
            return False

    @classmethod
    def model_specs(cls) -> tuple[ModelSpec, ...]:
        return tuple(
            ModelSpec(
                id=f"audiosr-{model_name}",
                backend=cls.name,
                model_name=model_name,
                name=f"AudioSR {model_name.title()}",
                description=f"AudioSR latent diffusion {model_name} model.",
                implementation="external_package",
                domain=("speech", "music", "sfx", "general"),
                architecture="latent-diffusion-audio-super-resolution",
                target_sample_rates=(AUDIOSR_SAMPLE_RATE,),
                weights_license="unknown",
                weights_source="Hugging Face download managed by the audiosr package",
                maturity="experimental",
                capability=DEFAULT_FILE_BACKEND_CAPABILITY,
            )
            for model_name in AUDIOSR_MODEL_NAMES
        )

    def enhance(self, audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
        raise RuntimeError("The audiosr backend requires file-based enhancement through AudioSuperResolver.enhance().")

    def enhance_file(self, input_path: str | Path, output_path: str | Path, target_sample_rate: int) -> None:
        if target_sample_rate != AUDIOSR_SAMPLE_RATE:
            raise ValueError("The audiosr backend outputs 48000 Hz audio; set target_sr=48000.")
        if self.config.device == "directml":
            raise ValueError("The audiosr backend does not support directml; use cpu, cuda, mps, or auto.")
        if self.config.precision not in {"float32", "auto"}:
            raise ValueError("The audiosr backend currently supports only float32 or auto precision.")
        if self.config.model_name not in AUDIOSR_MODEL_NAMES:
            choices = ", ".join(AUDIOSR_MODEL_NAMES)
            raise ValueError(f"The audiosr backend model_name must be one of: {choices}")

        audiosr = _import_audiosr()
        self._prepare_cache_environment()

        model = self._load_model(audiosr)
        waveform = audiosr.super_resolution(
            model,
            str(input_path),
            seed=self.config.seed,
            guidance_scale=self.config.guidance_scale,
            ddim_steps=self.config.ddim_steps,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, _waveform_to_audio_array(waveform), samplerate=AUDIOSR_SAMPLE_RATE)

    def _prepare_cache_environment(self) -> None:
        cache_dir = self.config.ensure_model_cache_dir()
        os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))

    def _load_model(self, audiosr: ModuleType):
        if self._model is None:
            self._model = audiosr.build_model(model_name=self.config.model_name, device=self.config.device)
        return self._model


def _import_audiosr() -> ModuleType:
    try:
        return importlib.import_module("audiosr")
    except ImportError as exc:
        raise RuntimeError(
            "The audiosr backend requires the optional audiosr dependency. "
            "Install it with `pip install audio-super-resolution[audiosr]`."
        ) from exc


def _waveform_to_audio_array(waveform) -> np.ndarray:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()

    audio = np.asarray(waveform, dtype=np.float32)

    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 2 and audio.shape[0] <= 8:
        audio = audio.T
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    if audio.ndim not in {1, 2}:
        raise ValueError(f"Unsupported AudioSR waveform shape: {audio.shape}")

    return audio
