from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

DEFAULT_LOWPASS_CUTOFF_HZ = 16000.0


def apply_preprocessing(
    audio: np.ndarray,
    sample_rate: int,
    mode: str = "none",
    lowpass_cutoff_hz: float | None = None,
    lowpass_order: int = 8,
) -> np.ndarray:
    """Apply optional preprocessing before enhancement."""

    normalized_mode = mode.lower()
    if normalized_mode == "none":
        return np.asarray(audio)
    if normalized_mode == "lowpass":
        return lowpass_filter(
            audio,
            sample_rate=sample_rate,
            cutoff_hz=lowpass_cutoff_hz,
            order=lowpass_order,
        )
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def lowpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float | None = None,
    order: int = 8,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter without changing sample rate."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if order <= 0:
        raise ValueError("lowpass_order must be greater than zero")

    nyquist_hz = sample_rate / 2
    resolved_cutoff_hz = cutoff_hz if cutoff_hz is not None else min(DEFAULT_LOWPASS_CUTOFF_HZ, sample_rate * 0.45)
    if resolved_cutoff_hz <= 0 or resolved_cutoff_hz >= nyquist_hz:
        raise ValueError("lowpass_cutoff_hz must be greater than zero and below Nyquist")

    audio_array = np.asarray(audio)
    sos = butter(order, resolved_cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")

    try:
        return sosfiltfilt(sos, audio_array, axis=0)
    except ValueError:
        return sosfilt(sos, audio_array, axis=0)
