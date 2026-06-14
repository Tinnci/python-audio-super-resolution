from __future__ import annotations

import numpy as np

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
    """Apply a low-pass filter without changing sample rate."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if order <= 0:
        raise ValueError("lowpass_order must be greater than zero")

    nyquist_hz = sample_rate / 2
    resolved_cutoff_hz = cutoff_hz if cutoff_hz is not None else min(DEFAULT_LOWPASS_CUTOFF_HZ, sample_rate * 0.45)
    if resolved_cutoff_hz <= 0 or resolved_cutoff_hz >= nyquist_hz:
        raise ValueError("lowpass_cutoff_hz must be greater than zero and below Nyquist")

    audio_array = np.asarray(audio)
    return _fft_lowpass(audio_array, sample_rate=sample_rate, cutoff_hz=resolved_cutoff_hz, order=order)


def _fft_lowpass(audio: np.ndarray, *, sample_rate: int, cutoff_hz: float, order: int) -> np.ndarray:
    if audio.shape[0] == 0:
        return audio.copy()

    output_dtype = audio.dtype if np.issubdtype(audio.dtype, np.floating) else np.float64
    audio_float = audio.astype(np.float64, copy=False)
    frequencies = np.fft.rfftfreq(audio_float.shape[0], d=1 / sample_rate)
    spectrum = np.fft.rfft(audio_float, axis=0)
    mask = _lowpass_mask(frequencies, cutoff_hz=cutoff_hz, nyquist_hz=sample_rate / 2, order=order)
    if spectrum.ndim > 1:
        mask = mask.reshape((mask.shape[0],) + (1,) * (spectrum.ndim - 1))
    spectrum *= mask
    return np.fft.irfft(spectrum, n=audio_float.shape[0], axis=0).astype(output_dtype, copy=False)


def _lowpass_mask(
    frequencies: np.ndarray,
    *,
    cutoff_hz: float,
    nyquist_hz: float,
    order: int,
) -> np.ndarray:
    transition_width_hz = min(nyquist_hz - cutoff_hz, max(cutoff_hz / order, 1.0))
    mask = np.ones_like(frequencies)
    stopband_start_hz = cutoff_hz + transition_width_hz
    mask[frequencies >= stopband_start_hz] = 0.0

    transition = (frequencies > cutoff_hz) & (frequencies < stopband_start_hz)
    if np.any(transition):
        phase = (frequencies[transition] - cutoff_hz) / transition_width_hz
        mask[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return mask
