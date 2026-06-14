"""Torch modules for LavaSR-compatible inference.

The Vocos-style block layout and same-padding ISTFT are adapted from Vocos
(MIT License, Copyright Charactr Inc.). The low/high-frequency merge follows
the LavaSR reference implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .lavasr_validation import LavaSRConfig


class LavaSRV2Model(nn.Module):
    """Self-contained LavaSR v2 BWE model structure compatible with Vocos-style weights."""

    def __init__(self, config: LavaSRConfig) -> None:
        super().__init__()
        self.feature_extractor = LavaSRMelSpectrogramFeatures(config)
        self.backbone = LavaSRVocosBackbone(config)
        self.head = LavaSRISTFTHead(config)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(audio)
        hidden = self.backbone(features)
        return self.head(hidden)


class LavaSRMelSpectrogramFeatures(nn.Module):
    def __init__(self, config: LavaSRConfig) -> None:
        super().__init__()
        feature_config = config.feature_extractor
        _require_padding(feature_config.padding)
        self.padding = feature_config.padding
        self.mel_spec = LavaSRMelSpectrogram(
            sample_rate=feature_config.sample_rate,
            n_fft=feature_config.n_fft,
            hop_length=feature_config.hop_length,
            n_mels=feature_config.n_mels,
            center=feature_config.padding == "center",
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.padding == "same":
            pad = self.mel_spec.spectrogram.win_length - self.mel_spec.spectrogram.hop_length
            audio = F.pad(audio, (pad // 2, pad // 2), mode="reflect")
        return safe_log(self.mel_spec(audio))


class LavaSRMelSpectrogram(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        center: bool,
    ) -> None:
        super().__init__()
        self.spectrogram = LavaSRSpectrogram(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=center)
        self.mel_scale = LavaSRMelScale(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.mel_scale(self.spectrogram(audio))


class LavaSRSpectrogram(nn.Module):
    def __init__(self, *, n_fft: int, hop_length: int, win_length: int, center: bool) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        window = self.window.to(dtype=audio.dtype, device=audio.device)
        spectrogram = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            return_complex=True,
        )
        return spectrogram.abs()


class LavaSRMelScale(nn.Module):
    def __init__(self, *, sample_rate: int, n_fft: int, n_mels: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.register_buffer("fb", torch.zeros(n_fft // 2 + 1, n_mels))

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        fb = self.fb.to(dtype=spectrogram.dtype, device=spectrogram.device)
        return torch.matmul(spectrogram.transpose(1, 2), fb).transpose(1, 2)


class LavaSRVocosBackbone(nn.Module):
    def __init__(self, config: LavaSRConfig) -> None:
        super().__init__()
        backbone_config = config.backbone
        self.input_channels = backbone_config.input_channels
        self.embed = nn.Conv1d(backbone_config.input_channels, backbone_config.dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(backbone_config.dim, eps=1e-6)
        layer_scale = 1 / backbone_config.num_layers
        self.convnext = nn.ModuleList(
            [
                LavaSRConvNeXtBlock(
                    dim=backbone_config.dim,
                    intermediate_dim=backbone_config.intermediate_dim,
                    layer_scale_init_value=layer_scale,
                )
                for _ in range(backbone_config.num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(backbone_config.dim, eps=1e-6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(features)
        hidden = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        for conv_block in self.convnext:
            hidden = conv_block(hidden)
        return self.final_layer_norm(hidden.transpose(1, 2))


class LavaSRConvNeXtBlock(nn.Module):
    def __init__(self, *, dim: int, intermediate_dim: int, layer_scale_init_value: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        hidden = self.dwconv(features)
        hidden = self.norm(hidden.transpose(1, 2))
        hidden = self.pwconv1(hidden)
        hidden = self.act(hidden)
        hidden = self.pwconv2(hidden)
        hidden = self.gamma * hidden
        return residual + hidden.transpose(1, 2)


class LavaSRISTFTHead(nn.Module):
    def __init__(self, config: LavaSRConfig) -> None:
        super().__init__()
        head_config = config.head
        _require_padding(head_config.padding)
        out_dim = head_config.n_fft + 2
        self.out = nn.Linear(head_config.dim, out_dim)
        self.istft = LavaSRISTFT(
            n_fft=head_config.n_fft,
            hop_length=head_config.hop_length,
            win_length=head_config.n_fft,
            padding=head_config.padding,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.out(hidden).transpose(1, 2)
        magnitude, phase = projected.chunk(2, dim=1)
        magnitude = torch.exp(magnitude).clamp(max=1e3)
        spectrum = magnitude * (torch.cos(phase) + 1j * torch.sin(phase))
        return self.istft(spectrum)


class LavaSRISTFT(nn.Module):
    def __init__(self, *, n_fft: int, hop_length: int, win_length: int, padding: str = "same") -> None:
        super().__init__()
        _require_padding(padding)
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.dim() != 3:
            raise ValueError("LavaSR ISTFT expects a 3D complex spectrogram")
        if self.padding == "center":
            window = self.window.to(dtype=spectrum.real.dtype, device=spectrum.device)
            return torch.istft(spectrum, self.n_fft, self.hop_length, self.win_length, window, center=True)

        pad = (self.win_length - self.hop_length) // 2
        window = self.window.to(dtype=spectrum.real.dtype, device=spectrum.device)
        frame_count = spectrum.shape[2]
        inverse_fft = torch.fft.irfft(spectrum, self.n_fft, dim=1, norm="backward")
        inverse_fft = inverse_fft * window.view(1, self.win_length, 1)

        output_size = (frame_count - 1) * self.hop_length + self.win_length
        audio = F.fold(
            inverse_fft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0]
        envelope = self._window_envelope(frame_count, output_size, window)
        if pad:
            audio = audio[:, pad:-pad]
            envelope = envelope[pad:-pad]
        if torch.any(envelope <= 1e-11):
            raise RuntimeError("LavaSR ISTFT window envelope contains zeros")
        return audio / envelope.unsqueeze(0)

    def _window_envelope(self, frame_count: int, output_size: int, window: torch.Tensor) -> torch.Tensor:
        window_sq = window.square().view(1, self.win_length, 1).expand(1, self.win_length, frame_count)
        return F.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[0, 0, 0]


class FastLRMerge:
    """Merge source low frequencies with predicted high frequencies."""

    def __init__(self, *, sample_rate: int = 48000, cutoff: float = 4000.0, transition_bins: int = 256) -> None:
        self.sample_rate = sample_rate
        self.cutoff = cutoff
        self.transition_bins = transition_bins
        self.mask_cache: dict[tuple[int, int, str, int | None], torch.Tensor] = {}

    def __call__(self, predicted_audio: torch.Tensor, source_audio: torch.Tensor) -> torch.Tensor:
        predicted_spectrum = torch.fft.rfft(predicted_audio, dim=-1)
        source_spectrum = torch.fft.rfft(source_audio, dim=-1)
        mask = self._mask(predicted_spectrum.size(-1), predicted_spectrum.ndim, predicted_spectrum.device)
        merged = source_spectrum + (predicted_spectrum - source_spectrum) * mask.to(dtype=predicted_spectrum.dtype)
        return torch.fft.irfft(merged, n=predicted_audio.size(-1), dim=-1)

    def _mask(self, n_bins: int, ndim: int, device: torch.device) -> torch.Tensor:
        key = (n_bins, ndim, device.type, device.index)
        cached = self.mask_cache.get(key)
        if cached is not None:
            return cached

        cutoff_bin = int((self.cutoff / (self.sample_rate / 2)) * n_bins)
        half_transition = self.transition_bins // 2
        start = max(0, cutoff_bin - half_transition)
        end = min(n_bins, cutoff_bin + half_transition)
        fade = _fade_curve(self.transition_bins, device=device)[: end - start]

        mask = torch.ones(n_bins, device=device, dtype=torch.complex64)
        mask[:start] = 0
        mask[start:end] = fade
        for _ in range(ndim - 1):
            mask = mask.unsqueeze(0)
        self.mask_cache[key] = mask
        return mask


def build_lavasr_v2_model(config: LavaSRConfig) -> LavaSRV2Model:
    return LavaSRV2Model(config)


def load_lavasr_v2_state_dict(model: LavaSRV2Model, checkpoint_path: str | Path) -> None:
    state_dict = _load_state_dict(checkpoint_path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"LavaSR v2 checkpoint does not match the self-contained model structure: {exc}") from exc


def safe_log(value: torch.Tensor, clip_value: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clip(value, min=clip_value))


def _load_state_dict(checkpoint_path: str | Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or any(not isinstance(key, str) for key in checkpoint):
        raise ValueError("LavaSR v2 checkpoint must contain a state-dict mapping with string keys")
    return checkpoint


def _require_padding(value: str) -> None:
    if value not in {"center", "same"}:
        raise ValueError("LavaSR padding must be 'center' or 'same'")


def _fade_curve(length: int, *, device: torch.device) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, device=device, dtype=torch.complex64)
    x = torch.linspace(-1, 1, steps=length, device=device)
    t = (x + 1) / 2
    return (3 * t**2 - 2 * t**3).to(torch.complex64)
