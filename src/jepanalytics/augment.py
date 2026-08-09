"""Physically motivated perturbations for analytical signals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class AugmentationConfig:
    noise_std: float = 0.015
    baseline_std: float = 0.04
    intensity_scale: float = 0.15
    shift_bins: int = 2
    broadening_probability: float = 0.35
    resolution_probability: float = 0.25
    missing_span_probability: float = 0.20
    missing_span_fraction: float = 0.03


class SpectralAugmenter:
    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def __call__(self, intensity: torch.Tensor, *, light: bool = False) -> torch.Tensor:
        if intensity.ndim != 2:
            raise ValueError("intensity must have shape [batch, samples]")
        config = self.config
        batch, length = intensity.shape
        strength = 0.35 if light else 1.0
        output = intensity.clone()

        scales = 1.0 + torch.randn(batch, 1, device=output.device) * config.intensity_scale * strength
        output = output * scales.clamp_min(0.2)

        position = torch.linspace(-1.0, 1.0, length, device=output.device).unsqueeze(0)
        intercept = torch.randn(batch, 1, device=output.device) * config.baseline_std * strength
        slope = torch.randn(batch, 1, device=output.device) * config.baseline_std * strength
        curvature = torch.randn(batch, 1, device=output.device) * config.baseline_std * 0.5 * strength
        output = output + intercept + slope * position + curvature * (position.square() - 1.0 / 3.0)

        output = output + torch.randn_like(output) * config.noise_std * strength

        if not light and config.shift_bins > 0:
            shifts = torch.randint(-config.shift_bins, config.shift_bins + 1, (batch,), device=output.device)
            shifted = torch.zeros_like(output)
            for row, shift_tensor in enumerate(shifts):
                shift = int(shift_tensor.item())
                if shift > 0:
                    shifted[row, shift:] = output[row, :-shift]
                elif shift < 0:
                    shifted[row, :shift] = output[row, -shift:]
                else:
                    shifted[row] = output[row]
            output = shifted

        if torch.rand((), device=output.device) < config.broadening_probability * strength:
            kernel = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=output.device)
            kernel = (kernel / kernel.sum()).view(1, 1, -1)
            output = F.conv1d(output.unsqueeze(1), kernel, padding=2).squeeze(1)

        if not light and torch.rand((), device=output.device) < config.resolution_probability:
            low_length = max(16, length // 4)
            output = F.interpolate(
                F.interpolate(output.unsqueeze(1), size=low_length, mode="linear", align_corners=False),
                size=length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        if not light and config.missing_span_probability > 0:
            span = max(1, round(length * config.missing_span_fraction))
            for row in range(batch):
                if torch.rand((), device=output.device) < config.missing_span_probability:
                    start = int(torch.randint(0, max(1, length - span + 1), (1,), device=output.device))
                    output[row, start : start + span] = 0.0
        return output

