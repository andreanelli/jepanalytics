"""Physically motivated perturbations for analytical signals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .signal import AcquisitionFamily


#: Acquisition families whose records are centroided peak lists rather than
#: dense traces.  A baseline, additive Gaussian noise, broadening, and
#: resolution loss are meaningless for them: those operations describe a
#: continuously sampled detector, while a centroided spectrum is a sparse list
#: of (m/z, abundance) pairs whose information lives entirely in which bins are
#: occupied and how strongly.
SPARSE_ACQUISITIONS = (
    int(AcquisitionFamily.MSMS_POSITIVE),
    int(AcquisitionFamily.MSMS_NEGATIVE),
)


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
    #: Sparse-signal view: probability that any individual peak is dropped.
    peak_dropout: float = 0.10
    #: Sparse-signal view: relative standard deviation of per-peak abundance.
    peak_intensity_jitter: float = 0.20
    #: Sparse-signal view: maximum whole-bin m/z miscalibration.
    peak_shift_bins: int = 1


class SpectralAugmenter:
    """Augment a batch of signals, branching on acquisition family.

    Every random decision is drawn per sample.  Drawing one scalar for the
    whole batch (as an earlier revision did for broadening and resolution)
    collapses augmentation diversity, which matters most in multi-view
    training where several views of one molecule are built in a single call.
    """

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def __call__(
        self,
        intensity: torch.Tensor,
        *,
        light: bool = False,
        acquisition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intensity.ndim != 2:
            raise ValueError("intensity must have shape [batch, samples]")
        batch, _ = intensity.shape
        if acquisition is None:
            sparse = torch.zeros(batch, dtype=torch.bool, device=intensity.device)
        else:
            if acquisition.shape[0] != batch:
                raise ValueError("acquisition must have one entry per row")
            sparse = torch.isin(
                acquisition.to(intensity.device),
                torch.tensor(SPARSE_ACQUISITIONS, device=intensity.device),
            )
        output = intensity.clone()
        if not bool(sparse.all()):
            output = torch.where(
                sparse.unsqueeze(1),
                output,
                self._augment_dense(intensity, light=light),
            )
        if bool(sparse.any()):
            output = torch.where(
                sparse.unsqueeze(1),
                self._augment_sparse(intensity, light=light),
                output,
            )
        return output

    def _row_shift(self, output: torch.Tensor, max_shift: int) -> torch.Tensor:
        """Shift each row independently by up to ``max_shift`` whole bins."""

        batch, length = output.shape
        shifts = torch.randint(
            -max_shift, max_shift + 1, (batch, 1), device=output.device
        )
        positions = torch.arange(length, device=output.device).unsqueeze(0) - shifts
        inside = (positions >= 0) & (positions < length)
        gathered = output.gather(1, positions.clamp(0, length - 1))
        return gathered * inside

    def _augment_dense(self, intensity: torch.Tensor, *, light: bool) -> torch.Tensor:
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
            output = self._row_shift(output, config.shift_bins)

        broaden = (
            torch.rand(batch, 1, device=output.device)
            < config.broadening_probability * strength
        )
        if bool(broaden.any()):
            kernel = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=output.device)
            kernel = (kernel / kernel.sum()).view(1, 1, -1)
            broadened = F.conv1d(output.unsqueeze(1), kernel, padding=2).squeeze(1)
            output = torch.where(broaden, broadened, output)

        if not light:
            coarse = torch.rand(batch, 1, device=output.device) < config.resolution_probability
            if bool(coarse.any()):
                low_length = max(16, length // 4)
                reduced = F.interpolate(
                    F.interpolate(
                        output.unsqueeze(1), size=low_length, mode="linear", align_corners=False
                    ),
                    size=length,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
                output = torch.where(coarse, reduced, output)

        if not light and config.missing_span_probability > 0:
            span = max(1, round(length * config.missing_span_fraction))
            starts = torch.randint(
                0, max(1, length - span + 1), (batch, 1), device=output.device
            )
            grid = torch.arange(length, device=output.device).unsqueeze(0)
            hole = (grid >= starts) & (grid < starts + span)
            drop = torch.rand(batch, 1, device=output.device) < config.missing_span_probability
            output = output.masked_fill(hole & drop, 0.0)
        return output

    def _augment_sparse(self, intensity: torch.Tensor, *, light: bool) -> torch.Tensor:
        """Perturb a centroided spectrum without destroying its sparsity.

        Peak dropout stands in for detection thresholds and collision-energy
        variation, abundance jitter for run-to-run intensity irreproducibility,
        and a whole-bin shift for m/z miscalibration.
        """

        config = self.config
        strength = 0.35 if light else 1.0
        output = intensity.clone()

        scales = 1.0 + torch.randn(
            intensity.shape[0], 1, device=output.device
        ) * config.intensity_scale * strength
        output = output * scales.clamp_min(0.2)

        jitter = 1.0 + torch.randn_like(output) * config.peak_intensity_jitter * strength
        output = output * jitter.clamp_min(0.0)

        if config.peak_dropout > 0:
            keep = torch.rand_like(output) >= config.peak_dropout * strength
            output = output * keep

        if not light and config.peak_shift_bins > 0:
            output = self._row_shift(output, config.peak_shift_bins)
        return output
