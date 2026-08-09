"""Peak-preserving canonicalization and model-ready feature construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signal import AxisType, SpectralSignal


@dataclass(frozen=True, slots=True)
class ScaleStatistics:
    baseline: float
    scale: float
    clipped_fraction: float


@dataclass(frozen=True, slots=True)
class ProcessedSignal:
    intensity: np.ndarray
    physical_coordinate: np.ndarray
    normalized_coordinate: np.ndarray
    continuous_metadata: np.ndarray
    axis_type: int
    axis_unit: int
    acquisition: int
    molecule_id: str
    source_id: str
    split: str | None
    scaffold_id: str | None
    labels: np.ndarray | None


def robust_scale(
    intensity: np.ndarray,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.99,
    clip: tuple[float, float] = (-2.0, 5.0),
) -> tuple[np.ndarray, ScaleStatistics]:
    y = np.asarray(intensity, dtype=np.float64)
    baseline = float(np.quantile(y, lower_quantile))
    high = float(np.quantile(y, upper_quantile))
    # Centroided peak lists are mostly zero after rasterization. Estimate the
    # upper scale from occupied bins rather than allowing q99 to remain zero.
    occupied = y[y > baseline + np.finfo(np.float64).eps]
    if high <= baseline and occupied.size:
        high = float(np.quantile(occupied, 0.95))
    scale = max(high - baseline, np.finfo(np.float64).eps)
    scaled = (y - baseline) / scale
    clipped = np.clip(scaled, clip[0], clip[1])
    fraction = float(np.mean(clipped != scaled))
    return clipped.astype(np.float32), ScaleStatistics(baseline, scale, fraction)


def resample_peak_preserving(
    coordinate: np.ndarray,
    intensity: np.ndarray,
    n_bins: int = 4096,
    *,
    peak_list: bool = False,
    target_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a dense trace or centroided peak list onto an even grid.

    Dense traces use linear interpolation. Peak lists use nearest-bin max
    aggregation, avoiding attenuation of narrow centroided peaks.
    """

    x = np.asarray(coordinate, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("coordinate and intensity must be matching 1-D arrays")
    if not np.all(np.diff(x) > 0):
        raise ValueError("coordinate must be strictly increasing before resampling")
    low, high = target_range or (float(x[0]), float(x[-1]))
    if low > x[0] or high < x[-1] or low >= high:
        raise ValueError("target_range must contain the full input coordinate range")
    target = np.linspace(low, high, n_bins, dtype=np.float64)
    if not peak_list:
        values = np.interp(target, x, y)
        return target, values.astype(np.float32)

    values = np.zeros(n_bins, dtype=np.float64)
    positions = np.rint((x - low) / (high - low) * (n_bins - 1)).astype(np.int64)
    np.maximum.at(values, positions, y)
    return target, values.astype(np.float32)


def _axis_scale(axis_type: AxisType) -> float:
    return {
        AxisType.WAVENUMBER: 4000.0,
        AxisType.CHEMICAL_SHIFT: 250.0,
        AxisType.MASS_TO_CHARGE: 2000.0,
    }[axis_type]


@dataclass(slots=True)
class SignalProcessor:
    n_bins: int = 4096

    def __call__(self, signal: SpectralSignal) -> ProcessedSignal:
        canonical = signal.canonical()
        peak_list = bool(canonical.metadata.get("representation") == "peak_list")
        target_range = None
        if peak_list:
            configured = canonical.metadata.get("coordinate_range")
            if configured is not None:
                target_range = (float(configured[0]), float(configured[1]))
            elif canonical.axis_type == AxisType.MASS_TO_CHARGE:
                precursor = float(canonical.metadata.get("precursor_mz", 0.0) or 0.0)
                target_range = (0.0, max(precursor, float(canonical.coordinate[-1])))
        coordinate, intensity = resample_peak_preserving(
            canonical.coordinate,
            canonical.intensity,
            self.n_bins,
            peak_list=peak_list,
            target_range=target_range,
        )
        scaled, stats = robust_scale(intensity)
        span = float(coordinate[-1] - coordinate[0])
        normalized = ((coordinate - coordinate[0]) / span).astype(np.float32)
        axis_scale = _axis_scale(canonical.axis_type)
        precursor = float(canonical.metadata.get("precursor_mz", 0.0) or 0.0)
        collision = float(canonical.metadata.get("collision_energy", 0.0) or 0.0)
        sampling = float(np.median(np.diff(canonical.coordinate)))
        continuous = np.asarray(
            [
                coordinate[0] / axis_scale,
                coordinate[-1] / axis_scale,
                span / axis_scale,
                np.sign(stats.baseline) * np.log1p(abs(stats.baseline)),
                np.log1p(stats.scale),
                np.log1p(max(precursor, 0.0)) / 8.0,
                collision / 100.0,
                float(canonical.original_descending),
                np.log1p(abs(sampling)) / 10.0,
                stats.clipped_fraction,
            ],
            dtype=np.float32,
        )
        return ProcessedSignal(
            intensity=scaled,
            physical_coordinate=coordinate.astype(np.float32),
            normalized_coordinate=normalized,
            continuous_metadata=continuous,
            axis_type=int(canonical.axis_type),
            axis_unit=int(canonical.axis_unit),
            acquisition=int(canonical.acquisition),
            molecule_id=canonical.molecule_id,
            source_id=canonical.source_id,
            split=canonical.split,
            scaffold_id=canonical.scaffold_id,
            labels=canonical.labels,
        )
