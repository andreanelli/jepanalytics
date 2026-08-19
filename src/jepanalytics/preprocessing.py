"""Peak-preserving canonicalization and model-ready feature construction."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .signal import AcquisitionFamily, AxisType, SpectralSignal


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
    #: Fraction of input peaks discarded for falling outside the fixed m/z
    #: window.  Zero for dense traces and for spectra fully inside the window.
    out_of_range_fraction: float = 0.0


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
        if x.size > n_bins:
            # Point-sampled interpolation silently drops NMR lines that are
            # only one to three native samples wide.  Aggregate every native
            # sample into its target bin instead, so a narrow line survives
            # decimation; bins with no native sample fall back to interpolation.
            values = np.interp(target, x, y)
            positions = np.rint((x - low) / (high - low) * (n_bins - 1)).astype(np.int64)
            np.clip(positions, 0, n_bins - 1, out=positions)
            occupied = np.zeros(n_bins, dtype=bool)
            occupied[positions] = True
            aggregated = np.zeros(n_bins, dtype=np.float64)
            np.maximum.at(aggregated, positions, y)
            values[occupied] = aggregated[occupied]
            return target, values.astype(np.float32)
        values = np.interp(target, x, y)
        return target, values.astype(np.float32)

    values = np.zeros(n_bins, dtype=np.float64)
    positions = np.rint((x - low) / (high - low) * (n_bins - 1)).astype(np.int64)
    np.maximum.at(values, positions, y)
    return target, values.astype(np.float32)


#: Fixed absolute m/z window every mass spectrum is rasterized onto.  A
#: per-molecule window makes bin *i* mean a different m/z for every molecule,
#: which leaves absolute fragment masses unlearnable and leaks the precursor
#: (hence the exact molecular mass) through the coordinate metadata.
DEFAULT_MS_MZ_RANGE = (0.0, 1000.0)


def _axis_scale(axis_type: AxisType, acquisition: AcquisitionFamily | int | None = None) -> float:
    """Return the normalizing constant for a physical coordinate axis.

    Chemical shift is acquisition dependent: a single 250 ppm constant maps the
    -2..10 ppm proton window onto 0.04 while carbon spans nearly the full unit
    interval, so the shared coordinate projection receives proton coordinates
    roughly twenty-five times smaller than carbon and infrared ones.
    """

    if axis_type == AxisType.CHEMICAL_SHIFT:
        return 12.0 if acquisition == AcquisitionFamily.H1_NMR else 250.0
    return {
        AxisType.WAVENUMBER: 4000.0,
        AxisType.MASS_TO_CHARGE: 2000.0,
    }[axis_type]


@dataclass(slots=True)
class SignalProcessor:
    """Rasterize a validated signal onto the shared model grid.

    ``ms_mz_range`` pins every mass spectrum to one absolute window so a bin
    index carries the same m/z for every molecule.  Peaks outside the window
    are dropped and counted rather than silently rescaling the axis.

    ``include_precursor_metadata`` controls whether the precursor m/z reaches
    the encoder.  The precursor is derived from the exact molecular mass, so
    feeding it in makes functional-group probing and MS+/MS- retrieval
    partially solvable without reading the spectrum; it stays off for
    representation-learning claims.
    """

    n_bins: int = 4096
    ms_mz_range: tuple[float, float] = DEFAULT_MS_MZ_RANGE
    include_precursor_metadata: bool = False

    def __call__(self, signal: SpectralSignal) -> ProcessedSignal:
        canonical = signal.canonical()
        peak_list = bool(canonical.metadata.get("representation") == "peak_list")
        target_range = None
        out_of_range = 0.0
        if peak_list and canonical.axis_type == AxisType.MASS_TO_CHARGE:
            target_range = (float(self.ms_mz_range[0]), float(self.ms_mz_range[1]))
            inside = (canonical.coordinate >= target_range[0]) & (
                canonical.coordinate <= target_range[1]
            )
            if not np.all(inside):
                out_of_range = float(np.mean(~inside))
                coordinate_in = canonical.coordinate[inside]
                intensity_in = canonical.intensity[inside]
                if coordinate_in.size < 2:
                    # Keep a well-formed, essentially empty spectrum rather
                    # than failing the whole record.
                    coordinate_in = np.asarray(target_range, dtype=np.float64)
                    intensity_in = np.zeros(2, dtype=np.float64)
                canonical = replace(
                    canonical, coordinate=coordinate_in, intensity=intensity_in
                )
        elif peak_list:
            configured = canonical.metadata.get("coordinate_range")
            if configured is not None:
                target_range = (float(configured[0]), float(configured[1]))
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
        axis_scale = _axis_scale(canonical.axis_type, canonical.acquisition)
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
                np.log1p(max(precursor, 0.0)) / 8.0
                if self.include_precursor_metadata
                else 0.0,
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
            out_of_range_fraction=out_of_range,
        )
