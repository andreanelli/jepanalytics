"""Validated, modality-neutral representation for one-dimensional spectra."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Mapping

import numpy as np


class AxisType(IntEnum):
    WAVENUMBER = 0
    CHEMICAL_SHIFT = 1
    MASS_TO_CHARGE = 2


class AxisUnit(IntEnum):
    INVERSE_CENTIMETER = 0
    PPM = 1
    MZ = 2


class AcquisitionFamily(IntEnum):
    IR = 0
    H1_NMR = 1
    C13_NMR = 2
    MSMS_POSITIVE = 3
    MSMS_NEGATIVE = 4


_EXPECTED_UNIT = {
    AxisType.WAVENUMBER: AxisUnit.INVERSE_CENTIMETER,
    AxisType.CHEMICAL_SHIFT: AxisUnit.PPM,
    AxisType.MASS_TO_CHARGE: AxisUnit.MZ,
}


@dataclass(frozen=True, slots=True)
class SpectralSignal:
    """A single analytical signal and the provenance required for safe reuse.

    Arrays are converted to float64 on construction so that preprocessing is
    deterministic. Model-ready tensors are produced by :class:`SignalProcessor`.
    """

    coordinate: np.ndarray
    intensity: np.ndarray
    axis_type: AxisType
    axis_unit: AxisUnit
    acquisition: AcquisitionFamily
    molecule_id: str
    source: str
    source_id: str
    split: str | None = None
    scaffold_id: str | None = None
    labels: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    original_descending: bool = False

    def __post_init__(self) -> None:
        x = np.asarray(self.coordinate, dtype=np.float64)
        y = np.asarray(self.intensity, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("coordinate and intensity must both be one-dimensional")
        if x.size != y.size:
            raise ValueError("coordinate and intensity must have the same length")
        if x.size < 2:
            raise ValueError("a signal must contain at least two samples")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("coordinate and intensity must contain only finite values")
        if not self.molecule_id or not self.source or not self.source_id:
            raise ValueError("molecule_id, source, and source_id are required")
        if _EXPECTED_UNIT[self.axis_type] != self.axis_unit:
            raise ValueError(f"{self.axis_unit.name} is incompatible with {self.axis_type.name}")
        labels = None if self.labels is None else np.asarray(self.labels, dtype=np.float32)
        if labels is not None and labels.ndim != 1:
            raise ValueError("labels must be a one-dimensional multi-hot vector")
        object.__setattr__(self, "coordinate", x)
        object.__setattr__(self, "intensity", y)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def canonical(self) -> "SpectralSignal":
        """Return a copy with a strictly increasing coordinate axis.

        Duplicate coordinates are merged by retaining the greatest intensity,
        which avoids attenuating centroided mass-spectral peaks.
        """

        order = np.argsort(self.coordinate, kind="stable")
        x = self.coordinate[order]
        y = self.intensity[order]
        unique_x, first, counts = np.unique(x, return_index=True, return_counts=True)
        if np.any(counts > 1):
            y = np.maximum.reduceat(y, first)
            x = unique_x
        if x.size < 2 or not np.all(np.diff(x) > 0):
            raise ValueError("coordinate axis cannot be made strictly increasing")
        differences = np.diff(self.coordinate)
        descending = bool(np.all(differences <= 0) and np.any(differences < 0))
        return replace(
            self,
            coordinate=x,
            intensity=y,
            original_descending=self.original_descending or descending,
        )


def default_axis(acquisition: AcquisitionFamily) -> tuple[AxisType, AxisUnit]:
    if acquisition == AcquisitionFamily.IR:
        return AxisType.WAVENUMBER, AxisUnit.INVERSE_CENTIMETER
    if acquisition in (AcquisitionFamily.H1_NMR, AcquisitionFamily.C13_NMR):
        return AxisType.CHEMICAL_SHIFT, AxisUnit.PPM
    return AxisType.MASS_TO_CHARGE, AxisUnit.MZ
