"""Opt-in adapters for licensed public analytical datasets.

Large downloads are intentionally not performed here. Each adapter consumes a
local export and requires an approved entry in ``docs/LICENSE_AUDIT.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from .signal import AcquisitionFamily, AxisType, AxisUnit, SpectralSignal


APPROVED_STATUSES = {"approved_for_training", "approved_for_evaluation"}


def load_license_audit(path: str | Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text())
    entries = payload.get("sources", [])
    result = {entry["id"]: entry for entry in entries}
    if len(result) != len(entries):
        raise ValueError("license audit contains duplicate source ids")
    return result


def require_approved_license(
    audit_path: str | Path, source_id: str, *, evaluation_only: bool = False
) -> Mapping[str, Any]:
    entries = load_license_audit(audit_path)
    if source_id not in entries:
        raise PermissionError(f"{source_id!r} has no license-audit entry")
    entry = entries[source_id]
    permitted = (
        {"approved_for_evaluation", "approved_for_training"}
        if evaluation_only
        else {"approved_for_training"}
    )
    if entry.get("status") not in permitted:
        raise PermissionError(
            f"{source_id!r} is {entry.get('status')!r}; required one of {sorted(permitted)}"
        )
    return entry


def chemistry_identifiers(smiles: str) -> tuple[str, str, str]:
    """Return canonical SMILES, InChIKey, and Bemis-Murcko scaffold.

    RDKit is optional at installation time but mandatory for chemistry-aware
    dataset preparation; the code never substitutes a string heuristic.
    """

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise RuntimeError("install jepanalytics[chem] for structure standardization") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    inchi_key = Chem.MolToInchiKey(molecule)
    if not inchi_key:
        raise ValueError(f"structure has no usable InChIKey: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
    return canonical, inchi_key, scaffold or f"__acyclic__:{inchi_key}"


class SmartsLabeler:
    """Functional-group labeler loaded from a versioned published SMARTS file."""

    def __init__(self, definitions: str | Path) -> None:
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise RuntimeError("install jepanalytics[chem] for SMARTS labeling") from exc
        payload = json.loads(Path(definitions).read_text())
        self.names = [item["name"] for item in payload["functional_groups"]]
        self.patterns = [Chem.MolFromSmarts(item["smarts"]) for item in payload["functional_groups"]]
        if any(pattern is None for pattern in self.patterns):
            raise ValueError("one or more functional-group SMARTS definitions are invalid")

    def __call__(self, smiles: str) -> np.ndarray:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        return self.label_molecule(molecule)

    def label_molecule(self, molecule: Any) -> np.ndarray:
        """Label an existing RDKit molecule without reparsing its SMILES."""

        return np.asarray(
            [molecule.HasSubstructMatch(pattern) for pattern in self.patterns], dtype=np.float32
        )


def _massbank_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    peaks: list[tuple[float, float]] = []
    in_peaks = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("PK$PEAK:"):
            in_peaks = True
            continue
        if line == "//":
            break
        if in_peaks and re.match(r"^[0-9]", line):
            pieces = line.split()
            if len(pieces) >= 2:
                peaks.append((float(pieces[0]), float(pieces[1])))
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            fields.setdefault(key, []).append(value.strip())
    fields["peaks"] = peaks
    return fields


def _first_number(values: Sequence[str], field: str) -> float:
    """Return the first number following ``field`` in MassBank subfields."""

    pattern = re.compile(rf"\b{re.escape(field)}\s+([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE)
    for value in values:
        match = pattern.search(value)
        if match:
            return float(match.group(1))
    return 0.0


def _license_is_allowed(value: str, allowed_licenses: Sequence[str]) -> bool:
    normalized = " ".join(value.upper().replace("_", "-").split())
    return any(
        normalized == " ".join(allowed.upper().replace("_", "-").split())
        or normalized.startswith(
            " ".join(allowed.upper().replace("_", "-").split()) + " "
        )
        for allowed in allowed_licenses
    )


def massbank_record_signal(
    path: str | Path,
    *,
    labeler: SmartsLabeler | None = None,
    source: str = "massbank",
    allowed_licenses: Sequence[str],
    rejection_counts: MutableMapping[str, int] | None = None,
) -> SpectralSignal | None:
    """Parse one licensed, structured MS2 MassBank record.

    The parser fails closed for missing ion mode, non-MS2 spectra, unusable
    structures, and unapproved record-level licenses. ``rejection_counts`` is
    optional so a dataset build can preserve an auditable filtering summary.
    """

    path = Path(path)
    fields = _massbank_fields(path.read_text(errors="replace"))

    def reject(reason: str) -> None:
        if rejection_counts is not None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    accession = (fields.get("ACCESSION") or [path.stem])[0]
    smiles = (fields.get("CH$SMILES") or [None])[0]
    license_value = " ".join(fields.get("LICENSE", []))
    peaks = fields["peaks"]
    mass_fields = fields.get("AC$MASS_SPECTROMETRY", [])
    mass_text = " ".join(mass_fields).upper()
    if not smiles or smiles.strip().upper() in {"N/A", "NA", "NONE", "UNKNOWN"}:
        reject("missing_smiles")
        return None
    if len(peaks) < 2:
        reject("fewer_than_two_peaks")
        return None
    if not license_value:
        reject("missing_license")
        return None
    if not _license_is_allowed(license_value, allowed_licenses):
        reject("license_not_allowed")
        return None
    if not re.search(r"\bMS_TYPE\s+MS2\b", mass_text):
        reject("not_ms2")
        return None
    if re.search(r"\bION_MODE\s+NEGATIVE\b", mass_text):
        acquisition = AcquisitionFamily.MSMS_NEGATIVE
    elif re.search(r"\bION_MODE\s+POSITIVE\b", mass_text):
        acquisition = AcquisitionFamily.MSMS_POSITIVE
    else:
        reject("missing_or_unknown_ion_mode")
        return None
    try:
        canonical_smiles, molecule_id, scaffold = chemistry_identifiers(smiles)
    except (RuntimeError, ValueError):
        reject("invalid_or_unkeyable_smiles")
        return None
    focused_fields = fields.get("MS$FOCUSED_ION", [])
    precursor_mz = _first_number(focused_fields, "PRECURSOR_M/Z")
    collision_energy = _first_number(mass_fields, "COLLISION_ENERGY")
    coordinate, intensity = map(np.asarray, zip(*peaks, strict=True))
    return SpectralSignal(
        coordinate=coordinate,
        intensity=intensity,
        axis_type=AxisType.MASS_TO_CHARGE,
        axis_unit=AxisUnit.MZ,
        acquisition=acquisition,
        molecule_id=molecule_id,
        source=source,
        source_id=accession,
        scaffold_id=scaffold,
        labels=None if labeler is None else labeler(canonical_smiles),
        metadata={
            "representation": "peak_list",
            "license": license_value,
            "record_path": str(path),
            "precursor_mz": precursor_mz,
            "collision_energy": collision_energy,
            "ms_type": "MS2",
            "canonical_smiles": canonical_smiles,
        },
    )


def iter_massbank_records(
    root: str | Path,
    *,
    labeler: SmartsLabeler | None = None,
    source: str = "massbank",
    allowed_licenses: Sequence[str],
    rejection_counts: MutableMapping[str, int] | None = None,
) -> Iterator[SpectralSignal]:
    """Parse open MassBank text records after a record-level license filter."""

    for path in sorted(Path(root).rglob("*.txt")):
        signal = massbank_record_signal(
            path,
            labeler=labeler,
            source=source,
            allowed_licenses=allowed_licenses,
            rejection_counts=rejection_counts,
        )
        if signal is not None:
            yield signal


def read_jcamp_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Read the common explicit ``XYDATA=(XY..XY)`` JCAMP-DX export form."""

    metadata: dict[str, str] = {}
    coordinates: list[float] = []
    intensities: list[float] = []
    in_xy = False
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("##") and "=" in line:
            key, value = line[2:].split("=", 1)
            metadata[key.strip().upper()] = value.strip()
            in_xy = key.strip().upper() in {"XYDATA", "PEAK TABLE", "PEAKTABLE"}
            continue
        if in_xy and line and not line.startswith("#"):
            numbers = [float(value) for value in re.split(r"[\s,;]+", line) if value]
            if len(numbers) == 2:
                coordinates.append(numbers[0])
                intensities.append(numbers[1])
            elif len(numbers) > 2:
                x = numbers[0]
                delta = float(metadata.get("DELTAX", "1"))
                for offset, value in enumerate(numbers[1:]):
                    coordinates.append(x + offset * delta)
                    intensities.append(value)
    if len(coordinates) < 2:
        raise ValueError(
            f"{path} is compressed or unsupported JCAMP-DX; export explicit XY pairs first"
        )
    x_factor = float(metadata.get("XFACTOR", "1"))
    y_factor = float(metadata.get("YFACTOR", "1"))
    return (
        np.asarray(coordinates, dtype=np.float64) * x_factor,
        np.asarray(intensities, dtype=np.float64) * y_factor,
        metadata,
    )


def specteach_jcamp_signal(
    path: str | Path,
    *,
    molecule_id: str,
    scaffold_id: str,
    acquisition: AcquisitionFamily,
    labels: np.ndarray | None = None,
) -> SpectralSignal:
    coordinate, intensity, metadata = read_jcamp_xy(path)
    if acquisition == AcquisitionFamily.IR:
        axis_type, axis_unit = AxisType.WAVENUMBER, AxisUnit.INVERSE_CENTIMETER
    elif acquisition in (AcquisitionFamily.H1_NMR, AcquisitionFamily.C13_NMR):
        axis_type, axis_unit = AxisType.CHEMICAL_SHIFT, AxisUnit.PPM
    else:
        axis_type, axis_unit = AxisType.MASS_TO_CHARGE, AxisUnit.MZ
    return SpectralSignal(
        coordinate=coordinate,
        intensity=intensity,
        axis_type=axis_type,
        axis_unit=axis_unit,
        acquisition=acquisition,
        molecule_id=molecule_id,
        source="specteach",
        source_id=Path(path).stem,
        scaffold_id=scaffold_id,
        labels=labels,
        metadata={"representation": "dense", "jcamp": metadata},
    )
