"""Native adapter for the NeurIPS 2024 multimodal spectroscopy Parquet schema."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .adapters import SmartsLabeler, chemistry_identifiers, require_approved_license
from .data import CanonicalStoreWriter
from .preprocessing import SignalProcessor
from .signal import AcquisitionFamily, AxisType, AxisUnit, SpectralSignal
from .splits import scaffold_split, split_digest


SMILES_COLUMN = "smiles"
ROW_ID_COLUMN = "__index_level_0__"
DENSE_COLUMNS = ("ir_spectra", "h_nmr_spectra", "c_nmr_spectra")
MS_COLUMNS = (
    "msms_cfmid_positive_10ev",
    "msms_cfmid_positive_20ev",
    "msms_cfmid_positive_40ev",
    "msms_cfmid_negative_10ev",
    "msms_cfmid_negative_20ev",
    "msms_cfmid_negative_40ev",
)
REQUIRED_COLUMNS = (SMILES_COLUMN, ROW_ID_COLUMN, "molecular_formula", *DENSE_COLUMNS, *MS_COLUMNS)


@dataclass(frozen=True, slots=True)
class MoleculeCandidate:
    row_id: int
    smiles: str
    molecule_id: str
    scaffold_id: str
    heavy_atoms: int
    exact_mass: float
    labels: np.ndarray


def _arrow_dataset(path: str | Path):
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("install jepanalytics[data] for Parquet preparation") from exc
    dataset = ds.dataset(str(path), format="parquet")
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise ValueError(f"NeurIPS Parquet schema is missing columns: {missing}")
    return dataset


def _molecular_properties(smiles: str) -> tuple[int, float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError as exc:
        raise RuntimeError("install jepanalytics[chem] for dataset preparation") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return int(molecule.GetNumHeavyAtoms()), float(Descriptors.ExactMolWt(molecule))


def scan_candidates(
    parquet_path: str | Path,
    labeler: SmartsLabeler,
    *,
    excluded_molecules: set[str] | None = None,
    excluded_scaffolds: set[str] | None = None,
    strict_scaffold: bool = False,
) -> dict[str, MoleculeCandidate]:
    dataset = _arrow_dataset(parquet_path)
    excluded_molecules = excluded_molecules or set()
    excluded_scaffolds = excluded_scaffolds or set()
    candidates: dict[str, MoleculeCandidate] = {}
    scanner = dataset.scanner(columns=[SMILES_COLUMN, ROW_ID_COLUMN], batch_size=4096)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            smiles = row[SMILES_COLUMN]
            if not smiles:
                continue
            try:
                canonical, molecule_id, scaffold_id = chemistry_identifiers(smiles)
            except ValueError:
                continue
            if molecule_id in excluded_molecules:
                continue
            if strict_scaffold and scaffold_id in excluded_scaffolds:
                continue
            heavy_atoms, exact_mass = _molecular_properties(canonical)
            candidate = MoleculeCandidate(
                row_id=int(row[ROW_ID_COLUMN]),
                smiles=canonical,
                molecule_id=molecule_id,
                scaffold_id=scaffold_id,
                heavy_atoms=heavy_atoms,
                exact_mass=exact_mass,
                labels=labeler(canonical),
            )
            previous = candidates.get(molecule_id)
            if previous is None or candidate.row_id < previous.row_id:
                candidates[molecule_id] = candidate
    return candidates


def _size_bucket(heavy_atoms: int) -> int:
    if heavy_atoms <= 10:
        return 0
    if heavy_atoms <= 20:
        return 1
    return 2


def stratified_candidate_sample(
    candidates: Mapping[str, MoleculeCandidate], max_molecules: int, seed: int
) -> dict[int, MoleculeCandidate]:
    """Proportionally sample heavy-atom/rarest-functional-group strata."""

    if max_molecules <= 0:
        raise ValueError("max_molecules must be positive")
    if max_molecules >= len(candidates):
        return {candidate.row_id: candidate for candidate in candidates.values()}
    label_counts = np.sum([candidate.labels for candidate in candidates.values()], axis=0)
    groups: dict[tuple[int, int], list[MoleculeCandidate]] = defaultdict(list)
    for candidate in candidates.values():
        positives = np.flatnonzero(candidate.labels)
        rarest = int(positives[np.argmin(label_counts[positives])]) if positives.size else -1
        groups[(_size_bucket(candidate.heavy_atoms), rarest)].append(candidate)

    group_items = sorted(groups.items())
    total = len(candidates)
    raw_quotas = {
        key: max_molecules * len(items) / total for key, items in group_items
    }
    if max_molecules >= len(group_items):
        # Preserve every observed stratum when the requested sample permits it.
        quotas = {key: 1 for key, _ in group_items}
    else:
        quotas = {key: int(np.floor(raw_quotas[key])) for key, _ in group_items}

    remaining = max_molecules - sum(quotas.values())
    while remaining > 0:
        eligible = [
            (key, items)
            for key, items in group_items
            if quotas[key] < len(items)
        ]
        if not eligible:
            break
        key, _ = max(
            eligible,
            key=lambda item: (
                raw_quotas[item[0]] - quotas[item[0]],
                len(item[1]) - quotas[item[0]],
                item[0],
            ),
        )
        quotas[key] += 1
        remaining -= 1

    selected: dict[int, MoleculeCandidate] = {}
    for key, items in group_items:
        ordered = sorted(
            items,
            key=lambda candidate: hashlib.sha256(
                f"{seed}:{candidate.molecule_id}".encode("utf-8")
            ).digest(),
        )
        for candidate in ordered[: quotas[key]]:
            selected[candidate.row_id] = candidate
    return selected


def _read_identifier_file(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _dense_signal(
    values: list[float],
    acquisition: AcquisitionFamily,
    candidate: MoleculeCandidate,
    split: str,
    source_id: str,
    formula: str,
) -> SpectralSignal:
    definitions = {
        AcquisitionFamily.IR: (
            np.linspace(400.0, 4000.0, 1800),
            AxisType.WAVENUMBER,
            AxisUnit.INVERSE_CENTIMETER,
        ),
        AcquisitionFamily.H1_NMR: (
            np.linspace(10.0, -2.0, 10_000),
            AxisType.CHEMICAL_SHIFT,
            AxisUnit.PPM,
        ),
        AcquisitionFamily.C13_NMR: (
            np.linspace(230.0, -20.0, 10_000),
            AxisType.CHEMICAL_SHIFT,
            AxisUnit.PPM,
        ),
    }
    coordinate, axis_type, unit = definitions[acquisition]
    if len(values) != coordinate.size:
        raise ValueError(f"{source_id} has {len(values)} points; expected {coordinate.size}")
    return SpectralSignal(
        coordinate=coordinate,
        intensity=np.asarray(values),
        axis_type=axis_type,
        axis_unit=unit,
        acquisition=acquisition,
        molecule_id=candidate.molecule_id,
        source="multimodal-spectroscopic-dataset",
        source_id=source_id,
        split=split,
        scaffold_id=candidate.scaffold_id,
        labels=candidate.labels,
        metadata={"representation": "dense", "molecular_formula": formula},
    )


def _ms_signal(
    peaks: list[list[float]],
    acquisition: AcquisitionFamily,
    energy: int,
    candidate: MoleculeCandidate,
    split: str,
    source_id: str,
    formula: str,
) -> SpectralSignal:
    peak_array = np.asarray(peaks, dtype=np.float64)
    if peak_array.ndim != 2 or peak_array.shape[1] != 2 or peak_array.shape[0] < 1:
        raise ValueError(f"{source_id} is not a valid [m/z, intensity] peak list")
    if peak_array.shape[0] == 1:
        peak_array = np.vstack((np.asarray([[0.0, 0.0]]), peak_array))
    adduct_mass = 1.007276
    expected_precursor = (
        candidate.exact_mass + adduct_mass
        if acquisition == AcquisitionFamily.MSMS_POSITIVE
        else max(candidate.exact_mass - adduct_mass, 0.0)
    )
    precursor = max(expected_precursor, float(peak_array[:, 0].max()))
    return SpectralSignal(
        coordinate=peak_array[:, 0],
        intensity=peak_array[:, 1],
        axis_type=AxisType.MASS_TO_CHARGE,
        axis_unit=AxisUnit.MZ,
        acquisition=acquisition,
        molecule_id=candidate.molecule_id,
        source="multimodal-spectroscopic-dataset",
        source_id=source_id,
        split=split,
        scaffold_id=candidate.scaffold_id,
        labels=candidate.labels,
        metadata={
            "representation": "peak_list",
            "coordinate_range": [0.0, precursor],
            "precursor_mz": precursor,
            "collision_energy": energy,
            "molecular_formula": formula,
        },
    )


def build_neurips_store(
    parquet_path: str | Path,
    output_root: str | Path,
    *,
    smarts_definitions: str | Path,
    license_audit: str | Path,
    max_molecules: int = 100_000,
    n_bins: int = 4096,
    seed: int = 17,
    excluded_molecules_file: str | Path | None = None,
    excluded_scaffolds_file: str | Path | None = None,
    strict_scaffold: bool = False,
) -> Path:
    audit = require_approved_license(license_audit, "multimodal-spectroscopic-dataset")
    labeler = SmartsLabeler(smarts_definitions)
    candidates = scan_candidates(
        parquet_path,
        labeler,
        excluded_molecules=_read_identifier_file(excluded_molecules_file),
        excluded_scaffolds=_read_identifier_file(excluded_scaffolds_file),
        strict_scaffold=strict_scaffold,
    )
    selected = stratified_candidate_sample(candidates, max_molecules, seed)
    assignments = scaffold_split(
        {item.molecule_id: item.scaffold_id for item in selected.values()}, seed=seed
    )
    writer = CanonicalStoreWriter(
        output_root,
        n_records=len(selected) * 9,
        n_bins=n_bins,
        n_labels=len(labeler.names),
    )
    processor = SignalProcessor(n_bins=n_bins)
    dataset = _arrow_dataset(parquet_path)
    scanner = dataset.scanner(columns=list(REQUIRED_COLUMNS), batch_size=32)
    written_molecules: set[str] = set()
    dense_mapping = {
        "ir_spectra": AcquisitionFamily.IR,
        "h_nmr_spectra": AcquisitionFamily.H1_NMR,
        "c_nmr_spectra": AcquisitionFamily.C13_NMR,
    }
    ms_mapping = {
        "msms_cfmid_positive_10ev": (AcquisitionFamily.MSMS_POSITIVE, 10),
        "msms_cfmid_positive_20ev": (AcquisitionFamily.MSMS_POSITIVE, 20),
        "msms_cfmid_positive_40ev": (AcquisitionFamily.MSMS_POSITIVE, 40),
        "msms_cfmid_negative_10ev": (AcquisitionFamily.MSMS_NEGATIVE, 10),
        "msms_cfmid_negative_20ev": (AcquisitionFamily.MSMS_NEGATIVE, 20),
        "msms_cfmid_negative_40ev": (AcquisitionFamily.MSMS_NEGATIVE, 40),
    }
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            row_id = int(row[ROW_ID_COLUMN])
            candidate = selected.get(row_id)
            if candidate is None or candidate.molecule_id in written_molecules:
                continue
            split = assignments[candidate.molecule_id]
            formula = row["molecular_formula"] or ""
            for column, acquisition in dense_mapping.items():
                signal = _dense_signal(
                    row[column],
                    acquisition,
                    candidate,
                    split,
                    f"{row_id}:{column}",
                    formula,
                )
                writer.append(processor(signal))
            for column, (acquisition, energy) in ms_mapping.items():
                signal = _ms_signal(
                    row[column],
                    acquisition,
                    energy,
                    candidate,
                    split,
                    f"{row_id}:{column}",
                    formula,
                )
                writer.append(processor(signal))
            written_molecules.add(candidate.molecule_id)
    if len(written_molecules) != len(selected):
        missing = len(selected) - len(written_molecules)
        raise RuntimeError(f"failed to locate {missing} selected molecules during the second pass")
    return writer.finalize(
        provenance={
            "kind": "neurips_2024_multimodal_spectroscopy",
            "source": str(Path(parquet_path).resolve()),
            "seed": seed,
            "molecule_candidates": len(candidates),
            "selected_molecules": len(selected),
            "functional_groups": labeler.names,
            "split_sha256": split_digest(assignments),
            "license_audit_entry": dict(audit),
            "strict_scaffold_exclusion": strict_scaffold,
        }
    )
