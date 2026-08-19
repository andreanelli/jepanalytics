"""Reproducible experimental MassBank evaluation-store preparation."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .adapters import (
    SmartsLabeler,
    massbank_record_signal,
    require_approved_license,
)
from .data import CanonicalSpectraDataset, CanonicalStoreWriter, SPLIT_TO_ID
from .manifest import sha256_file
from .preprocessing import DEFAULT_MS_MZ_RANGE, SignalProcessor


DEFAULT_ALLOWED_LICENSES = (
    "CC0",
    "CC BY",
    "CC BY-SA",
    "CC BY-NC",
    "CC BY-NC-SA",
)


def _reference_train_identifiers(root: str | Path) -> tuple[set[str], set[str]]:
    dataset = CanonicalSpectraDataset(root)
    rows = np.flatnonzero(dataset.arrays["split"] == SPLIT_TO_ID["train"])
    molecule_indices = np.unique(dataset.arrays["molecule_index"][rows])
    scaffold_indices = np.unique(dataset.arrays["scaffold_index"][rows])
    return (
        {dataset.vocabularies["molecules"][int(index)] for index in molecule_indices},
        {dataset.vocabularies["scaffolds"][int(index)] for index in scaffold_indices},
    )


def build_massbank_store(
    input_root: str | Path,
    output_root: str | Path,
    *,
    smarts_definitions: str | Path,
    license_audit: str | Path,
    release: str,
    release_commit: str,
    source_archive: str | Path | None = None,
    reference_data: str | Path | None = None,
    allowed_licenses: Sequence[str] = DEFAULT_ALLOWED_LICENSES,
    n_bins: int = 4096,
    ms_mz_range: tuple[float, float] = DEFAULT_MS_MZ_RANGE,
    include_precursor_metadata: bool = False,
) -> Path:
    """Build an evaluation-only canonical store from open MassBank records."""

    audit_entry = require_approved_license(
        license_audit, "massbank", evaluation_only=True
    )
    input_root = Path(input_root)
    paths = sorted(input_root.rglob("*.txt"))
    if not paths:
        raise ValueError(f"no MassBank text records found under {input_root}")

    rejection_counts: dict[str, int] = {}
    accepted_paths: list[Path] = []
    molecule_to_scaffold: dict[str, str] = {}
    smiles_by_molecule: dict[str, Counter[str]] = {}
    acquisition_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    for path in paths:
        signal = massbank_record_signal(
            path,
            source=f"massbank-{release}",
            allowed_licenses=allowed_licenses,
            rejection_counts=rejection_counts,
        )
        if signal is None:
            continue
        accepted_paths.append(path)
        molecule_to_scaffold[signal.molecule_id] = signal.scaffold_id or ""
        smiles_by_molecule.setdefault(signal.molecule_id, Counter())[
            str(signal.metadata["canonical_smiles"])
        ] += 1
        acquisition_counts[signal.acquisition.name] += 1
        license_counts[str(signal.metadata["license"])] += 1
    if not accepted_paths:
        raise ValueError("no MassBank records passed the evaluation filters")

    reference_molecules: set[str] = set()
    reference_scaffolds: set[str] = set()
    if reference_data is not None:
        reference_molecules, reference_scaffolds = _reference_train_identifiers(
            reference_data
        )
    exact_overlap = set(molecule_to_scaffold) & reference_molecules
    scaffold_overlap = set(molecule_to_scaffold.values()) & reference_scaffolds

    labeler = SmartsLabeler(smarts_definitions)
    representative_smiles = {
        molecule: sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for molecule, counts in smiles_by_molecule.items()
    }
    labels_by_molecule = {
        molecule: labeler(smiles)
        for molecule, smiles in representative_smiles.items()
    }
    writer = CanonicalStoreWriter(
        output_root,
        n_records=len(accepted_paths),
        n_bins=n_bins,
        n_labels=len(labeler.names),
    )
    processor = SignalProcessor(
        n_bins=n_bins,
        ms_mz_range=ms_mz_range,
        include_precursor_metadata=include_precursor_metadata,
    )
    for path in accepted_paths:
        signal = massbank_record_signal(
            path,
            labeler=None,
            source=f"massbank-{release}",
            allowed_licenses=allowed_licenses,
        )
        if signal is None:
            raise RuntimeError(f"accepted MassBank record changed between passes: {path}")
        writer.append(
            processor(
                replace(
                    signal,
                    split="test",
                    labels=labels_by_molecule[signal.molecule_id],
                )
            )
        )

    archive = None if source_archive is None else Path(source_archive)
    provenance = {
        "ms_mz_range": list(ms_mz_range),
        "include_precursor_metadata": include_precursor_metadata,
        "kind": "massbank_experimental_evaluation",
        "evaluation_only": True,
        "release": release,
        "release_commit": release_commit,
        "source": str(input_root.resolve()),
        "source_archive": None if archive is None else str(archive.resolve()),
        "source_archive_sha256": (
            None if archive is None else sha256_file(archive)
        ),
        "license_audit_entry": audit_entry,
        "allowed_record_licenses": list(allowed_licenses),
        "record_license_counts": dict(sorted(license_counts.items())),
        "input_records": len(paths),
        "accepted_records": len(accepted_paths),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "acquisition_counts": dict(sorted(acquisition_counts.items())),
        "molecules": len(molecule_to_scaffold),
        "structure_representative_rule": (
            "most frequent canonical SMILES per standard InChIKey; lexical tie-break"
        ),
        "molecules_with_multiple_canonical_smiles": sum(
            len(counts) > 1 for counts in smiles_by_molecule.values()
        ),
        "functional_groups": labeler.names,
        "reference_data": (
            None if reference_data is None else str(Path(reference_data).resolve())
        ),
        "exact_molecule_overlap_count": len(exact_overlap),
        "scaffold_overlap_count": len(scaffold_overlap),
        "exact_molecule_unseen_count": len(molecule_to_scaffold) - len(exact_overlap),
        "scaffold_unseen_count": sum(
            scaffold not in reference_scaffolds
            for scaffold in molecule_to_scaffold.values()
        ),
    }
    return writer.finalize(provenance=provenance)
