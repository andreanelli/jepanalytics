from pathlib import Path
from collections import Counter

import numpy as np
import pytest

from jepanalytics.data import (
    CanonicalSpectraDataset,
    MultiViewSpectrumDataset,
    PairedSpectrumDataset,
    build_jsonl_store,
    build_synthetic_store,
    verify_canonical_store,
)
from jepanalytics.splits import MoleculeRecord, assert_no_split_leakage, scaffold_split


def test_scaffolds_are_never_split():
    mapping = {f"m{i}": f"s{i // 3}" for i in range(30)}
    assignment = scaffold_split(mapping, seed=4)
    for scaffold in set(mapping.values()):
        assert len({assignment[m] for m, s in mapping.items() if s == scaffold}) == 1
    assert set(assignment.values()) == {"train", "validation", "test"}


def test_leakage_assertion_detects_bad_external_assignment():
    records = [
        MoleculeRecord("m1", "s", "r1", 0),
        MoleculeRecord("m2", "s", "r2", 1),
    ]
    with pytest.raises(AssertionError, match="scaffold"):
        assert_no_split_leakage(records, {"m1": "train", "m2": "test"})


def test_synthetic_store_hashes_splits_and_cross_modal_pairs(tmp_path: Path):
    root = tmp_path / "store"
    build_synthetic_store(root, n_molecules=24, n_bins=128, seed=3)
    verification = verify_canonical_store(root)
    assert verification["valid"]
    assert verification["n_records"] == 24 * 5
    train = CanonicalSpectraDataset(root, "train")
    pairs = PairedSpectrumDataset(train)
    first, second = pairs[0]
    assert first["molecule_index"].item() == second["molecule_index"].item()
    assert first["acquisition"].item() != second["acquisition"].item()


def test_pair_schedule_balances_acquisition_families(tmp_path: Path):
    root = tmp_path / "balanced-store"
    build_synthetic_store(root, n_molecules=160, n_bins=64, seed=7)
    train = CanonicalSpectraDataset(root, "train")
    pairs = PairedSpectrumDataset(train, seed=7)
    counts = Counter()
    for index in range(len(pairs)):
        first, second = pairs[index]
        counts[tuple(sorted((first["acquisition"].item(), second["acquisition"].item())))] += 1
    assert len(counts) == 10
    assert max(counts.values()) - min(counts.values()) <= 1


def test_pair_schedule_samples_one_collision_energy_within_selected_family(
    tmp_path: Path,
):
    root = tmp_path / "replicate-store"
    build_synthetic_store(root, n_molecules=40, n_bins=64, seed=4)
    train = CanonicalSpectraDataset(root, "train")
    pairs = PairedSpectrumDataset(train, seed=4)
    molecule = pairs.groups[0]
    assert all(len(records) == 1 for records in molecule.values())
    # Synthetic smoke data have one record per family; the real pilot has three
    # MS collision energies. The sampler's invariant is still one record from
    # each family, never weighting a family by its replicate count.
    first, second = pairs[0]
    assert first["record_index"].ndim == second["record_index"].ndim == 0


def test_multiview_sampler_returns_one_record_per_family(tmp_path: Path):
    root = tmp_path / "multiview-store"
    build_synthetic_store(root, n_molecules=40, n_bins=64, seed=6)
    train = CanonicalSpectraDataset(root, "train")
    samples = MultiViewSpectrumDataset(train, views_per_molecule=5, seed=6)
    views = samples[0]
    assert len(views) == 5
    assert {view["acquisition"].item() for view in views} == set(range(5))
    assert len({view["molecule_index"].item() for view in views}) == 1


def test_jsonl_import_assigns_scaffold_splits_and_supports_missing_labels(tmp_path: Path):
    import json

    path = tmp_path / "signals.jsonl"
    rows = []
    for molecule in range(12):
        for acquisition in ("IR", "H1_NMR"):
            rows.append(
                {
                    "coordinate": [1, 2, 3],
                    "intensity": [0, 1, 0],
                    "axis_type": "WAVENUMBER" if acquisition == "IR" else "CHEMICAL_SHIFT",
                    "axis_unit": "INVERSE_CENTIMETER" if acquisition == "IR" else "PPM",
                    "acquisition": acquisition,
                    "molecule_id": f"m{molecule}",
                    "source": "fixture",
                    "source_id": f"m{molecule}-{acquisition}",
                    "scaffold_id": f"s{molecule // 2}",
                    "labels": [1, 0] if molecule % 2 else None,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    root = tmp_path / "jsonl-store"
    build_jsonl_store(path, root, n_bins=32)
    assert verify_canonical_store(root)["valid"]
    dataset = CanonicalSpectraDataset(root)
    assert "label_available" in dataset.arrays
    assert dataset.arrays["label_available"].sum() == 12
