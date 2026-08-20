"""Leakage-audited molecular-formula targets for weak chemistry distillation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .data import ID_TO_SPLIT
from .manifest import sha256_file, write_json_atomic
from .neurips import _arrow_dataset, _iter_dataset_rows


FORMULA_ELEMENTS = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I")
FORMULA_FEATURES = (*FORMULA_ELEMENTS, "other", "total_atoms", "heavy_atoms")
ATOMIC_MASSES = {
    "H": 1.00794,
    "B": 10.811,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.998403,
    "Si": 28.0855,
    "P": 30.973762,
    "S": 32.065,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.90447,
}
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_molecular_formula(formula: str) -> dict[str, int]:
    """Parse a conventional sum formula into element counts."""

    counts: dict[str, int] = {}
    for element, count in _FORMULA_TOKEN.findall(formula or ""):
        counts[element] = counts.get(element, 0) + (int(count) if count else 1)
    return counts


def formula_feature_vector(formula: str) -> np.ndarray:
    counts = parse_molecular_formula(formula)
    known = [counts.get(element, 0) for element in FORMULA_ELEMENTS]
    other = sum(count for element, count in counts.items() if element not in FORMULA_ELEMENTS)
    total = sum(counts.values())
    heavy = total - counts.get("H", 0)
    return np.asarray([*known, other, total, heavy], dtype=np.float32)


def build_formula_targets(
    data_root: str | Path,
    parquet_root: str | Path,
    output_root: str | Path,
) -> Path:
    """Build train-standardized formula targets aligned to molecule indices.

    Canonical source IDs retain the upstream global Parquet row identifier, so
    formulas can be recovered without rescanning SMILES or reconstructing the
    downstream SMARTS labels.
    """

    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    target_path = output_root / "targets.npy"
    manifest_path = output_root / "manifest.json"
    if target_path.exists() or manifest_path.exists():
        raise FileExistsError(f"formula target output must be empty: {output_root}")

    vocabularies_path = data_root / "vocabularies.json"
    vocabularies = json.loads(vocabularies_path.read_text())
    molecules = np.load(data_root / "molecule_index.npy", mmap_mode="r")
    sources = np.load(data_root / "source_index.npy", mmap_mode="r")
    split = np.load(data_root / "split.npy", mmap_mode="r")
    molecule_ids, first = np.unique(molecules, return_index=True)
    if not np.array_equal(molecule_ids, np.arange(len(vocabularies["molecules"]))):
        raise ValueError("canonical molecule indices are not contiguous")
    upstream_rows: dict[int, int] = {}
    for molecule_index, record in zip(molecule_ids, first, strict=True):
        source_id = vocabularies["sources"][int(sources[record])]
        try:
            upstream_row = int(source_id.split(":", 1)[0])
        except ValueError as error:
            raise ValueError(f"source ID does not contain an upstream row: {source_id}") from error
        upstream_rows[upstream_row] = int(molecule_index)

    raw = np.zeros((molecule_ids.size, len(FORMULA_FEATURES)), dtype=np.float32)
    found = np.zeros(molecule_ids.size, dtype=bool)
    dataset = _arrow_dataset(parquet_root)
    for row_id, row in _iter_dataset_rows(
        dataset, ["molecular_formula"], batch_size=8192
    ):
        molecule_index = upstream_rows.get(row_id)
        if molecule_index is None:
            continue
        raw[molecule_index] = formula_feature_vector(row["molecular_formula"] or "")
        found[molecule_index] = True
        if bool(found.all()):
            break
    if not bool(found.all()):
        raise RuntimeError(f"failed to recover {int((~found).sum())} molecular formulas")

    molecule_split = np.asarray(split[first])
    train_id = next(key for key, value in ID_TO_SPLIT.items() if value == "train")
    transformed = np.log1p(raw)
    train = transformed[molecule_split == train_id]
    raw_mean = train.mean(axis=0)
    raw_std = train.std(axis=0)
    active = raw_std > 1e-6
    if not bool(active.any()):
        raise ValueError("every formula feature is constant in the training split")
    mean = raw_mean[active]
    std = raw_std[active]
    targets = ((transformed[:, active] - mean) / std).astype(np.float32)
    np.save(target_path, targets, allow_pickle=False)
    manifest: dict[str, Any] = {
        "format_version": 1,
        "kind": "molecular_formula_distillation_targets",
        "data_root": str(data_root.resolve()),
        "dataset_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "vocabularies_sha256": sha256_file(vocabularies_path),
        "parquet_root": str(Path(parquet_root).resolve()),
        "n_molecules": int(targets.shape[0]),
        "dimension": int(targets.shape[1]),
        "features": [
            feature for feature, keep in zip(FORMULA_FEATURES, active, strict=True) if keep
        ],
        "excluded_constant_features": [
            feature for feature, keep in zip(FORMULA_FEATURES, active, strict=True) if not keep
        ],
        "transform": "train-split z-score of log1p counts",
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
        "targets_sha256": sha256_file(target_path),
        "downstream_smarts_used": False,
    }
    write_json_atomic(manifest, manifest_path)
    return manifest_path


def load_formula_targets(
    root: str | Path, data_root: str | Path
) -> tuple[np.ndarray, dict[str, Any]]:
    root = Path(root)
    data_root = Path(data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["dataset_manifest_sha256"] != sha256_file(data_root / "manifest.json"):
        raise ValueError("formula targets do not match the canonical dataset manifest")
    if manifest["vocabularies_sha256"] != sha256_file(data_root / "vocabularies.json"):
        raise ValueError("formula targets do not match the molecule vocabulary")
    target_path = root / "targets.npy"
    if manifest["targets_sha256"] != sha256_file(target_path):
        raise ValueError("formula target hash does not match")
    targets = np.load(target_path, mmap_mode="r")
    if list(targets.shape) != [manifest["n_molecules"], manifest["dimension"]]:
        raise ValueError("formula target shape does not match its manifest")
    return targets, manifest


def formula_matching_features(
    targets: np.ndarray, manifest: dict[str, Any]
) -> np.ndarray:
    """Build formula/mass features used only to choose contrastive negatives.

    Formula targets are already train-standardized log-counts. Dividing their
    block by its square-root dimension gives composition and mass comparable
    influence in Euclidean neighbor selection. The mass coordinate is recovered
    from the manifest's train-only normalization and uses a fixed 100 Da scale,
    so no validation/test statistic enters pretraining.
    """

    values = np.asarray(targets, dtype=np.float32)
    features = list(manifest["features"])
    mean = np.asarray(manifest["train_mean"], dtype=np.float32)
    std = np.asarray(manifest["train_std"], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(features):
        raise ValueError("formula targets and manifest features do not match")
    transformed = values * std + mean
    counts = np.maximum(0.0, np.expm1(transformed))
    mass = np.zeros(values.shape[0], dtype=np.float32)
    for index, feature in enumerate(features):
        if feature in ATOMIC_MASSES:
            mass += counts[:, index] * ATOMIC_MASSES[feature]
    composition = values / np.sqrt(float(values.shape[1]))
    return np.concatenate((composition, (mass / 100.0)[:, None]), axis=1).astype(
        np.float32
    )
