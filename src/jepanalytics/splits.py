"""Leakage-resistant molecule/scaffold splitting and audit utilities."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class MoleculeRecord:
    molecule_id: str
    scaffold_id: str
    source_id: str
    acquisition: int


def scaffold_split(
    molecule_to_scaffold: Mapping[str, str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 17,
) -> dict[str, str]:
    """Assign whole scaffold groups while approximately balancing molecules."""

    if abs(sum(fractions) - 1.0) > 1e-8 or any(f <= 0 for f in fractions):
        raise ValueError("fractions must be positive and sum to one")
    groups: dict[str, list[str]] = defaultdict(list)
    for molecule, scaffold in molecule_to_scaffold.items():
        groups[scaffold or f"__singleton__:{molecule}"].append(molecule)
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    names = ("train", "validation", "test")
    targets = [fractions[i] * len(molecule_to_scaffold) for i in range(3)]
    counts = [0, 0, 0]
    result: dict[str, str] = {}
    for _, molecules in items:
        index = min(range(3), key=lambda i: counts[i] / max(targets[i], 1e-8))
        for molecule in molecules:
            result[molecule] = names[index]
        counts[index] += len(molecules)
    return result


def assert_no_split_leakage(records: Sequence[MoleculeRecord], assignments: Mapping[str, str]) -> None:
    molecule_splits: dict[str, set[str]] = defaultdict(set)
    scaffold_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.molecule_id not in assignments:
            raise AssertionError(f"missing split for molecule {record.molecule_id}")
        split = assignments[record.molecule_id]
        molecule_splits[record.molecule_id].add(split)
        scaffold_splits[record.scaffold_id].add(split)
        source_splits[record.source_id].add(split)
    violations = {
        "molecule": [key for key, values in molecule_splits.items() if len(values) > 1],
        "scaffold": [key for key, values in scaffold_splits.items() if key and len(values) > 1],
        "source": [key for key, values in source_splits.items() if len(values) > 1],
    }
    if any(violations.values()):
        raise AssertionError(f"split leakage detected: {violations}")


def split_digest(assignments: Mapping[str, str]) -> str:
    payload = "\n".join(f"{key}\t{assignments[key]}" for key in sorted(assignments))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exclude_external_structures(
    molecule_to_scaffold: Mapping[str, str],
    *,
    external_molecules: Iterable[str],
    external_scaffolds: Iterable[str] = (),
    strict_scaffold: bool = False,
) -> set[str]:
    excluded = set(external_molecules)
    if strict_scaffold:
        blocked = set(external_scaffolds)
        excluded.update(
            molecule for molecule, scaffold in molecule_to_scaffold.items() if scaffold in blocked
        )
    return excluded
