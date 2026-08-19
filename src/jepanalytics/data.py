"""Canonical on-disk store, paired sampling, and deterministic smoke data."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .preprocessing import ProcessedSignal, SignalProcessor
from .signal import AcquisitionFamily, AxisType, AxisUnit, SpectralSignal, default_axis
from .splits import scaffold_split, split_digest


SPLIT_TO_ID = {None: 255, "train": 0, "validation": 1, "test": 2}
ID_TO_SPLIT = {value: key for key, value in SPLIT_TO_ID.items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CanonicalStoreWriter:
    """Streaming writer for memory-mappable, model-ready arrays."""

    def __init__(
        self,
        root: str | Path,
        *,
        n_records: int,
        n_bins: int,
        n_labels: int = 0,
        metadata_dim: int = 10,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise FileExistsError(f"canonical store must be empty: {self.root}")
        self.n_records = n_records
        self.n_bins = n_bins
        self.n_labels = n_labels
        self.metadata_dim = metadata_dim
        self.position = 0
        self._arrays: dict[str, np.memmap] = {
            "intensity": np.lib.format.open_memmap(
                self.root / "intensity.npy", mode="w+", dtype=np.float16, shape=(n_records, n_bins)
            ),
            "continuous_metadata": np.lib.format.open_memmap(
                self.root / "continuous_metadata.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_records, metadata_dim),
            ),
            "axis_type": np.lib.format.open_memmap(
                self.root / "axis_type.npy", mode="w+", dtype=np.uint8, shape=(n_records,)
            ),
            "axis_unit": np.lib.format.open_memmap(
                self.root / "axis_unit.npy", mode="w+", dtype=np.uint8, shape=(n_records,)
            ),
            "acquisition": np.lib.format.open_memmap(
                self.root / "acquisition.npy", mode="w+", dtype=np.uint8, shape=(n_records,)
            ),
            "split": np.lib.format.open_memmap(
                self.root / "split.npy", mode="w+", dtype=np.uint8, shape=(n_records,)
            ),
            "molecule_index": np.lib.format.open_memmap(
                self.root / "molecule_index.npy", mode="w+", dtype=np.int32, shape=(n_records,)
            ),
            "source_index": np.lib.format.open_memmap(
                self.root / "source_index.npy", mode="w+", dtype=np.int32, shape=(n_records,)
            ),
            "scaffold_index": np.lib.format.open_memmap(
                self.root / "scaffold_index.npy", mode="w+", dtype=np.int32, shape=(n_records,)
            ),
        }
        if n_labels:
            self._arrays["labels"] = np.lib.format.open_memmap(
                self.root / "labels.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_records, n_labels),
            )
            self._arrays["label_available"] = np.lib.format.open_memmap(
                self.root / "label_available.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(n_records,),
            )
        self._vocab: dict[str, list[str]] = {"molecules": [], "sources": [], "scaffolds": []}
        self._lookup: dict[str, dict[str, int]] = {key: {} for key in self._vocab}

    def _index(self, vocabulary: str, value: str) -> int:
        if value not in self._lookup[vocabulary]:
            self._lookup[vocabulary][value] = len(self._vocab[vocabulary])
            self._vocab[vocabulary].append(value)
        return self._lookup[vocabulary][value]

    def append(self, signal: ProcessedSignal) -> None:
        if self.position >= self.n_records:
            raise IndexError("more records appended than declared")
        if signal.intensity.shape != (self.n_bins,):
            raise ValueError(f"expected {self.n_bins} intensity bins")
        if signal.continuous_metadata.shape != (self.metadata_dim,):
            raise ValueError(f"expected {self.metadata_dim} continuous metadata values")
        if self.n_labels and signal.labels is not None and signal.labels.shape != (self.n_labels,):
            raise ValueError(f"expected {self.n_labels} labels when labels are available")
        row = self.position
        self._arrays["intensity"][row] = signal.intensity
        self._arrays["continuous_metadata"][row] = signal.continuous_metadata
        self._arrays["axis_type"][row] = signal.axis_type
        self._arrays["axis_unit"][row] = signal.axis_unit
        self._arrays["acquisition"][row] = signal.acquisition
        self._arrays["split"][row] = SPLIT_TO_ID[signal.split]
        self._arrays["molecule_index"][row] = self._index("molecules", signal.molecule_id)
        self._arrays["source_index"][row] = self._index("sources", signal.source_id)
        self._arrays["scaffold_index"][row] = self._index(
            "scaffolds", signal.scaffold_id or f"__unknown__:{signal.molecule_id}"
        )
        if self.n_labels:
            self._arrays["labels"][row] = (
                np.zeros(self.n_labels, dtype=np.float32) if signal.labels is None else signal.labels
            )
            self._arrays["label_available"][row] = int(signal.labels is not None)
        self.position += 1

    def finalize(self, *, provenance: Mapping[str, Any] | None = None) -> Path:
        if self.position != self.n_records:
            raise RuntimeError(f"expected {self.n_records} records, received {self.position}")
        for array in self._arrays.values():
            array.flush()
        vocab_path = self.root / "vocabularies.json"
        vocab_path.write_text(json.dumps(self._vocab, indent=2, sort_keys=True) + "\n")
        array_hashes = {
            f"{name}.npy": _file_sha256(self.root / f"{name}.npy") for name in self._arrays
        }
        metadata = {
            "format_version": 1,
            "n_records": self.n_records,
            "n_bins": self.n_bins,
            "n_labels": self.n_labels,
            "metadata_dim": self.metadata_dim,
            "arrays": array_hashes,
            "vocabularies_sha256": _file_sha256(vocab_path),
            "provenance": dict(provenance or {}),
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return path


class CanonicalSpectraDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        *,
        molecule_targets: np.ndarray | None = None,
        molecule_matching_targets: np.ndarray | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.vocabularies = json.loads((self.root / "vocabularies.json").read_text())
        array_names = self.manifest["arrays"]
        self.arrays = {
            name.removesuffix(".npy"): np.load(self.root / name, mmap_mode="r")
            for name in array_names
        }
        self.molecule_targets = molecule_targets
        self.molecule_matching_targets = molecule_matching_targets
        if molecule_targets is not None and molecule_targets.shape[0] != len(
            self.vocabularies["molecules"]
        ):
            raise ValueError("molecule target rows must match the molecule vocabulary")
        if molecule_matching_targets is not None and molecule_matching_targets.shape[0] != len(
            self.vocabularies["molecules"]
        ):
            raise ValueError("molecule matching rows must match the molecule vocabulary")
        if split is None:
            self.indices = np.arange(self.manifest["n_records"], dtype=np.int64)
        else:
            if split not in SPLIT_TO_ID or split is None:
                raise ValueError(f"unknown split {split!r}")
            self.indices = np.flatnonzero(self.arrays["split"] == SPLIT_TO_ID[split])

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        item = {
            "intensity": torch.tensor(self.arrays["intensity"][row], dtype=torch.float32),
            "continuous_metadata": torch.tensor(
                self.arrays["continuous_metadata"][row], dtype=torch.float32
            ),
            "axis_type": torch.tensor(int(self.arrays["axis_type"][row]), dtype=torch.long),
            "axis_unit": torch.tensor(int(self.arrays["axis_unit"][row]), dtype=torch.long),
            "acquisition": torch.tensor(int(self.arrays["acquisition"][row]), dtype=torch.long),
            "molecule_index": torch.tensor(
                int(self.arrays["molecule_index"][row]), dtype=torch.long
            ),
            "record_index": torch.tensor(row, dtype=torch.long),
        }
        if "labels" in self.arrays:
            item["labels"] = torch.tensor(self.arrays["labels"][row], dtype=torch.float32)
            item["label_available"] = torch.tensor(
                bool(self.arrays["label_available"][row]), dtype=torch.bool
            )
        if self.molecule_targets is not None:
            item["chemistry_target"] = torch.tensor(
                self.molecule_targets[int(self.arrays["molecule_index"][row])],
                dtype=torch.float32,
            )
        if self.molecule_matching_targets is not None:
            item["prototype_matching_target"] = torch.tensor(
                self.molecule_matching_targets[
                    int(self.arrays["molecule_index"][row])
                ],
                dtype=torch.float32,
            )
        return item


class PairedSpectrumDataset(Dataset[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]):
    """Choose a balanced acquisition pair from one molecule per epoch.

    The canonical NeurIPS store has three records per MS/MS family and one per
    IR/NMR family. Sampling records directly therefore makes MS+/MS- pairs about
    twelve times more common than IR/H1 pairs. Pair families first, then sample
    one record within each selected family so collision energies do not distort
    the cross-technique objective.
    """

    def __init__(self, base: CanonicalSpectraDataset, seed: int = 17) -> None:
        self.base = base
        self.seed = seed
        self.epoch = 0
        grouped: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for local_index, row in enumerate(base.indices):
            molecule = int(base.arrays["molecule_index"][row])
            acquisition = int(base.arrays["acquisition"][row])
            grouped[molecule][acquisition].append(local_index)
        self.groups = [dict(families) for families in grouped.values()]
        self.pair_schedules: list[list[tuple[int, int]]] = []
        for families in self.groups:
            acquisitions = sorted(families)
            if len(acquisitions) == 1:
                pairs = [(acquisitions[0], acquisitions[0])]
            else:
                pairs = list(itertools.combinations(acquisitions, 2))
                random.Random(self.seed).shuffle(pairs)
            self.pair_schedules.append(pairs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        families = self.groups[index]
        schedule = self.pair_schedules[index]
        first_acquisition, second_acquisition = schedule[(index + self.epoch) % len(schedule)]
        rng = random.Random(
            self.seed + self.epoch * len(self.groups) + index * 9973
        )
        if first_acquisition != second_acquisition and rng.random() < 0.5:
            first_acquisition, second_acquisition = (
                second_acquisition,
                first_acquisition,
            )
        first = rng.choice(families[first_acquisition])
        second_choices = families[second_acquisition]
        if first_acquisition == second_acquisition and len(second_choices) > 1:
            second = rng.choice([candidate for candidate in second_choices if candidate != first])
        else:
            second = rng.choice(second_choices)
        return self.base[first], self.base[second]


class MultiViewSpectrumDataset(Dataset[list[dict[str, torch.Tensor]]]):
    """Sample one record from each requested acquisition family per molecule."""

    def __init__(
        self,
        base: CanonicalSpectraDataset,
        *,
        views_per_molecule: int,
        seed: int = 17,
    ) -> None:
        if views_per_molecule < 2:
            raise ValueError("multi-view sampling requires at least two views")
        self.base = base
        self.views_per_molecule = views_per_molecule
        self.seed = seed
        self.epoch = 0
        grouped: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for local_index, row in enumerate(base.indices):
            molecule = int(base.arrays["molecule_index"][row])
            acquisition = int(base.arrays["acquisition"][row])
            grouped[molecule][acquisition].append(local_index)
        retained = [
            (molecule, dict(families))
            for molecule, families in grouped.items()
            if len(families) >= views_per_molecule
        ]
        self.molecule_indices = np.asarray(
            [molecule for molecule, _families in retained], dtype=np.int64
        )
        self.groups = [families for _molecule, families in retained]
        if not self.groups:
            raise ValueError("no molecule has enough acquisition families for multi-view training")
        self.view_schedules = []
        for families in self.groups:
            schedules = list(
                itertools.combinations(sorted(families), views_per_molecule)
            )
            random.Random(seed).shuffle(schedules)
            self.view_schedules.append(schedules)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> list[dict[str, torch.Tensor]]:
        families = self.groups[index]
        schedules = self.view_schedules[index]
        acquisitions = schedules[(index + self.epoch) % len(schedules)]
        rng = random.Random(
            self.seed + self.epoch * len(self.groups) + index * 9973
        )
        return [self.base[rng.choice(families[acquisition])] for acquisition in acquisitions]


class FormulaMassBatchSampler(Sampler[list[int]]):
    """Form unique-molecule batches within local molecular-mass windows.

    Sorting by the fixed mass feature is O(N log N), unlike an all-pairs formula
    search. Shuffling inside nearby windows changes batches between epochs while
    preserving mass-matched candidates for the in-batch composition search.
    """

    def __init__(
        self,
        molecule_indices: np.ndarray,
        matching_targets: np.ndarray,
        *,
        batch_size: int,
        seed: int,
        window_batches: int = 4,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 2 or window_batches < 1:
            raise ValueError("batch_size and window_batches must be valid")
        molecules = np.asarray(molecule_indices, dtype=np.int64)
        features = np.asarray(matching_targets, dtype=np.float32)
        if molecules.ndim != 1 or features.ndim != 2:
            raise ValueError("molecule indices and matching targets have invalid shapes")
        if not molecules.size or molecules.max() >= features.shape[0]:
            raise ValueError("matching targets do not cover every sampled molecule")
        self.mass = features[molecules, -1]
        self.batch_size = batch_size
        self.seed = seed
        self.window_size = batch_size * window_batches
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self.mass.size // self.batch_size
        return math.ceil(self.mass.size / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 104729)
        order = np.argsort(self.mass, kind="stable")
        windows = [
            order[start : start + self.window_size].copy()
            for start in range(0, order.size, self.window_size)
        ]
        for window in windows:
            rng.shuffle(window)
        rng.shuffle(windows)
        shuffled = np.concatenate(windows)
        for start in range(0, shuffled.size, self.batch_size):
            batch = shuffled[start : start + self.batch_size]
            if batch.size < self.batch_size and self.drop_last:
                continue
            yield batch.tolist()


def paired_collate(
    samples: Sequence[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    def stack(side: int) -> dict[str, torch.Tensor]:
        keys = samples[0][side].keys()
        return {key: torch.stack([sample[side][key] for sample in samples]) for key in keys}

    return stack(0), stack(1)


def multiview_collate(
    samples: Sequence[list[dict[str, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    flattened = [view for sample in samples for view in sample]
    keys = flattened[0].keys()
    return {key: torch.stack([view[key] for view in flattened]) for key in keys}


def _gaussian_trace(axis: np.ndarray, peaks: Iterable[tuple[float, float, float]]) -> np.ndarray:
    output = np.zeros_like(axis)
    for center, height, width in peaks:
        output += height * np.exp(-0.5 * ((axis - center) / width) ** 2)
    return output


def _synthetic_signal(
    molecule: int,
    acquisition: AcquisitionFamily,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> SpectralSignal:
    axis_type, unit = default_axis(acquisition)
    ranges = {
        AcquisitionFamily.IR: (400.0, 4000.0),
        AcquisitionFamily.H1_NMR: (-2.0, 10.0),
        AcquisitionFamily.C13_NMR: (-20.0, 230.0),
        AcquisitionFamily.MSMS_POSITIVE: (20.0, 600.0),
        AcquisitionFamily.MSMS_NEGATIVE: (20.0, 600.0),
    }
    low, high = ranges[acquisition]
    is_ms = acquisition in (AcquisitionFamily.MSMS_POSITIVE, AcquisitionFamily.MSMS_NEGATIVE)
    family_offset = int(acquisition) * 0.071
    peaks: list[tuple[float, float, float]] = []
    for label_index in np.flatnonzero(labels):
        fraction = (0.11 + label_index * 0.097 + family_offset) % 0.86 + 0.06
        center = low + fraction * (high - low) + rng.normal(0, (high - low) * 0.002)
        height = 0.6 + 0.5 * rng.random()
        width = (high - low) * (0.0015 if is_ms else 0.004 + 0.002 * rng.random())
        peaks.append((center, height, width))
    for _ in range(2):
        peaks.append(
            (
                rng.uniform(low, high),
                rng.uniform(0.15, 0.45),
                (high - low) * rng.uniform(0.001, 0.006),
            )
        )
    if is_ms:
        coordinate = np.asarray(sorted(center for center, _, _ in peaks), dtype=np.float64)
        intensity = np.asarray(
            [next(height for center2, height, _ in peaks if center2 == center) for center in coordinate]
        )
        representation = "peak_list"
    else:
        coordinate = np.linspace(low, high, 768)
        intensity = _gaussian_trace(coordinate, peaks)
        intensity += rng.normal(0, 0.005, coordinate.size)
        representation = "dense"
    return SpectralSignal(
        coordinate=coordinate,
        intensity=intensity,
        axis_type=axis_type,
        axis_unit=unit,
        acquisition=acquisition,
        molecule_id=f"synthetic-{molecule:06d}",
        source="jepanalytics-synthetic",
        source_id=f"synthetic-{molecule:06d}-{acquisition.name.lower()}",
        scaffold_id=f"scaffold-{molecule // 4:05d}",
        labels=labels,
        metadata={
            "representation": representation,
            "precursor_mz": 600.0 if is_ms else 0.0,
            "collision_energy": 20.0 if is_ms else 0.0,
            "coordinate_range": [low, high],
        },
    )


def build_synthetic_store(
    root: str | Path,
    *,
    n_molecules: int = 64,
    n_bins: int = 256,
    n_labels: int = 8,
    seed: int = 17,
) -> Path:
    rng = np.random.default_rng(seed)
    molecule_to_scaffold = {
        f"synthetic-{index:06d}": f"scaffold-{index // 4:05d}" for index in range(n_molecules)
    }
    assignments = scaffold_split(molecule_to_scaffold, seed=seed)
    acquisitions = list(AcquisitionFamily)
    writer = CanonicalStoreWriter(
        root,
        n_records=n_molecules * len(acquisitions),
        n_bins=n_bins,
        n_labels=n_labels,
    )
    processor = SignalProcessor(n_bins=n_bins)
    for molecule in range(n_molecules):
        labels = (rng.random(n_labels) < 0.3).astype(np.float32)
        if not labels.any():
            labels[rng.integers(0, n_labels)] = 1.0
        for acquisition in acquisitions:
            signal = _synthetic_signal(molecule, acquisition, labels, rng)
            processed = processor(signal)
            processed = ProcessedSignal(
                **{
                    **asdict(processed),
                    "split": assignments[signal.molecule_id],
                }
            )
            writer.append(processed)
    return writer.finalize(
        provenance={
            "kind": "deterministic_synthetic_smoke_corpus",
            "seed": seed,
            "split_sha256": split_digest(assignments),
        }
    )


def iter_jsonl_signals(path: str | Path) -> Iterator[SpectralSignal]:
    """Read the documented interchange format used by external adapters."""

    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                yield SpectralSignal(
                    coordinate=np.asarray(item["coordinate"]),
                    intensity=np.asarray(item["intensity"]),
                    axis_type=AxisType[item["axis_type"]],
                    axis_unit=AxisUnit[item["axis_unit"]],
                    acquisition=AcquisitionFamily[item["acquisition"]],
                    molecule_id=item["molecule_id"],
                    source=item["source"],
                    source_id=item["source_id"],
                    split=item.get("split"),
                    scaffold_id=item.get("scaffold_id"),
                    labels=None if item.get("labels") is None else np.asarray(item["labels"]),
                    metadata=item.get("metadata", {}),
                )
            except Exception as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc


def build_jsonl_store(
    input_path: str | Path,
    output_root: str | Path,
    *,
    n_bins: int = 4096,
) -> Path:
    signals = list(iter_jsonl_signals(input_path))
    if not signals:
        raise ValueError("JSONL input contains no records")
    split_presence = {signal.split is not None for signal in signals}
    if len(split_presence) > 1:
        raise ValueError("either every JSONL record must have a split or none may have one")
    if split_presence == {False}:
        molecule_to_scaffold: dict[str, str] = {}
        for signal in signals:
            if not signal.scaffold_id:
                raise ValueError(
                    "scaffold_id is required when JSONL records do not carry precomputed splits"
                )
            previous = molecule_to_scaffold.setdefault(signal.molecule_id, signal.scaffold_id)
            if previous != signal.scaffold_id:
                raise ValueError(f"molecule {signal.molecule_id} has inconsistent scaffolds")
        assignments = scaffold_split(molecule_to_scaffold)
        signals = [replace(signal, split=assignments[signal.molecule_id]) for signal in signals]
    n_labels = max((0 if item.labels is None else item.labels.size for item in signals), default=0)
    if any(item.labels is not None and item.labels.size != n_labels for item in signals):
        raise ValueError("all available label vectors must have the same length")
    writer = CanonicalStoreWriter(
        output_root, n_records=len(signals), n_bins=n_bins, n_labels=n_labels
    )
    processor = SignalProcessor(n_bins=n_bins)
    for signal in signals:
        writer.append(processor(signal))
    return writer.finalize(
        provenance={"kind": "jsonl_import", "input_sha256": _file_sha256(Path(input_path))}
    )


def verify_canonical_store(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    hash_failures = []
    for filename, expected in manifest["arrays"].items():
        actual = _file_sha256(root / filename)
        if actual != expected:
            hash_failures.append({"file": filename, "expected": expected, "actual": actual})
    dataset = CanonicalSpectraDataset(root)
    molecule_splits: dict[int, set[int]] = defaultdict(set)
    scaffold_splits: dict[int, set[int]] = defaultdict(set)
    source_splits: dict[int, set[int]] = defaultdict(set)
    for row in dataset.indices:
        split = int(dataset.arrays["split"][row])
        molecule_splits[int(dataset.arrays["molecule_index"][row])].add(split)
        scaffold_splits[int(dataset.arrays["scaffold_index"][row])].add(split)
        source_splits[int(dataset.arrays["source_index"][row])].add(split)
    leakage = {
        "molecules": [key for key, values in molecule_splits.items() if len(values) > 1],
        "scaffolds": [key for key, values in scaffold_splits.items() if len(values) > 1],
        "sources": [key for key, values in source_splits.items() if len(values) > 1],
    }
    split_counts = {
        str(ID_TO_SPLIT[split_id]): int(np.sum(dataset.arrays["split"] == split_id))
        for split_id in sorted(ID_TO_SPLIT)
    }
    valid = not hash_failures and not any(leakage.values())
    return {
        "valid": valid,
        "hash_failures": hash_failures,
        "leakage": leakage,
        "split_counts": split_counts,
        "n_records": manifest["n_records"],
        "n_molecules": len(dataset.vocabularies["molecules"]),
    }
