"""Validation-tuned supervised raw-signal ceilings for representation audits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .baselines import CoordinateAwareSpectrumCNN
from .data import CanonicalSpectraDataset
from .evaluation import SPLIT_SUBSET_OFFSETS, restrict_dataset_molecules
from .metrics import macro_auprc, macro_f1
from .probe_audit import ACQUISITION_NAMES, per_label_metrics, random_molecule_indices
from .training import move_batch, resolve_device, seed_everything


@dataclass(frozen=True, slots=True)
class CNNFitSpec:
    learning_rate: float
    weight_decay: float


DEFAULT_CNN_SPECS = (
    CNNFitSpec(3e-4, 1e-3),
    CNNFitSpec(1e-3, 1e-3),
)


@dataclass(slots=True)
class FittedCNN:
    model: nn.Module
    best_epoch: int
    validation_auprc: float


def _family_local_indices(dataset: CanonicalSpectraDataset, acquisition: int) -> np.ndarray:
    rows = dataset.indices
    mask = np.asarray(dataset.arrays["acquisition"][rows]) == acquisition
    if "label_available" in dataset.arrays:
        mask &= np.asarray(dataset.arrays["label_available"][rows], dtype=bool)
    return np.flatnonzero(mask)


def _selected_family_indices(
    dataset: CanonicalSpectraDataset,
    family_indices: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    rows = dataset.indices[family_indices]
    local = random_molecule_indices(
        np.asarray(dataset.arrays["molecule_index"][rows]), fraction, seed
    )
    return family_indices[local]


def _labels_for_indices(
    dataset: CanonicalSpectraDataset, local_indices: np.ndarray
) -> np.ndarray:
    rows = dataset.indices[local_indices]
    return np.asarray(dataset.arrays["labels"][rows], dtype=np.float32)


def _loader(
    dataset: CanonicalSpectraDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices.astype(np.int64).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


@torch.inference_mode()
def _predict(
    model: nn.Module,
    dataset: CanonicalSpectraDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in _loader(
        dataset, indices, batch_size=batch_size, shuffle=False, seed=0
    ):
        batch = move_batch(batch, device)
        probabilities.append(torch.sigmoid(model(batch["intensity"])).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def fit_coordinate_cnn(
    train_dataset: CanonicalSpectraDataset,
    train_indices: np.ndarray,
    validation_dataset: CanonicalSpectraDataset,
    validation_indices: np.ndarray,
    *,
    spec: CNNFitSpec,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    evaluation_interval: int = 5,
    patience: int = 5,
) -> FittedCNN:
    """Fit one family-specific CNN and retain the best scaffold-validation epoch."""

    seed_everything(seed)
    target_device = torch.device(device)
    labels = _labels_for_indices(train_dataset, train_indices)
    n_labels = labels.shape[1]
    model = CoordinateAwareSpectrumCNN(n_labels).to(target_device)
    positive = torch.as_tensor(labels.sum(axis=0), dtype=torch.float32, device=target_device)
    pos_weight = ((labels.shape[0] - positive) / positive.clamp_min(1.0)).clamp(max=20.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    best_score = -float("inf")
    best_epoch = evaluation_interval
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    train_loader = _loader(
        train_dataset,
        train_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(batch["intensity"]), batch["labels"], pos_weight=pos_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch % evaluation_interval and epoch != epochs:
            continue
        probability, validation_y = _predict(
            model,
            validation_dataset,
            validation_indices,
            batch_size=batch_size,
            device=target_device,
        )
        score = macro_auprc(validation_y, probability)
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("CNN training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return FittedCNN(model, best_epoch, float(best_score))


def evaluate_supervised_cnn_ceiling(
    data_root: str | Path,
    *,
    fraction: float = 0.01,
    seeds: Sequence[int] = (11, 17, 23, 31, 47),
    tuning_seed: int = 17,
    epochs: int = 100,
    batch_size: int = 128,
    specs: Sequence[CNNFitSpec] = DEFAULT_CNN_SPECS,
    acquisitions: Sequence[int] = tuple(range(len(ACQUISITION_NAMES))),
    max_validation_molecules: int | None = None,
    max_molecules_per_split: Mapping[str, int] | None = None,
    subset_seed: int = 20260810,
    device: str = "auto",
) -> dict[str, Any]:
    """Measure a strong supervised specialist ceiling at a fixed label budget.

    Pass the same ``max_molecules_per_split`` and ``subset_seed`` as the
    frozen-probe audit being compared against: the split-level molecule
    subsets then match the audited embedding cache exactly, so the CNN and
    the probe draw identical labeled molecules per seed and are scored on
    an identical test population.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    if max_validation_molecules is not None and max_molecules_per_split is not None:
        raise ValueError(
            "pass either max_molecules_per_split (cache-matched subsets) or the "
            "legacy max_validation_molecules, not both"
        )
    target_device = resolve_device(device)
    train = CanonicalSpectraDataset(data_root, "train")
    validation = CanonicalSpectraDataset(data_root, "validation")
    test = CanonicalSpectraDataset(data_root, "test")
    if "labels" not in train.arrays:
        raise ValueError("supervised CNN evaluation requires labels")
    limits = dict(max_molecules_per_split or {})
    for split_name, dataset in (("train", train), ("validation", validation), ("test", test)):
        restrict_dataset_molecules(
            dataset, limits.get(split_name), subset_seed + SPLIT_SUBSET_OFFSETS[split_name]
        )

    tuning: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    if any(index < 0 or index >= len(ACQUISITION_NAMES) for index in acquisitions):
        raise ValueError("acquisitions must contain indices between 0 and 4")
    for acquisition in acquisitions:
        acquisition_name = ACQUISITION_NAMES[acquisition]
        train_family = _family_local_indices(train, acquisition)
        validation_family = _family_local_indices(validation, acquisition)
        test_family = _family_local_indices(test, acquisition)
        if max_validation_molecules is not None:
            rows = validation.indices[validation_family]
            molecules = np.asarray(validation.arrays["molecule_index"][rows])
            unique = np.unique(molecules)
            if unique.size > max_validation_molecules:
                rng = np.random.default_rng(tuning_seed)
                chosen = rng.choice(
                    unique, size=max_validation_molecules, replace=False
                )
                validation_family = validation_family[np.isin(molecules, chosen)]
        tuning_indices = _selected_family_indices(
            train, train_family, fraction, tuning_seed
        )
        candidates: list[tuple[float, int, CNNFitSpec, FittedCNN]] = []
        for spec in specs:
            fitted = fit_coordinate_cnn(
                train,
                tuning_indices,
                validation,
                validation_family,
                spec=spec,
                epochs=epochs,
                batch_size=batch_size,
                seed=tuning_seed,
                device=target_device,
            )
            candidates.append(
                (fitted.validation_auprc, fitted.best_epoch, spec, fitted)
            )
        candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected_spec = candidates[0][2]
        tuning.append(
            {
                "acquisition": acquisition_name,
                "selected_spec": asdict(selected_spec),
                "candidates": [
                    {
                        "validation_auprc": score,
                        "best_epoch": best_epoch,
                        **asdict(spec),
                    }
                    for score, best_epoch, spec, _ in candidates
                ],
            }
        )
        for seed in seeds:
            selected = _selected_family_indices(train, train_family, fraction, seed)
            # Refit every seed, including the tuning seed: reusing the
            # spec-selection winner would report a max over specs for one of
            # the five seeds and bias the baseline mean upward.
            fitted = fit_coordinate_cnn(
                train,
                selected,
                validation,
                validation_family,
                spec=selected_spec,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                device=target_device,
            )
            probability, test_y = _predict(
                fitted.model,
                test,
                test_family,
                batch_size=batch_size,
                device=target_device,
            )
            reports = per_label_metrics(test_y, probability)
            rows = train.indices[selected]
            results.append(
                {
                    "acquisition": acquisition_name,
                    "fraction": fraction,
                    "seed": seed,
                    "labeled_molecules": int(
                        np.unique(train.arrays["molecule_index"][rows]).size
                    ),
                    "selected_spec": asdict(selected_spec),
                    "best_epoch": fitted.best_epoch,
                    "validation_auprc": fitted.validation_auprc,
                    "macro_auprc": macro_auprc(test_y, probability),
                    "macro_f1": macro_f1(test_y, probability),
                    "per_label": reports,
                }
            )

    manifest = json.loads((Path(data_root) / "manifest.json").read_text())
    label_names = manifest.get("provenance", {}).get("functional_groups")
    if label_names is None:
        label_names = [f"label_{index}" for index in range(train.arrays["labels"].shape[1])]
    return {
        "kind": "supervised_coordinate_cnn_ceiling",
        "data_root": str(Path(data_root).resolve()),
        "fraction": fraction,
        "seeds": list(seeds),
        "tuning_seed": tuning_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "acquisitions": list(acquisitions),
        "max_validation_molecules": max_validation_molecules,
        "max_molecules_per_split": limits,
        "subset_seed": subset_seed,
        "label_names": label_names,
        "tuning": tuning,
        "results": results,
    }
