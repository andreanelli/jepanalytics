"""Simulated-to-experimental frozen functional-group transfer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import CanonicalSpectraDataset, SPLIT_TO_ID
from .evaluation import accept_cache_subset, load_embedding_cache
from .metrics import macro_auprc, macro_f1
from .probe_audit import (
    ACQUISITION_NAMES,
    DEFAULT_LINEAR_SPECS,
    ProbeSpec,
    compose_representation,
    fit_validation_probe,
    per_label_metrics,
    predict_probe,
    random_molecule_indices,
)
from .training import resolve_device


def _label_names(root: str | Path) -> list[str]:
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    names = manifest.get("provenance", {}).get("functional_groups")
    if not names:
        raise ValueError(f"canonical store has no functional-group names: {root}")
    return list(names)


def _training_identifiers(root: str | Path) -> tuple[set[str], set[str]]:
    dataset = CanonicalSpectraDataset(root)
    rows = np.flatnonzero(dataset.arrays["split"] == SPLIT_TO_ID["train"])
    molecules = np.unique(dataset.arrays["molecule_index"][rows])
    scaffolds = np.unique(dataset.arrays["scaffold_index"][rows])
    return (
        {dataset.vocabularies["molecules"][int(index)] for index in molecules},
        {dataset.vocabularies["scaffolds"][int(index)] for index in scaffolds},
    )


def _experimental_identifiers(
    root: str | Path, record_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dataset = CanonicalSpectraDataset(root)
    molecule_indices = dataset.arrays["molecule_index"][record_indices]
    scaffold_indices = dataset.arrays["scaffold_index"][record_indices]
    molecules = np.asarray(
        [dataset.vocabularies["molecules"][int(index)] for index in molecule_indices]
    )
    scaffolds = np.asarray(
        [dataset.vocabularies["scaffolds"][int(index)] for index in scaffold_indices]
    )
    return molecules, scaffolds


def _aggregate_molecules(
    encoded: Mapping[str, np.ndarray],
    indices: np.ndarray,
    molecule_ids: np.ndarray,
    scaffold_ids: np.ndarray,
    representation: str,
) -> dict[str, np.ndarray]:
    """Mean-pool replicate spectra so heavily measured molecules count once."""

    ids = molecule_ids[indices]
    scaffolds = scaffold_ids[indices]
    unique, inverse = np.unique(ids, return_inverse=True)
    values = compose_representation(encoded, representation, indices)
    labels = np.asarray(encoded["labels"][indices], dtype=np.float32)
    pooled = np.zeros((unique.size, values.shape[1]), dtype=np.float32)
    counts = np.bincount(inverse).astype(np.float32)
    np.add.at(pooled, inverse, values)
    pooled /= counts[:, None]
    first = np.unique(inverse, return_index=True)[1]
    molecule_labels = labels[first]
    if not all(np.array_equal(labels[row], molecule_labels[inverse[row]]) for row in range(labels.shape[0])):
        raise ValueError("experimental records for one molecule have inconsistent labels")
    return {
        "x": pooled,
        "y": molecule_labels,
        "molecule_id": unique,
        "scaffold_id": scaffolds[first],
        "replicate_count": counts.astype(np.int64),
    }


def _score_stratum(
    y_true: np.ndarray, probability: np.ndarray
) -> dict[str, Any]:
    reports = per_label_metrics(y_true, probability)
    prevalence_probability = np.broadcast_to(
        y_true.mean(axis=0, keepdims=True), y_true.shape
    )
    return {
        "molecules": int(y_true.shape[0]),
        "labels_with_positive": int(np.sum(y_true.sum(axis=0) > 0)),
        "macro_auprc": macro_auprc(y_true, probability),
        "macro_f1": macro_f1(y_true, probability),
        "prevalence_baseline_macro_auprc": macro_auprc(
            y_true, prevalence_probability
        ),
        "per_label": reports,
    }


def evaluate_experimental_transfer(
    checkpoint: str | Path,
    simulated_data: str | Path,
    simulated_embeddings: str | Path,
    experimental_data: str | Path,
    experimental_embeddings: str | Path,
    *,
    representation: str = "general",
    fraction: float = 0.01,
    seeds: Sequence[int] = (11, 17, 23, 31, 47),
    tuning_seed: int = 17,
    epochs: int = 500,
    specs: Sequence[ProbeSpec] = DEFAULT_LINEAR_SPECS,
    device: str = "auto",
) -> dict[str, Any]:
    """Train probes on simulated spectra and test on experimental spectra."""

    simulated_names = _label_names(simulated_data)
    experimental_names = _label_names(experimental_data)
    if simulated_names != experimental_names:
        raise ValueError("simulated and experimental SMARTS label orders differ")
    target_device = resolve_device(device)
    simulated_limits, simulated_subset_seed = accept_cache_subset(simulated_embeddings)
    experimental_limits, experimental_subset_seed = accept_cache_subset(
        experimental_embeddings
    )
    train = load_embedding_cache(
        simulated_embeddings,
        "train",
        checkpoint=checkpoint,
        data_root=simulated_data,
        max_molecules_per_split=simulated_limits,
        subset_seed=simulated_subset_seed,
    )
    validation = load_embedding_cache(
        simulated_embeddings,
        "validation",
        checkpoint=checkpoint,
        data_root=simulated_data,
        max_molecules_per_split=simulated_limits,
        subset_seed=simulated_subset_seed,
    )
    experimental = load_embedding_cache(
        experimental_embeddings,
        "test",
        checkpoint=checkpoint,
        data_root=experimental_data,
        max_molecules_per_split=experimental_limits,
        subset_seed=experimental_subset_seed,
    )
    if not all("labels" in values for values in (train, validation, experimental)):
        raise ValueError("experimental transfer requires labels in every cache")

    reference_molecules, reference_scaffolds = _training_identifiers(simulated_data)
    experimental_molecules, experimental_scaffolds = _experimental_identifiers(
        experimental_data, np.asarray(experimental["record_index"])
    )
    results: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    for acquisition in (3, 4):
        acquisition_name = ACQUISITION_NAMES[acquisition]
        train_family = np.flatnonzero(np.asarray(train["acquisition"]) == acquisition)
        validation_family = np.flatnonzero(
            np.asarray(validation["acquisition"]) == acquisition
        )
        experimental_family = np.flatnonzero(
            np.asarray(experimental["acquisition"]) == acquisition
        )
        if not train_family.size or not validation_family.size or not experimental_family.size:
            continue
        pooled = _aggregate_molecules(
            experimental,
            experimental_family,
            experimental_molecules,
            experimental_scaffolds,
            representation,
        )
        validation_x = compose_representation(
            validation, representation, validation_family
        )
        validation_y = np.asarray(validation["labels"][validation_family])
        tuning_local = random_molecule_indices(
            np.asarray(train["molecule_index"][train_family]), fraction, tuning_seed
        )
        tuning_indices = train_family[tuning_local]
        tuning_x = compose_representation(train, representation, tuning_indices)
        tuning_y = np.asarray(train["labels"][tuning_indices])
        candidates = []
        for spec in specs:
            fitted = fit_validation_probe(
                tuning_x,
                tuning_y,
                validation_x,
                validation_y,
                kind="linear",
                spec=spec,
                epochs=epochs,
                seed=tuning_seed,
                device=target_device,
            )
            candidates.append((fitted.validation_auprc, fitted.best_epoch, spec))
        candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        _, _, selected_spec = candidates[0]
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
                    for score, best_epoch, spec in candidates
                ],
            }
        )
        strata = {
            "all": np.ones(pooled["molecule_id"].shape[0], dtype=bool),
            "exact_molecule_unseen": np.asarray(
                [value not in reference_molecules for value in pooled["molecule_id"]]
            ),
            "scaffold_unseen": np.asarray(
                [value not in reference_scaffolds for value in pooled["scaffold_id"]]
            ),
        }
        for seed in seeds:
            selected_local = random_molecule_indices(
                np.asarray(train["molecule_index"][train_family]), fraction, seed
            )
            selected = train_family[selected_local]
            fitted = fit_validation_probe(
                compose_representation(train, representation, selected),
                np.asarray(train["labels"][selected]),
                validation_x,
                validation_y,
                kind="linear",
                spec=selected_spec,
                epochs=epochs,
                seed=seed,
                device=target_device,
            )
            probability = predict_probe(fitted, pooled["x"], device=target_device)
            for stratum, mask in strata.items():
                if not mask.any():
                    continue
                results.append(
                    {
                        "acquisition": acquisition_name,
                        "representation": representation,
                        "fraction": fraction,
                        "seed": seed,
                        "labeled_simulated_molecules": int(
                            np.unique(train["molecule_index"][selected]).size
                        ),
                        "stratum": stratum,
                        "best_epoch": fitted.best_epoch,
                        "validation_auprc": fitted.validation_auprc,
                        **_score_stratum(pooled["y"][mask], probability[mask]),
                    }
                )
    return {
        "kind": "simulated_to_experimental_functional_group_transfer",
        "checkpoint": str(Path(checkpoint).resolve()),
        "simulated_data": str(Path(simulated_data).resolve()),
        "simulated_embeddings": str(Path(simulated_embeddings).resolve()),
        "experimental_data": str(Path(experimental_data).resolve()),
        "experimental_embeddings": str(Path(experimental_embeddings).resolve()),
        "representation": representation,
        "fraction": fraction,
        "seeds": list(seeds),
        "tuning_seed": tuning_seed,
        "simulated_cache_subset": {
            "max_molecules_per_split": simulated_limits,
            "subset_seed": simulated_subset_seed,
        },
        "experimental_cache_subset": {
            "max_molecules_per_split": experimental_limits,
            "subset_seed": experimental_subset_seed,
        },
        "label_names": simulated_names,
        "tuning": tuning,
        "results": results,
    }
