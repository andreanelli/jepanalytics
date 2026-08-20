"""Validation-tuned diagnostics for chemistry content in frozen embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .evaluation import load_embedding_cache
from .metrics import binary_average_precision, macro_auprc, macro_f1
from .probes import (
    fit_linear_probe_converged,
    positive_weight,
    standardization_stats,
)
from .training import resolve_device, seed_everything


ACQUISITION_NAMES = ("IR", "H1_NMR", "C13_NMR", "MSMS_POSITIVE", "MSMS_NEGATIVE")
REPRESENTATIONS = ("general", "aligned", "general_aligned", "patch_pool")
PROBE_KINDS = ("linear", "mlp")
SAMPLING_MODES = ("random", "coverage")


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    learning_rate: float
    weight_decay: float


# Linear probes are fit to convergence with L-BFGS, so only the L2 strength
# matters; the grid brackets the sklearn default (lambda = 1) on both sides.
DEFAULT_LINEAR_SPECS = (
    ProbeSpec(1.0, 0.01),
    ProbeSpec(1.0, 0.1),
    ProbeSpec(1.0, 1.0),
    ProbeSpec(1.0, 10.0),
)

# MLP probes remain first-order and mini-batched, so both fields are live.
DEFAULT_SPECS = (
    ProbeSpec(0.003, 1e-3),
    ProbeSpec(0.01, 1e-4),
    ProbeSpec(0.03, 1e-4),
    ProbeSpec(0.01, 1e-2),
)


def default_specs_for_kind(kind: str) -> tuple[ProbeSpec, ...]:
    return DEFAULT_LINEAR_SPECS if kind == "linear" else DEFAULT_SPECS


@dataclass(slots=True)
class FittedProbe:
    model: nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    best_epoch: int
    validation_auprc: float


def compose_representation(
    encoded: Mapping[str, np.ndarray],
    representation: str,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Materialize one requested representation, slicing before concatenation."""

    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")

    def values(name: str) -> np.ndarray:
        if name not in encoded:
            raise ValueError(f"embedding cache does not contain {name!r}")
        array = encoded[name]
        return np.asarray(array if indices is None else array[indices], dtype=np.float32)

    if representation == "general_aligned":
        return np.concatenate((values("general"), values("aligned")), axis=1)
    return values(representation)


def random_molecule_indices(
    molecules: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    unique = np.unique(molecules)
    count = max(1, round(unique.size * fraction))
    rng = np.random.default_rng(seed)
    selected = rng.choice(unique, size=count, replace=False)
    return np.flatnonzero(np.isin(molecules, selected))


def coverage_controlled_molecule_indices(
    molecules: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    seed: int,
    *,
    minimum_positives: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a fixed-size molecule subset while covering feasible rare labels.

    One label vector is used per molecule even when an acquisition contains
    multiple collision energies. Rare labels are filled first with inverse-
    frequency weighting; the remaining budget is sampled uniformly.
    """

    molecules = np.asarray(molecules)
    labels = np.asarray(labels)
    unique, first = np.unique(molecules, return_index=True)
    molecule_labels = labels[first] > 0.5
    budget = max(1, round(unique.size * fraction))
    if budget > unique.size:
        raise ValueError("coverage-controlled budget exceeds available molecules")
    available = molecule_labels.sum(axis=0).astype(np.int64)
    targets = np.minimum(available, minimum_positives)
    counts = np.zeros(molecule_labels.shape[1], dtype=np.int64)
    selected = np.zeros(unique.size, dtype=bool)
    rng = np.random.default_rng(seed)
    tie_break = rng.random(unique.size)

    while selected.sum() < budget and np.any(counts < targets):
        deficit = counts < targets
        candidates = np.flatnonzero(~selected & molecule_labels[:, deficit].any(axis=1))
        if not candidates.size:
            break
        weights = np.zeros_like(available, dtype=np.float64)
        weights[deficit] = 1.0 / np.maximum(available[deficit], 1)
        scores = molecule_labels[candidates] @ weights
        best_score = scores.max()
        best = candidates[np.isclose(scores, best_score)]
        choice = best[np.argmin(tie_break[best])]
        selected[choice] = True
        counts += molecule_labels[choice]

    remaining = np.flatnonzero(~selected)
    needed = budget - int(selected.sum())
    if needed:
        selected[rng.choice(remaining, size=needed, replace=False)] = True
    selected_molecules = unique[selected]
    record_indices = np.flatnonzero(np.isin(molecules, selected_molecules))
    selected_counts = molecule_labels[selected].sum(axis=0).astype(np.int64)
    diagnostics = {
        "selected_molecules": int(selected.sum()),
        "minimum_requested": minimum_positives,
        "zero_positive_labels": int(np.sum(selected_counts == 0)),
        "labels_below_minimum": int(np.sum(selected_counts < targets)),
        "positive_counts": selected_counts.tolist(),
        "targets": targets.tolist(),
    }
    return record_indices, diagnostics


def _probe_model(kind: str, input_dim: int, output_dim: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(input_dim, output_dim)
    if kind == "mlp":
        hidden = min(256, max(64, input_dim // 2))
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )
    raise ValueError(f"unknown probe kind {kind!r}")


def fit_validation_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    kind: str,
    spec: ProbeSpec,
    epochs: int,
    seed: int,
    device: str | torch.device,
    evaluation_interval: int = 10,
    patience: int = 4,
    batch_size: int = 256,
) -> FittedProbe:
    """Fit one probe to convergence and score it on scaffold validation.

    Linear probes solve the convex problem directly with L-BFGS on CPU;
    validation only selects ``spec.weight_decay``.  MLP probes use
    mini-batch AdamW (the previous implementation took one full-batch step
    per epoch, so a 120-"epoch" budget meant 120 gradient steps).
    """

    if kind == "linear":
        model, mean, std, iterations = fit_linear_probe_converged(
            train_x,
            train_y,
            weight_decay=spec.weight_decay,
            max_iterations=max(epochs, 200),
            seed=seed,
        )
        with torch.inference_mode():
            validation = torch.as_tensor(
                np.asarray(validation_x), dtype=torch.float32
            )
            probability = torch.sigmoid(model((validation - mean) / std)).numpy()
        score = macro_auprc(np.asarray(validation_y), probability)
        return FittedProbe(model, mean, std, iterations, float(score))

    seed_everything(seed)
    target_device = torch.device(device)
    x = torch.as_tensor(np.asarray(train_x), dtype=torch.float32, device=target_device)
    y = torch.as_tensor(np.asarray(train_y), dtype=torch.float32, device=target_device)
    validation = torch.as_tensor(
        np.asarray(validation_x), dtype=torch.float32, device=target_device
    )
    mean, std = standardization_stats(x)
    x = (x - mean) / std
    model = _probe_model(kind, x.shape[1], y.shape[1]).to(target_device)
    pos_weight = positive_weight(y)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    best_score = -float("inf")
    best_epoch = evaluation_interval
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for chunk in torch.randperm(x.shape[0], generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[chunk.to(target_device)])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y[chunk.to(target_device)], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
        if epoch % evaluation_interval and epoch != epochs:
            continue
        model.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(model((validation - mean) / std)).cpu().numpy()
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
        raise RuntimeError("probe training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return FittedProbe(model, mean, std, best_epoch, float(best_score))


@torch.inference_mode()
def predict_probe(
    fitted: FittedProbe,
    values: np.ndarray,
    *,
    device: str | torch.device,
) -> np.ndarray:
    target_device = torch.device(device)
    x = torch.as_tensor(np.asarray(values), dtype=torch.float32, device=target_device)
    fitted.model.to(target_device).eval()
    mean = fitted.mean.to(target_device)
    std = fitted.std.to(target_device)
    probability = torch.sigmoid(fitted.model((x - mean) / std))
    return probability.cpu().numpy()


def per_label_metrics(y_true: np.ndarray, probability: np.ndarray) -> list[dict[str, float]]:
    reports = []
    for label in range(y_true.shape[1]):
        prevalence = float(np.mean(y_true[:, label]))
        average_precision = binary_average_precision(y_true[:, label], probability[:, label])
        normalized_lift = (
            float("nan")
            if not np.isfinite(average_precision) or prevalence >= 1.0
            else (average_precision - prevalence) / (1.0 - prevalence)
        )
        reports.append(
            {
                "label": label,
                "prevalence": prevalence,
                "auprc": average_precision,
                "normalized_lift": normalized_lift,
            }
        )
    return reports


def _family_indices(encoded: Mapping[str, np.ndarray], acquisition: int) -> np.ndarray:
    mask = np.asarray(encoded["acquisition"]) == acquisition
    if "label_available" in encoded:
        mask &= np.asarray(encoded["label_available"], dtype=bool)
    return np.flatnonzero(mask)


def subset_encoded_molecules(
    encoded: Mapping[str, np.ndarray], maximum: int | None, seed: int
) -> dict[str, np.ndarray]:
    """Apply the same deterministic molecule subset to any compatible cache."""

    if maximum is None:
        return dict(encoded)
    molecules = np.asarray(encoded["molecule_index"])
    unique = np.unique(molecules)
    if unique.size <= maximum:
        return dict(encoded)
    selected = np.random.default_rng(seed).choice(unique, size=maximum, replace=False)
    indices = np.flatnonzero(np.isin(molecules, selected))
    return {name: np.asarray(values[indices]) for name, values in encoded.items()}


def evaluate_probe_audit(
    checkpoint: str | Path,
    data_root: str | Path,
    embedding_cache: str | Path,
    *,
    representations: Sequence[str] = REPRESENTATIONS,
    probe_kinds: Sequence[str] = PROBE_KINDS,
    sampling_modes: Sequence[str] = SAMPLING_MODES,
    fraction: float = 0.01,
    seeds: Sequence[int] = (11, 17, 23, 31, 47),
    tuning_seed: int = 17,
    minimum_positives: int = 5,
    epochs: int = 500,
    specs: Sequence[ProbeSpec] | None = None,
    max_molecules_per_split: Mapping[str, int] | None = None,
    subset_seed: int = 20260810,
    device: str = "auto",
) -> dict[str, Any]:
    """Audit frozen chemistry information without changing the encoder."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    for representation in representations:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unknown representation {representation!r}")
    for kind in probe_kinds:
        if kind not in PROBE_KINDS:
            raise ValueError(f"unknown probe kind {kind!r}")
    for mode in sampling_modes:
        if mode not in SAMPLING_MODES:
            raise ValueError(f"unknown sampling mode {mode!r}")

    target_device = resolve_device(device)
    limits = max_molecules_per_split or {}
    require_patch_pool = "patch_pool" in representations
    train, validation, test = (
        load_embedding_cache(
            embedding_cache,
            split,
            checkpoint=checkpoint,
            data_root=data_root,
            max_molecules_per_split=limits,
            subset_seed=subset_seed,
            require_patch_pool=require_patch_pool,
        )
        for split in ("train", "validation", "test")
    )
    train = subset_encoded_molecules(train, limits.get("train"), subset_seed)
    validation = subset_encoded_molecules(
        validation, limits.get("validation"), subset_seed + 1
    )
    test = subset_encoded_molecules(test, limits.get("test"), subset_seed + 2)
    if "labels" not in train or "labels" not in validation or "labels" not in test:
        raise ValueError("probe audit requires labels in every split")

    results: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    for representation in representations:
        for acquisition, acquisition_name in enumerate(ACQUISITION_NAMES):
            train_family = _family_indices(train, acquisition)
            validation_family = _family_indices(validation, acquisition)
            test_family = _family_indices(test, acquisition)
            validation_x = compose_representation(
                validation, representation, validation_family
            )
            validation_y = np.asarray(validation["labels"][validation_family], dtype=np.float32)
            test_x = compose_representation(test, representation, test_family)
            test_y = np.asarray(test["labels"][test_family], dtype=np.float32)

            tuning_local = random_molecule_indices(
                np.asarray(train["molecule_index"][train_family]), fraction, tuning_seed
            )
            tuning_indices = train_family[tuning_local]
            tuning_x = compose_representation(train, representation, tuning_indices)
            tuning_y = np.asarray(train["labels"][tuning_indices], dtype=np.float32)
            selected_specs: dict[str, ProbeSpec] = {}
            for kind in probe_kinds:
                candidates = []
                for spec in specs or default_specs_for_kind(kind):
                    fitted = fit_validation_probe(
                        tuning_x,
                        tuning_y,
                        validation_x,
                        validation_y,
                        kind=kind,
                        spec=spec,
                        epochs=epochs,
                        seed=tuning_seed,
                        device=target_device,
                    )
                    candidates.append((fitted.validation_auprc, fitted.best_epoch, spec))
                candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                score, best_epoch, selected_spec = candidates[0]
                selected_specs[kind] = selected_spec
                tuning.append(
                    {
                        "representation": representation,
                        "acquisition": acquisition_name,
                        "probe_kind": kind,
                        "selected_spec": asdict(selected_spec),
                        "validation_auprc": score,
                        "best_epoch": best_epoch,
                        "candidates": [
                            {
                                "validation_auprc": candidate_score,
                                "best_epoch": candidate_epoch,
                                **asdict(candidate_spec),
                            }
                            for candidate_score, candidate_epoch, candidate_spec in candidates
                        ],
                    }
                )

            for sampling_mode in sampling_modes:
                for seed in seeds:
                    family_molecules = np.asarray(train["molecule_index"][train_family])
                    family_labels = np.asarray(train["labels"][train_family])
                    if sampling_mode == "random":
                        selected_local = random_molecule_indices(
                            family_molecules, fraction, seed
                        )
                        selected_counts = family_labels[selected_local][
                            np.unique(family_molecules[selected_local], return_index=True)[1]
                        ].sum(axis=0)
                        coverage = {
                            "selected_molecules": int(
                                np.unique(family_molecules[selected_local]).size
                            ),
                            "minimum_requested": None,
                            "zero_positive_labels": int(np.sum(selected_counts == 0)),
                            "labels_below_minimum": None,
                            "positive_counts": selected_counts.astype(int).tolist(),
                            "targets": None,
                        }
                    else:
                        selected_local, coverage = coverage_controlled_molecule_indices(
                            family_molecules,
                            family_labels,
                            fraction,
                            seed,
                            minimum_positives=minimum_positives,
                        )
                    selected_indices = train_family[selected_local]
                    train_x = compose_representation(
                        train, representation, selected_indices
                    )
                    train_y = np.asarray(
                        train["labels"][selected_indices], dtype=np.float32
                    )
                    for kind in probe_kinds:
                        fitted = fit_validation_probe(
                            train_x,
                            train_y,
                            validation_x,
                            validation_y,
                            kind=kind,
                            spec=selected_specs[kind],
                            epochs=epochs,
                            seed=seed,
                            device=target_device,
                        )
                        probability = predict_probe(
                            fitted, test_x, device=target_device
                        )
                        label_reports = per_label_metrics(test_y, probability)
                        results.append(
                            {
                                "representation": representation,
                                "dimension": int(train_x.shape[1]),
                                "acquisition": acquisition_name,
                                "sampling": sampling_mode,
                                "fraction": fraction,
                                "seed": seed,
                                "probe_kind": kind,
                                "labeled_molecules": coverage["selected_molecules"],
                                "coverage": coverage,
                                "selected_spec": asdict(selected_specs[kind]),
                                "best_epoch": fitted.best_epoch,
                                "validation_auprc": fitted.validation_auprc,
                                "macro_auprc": macro_auprc(test_y, probability),
                                "macro_f1": macro_f1(test_y, probability),
                                "macro_normalized_lift": float(
                                    np.nanmean(
                                        [report["normalized_lift"] for report in label_reports]
                                    )
                                ),
                                "per_label": label_reports,
                            }
                        )
    dataset_manifest = json.loads((Path(data_root) / "manifest.json").read_text())
    label_names = dataset_manifest.get("provenance", {}).get("functional_groups")
    if label_names is None:
        label_names = [f"label_{index}" for index in range(train["labels"].shape[1])]
    return {
        "kind": "frozen_probe_audit",
        "checkpoint": str(Path(checkpoint).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "embedding_cache": str(Path(embedding_cache).resolve()),
        "fraction": fraction,
        "seeds": list(seeds),
        "tuning_seed": tuning_seed,
        "minimum_positives": minimum_positives,
        "max_molecules_per_split": dict(limits),
        "subset_seed": subset_seed,
        "label_names": label_names,
        "representations": list(representations),
        "probe_kinds": list(probe_kinds),
        "sampling_modes": list(sampling_modes),
        "tuning": tuning,
        "results": results,
    }
