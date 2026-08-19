"""Controlled local baselines and optional external encoder adapters."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


class SupervisedSpectrumCNN(nn.Module):
    """Compact supervised 1-D CNN used in the published benchmark comparison."""

    def __init__(self, n_labels: int, channels: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(channels, channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(channels * 2, n_labels)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(intensity.unsqueeze(1)).squeeze(-1))


class CoordinateAwareSpectrumCNN(nn.Module):
    """Strong family-specific raw-signal ceiling that retains peak location.

    Global average pooling makes a spectrum classifier nearly translation
    invariant, even though chemical shift, wavenumber, and m/z are essential
    to interpretation.  This baseline downsamples locally and retains a small
    ordered coordinate grid before classification.  It remains compact enough
    for repeated few-shot fits on Apple Silicon.
    """

    def __init__(
        self,
        n_labels: int,
        *,
        channels: int = 48,
        coordinate_bins: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(channels, channels * 2, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(coordinate_bins),
        )
        hidden = channels * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(channels * 2 * coordinate_bins),
            nn.Dropout(dropout),
            nn.Linear(channels * 2 * coordinate_bins, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        if intensity.ndim != 2:
            raise ValueError("intensity must have shape [batch, coordinate]")
        return self.classifier(self.features(intensity.unsqueeze(1)))


def _pca_projection(
    intensities: np.ndarray, components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and top principal directions of raw intensity rows."""

    x = np.asarray(intensities, dtype=np.float32)
    mean = x.mean(axis=0)
    centered = x - mean
    covariance = (centered.T @ centered) / max(1, x.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    order = np.argsort(eigenvalues)[::-1][:components]
    return mean, eigenvectors[:, order].astype(np.float32)


def evaluate_raw_signal_pca_probe(
    data_root: str | Path,
    *,
    components: int = 256,
    fraction: float = 0.01,
    seeds: Sequence[int] = (11, 17, 23, 31, 47),
    tuning_seed: int = 17,
    epochs: int = 500,
    fit_sample: int = 20000,
    max_molecules_per_split: Mapping[str, int] | None = None,
    subset_seed: int = 20260810,
    device: str = "auto",
) -> dict[str, Any]:
    """Linear-probe a per-family PCA of the raw binned signal.

    This is the untrained-feature control the probe audit was missing: PCA is
    fit on unlabeled train-split intensities (the fair analog of
    self-supervised pretraining), and the identical validation-tuned linear
    probe protocol is applied at the identical label budget.  A pretrained
    encoder that cannot beat this has not extracted chemistry beyond raw
    signal covariance.
    """

    from .data import CanonicalSpectraDataset
    from .evaluation import SPLIT_SUBSET_OFFSETS, restrict_dataset_molecules
    from .probe_audit import (
        ACQUISITION_NAMES,
        default_specs_for_kind,
        fit_validation_probe,
        per_label_metrics,
        predict_probe,
        random_molecule_indices,
    )
    from .metrics import macro_auprc, macro_f1
    from .training import resolve_device

    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    target_device = resolve_device(device)
    datasets = {
        split: CanonicalSpectraDataset(data_root, split)
        for split in ("train", "validation", "test")
    }
    limits = dict(max_molecules_per_split or {})
    for split, dataset in datasets.items():
        restrict_dataset_molecules(
            dataset, limits.get(split), subset_seed + SPLIT_SUBSET_OFFSETS[split]
        )
    if "labels" not in datasets["train"].arrays:
        raise ValueError("raw-signal PCA probe requires labels")

    # Every split view shares the same full-store memory-mapped arrays and
    # holds absolute row indices into them.
    store_arrays = datasets["train"].arrays

    def family_rows(split: str, acquisition: int) -> np.ndarray:
        rows = datasets[split].indices
        mask = np.asarray(store_arrays["acquisition"][rows]) == acquisition
        if "label_available" in store_arrays:
            mask &= np.asarray(store_arrays["label_available"][rows], dtype=bool)
        return rows[np.flatnonzero(mask)]

    def project(rows: np.ndarray, mean: np.ndarray, directions: np.ndarray) -> np.ndarray:
        x = np.asarray(store_arrays["intensity"][rows], dtype=np.float32)
        return (x - mean) @ directions
    results: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    for acquisition, acquisition_name in enumerate(ACQUISITION_NAMES):
        train_rows = family_rows("train", acquisition)
        validation_rows = family_rows("validation", acquisition)
        test_rows = family_rows("test", acquisition)
        if not train_rows.size or not validation_rows.size or not test_rows.size:
            continue
        rng = np.random.default_rng(tuning_seed)
        sample = train_rows
        if sample.size > fit_sample:
            sample = np.sort(rng.choice(sample, size=fit_sample, replace=False))
        mean, directions = _pca_projection(
            datasets["train"].arrays["intensity"][sample], components
        )
        validation_x = project(validation_rows, mean, directions)
        validation_y = np.asarray(
            datasets["validation"].arrays["labels"][validation_rows], dtype=np.float32
        )
        test_x = project(test_rows, mean, directions)
        test_y = np.asarray(
            datasets["test"].arrays["labels"][test_rows], dtype=np.float32
        )
        train_molecules = np.asarray(
            datasets["train"].arrays["molecule_index"][train_rows]
        )
        train_labels = np.asarray(
            datasets["train"].arrays["labels"][train_rows], dtype=np.float32
        )

        specs = default_specs_for_kind("linear")
        tuning_local = random_molecule_indices(train_molecules, fraction, tuning_seed)
        tuning_x = project(train_rows[tuning_local], mean, directions)
        tuning_y = train_labels[tuning_local]
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
        for seed in seeds:
            selected_local = random_molecule_indices(train_molecules, fraction, seed)
            fitted = fit_validation_probe(
                project(train_rows[selected_local], mean, directions),
                train_labels[selected_local],
                validation_x,
                validation_y,
                kind="linear",
                spec=selected_spec,
                epochs=epochs,
                seed=seed,
                device=target_device,
            )
            probability = predict_probe(fitted, test_x, device=target_device)
            label_reports = per_label_metrics(test_y, probability)
            results.append(
                {
                    "acquisition": acquisition_name,
                    "representation": f"raw_pca_{components}",
                    "sampling": "random",
                    "probe_kind": "linear",
                    "fraction": fraction,
                    "seed": seed,
                    "labeled_molecules": int(
                        np.unique(train_molecules[selected_local]).size
                    ),
                    "selected_spec": asdict(selected_spec),
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
    manifest = json.loads((Path(data_root) / "manifest.json").read_text())
    label_names = manifest.get("provenance", {}).get("functional_groups")
    return {
        "kind": "raw_signal_pca_probe",
        "data_root": str(Path(data_root).resolve()),
        "components": components,
        "fraction": fraction,
        "seeds": list(seeds),
        "tuning_seed": tuning_seed,
        "fit_sample": fit_sample,
        "max_molecules_per_split": limits,
        "subset_seed": subset_seed,
        "label_names": label_names,
        "tuning": tuning,
        "results": results,
    }


@dataclass(frozen=True, slots=True)
class ExternalBaselineStatus:
    name: str
    available: bool
    package: str
    purpose: str


def external_baseline_status() -> list[ExternalBaselineStatus]:
    """Report optional integrations without silently substituting another model."""

    return [
        ExternalBaselineStatus(
            name="MOMENT",
            available=importlib.util.find_spec("momentfm") is not None,
            package="momentfm",
            purpose="Open time-series foundation-model embedding baseline",
        ),
        ExternalBaselineStatus(
            name="TS2Vec",
            available=importlib.util.find_spec("ts2vec") is not None,
            package="ts2vec",
            purpose="Universal contrastive time-series representation baseline",
        ),
    ]
