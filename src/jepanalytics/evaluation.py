"""Frozen probes, retrieval, robustness, and shortcut evaluations."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CanonicalSpectraDataset
from .manifest import sha256_file, write_json_atomic
from .metrics import macro_auprc, macro_f1
from .model import UniversalSpectrumEncoder, batch_to_encoder_kwargs
from .training import load_encoder_checkpoint, move_batch, resolve_device, seed_everything


@torch.inference_mode()
def extract_embeddings(
    encoder: UniversalSpectrumEncoder,
    dataset: CanonicalSpectraDataset,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    target_device = torch.device(device)
    encoder.to(target_device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    general, aligned, labels, availabilities, molecules, acquisitions, records = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for batch in loader:
        batch = move_batch(batch, target_device)
        output = encoder(**batch_to_encoder_kwargs(batch))
        general.append(output.general.cpu().numpy())
        aligned.append(output.aligned.cpu().numpy())
        molecules.append(batch["molecule_index"].cpu().numpy())
        acquisitions.append(batch["acquisition"].cpu().numpy())
        records.append(batch["record_index"].cpu().numpy())
        if "labels" in batch:
            labels.append(batch["labels"].cpu().numpy())
            if "label_available" in batch:
                # Availability is returned alongside labels so mixed public
                # corpora never silently treat an unlabeled record as negative.
                availabilities.append(batch["label_available"].cpu().numpy())
    result = {
        "general": np.concatenate(general),
        "aligned": np.concatenate(aligned),
        "molecule_index": np.concatenate(molecules),
        "acquisition": np.concatenate(acquisitions),
        "record_index": np.concatenate(records),
    }
    if labels:
        result["labels"] = np.concatenate(labels)
        if availabilities:
            result["label_available"] = np.concatenate(availabilities).astype(bool)
    return result


def build_embedding_cache(
    checkpoint: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
    batch_size: int = 256,
    device: str = "auto",
) -> Path:
    """Encode each requested split once and persist validated NumPy arrays."""

    checkpoint = Path(checkpoint)
    data_root = Path(data_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target_device = resolve_device(device)
    encoder = load_encoder_checkpoint(checkpoint, target_device)
    manifest: dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_manifest": str((data_root / "manifest.json").resolve()),
        "dataset_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "splits": {},
    }
    for split in splits:
        encoded = extract_embeddings(
            encoder,
            CanonicalSpectraDataset(data_root, split),
            batch_size=batch_size,
            device=target_device,
        )
        split_root = output / split
        split_root.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for name, values in encoded.items():
            path = split_root / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            arrays[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
        manifest["splits"][split] = {
            "n_records": int(encoded["record_index"].size),
            "arrays": arrays,
        }
    manifest_path = output / "manifest.json"
    write_json_atomic(manifest, manifest_path)
    return manifest_path


def load_embedding_cache(
    root: str | Path,
    split: str,
    *,
    checkpoint: str | Path,
    data_root: str | Path,
) -> dict[str, np.ndarray]:
    """Load a cache only after its model and dataset provenance match."""

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("embedding cache checkpoint hash does not match")
    if manifest["dataset_manifest_sha256"] != sha256_file(
        Path(data_root) / "manifest.json"
    ):
        raise ValueError("embedding cache dataset hash does not match")
    if split not in manifest["splits"]:
        raise ValueError(f"embedding cache does not contain split {split!r}")
    return {
        name: np.load(root / split / f"{name}.npy", mmap_mode="r")
        for name in manifest["splits"][split]["arrays"]
    }


def _sample_molecules(molecules: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    unique = np.unique(molecules)
    count = max(1, round(unique.size * fraction))
    rng = np.random.default_rng(seed)
    selected = rng.choice(unique, size=count, replace=False)
    return np.flatnonzero(np.isin(molecules, selected))


def fit_multilabel_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    epochs: int = 150,
    learning_rate: float = 0.03,
    seed: int = 17,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    seed_everything(seed)
    target_device = torch.device(device)
    x = torch.tensor(train_x, dtype=torch.float32, device=target_device)
    y = torch.tensor(train_y, dtype=torch.float32, device=target_device)
    test = torch.tensor(test_x, dtype=torch.float32, device=target_device)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    model = nn.Linear(x.shape[1], y.shape[1]).to(target_device)
    positives = y.sum(dim=0)
    pos_weight = ((y.shape[0] - positives) / positives.clamp_min(1.0)).clamp(max=20.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model((x - mean) / std)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        return torch.sigmoid(model((test - mean) / std)).cpu().numpy()


def evaluate_few_shot_probes(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    fractions: Sequence[float] = (0.01, 0.05, 0.10, 1.0),
    seeds: Sequence[int] = (11, 17, 23, 31, 47),
    probe_epochs: int = 150,
    batch_size: int = 64,
    device: str = "auto",
    embedding_cache: str | Path | None = None,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    if embedding_cache:
        train = load_embedding_cache(
            embedding_cache, "train", checkpoint=checkpoint, data_root=data_root
        )
        test = load_embedding_cache(
            embedding_cache, "test", checkpoint=checkpoint, data_root=data_root
        )
    else:
        encoder = load_encoder_checkpoint(checkpoint, target_device)
        train_dataset = CanonicalSpectraDataset(data_root, split="train")
        test_dataset = CanonicalSpectraDataset(data_root, split="test")
        train = extract_embeddings(
            encoder, train_dataset, batch_size=batch_size, device=target_device
        )
        test = extract_embeddings(
            encoder, test_dataset, batch_size=batch_size, device=target_device
        )
    if "labels" not in train or "labels" not in test:
        raise ValueError("functional-group evaluation requires labels in the canonical store")
    results: list[dict[str, Any]] = []
    acquisition_names = ["IR", "H1_NMR", "C13_NMR", "MSMS_POSITIVE", "MSMS_NEGATIVE"]
    for acquisition, acquisition_name in enumerate(acquisition_names):
        train_family = train["acquisition"] == acquisition
        test_family = test["acquisition"] == acquisition
        if "label_available" in train:
            train_family &= train["label_available"]
        if "label_available" in test:
            test_family &= test["label_available"]
        if not train_family.any() or not test_family.any():
            continue
        for fraction in fractions:
            for seed in seeds:
                family_indices = np.flatnonzero(train_family)
                sampled_local = _sample_molecules(
                    train["molecule_index"][family_indices], fraction, seed
                )
                selected = family_indices[sampled_local]
                probability = fit_multilabel_linear_probe(
                    train["general"][selected],
                    train["labels"][selected],
                    test["general"][test_family],
                    epochs=probe_epochs,
                    seed=seed,
                    device=target_device,
                )
                results.append(
                    {
                        "acquisition": acquisition_name,
                        "fraction": fraction,
                        "seed": seed,
                        "labeled_molecules": int(
                            np.unique(train["molecule_index"][selected]).size
                        ),
                        "macro_auprc": macro_auprc(test["labels"][test_family], probability),
                        "macro_f1": macro_f1(test["labels"][test_family], probability),
                    }
                )
    return {
        "kind": "few_shot_functional_group_probe",
        "checkpoint": str(Path(checkpoint).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "results": results,
    }


def evaluate_cross_modal_retrieval(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    split: str = "test",
    max_per_acquisition: int = 5000,
    batch_size: int = 64,
    device: str = "auto",
    embedding_cache: str | Path | None = None,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    if embedding_cache:
        encoded = load_embedding_cache(
            embedding_cache, split, checkpoint=checkpoint, data_root=data_root
        )
    else:
        encoder = load_encoder_checkpoint(checkpoint, target_device)
        dataset = CanonicalSpectraDataset(data_root, split=split)
        encoded = extract_embeddings(
            encoder, dataset, batch_size=batch_size, device=target_device
        )
    reports = []
    acquisitions = sorted(np.unique(encoded["acquisition"]).tolist())
    for source in acquisitions:
        for target in acquisitions:
            if source >= target:
                continue
            source_indices = np.flatnonzero(encoded["acquisition"] == source)[:max_per_acquisition]
            target_indices = np.flatnonzero(encoded["acquisition"] == target)[:max_per_acquisition]
            if not source_indices.size or not target_indices.size:
                continue
            queries = encoded["aligned"][source_indices]
            candidates = encoded["aligned"][target_indices]
            similarity = queries @ candidates.T
            order = np.argsort(-similarity, axis=1)
            target_molecules = encoded["molecule_index"][target_indices]
            query_molecules = encoded["molecule_index"][source_indices]
            ranks = []
            for row, molecule in enumerate(query_molecules):
                matches = np.flatnonzero(target_molecules[order[row]] == molecule)
                if matches.size:
                    ranks.append(int(matches[0]) + 1)
            rank_array = np.asarray(ranks, dtype=np.int64)
            reports.append(
                {
                    "source_acquisition": int(source),
                    "target_acquisition": int(target),
                    "queries_with_match": int(rank_array.size),
                    "recall_at_1": float(np.mean(rank_array <= 1)) if rank_array.size else 0.0,
                    "recall_at_5": float(np.mean(rank_array <= 5)) if rank_array.size else 0.0,
                    "recall_at_10": float(np.mean(rank_array <= 10)) if rank_array.size else 0.0,
                    "median_rank": float(np.median(rank_array)) if rank_array.size else None,
                }
            )
    return {"kind": "cross_modal_retrieval", "split": split, "results": reports}


def evaluate_modality_shortcut(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    representation: str = "general",
    epochs: int = 100,
    device: str = "auto",
    embedding_cache: str | Path | None = None,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    if embedding_cache:
        train = load_embedding_cache(
            embedding_cache, "train", checkpoint=checkpoint, data_root=data_root
        )
        test = load_embedding_cache(
            embedding_cache, "test", checkpoint=checkpoint, data_root=data_root
        )
    else:
        encoder = load_encoder_checkpoint(checkpoint, target_device)
        train = extract_embeddings(
            encoder, CanonicalSpectraDataset(data_root, "train"), device=target_device
        )
        test = extract_embeddings(
            encoder, CanonicalSpectraDataset(data_root, "test"), device=target_device
        )
    x = torch.tensor(train[representation], dtype=torch.float32, device=target_device)
    y = torch.tensor(train["acquisition"], dtype=torch.long, device=target_device)
    model = nn.Linear(x.shape[1], 5).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        test_x = torch.tensor(test[representation], dtype=torch.float32, device=target_device)
        predictions = model(test_x).argmax(dim=-1).cpu().numpy()
    return {
        "kind": "modality_shortcut",
        "representation": representation,
        "accuracy": float(np.mean(predictions == test["acquisition"])),
        "chance_accuracy": 0.2,
    }


def _perturb(intensity: torch.Tensor, kind: str, seed: int = 17) -> torch.Tensor:
    if kind == "coordinate_shift":
        output = torch.zeros_like(intensity)
        output[:, 2:] = intensity[:, :-2]
        return output
    if kind == "broadening":
        kernel = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=intensity.device)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        return F.conv1d(intensity.unsqueeze(1), kernel, padding=2).squeeze(1)
    if kind == "baseline_drift":
        position = torch.linspace(-1.0, 1.0, intensity.shape[1], device=intensity.device)
        return intensity + 0.1 * position.unsqueeze(0)
    if kind == "noise":
        generator = torch.Generator(device=intensity.device).manual_seed(seed)
        return intensity + torch.randn(
            intensity.shape,
            generator=generator,
            device=intensity.device,
            dtype=intensity.dtype,
        ) * 0.05
    if kind == "resolution":
        low = F.interpolate(
            intensity.unsqueeze(1), size=max(16, intensity.shape[1] // 4), mode="linear"
        )
        return F.interpolate(low, size=intensity.shape[1], mode="linear").squeeze(1)
    raise ValueError(f"unknown perturbation {kind!r}")


@torch.inference_mode()
def evaluate_robustness(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    split: str = "test",
    batch_size: int = 64,
    device: str = "auto",
) -> dict[str, Any]:
    target_device = resolve_device(device)
    encoder = load_encoder_checkpoint(checkpoint, target_device)
    dataset = CanonicalSpectraDataset(data_root, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    variants = ("coordinate_shift", "broadening", "baseline_drift", "noise", "resolution")
    similarities: dict[str, dict[int, list[float]]] = {
        variant: defaultdict(list) for variant in variants
    }
    for batch in loader:
        batch = move_batch(batch, target_device)
        reference = encoder(**batch_to_encoder_kwargs(batch)).general
        for variant in variants:
            changed = {**batch, "intensity": _perturb(batch["intensity"], variant)}
            embedding = encoder(**batch_to_encoder_kwargs(changed)).general
            cosine = F.cosine_similarity(reference, embedding).cpu().numpy()
            for acquisition, value in zip(batch["acquisition"].cpu().numpy(), cosine, strict=True):
                similarities[variant][int(acquisition)].append(float(value))
    results = []
    for variant, families in similarities.items():
        for acquisition, values in families.items():
            results.append(
                {
                    "perturbation": variant,
                    "acquisition": acquisition,
                    "mean_cosine_stability": float(np.mean(values)),
                    "fifth_percentile": float(np.quantile(values, 0.05)),
                    "n": len(values),
                }
            )
    return {"kind": "robustness", "split": split, "results": results}


def write_evaluation(result: Mapping[str, Any], path: str | Path) -> None:
    write_json_atomic(result, path)
