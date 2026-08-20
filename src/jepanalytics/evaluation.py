"""Frozen probes, retrieval, robustness, and shortcut evaluations."""

from __future__ import annotations

import json
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
from .probes import fit_linear_probe_converged
from .training import load_encoder_checkpoint, move_batch, resolve_device


SPLIT_SUBSET_OFFSETS = {"train": 0, "validation": 1, "test": 2}


def restrict_dataset_molecules(
    dataset: CanonicalSpectraDataset, maximum: int | None, seed: int
) -> None:
    """Deterministically restrict a dataset view to at most ``maximum`` molecules.

    Uses the same selection scheme as the embedding cache (sorted unique
    molecules, one ``default_rng(seed)`` draw), so raw-signal baselines can be
    evaluated on exactly the molecules an audited cache contains.
    """

    if maximum is None:
        return
    molecules = np.asarray(dataset.arrays["molecule_index"][dataset.indices])
    unique = np.unique(molecules)
    if unique.size <= maximum:
        return
    selected = np.random.default_rng(seed).choice(unique, size=maximum, replace=False)
    dataset.indices = dataset.indices[np.isin(molecules, selected)]


@torch.inference_mode()
def extract_embeddings(
    encoder: UniversalSpectrumEncoder,
    dataset: CanonicalSpectraDataset,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    include_patch_pool: bool = False,
) -> dict[str, np.ndarray]:
    target_device = torch.device(device)
    encoder.to(target_device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    general, aligned, patch_pool, labels, availabilities, molecules, acquisitions, records = (
        [],
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
        if include_patch_pool:
            patch_pool.append(output.patches.mean(dim=1).cpu().numpy())
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
    if patch_pool:
        result["patch_pool"] = np.concatenate(patch_pool)
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
    include_patch_pool: bool = False,
    max_molecules_per_split: Mapping[str, int] | None = None,
    subset_seed: int = 20260810,
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
        "include_patch_pool": include_patch_pool,
        "max_molecules_per_split": dict(max_molecules_per_split or {}),
        "subset_seed": subset_seed,
        "splits": {},
    }
    for split in splits:
        dataset = CanonicalSpectraDataset(data_root, split)
        restrict_dataset_molecules(
            dataset,
            (max_molecules_per_split or {}).get(split),
            subset_seed + SPLIT_SUBSET_OFFSETS[split],
        )
        encoded = extract_embeddings(
            encoder,
            dataset,
            batch_size=batch_size,
            device=target_device,
            include_patch_pool=include_patch_pool,
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
            "n_molecules": int(np.unique(encoded["molecule_index"]).size),
            "arrays": arrays,
        }
    manifest_path = output / "manifest.json"
    write_json_atomic(manifest, manifest_path)
    return manifest_path


def read_cache_manifest(root: str | Path) -> dict[str, Any]:
    return json.loads((Path(root) / "manifest.json").read_text())


def load_embedding_cache(
    root: str | Path,
    split: str,
    *,
    checkpoint: str | Path,
    data_root: str | Path,
    max_molecules_per_split: Mapping[str, int] | None = None,
    subset_seed: int | None = None,
    require_patch_pool: bool = False,
) -> dict[str, np.ndarray]:
    """Load a cache only after its model, dataset, and subset provenance match.

    ``max_molecules_per_split`` states the molecule budget the evaluation
    expects to be in effect for each split.  A cache that was already
    restricted at build time is accepted only when its recorded limit and
    subset seed match the expectation; a cache built on the full split is
    always accepted (the evaluation may still subset at read time).  This
    prevents a cache built on a molecule subset from being silently reused
    by an evaluation that assumes the full store.
    """

    root = Path(root)
    manifest = read_cache_manifest(root)
    if manifest["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("embedding cache checkpoint hash does not match")
    if manifest["dataset_manifest_sha256"] != sha256_file(
        Path(data_root) / "manifest.json"
    ):
        raise ValueError("embedding cache dataset hash does not match")
    if split not in manifest["splits"]:
        raise ValueError(f"embedding cache does not contain split {split!r}")
    cache_limit = (manifest.get("max_molecules_per_split") or {}).get(split)
    if cache_limit is not None:
        expected_limit = (max_molecules_per_split or {}).get(split)
        if expected_limit != cache_limit:
            raise ValueError(
                f"embedding cache split {split!r} was built with a "
                f"{cache_limit}-molecule subset but the evaluation expects "
                f"{expected_limit if expected_limit is not None else 'the full split'}"
            )
        if subset_seed != manifest.get("subset_seed"):
            raise ValueError(
                f"embedding cache split {split!r} was subset with seed "
                f"{manifest.get('subset_seed')} but the evaluation expects {subset_seed}"
            )
    if require_patch_pool and not manifest.get("include_patch_pool"):
        raise ValueError(
            "embedding cache was built without --include-patch-pool but the "
            "evaluation requires the patch_pool representation"
        )
    return {
        name: np.load(root / split / f"{name}.npy", mmap_mode="r")
        for name in manifest["splits"][split]["arrays"]
    }


def accept_cache_subset(root: str | Path) -> tuple[dict[str, int], int | None]:
    """Adopt a cache's recorded subset so diagnostics can run on any cache.

    Budget-critical evaluations must instead state their expected subset when
    calling :func:`load_embedding_cache`; diagnostics that merely characterize
    an embedding space adopt whatever subset the cache was built with and are
    responsible for recording it in their result payload.
    """

    manifest = read_cache_manifest(root)
    return (
        dict(manifest.get("max_molecules_per_split") or {}),
        manifest.get("subset_seed"),
    )


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
    epochs: int = 500,
    weight_decay: float = 1.0,
    seed: int = 17,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Fit a converged multilabel logistic probe and return test probabilities.

    This legacy entry point has no validation split, so it uses a fixed L2
    strength (the sklearn default of 1.0); ``epochs`` bounds the L-BFGS
    iteration count.  Budget-sensitive comparisons should prefer the
    validation-tuned probe in :mod:`jepanalytics.probe_audit`.
    """

    del device  # the convex fit is deterministic on CPU
    model, mean, std, _ = fit_linear_probe_converged(
        train_x,
        train_y,
        weight_decay=weight_decay,
        max_iterations=max(epochs, 200),
        seed=seed,
    )
    with torch.inference_mode():
        test = torch.as_tensor(np.asarray(test_x), dtype=torch.float32)
        return torch.sigmoid(model((test - mean) / std)).numpy()


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
    cache_subset: dict[str, Any] | None = None
    if embedding_cache:
        limits, cache_seed = accept_cache_subset(embedding_cache)
        cache_subset = {"max_molecules_per_split": limits, "subset_seed": cache_seed}
        train = load_embedding_cache(
            embedding_cache,
            "train",
            checkpoint=checkpoint,
            data_root=data_root,
            max_molecules_per_split=limits,
            subset_seed=cache_seed,
        )
        test = load_embedding_cache(
            embedding_cache,
            "test",
            checkpoint=checkpoint,
            data_root=data_root,
            max_molecules_per_split=limits,
            subset_seed=cache_seed,
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
        "embedding_cache_subset": cache_subset,
        "results": results,
    }


def _one_record_per_molecule(
    indices: np.ndarray,
    record_molecules: np.ndarray,
    wanted: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick one random record per wanted molecule, ordered like ``wanted``.

    ``indices`` and ``record_molecules`` are aligned arrays: entry ``i`` of
    ``record_molecules`` is the molecule of record ``indices[i]``.
    """

    permutation = rng.permutation(indices.size)
    permuted = np.asarray(indices)[permutation]
    permuted_molecules = np.asarray(record_molecules)[permutation]
    keep = np.isin(permuted_molecules, wanted)
    permuted, permuted_molecules = permuted[keep], permuted_molecules[keep]
    unique, first = np.unique(permuted_molecules, return_index=True)
    chosen = dict(zip(unique.tolist(), permuted[first].tolist()))
    return np.asarray([chosen[molecule] for molecule in wanted.tolist()], dtype=np.int64)


def evaluate_cross_modal_retrieval(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    split: str = "test",
    max_per_acquisition: int = 5000,
    batch_size: int = 64,
    device: str = "auto",
    seed: int = 17,
    embedding_cache: str | Path | None = None,
) -> dict[str, Any]:
    """Molecule-matched cross-technique retrieval with explicit chance levels.

    Every pair uses one randomly chosen record per molecule on both sides and
    restricts to molecules present in both acquisitions, so each query has
    exactly one match, pools are the same size across pairs, and chance
    recall@k is exactly ``k / pool``.  (The previous protocol truncated pools
    by store order, so MS pools held three collision-energy records per
    molecule and recall was not comparable across pairs.)
    """

    target_device = resolve_device(device)
    cache_subset: dict[str, Any] | None = None
    if embedding_cache:
        limits, cache_seed = accept_cache_subset(embedding_cache)
        cache_subset = {"max_molecules_per_split": limits, "subset_seed": cache_seed}
        encoded = load_embedding_cache(
            embedding_cache,
            split,
            checkpoint=checkpoint,
            data_root=data_root,
            max_molecules_per_split=limits,
            subset_seed=cache_seed,
        )
    else:
        encoder = load_encoder_checkpoint(checkpoint, target_device)
        dataset = CanonicalSpectraDataset(data_root, split=split)
        encoded = extract_embeddings(
            encoder, dataset, batch_size=batch_size, device=target_device
        )
    reports = []
    molecules = np.asarray(encoded["molecule_index"])
    acquisitions = sorted(np.unique(encoded["acquisition"]).tolist())
    for source in acquisitions:
        for target in acquisitions:
            if source >= target:
                continue
            rng = np.random.default_rng(seed + 1000 * source + target)
            source_indices = np.flatnonzero(encoded["acquisition"] == source)
            target_indices = np.flatnonzero(encoded["acquisition"] == target)
            common = np.intersect1d(
                molecules[source_indices], molecules[target_indices]
            )
            if not common.size:
                continue
            if common.size > max_per_acquisition:
                common = rng.choice(common, size=max_per_acquisition, replace=False)
                common.sort()
            query_rows = _one_record_per_molecule(
                source_indices, molecules[source_indices], common, rng
            )
            candidate_rows = _one_record_per_molecule(
                target_indices, molecules[target_indices], common, rng
            )
            queries = np.asarray(encoded["aligned"][query_rows], dtype=np.float32)
            candidates = np.asarray(
                encoded["aligned"][candidate_rows], dtype=np.float32
            )
            similarity = queries @ candidates.T
            # Query i's unique match sits at candidate row i by construction.
            match_similarity = np.diagonal(similarity)
            ranks = 1 + np.sum(similarity > match_similarity[:, None], axis=1)
            pool = int(common.size)
            reports.append(
                {
                    "source_acquisition": int(source),
                    "target_acquisition": int(target),
                    "pool_molecules": pool,
                    "queries_with_match": pool,
                    "recall_at_1": float(np.mean(ranks <= 1)),
                    "recall_at_5": float(np.mean(ranks <= 5)),
                    "recall_at_10": float(np.mean(ranks <= 10)),
                    "median_rank": float(np.median(ranks)),
                    "chance_recall_at_1": 1.0 / pool,
                    "chance_recall_at_5": min(1.0, 5.0 / pool),
                    "chance_recall_at_10": min(1.0, 10.0 / pool),
                    "chance_median_rank": (pool + 1) / 2.0,
                }
            )
    return {
        "kind": "cross_modal_retrieval",
        "split": split,
        "seed": seed,
        "embedding_cache_subset": cache_subset,
        "results": reports,
    }


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
    cache_subset: dict[str, Any] | None = None
    if embedding_cache:
        limits, cache_seed = accept_cache_subset(embedding_cache)
        cache_subset = {"max_molecules_per_split": limits, "subset_seed": cache_seed}
        train = load_embedding_cache(
            embedding_cache,
            "train",
            checkpoint=checkpoint,
            data_root=data_root,
            max_molecules_per_split=limits,
            subset_seed=cache_seed,
        )
        test = load_embedding_cache(
            embedding_cache,
            "test",
            checkpoint=checkpoint,
            data_root=data_root,
            max_molecules_per_split=limits,
            subset_seed=cache_seed,
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
    # Chance is the majority-class record prior, not 1/n_classes: with three
    # collision energies per MS polarity the record prior is (1,1,1,3,3)/9.
    test_acquisitions = np.asarray(test["acquisition"])
    counts = np.bincount(test_acquisitions, minlength=5)
    return {
        "kind": "modality_shortcut",
        "representation": representation,
        "accuracy": float(np.mean(predictions == test_acquisitions)),
        "chance_accuracy": float(counts.max() / counts.sum()),
        "class_prior": (counts / counts.sum()).tolist(),
        "embedding_cache_subset": cache_subset,
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
    controls: dict[int, list[float]] = defaultdict(list)
    for batch in loader:
        batch = move_batch(batch, target_device)
        reference = encoder(**batch_to_encoder_kwargs(batch)).general
        for variant in variants:
            changed = {**batch, "intensity": _perturb(batch["intensity"], variant)}
            embedding = encoder(**batch_to_encoder_kwargs(changed)).general
            cosine = F.cosine_similarity(reference, embedding).cpu().numpy()
            for acquisition, value in zip(batch["acquisition"].cpu().numpy(), cosine, strict=True):
                similarities[variant][int(acquisition)].append(float(value))
        # Negative control: cosine between unperturbed embeddings of records
        # from the same acquisition but different molecules.  A perturbation
        # cosine is only meaningful relative to this between-molecule floor;
        # if both sit near 1.0 the embedding barely responds to the signal.
        batch_acquisitions = batch["acquisition"].cpu().numpy()
        batch_molecules = batch["molecule_index"].cpu().numpy()
        for acquisition in np.unique(batch_acquisitions):
            rows = np.flatnonzero(batch_acquisitions == acquisition)
            if rows.size < 2:
                continue
            partners = np.roll(rows, 1)
            different = batch_molecules[rows] != batch_molecules[partners]
            if not different.any():
                continue
            control = F.cosine_similarity(
                reference[rows[different]], reference[partners[different]]
            ).cpu().numpy()
            controls[int(acquisition)].extend(float(value) for value in control)
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
    negative_controls = [
        {
            "acquisition": acquisition,
            "mean_between_molecule_cosine": float(np.mean(values)),
            "ninety_fifth_percentile": float(np.quantile(values, 0.95)),
            "n": len(values),
        }
        for acquisition, values in sorted(controls.items())
    ]
    return {
        "kind": "robustness",
        "split": split,
        "results": results,
        "negative_controls": negative_controls,
    }


def write_evaluation(result: Mapping[str, Any], path: str | Path) -> None:
    write_json_atomic(result, path)
