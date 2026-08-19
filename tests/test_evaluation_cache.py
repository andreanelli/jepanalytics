import json
from pathlib import Path

import torch

from jepanalytics.data import build_synthetic_store
from jepanalytics.evaluation import build_embedding_cache, load_embedding_cache
from jepanalytics.model import EncoderConfig, UniversalSpectrumEncoder


def test_embedding_cache_records_and_validates_provenance(tmp_path: Path):
    data = tmp_path / "data"
    checkpoint = tmp_path / "model.pt"
    cache = tmp_path / "embeddings"
    build_synthetic_store(data, n_molecules=24, n_bins=64, n_labels=4, seed=17)
    config = EncoderConfig(
        n_bins=64,
        patch_size=8,
        hidden_dim=32,
        depth=1,
        heads=4,
        mlp_ratio=2,
        aligned_dim=16,
    )
    model = UniversalSpectrumEncoder(config)
    torch.save(
        {
            "encoder_config": config.to_dict(),
            "online_encoder": model.state_dict(),
        },
        checkpoint,
    )

    manifest_path = build_embedding_cache(
        checkpoint,
        data,
        cache,
        splits=("train", "validation", "test"),
        batch_size=8,
        device="cpu",
        include_patch_pool=True,
    )
    manifest = json.loads(manifest_path.read_text())
    train = load_embedding_cache(
        cache,
        "train",
        checkpoint=checkpoint,
        data_root=data,
    )

    assert manifest["checkpoint_sha256"]
    assert manifest["dataset_manifest_sha256"]
    assert train["general"].shape[1] == 32
    assert train["aligned"].shape[1] == 16
    assert train["patch_pool"].shape[1] == 32
    assert "validation" in manifest["splits"]
    assert manifest["include_patch_pool"] is True
    assert train["record_index"].size == manifest["splits"]["train"]["n_records"]


def test_subset_cache_is_rejected_unless_expected(tmp_path: Path):
    import numpy as np
    import pytest

    data = tmp_path / "data"
    checkpoint = tmp_path / "model.pt"
    cache = tmp_path / "embeddings"
    build_synthetic_store(data, n_molecules=24, n_bins=64, n_labels=4, seed=17)
    config = EncoderConfig(
        n_bins=64,
        patch_size=8,
        hidden_dim=32,
        depth=1,
        heads=4,
        mlp_ratio=2,
        aligned_dim=16,
    )
    model = UniversalSpectrumEncoder(config)
    torch.save(
        {"encoder_config": config.to_dict(), "online_encoder": model.state_dict()},
        checkpoint,
    )
    build_embedding_cache(
        checkpoint,
        data,
        cache,
        splits=("train",),
        batch_size=8,
        device="cpu",
        max_molecules_per_split={"train": 4},
        subset_seed=7,
    )

    # A cache built on a molecule subset must not be silently accepted by an
    # evaluation that assumes the full split.
    with pytest.raises(ValueError, match="subset"):
        load_embedding_cache(cache, "train", checkpoint=checkpoint, data_root=data)
    with pytest.raises(ValueError, match="seed"):
        load_embedding_cache(
            cache,
            "train",
            checkpoint=checkpoint,
            data_root=data,
            max_molecules_per_split={"train": 4},
            subset_seed=99,
        )
    with pytest.raises(ValueError, match="patch-pool|patch_pool"):
        load_embedding_cache(
            cache,
            "train",
            checkpoint=checkpoint,
            data_root=data,
            max_molecules_per_split={"train": 4},
            subset_seed=7,
            require_patch_pool=True,
        )
    train = load_embedding_cache(
        cache,
        "train",
        checkpoint=checkpoint,
        data_root=data,
        max_molecules_per_split={"train": 4},
        subset_seed=7,
    )
    assert np.unique(train["molecule_index"]).size == 4
