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
        batch_size=8,
        device="cpu",
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
    assert train["record_index"].size == manifest["splits"]["train"]["n_records"]
