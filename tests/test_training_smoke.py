import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from jepanalytics.augment import AugmentationConfig
from jepanalytics.config import TrainingConfig
from jepanalytics.data import build_synthetic_store
from jepanalytics.model import EncoderConfig
from jepanalytics.training import train


def test_tiny_synthetic_training_is_finite_and_improves(tmp_path: Path):
    data = tmp_path / "data"
    run = tmp_path / "run"
    build_synthetic_store(data, n_molecules=24, n_bins=64, n_labels=4, seed=9)
    config = TrainingConfig(
        data_root=str(data),
        output_dir=str(run),
        encoder=EncoderConfig(
            n_bins=64,
            patch_size=8,
            hidden_dim=32,
            depth=1,
            heads=4,
            mlp_ratio=2,
            aligned_dim=16,
        ),
        augmentation=AugmentationConfig(
            noise_std=0,
            baseline_std=0,
            intensity_scale=0,
            shift_bins=0,
            broadening_probability=0,
            resolution_probability=0,
            missing_span_probability=0,
        ),
        epochs=4,
        batch_size=4,
        learning_rate=0.003,
        mask_ratio=0.5,
        alignment_weight=0.05,
        ema_decay=0.95,
        collapse_min_rank=1.0,
        device="cpu",
        log_every=1,
    )
    checkpoint = train(config, repository=tmp_path)
    assert checkpoint.exists()
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    losses = np.asarray([row["loss"] for row in rows])
    assert np.all(np.isfinite(losses))
    assert np.min(losses[-3:]) < np.mean(losses[:3])
    assert all(row["effective_rank"] > 1 for row in rows)
    assert all(row["aligned_effective_rank"] > 1 for row in rows)
    assert all(row["target_signal_residual_norm"] > 0 for row in rows)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert len(manifest["dataset_manifest_sha256"]) == 64
    assert len(manifest["artifacts"]["best_checkpoint"]["sha256"]) == 64
    assert len(manifest["artifacts"]["metrics"]["sha256"]) == 64


def test_training_resume_appends_metrics_and_records_provenance(tmp_path: Path):
    data = tmp_path / "data"
    run = tmp_path / "run"
    build_synthetic_store(data, n_molecules=16, n_bins=64, n_labels=3, seed=5)
    config = TrainingConfig(
        data_root=str(data),
        output_dir=str(run),
        encoder=EncoderConfig(
            n_bins=64,
            patch_size=8,
            hidden_dim=32,
            depth=1,
            heads=4,
            mlp_ratio=2,
            aligned_dim=16,
        ),
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        device="cpu",
        log_every=1,
    )
    train(config, repository=tmp_path)
    initial_rows = (run / "metrics.jsonl").read_text().splitlines()
    resumed = replace(config, epochs=2, resume_from=str(run / "last.pt"))
    train(resumed, repository=tmp_path)
    final_rows = (run / "metrics.jsonl").read_text().splitlines()
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert len(final_rows) > len(initial_rows)
    assert manifest["resumed_from"]["start_epoch"] == 1
    assert len(manifest["resumed_from"]["sha256"]) == 64
