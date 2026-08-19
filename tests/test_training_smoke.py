import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from jepanalytics.augment import AugmentationConfig
from jepanalytics.config import TrainingConfig
from jepanalytics.data import (
    CanonicalSpectraDataset,
    FormulaMassBatchSampler,
    MultiViewSpectrumDataset,
    build_synthetic_store,
    multiview_collate,
)
from jepanalytics.losses import (
    covariance_regularization,
    group_centroid_regularization,
    modality_centroid_alignment_loss,
    multi_positive_alignment_loss,
    multiview_alignment_diagnostics,
    variance_regularization,
)
from jepanalytics.model import (
    EncoderConfig,
    UniversalSpectrumEncoder,
    batch_to_encoder_kwargs,
)
from jepanalytics.training import (
    JEPAExperiment,
    _restore_rng_state,
    jepa_token_mask,
    train,
    warm_start_encoder,
)


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
        summary_jepa_weight=1.0,
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
    assert all(row["target_summary_signal_residual_norm"] > 0 for row in rows)
    assert all(np.isfinite(row["summary_jepa_loss"]) for row in rows)
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


def test_mps_rng_restore_moves_checkpoint_state_to_cpu(monkeypatch):
    cpu_state = object()

    class LoadedMpsState:
        def cpu(self):
            return cpu_state

    restored = []
    monkeypatch.setattr(torch.mps, "set_rng_state", restored.append)
    _restore_rng_state(
        {"mps_random_state": LoadedMpsState()},
        torch.device("mps"),
    )
    assert restored == [cpu_state]


def test_multiview_alignment_overfits_known_pairs_and_retrieves_in_eval_mode(
    tmp_path: Path,
):
    data = tmp_path / "alignment-data"
    build_synthetic_store(
        data, n_molecules=24, n_bins=128, n_labels=32, seed=17
    )
    base = CanonicalSpectraDataset(data, "train")
    samples = MultiViewSpectrumDataset(
        base, views_per_molecule=5, seed=17
    )
    batch = next(
        iter(
            DataLoader(
                samples,
                batch_size=len(samples),
                shuffle=False,
                collate_fn=multiview_collate,
            )
        )
    )
    torch.manual_seed(17)
    model = UniversalSpectrumEncoder(
        EncoderConfig(
            n_bins=128,
            patch_size=8,
            hidden_dim=64,
            depth=2,
            heads=4,
            mlp_ratio=2,
            aligned_dim=32,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(110):
        output = model(**batch_to_encoder_kwargs(batch))
        aligned = model.normalize_alignment_logits(
            output.alignment_logits, update_stats=True
        )
        alignment = multi_positive_alignment_loss(
            aligned, batch["molecule_index"], temperature=0.1
        )
        regularization = (
            variance_regularization(aligned, target_std=0.1)
            + covariance_regularization(aligned)
            + group_centroid_regularization(aligned, batch["acquisition"])
            + modality_centroid_alignment_loss(
                aligned, batch["acquisition"]
            )
        )
        optimizer.zero_grad(set_to_none=True)
        (alignment + 0.1 * regularization).backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        aligned = model(**batch_to_encoder_kwargs(batch)).aligned
    diagnostics = multiview_alignment_diagnostics(
        aligned, batch["molecule_index"]
    )
    assert diagnostics["alignment_batch_top1"] > 0.95
    assert diagnostics["alignment_cosine_margin"] > 0.25


def test_multiview_jepa_training_logs_balanced_positive_pairs(tmp_path: Path):
    data = tmp_path / "multiview-data"
    run = tmp_path / "multiview-run"
    build_synthetic_store(data, n_molecules=24, n_bins=64, n_labels=8, seed=8)
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
        views_per_molecule=5,
        learning_rate=0.001,
        alignment_weight=0.2,
        collapse_min_rank=1.0,
        device="cpu",
        log_every=1,
    )
    checkpoint = train(config, repository=tmp_path)
    assert checkpoint.exists()
    rows = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows
    assert all(row["alignment_positives_per_anchor"] == 4.0 for row in rows)
    for first in range(5):
        for second in range(first + 1, 5):
            assert all(
                abs(row[f"pair_fraction_{first}_{second}"] - 0.1) < 1e-6
                for row in rows
            )


def test_formula_distillation_loss_is_finite_on_multiview_batch(tmp_path: Path):
    data = tmp_path / "chemistry-data"
    build_synthetic_store(data, n_molecules=16, n_bins=64, n_labels=4, seed=12)
    targets = np.random.default_rng(12).normal(size=(16, 3)).astype(np.float32)
    base = CanonicalSpectraDataset(data, "train", molecule_targets=targets)
    samples = MultiViewSpectrumDataset(base, views_per_molecule=5, seed=12)
    loaded = next(
        iter(
            DataLoader(
                samples,
                batch_size=4,
                shuffle=False,
                collate_fn=multiview_collate,
            )
        )
    )
    config = TrainingConfig(
        data_root=str(data),
        output_dir=str(tmp_path / "unused"),
        encoder=EncoderConfig(
            n_bins=64,
            patch_size=8,
            hidden_dim=32,
            depth=1,
            heads=4,
            mlp_ratio=2,
            aligned_dim=16,
        ),
        views_per_molecule=5,
        alignment_weight=0.2,
        chemistry_weight=1.0,
        chemistry_target_root="unused",
        chemistry_target_dim=3,
    )
    experiment = JEPAExperiment(config)
    loss, metrics = experiment.forward_multiview(loaded)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["chemistry_loss"] > 0


def test_formula_matched_prototype_loss_is_finite_on_multiview_batch(tmp_path: Path):
    data = tmp_path / "prototype-data"
    build_synthetic_store(data, n_molecules=16, n_bins=64, n_labels=4, seed=14)
    rng = np.random.default_rng(14)
    targets = rng.normal(size=(16, 3)).astype(np.float32)
    matching = rng.normal(size=(16, 4)).astype(np.float32)
    base = CanonicalSpectraDataset(
        data,
        "train",
        molecule_targets=targets,
        molecule_matching_targets=matching,
    )
    samples = MultiViewSpectrumDataset(base, views_per_molecule=5, seed=14)
    loaded = next(
        iter(
            DataLoader(
                samples,
                batch_size=4,
                shuffle=False,
                collate_fn=multiview_collate,
            )
        )
    )
    config = TrainingConfig(
        data_root=str(data),
        output_dir=str(tmp_path / "unused-prototype"),
        encoder=EncoderConfig(
            n_bins=64,
            patch_size=8,
            hidden_dim=32,
            depth=1,
            heads=4,
            mlp_ratio=2,
            aligned_dim=16,
        ),
        views_per_molecule=5,
        alignment_weight=0.2,
        chemistry_weight=1.0,
        chemistry_target_root="unused",
        chemistry_target_dim=3,
        prototype_alignment_weight=0.2,
        prototype_hard_negatives=2,
    )
    experiment = JEPAExperiment(config)
    loss, metrics = experiment.forward_multiview(loaded)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["prototype_alignment_loss"] > 0
    assert metrics["prototype_negatives_per_anchor"] == 2.0


def test_formula_mass_batch_sampler_is_complete_deterministic_and_local():
    molecules = np.arange(24, dtype=np.int64)
    matching = np.zeros((24, 3), dtype=np.float32)
    matching[:, -1] = np.arange(24, dtype=np.float32)
    sampler = FormulaMassBatchSampler(
        molecules,
        matching,
        batch_size=4,
        seed=17,
        window_batches=2,
        drop_last=False,
    )
    first = list(sampler)
    assert list(sampler) == first
    assert sorted(index for batch in first for index in batch) == list(range(24))
    assert all(np.ptp(matching[batch, -1]) <= 7 for batch in first)
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_hybrid_mask_reserves_peak_tokens_and_masks_half_the_dense_tokens():
    config = EncoderConfig(
        n_bins=96,
        patch_size=8,
        hidden_dim=36,
        depth=1,
        heads=4,
        aligned_dim=16,
        tokenizer_type="hybrid_peak_multiscale",
        multiscale_kernel_sizes=(5, 9, 15),
        hybrid_peak_tokens=3,
        peak_window_size=5,
        peak_suppression_size=5,
    )
    mask = jepa_token_mask(torch.randn(2, 96), config, 0.5)
    assert mask.shape == (2, 12)
    assert torch.all(mask[:, :9].sum(dim=1) == 4)
    assert not mask[:, 9:].any()


def test_warm_start_reuses_backbone_when_tokenizer_changes():
    common = dict(
        n_bins=96,
        patch_size=8,
        hidden_dim=36,
        depth=1,
        heads=4,
        aligned_dim=16,
        multiscale_kernel_sizes=(5, 9, 15),
        hybrid_peak_tokens=3,
        peak_window_size=5,
        peak_suppression_size=5,
    )
    source = UniversalSpectrumEncoder(EncoderConfig(**common))
    destination = UniversalSpectrumEncoder(
        EncoderConfig(**common, tokenizer_type="hybrid_peak_multiscale")
    )
    details = warm_start_encoder(destination, source.state_dict())
    assert "patch_projection.weight" in details["ignored_keys"]
    assert any(
        key.startswith("peak_projection.")
        for key in details["initialized_keys"]
    )
    source_weight = source.transformer.layers[0].linear1.weight
    destination_weight = destination.transformer.layers[0].linear1.weight
    assert torch.equal(source_weight, destination_weight)
