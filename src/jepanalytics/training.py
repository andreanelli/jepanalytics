"""JEPA and controlled masked-reconstruction training loops."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .augment import SpectralAugmenter
from .config import TrainingConfig, save_config
from .data import CanonicalSpectraDataset, PairedSpectrumDataset, paired_collate
from .losses import (
    embedding_diagnostics,
    masked_latent_loss,
    symmetric_alignment_loss,
    variance_regularization,
)
from .manifest import build_run_manifest, sha256_file, write_json_atomic
from .masking import mixed_patch_mask
from .model import LatentPredictor, UniversalSpectrumEncoder, batch_to_encoder_kwargs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class JEPAExperiment(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.online = UniversalSpectrumEncoder(config.encoder)
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.target.eval()
        self.predictor = LatentPredictor(hidden_dim=config.encoder.hidden_dim)
        self.augment = SpectralAugmenter(config.augmentation)

    def train(self, mode: bool = True) -> "JEPAExperiment":
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def update_target(self) -> None:
        decay = self.config.ema_decay
        for target, online in zip(self.target.parameters(), self.online.parameters(), strict=True):
            target.mul_(decay).add_(online, alpha=1.0 - decay)

    def _side(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, Any, Any]:
        context = self.augment(batch["intensity"], light=False)
        target_intensity = self.augment(batch["intensity"], light=True)
        patches = context.unfold(
            1, self.config.encoder.patch_size, self.config.encoder.patch_size
        ).abs().amax(dim=-1)
        mask = mixed_patch_mask(patches, self.config.mask_ratio)
        context_kwargs = {**batch_to_encoder_kwargs(batch), "intensity": context, "patch_mask": mask}
        online = self.online(**context_kwargs)
        with torch.no_grad():
            target = self.target(**{**batch_to_encoder_kwargs(batch), "intensity": target_intensity})
        prediction = self.predictor(online.patches)
        return masked_latent_loss(prediction, target.patches, mask), online, mask

    def forward(
        self, first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        first_jepa, first_embedding, first_mask = self._side(first)
        second_jepa, second_embedding, second_mask = self._side(second)
        jepa = 0.5 * (first_jepa + second_jepa)
        alignment = symmetric_alignment_loss(
            first_embedding.aligned,
            second_embedding.aligned,
            self.config.alignment_temperature,
        )
        combined = torch.cat((first_embedding.general, second_embedding.general), dim=0)
        diagnostics = embedding_diagnostics(
            combined,
            min_std=self.config.collapse_min_std,
            min_effective_rank=self.config.collapse_min_rank,
        )
        variance = variance_regularization(combined) if diagnostics.collapsed else combined.new_zeros(())
        total = (
            jepa
            + self.config.alignment_weight * alignment
            + self.config.variance_weight * variance
        )
        metrics = {
            "loss": float(total.detach()),
            "jepa_loss": float(jepa.detach()),
            "alignment_loss": float(alignment.detach()),
            "variance_loss": float(variance.detach()),
            "embedding_std": diagnostics.mean_feature_std,
            "effective_rank": diagnostics.effective_rank,
            "collapsed": float(diagnostics.collapsed),
            "masked_fraction": float(
                torch.cat((first_mask.flatten(), second_mask.flatten())).float().mean()
            ),
        }
        return total, metrics


class MaskedAutoencoderExperiment(nn.Module):
    """Raw-intensity baseline with the identical shared encoder."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.online = UniversalSpectrumEncoder(config.encoder)
        self.decoder = nn.Linear(config.encoder.hidden_dim, config.encoder.patch_size)
        self.augment = SpectralAugmenter(config.augmentation)

    def forward(
        self, first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses = []
        embeddings = []
        masks = []
        for batch in (first, second):
            intensity = self.augment(batch["intensity"], light=False)
            raw_patches = batch["intensity"].unfold(
                1, self.config.encoder.patch_size, self.config.encoder.patch_size
            )
            scores = raw_patches.abs().amax(dim=-1)
            mask = mixed_patch_mask(scores, self.config.mask_ratio)
            encoded = self.online(
                **{**batch_to_encoder_kwargs(batch), "intensity": intensity, "patch_mask": mask}
            )
            reconstruction = self.decoder(encoded.patches)
            losses.append((reconstruction[mask] - raw_patches[mask]).square().mean())
            embeddings.append(encoded.general)
            masks.append(mask)
        loss = torch.stack(losses).mean()
        diagnostics = embedding_diagnostics(torch.cat(embeddings))
        return loss, {
            "loss": float(loss.detach()),
            "reconstruction_loss": float(loss.detach()),
            "embedding_std": diagnostics.mean_feature_std,
            "effective_rank": diagnostics.effective_rank,
            "collapsed": float(diagnostics.collapsed),
            "masked_fraction": float(torch.cat([mask.flatten() for mask in masks]).float().mean()),
        }


def _filter_acquisition(
    dataset: CanonicalSpectraDataset,
    acquisition: int | None,
    excluded_acquisitions: tuple[int, ...] = (),
) -> None:
    if acquisition is not None and excluded_acquisitions:
        raise ValueError("acquisition and excluded_acquisitions are mutually exclusive")
    if acquisition is not None:
        dataset.indices = dataset.indices[dataset.arrays["acquisition"][dataset.indices] == acquisition]
    elif excluded_acquisitions:
        dataset.indices = dataset.indices[
            ~np.isin(dataset.arrays["acquisition"][dataset.indices], excluded_acquisitions)
        ]
    if not len(dataset):
        raise ValueError("acquisition filtering removed every training record")


def _atomic_checkpoint(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def train(config: TrainingConfig, *, repository: str | Path | None = None) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.json")
    dataset = CanonicalSpectraDataset(config.data_root, split="train")
    if dataset.manifest["n_bins"] != config.encoder.n_bins:
        raise ValueError("dataset n_bins does not match encoder configuration")
    _filter_acquisition(dataset, config.acquisition, config.excluded_acquisitions)
    paired = PairedSpectrumDataset(dataset, seed=config.seed)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        paired,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=config.num_workers,
        collate_fn=paired_collate,
        drop_last=len(paired) >= config.batch_size,
        pin_memory=device.type == "cuda",
    )
    experiment: JEPAExperiment | MaskedAutoencoderExperiment
    experiment = JEPAExperiment(config) if config.objective == "jepa" else MaskedAutoencoderExperiment(config)
    experiment.to(device)
    parameters = [parameter for parameter in experiment.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs * len(loader))
    )
    repository = Path(repository or Path.cwd())
    manifest = build_run_manifest(
        config=config.to_dict(),
        dataset_manifest=Path(config.data_root) / "manifest.json",
        repository=repository,
    )
    write_json_atomic(manifest, output / "run_manifest.json")
    metrics_path = output / "metrics.jsonl"
    global_step = 0
    best_loss = math.inf
    start_time = time.monotonic()
    with metrics_path.open("w") as metrics_file:
        for epoch in range(config.epochs):
            paired.set_epoch(epoch)
            experiment.train()
            running = 0.0
            for first, second in loader:
                first = move_batch(first, device)
                second = move_batch(second, device)
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = experiment(first, second)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {global_step}: {loss}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
                optimizer.step()
                scheduler.step()
                if isinstance(experiment, JEPAExperiment):
                    experiment.update_target()
                running += float(loss.detach())
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.monotonic() - start_time,
                    **metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                if global_step % config.log_every == 0:
                    metrics_file.flush()
                global_step += 1
            epoch_loss = running / max(1, len(loader))
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "encoder_config": config.encoder.to_dict(),
                "training_config": config.to_dict(),
                "online_encoder": experiment.online.state_dict(),
                "experiment": experiment.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch_loss": epoch_loss,
            }
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                _atomic_checkpoint(checkpoint, output / "best.pt")
            if (epoch + 1) % config.checkpoint_every == 0:
                _atomic_checkpoint(checkpoint, output / "last.pt")
    best_checkpoint = output / "best.pt"
    last_checkpoint = output / "last.pt"
    summary_path = output / "summary.json"
    write_json_atomic(
        {
            "status": "complete",
            "best_epoch_loss": best_loss,
            "epochs": config.epochs,
            "steps": global_step,
            "elapsed_seconds": time.monotonic() - start_time,
            "checkpoint": str(best_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(best_checkpoint),
        },
        summary_path,
    )
    manifest["completed_at"] = time.time()
    artifacts = {
        "best_checkpoint": {
            "path": str(best_checkpoint.resolve()),
            "sha256": sha256_file(best_checkpoint),
        },
        "metrics": {
            "path": str(metrics_path.resolve()),
            "sha256": sha256_file(metrics_path),
        },
        "summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
        },
    }
    if last_checkpoint.exists():
        artifacts["last_checkpoint"] = {
            "path": str(last_checkpoint.resolve()),
            "sha256": sha256_file(last_checkpoint),
        }
    manifest["artifacts"] = artifacts
    write_json_atomic(manifest, output / "run_manifest.json")
    return best_checkpoint


def load_encoder_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> UniversalSpectrumEncoder:
    payload = torch.load(path, map_location=device, weights_only=False)
    from .model import EncoderConfig

    encoder = UniversalSpectrumEncoder(config=EncoderConfig(**payload["encoder_config"]))
    encoder.load_state_dict(payload["online_encoder"])
    encoder.to(device)
    encoder.eval()
    return encoder
