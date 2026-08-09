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
    covariance_regularization,
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

    def _side(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Any, Any, torch.Tensor]:
        context = self.augment(batch["intensity"], light=False)
        target_intensity = self.augment(batch["intensity"], light=True)
        patches = context.unfold(
            1, self.config.encoder.patch_size, self.config.encoder.patch_size
        ).abs().amax(dim=-1)
        mask = mixed_patch_mask(patches, self.config.mask_ratio)
        context_kwargs = {**batch_to_encoder_kwargs(batch), "intensity": context, "patch_mask": mask}
        online = self.online(**context_kwargs)
        with torch.no_grad():
            target = self.target(
                **{
                    **batch_to_encoder_kwargs(batch),
                    "intensity": target_intensity,
                }
            )
            metadata_only = self.target(
                **{
                    **batch_to_encoder_kwargs(batch),
                    "intensity": torch.zeros_like(target_intensity),
                }
            )
            # Coordinate, unit, and acquisition tokens are intentionally part
            # of the shared encoder, but they otherwise make the JEPA target
            # predictable without looking at the spectrum.  Predict the latent
            # change caused by intensity so a metadata-only solution has loss 2.
            target_patches = target.patches - metadata_only.patches
        prediction = self.predictor(online.patches)
        residual_norm = target_patches[mask].norm(dim=-1).mean()
        return (
            masked_latent_loss(prediction, target_patches, mask),
            online,
            mask,
            residual_norm,
        )

    def forward(
        self, first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        first_jepa, first_embedding, first_mask, first_residual_norm = self._side(
            first
        )
        second_jepa, second_embedding, second_mask, second_residual_norm = self._side(
            second
        )
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
        aligned_combined = torch.cat(
            (first_embedding.aligned, second_embedding.aligned), dim=0
        )
        aligned_diagnostics = embedding_diagnostics(
            aligned_combined,
            min_std=self.config.collapse_min_std,
            min_effective_rank=self.config.collapse_min_rank,
        )
        if diagnostics.collapsed:
            variance = variance_regularization(combined)
            covariance = covariance_regularization(combined)
        else:
            variance = combined.new_zeros(())
            covariance = combined.new_zeros(())
        if aligned_diagnostics.collapsed:
            # Unit-normalized 256-D projections have a healthy per-feature
            # standard deviation near 1/sqrt(256), so use a compatible floor.
            variance = variance + variance_regularization(
                aligned_combined, target_std=0.04
            )
            covariance = covariance + covariance_regularization(aligned_combined)
        total = (
            jepa
            + self.config.alignment_weight * alignment
            + self.config.variance_weight * (variance + covariance)
        )
        metrics = {
            "loss": float(total.detach()),
            "jepa_loss": float(jepa.detach()),
            "alignment_loss": float(alignment.detach()),
            "variance_loss": float(variance.detach()),
            "covariance_loss": float(covariance.detach()),
            "embedding_std": diagnostics.mean_feature_std,
            "effective_rank": diagnostics.effective_rank,
            "aligned_embedding_std": aligned_diagnostics.mean_feature_std,
            "aligned_effective_rank": aligned_diagnostics.effective_rank,
            "collapsed": float(diagnostics.collapsed or aligned_diagnostics.collapsed),
            "masked_fraction": float(
                torch.cat((first_mask.flatten(), second_mask.flatten())).float().mean()
            ),
            "target_signal_residual_norm": float(
                0.5 * (first_residual_norm + second_residual_norm)
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


def _restore_rng_state(payload: Mapping[str, Any], device: torch.device) -> None:
    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
    if "numpy_random_state" in payload:
        np.random.set_state(payload["numpy_random_state"])
    if "torch_random_state" in payload:
        torch.set_rng_state(payload["torch_random_state"].cpu())
    if device.type == "cuda" and "cuda_random_state" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_random_state"])
    if (
        device.type == "mps"
        and "mps_random_state" in payload
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(payload["mps_random_state"])


def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda_random_state"] = torch.cuda.get_rng_state_all()
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["mps_random_state"] = torch.mps.get_rng_state()
    return state


def _device_memory_metrics(device: torch.device) -> dict[str, int]:
    if device.type == "cuda":
        return {
            "accelerator_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "accelerator_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        metrics = {
            "accelerator_allocated_bytes": int(torch.mps.current_allocated_memory()),
        }
        if hasattr(torch.mps, "driver_allocated_memory"):
            metrics["accelerator_driver_allocated_bytes"] = int(
                torch.mps.driver_allocated_memory()
            )
        return metrics
    return {}


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
    start_epoch = 0
    best_loss = math.inf
    metrics_mode = "w"
    if config.resume_from:
        resume_path = Path(config.resume_from)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        experiment.load_state_dict(payload["experiment"])
        optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if "loader_generator_state" in payload:
            loader_generator.set_state(payload["loader_generator_state"].cpu())
        _restore_rng_state(payload, device)
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"]) + 1
        best_loss = float(payload.get("best_loss", payload.get("epoch_loss", math.inf)))
        metrics_mode = "a"
        manifest["resumed_from"] = {
            "path": str(resume_path.resolve()),
            "sha256": sha256_file(resume_path),
            "start_epoch": start_epoch,
            "global_step": global_step,
        }
        write_json_atomic(manifest, output / "run_manifest.json")
    if start_epoch >= config.epochs:
        raise ValueError(
            f"checkpoint already completed epoch {start_epoch}; config requests {config.epochs} epochs"
        )
    start_time = time.monotonic()
    with metrics_path.open(metrics_mode) as metrics_file:
        for epoch in range(start_epoch, config.epochs):
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
                    **_device_memory_metrics(device),
                    **metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                if global_step % config.log_every == 0:
                    metrics_file.flush()
                global_step += 1
            epoch_loss = running / max(1, len(loader))
            is_best = epoch_loss < best_loss
            best_loss = min(best_loss, epoch_loss)
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "encoder_config": config.encoder.to_dict(),
                "training_config": config.to_dict(),
                "online_encoder": experiment.online.state_dict(),
                "experiment": experiment.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loader_generator_state": loader_generator.get_state(),
                "epoch_loss": epoch_loss,
                "best_loss": best_loss,
                **_rng_state(device),
            }
            if is_best:
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
