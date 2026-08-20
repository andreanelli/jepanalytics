"""Strict JSON configuration objects for reproducible experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .augment import AugmentationConfig
from .model import EncoderConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    data_root: str
    output_dir: str
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    objective: str = "jepa"
    epochs: int = 20
    batch_size: int = 32
    views_per_molecule: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    mask_ratio: float = 0.50
    alignment_weight: float = 0.05
    alignment_temperature: float = 0.07
    alignment_clean_view: bool = True
    alignment_uniformity_weight: float = 0.10
    alignment_target_std: float = 0.04
    summary_jepa_weight: float = 0.0
    chemistry_weight: float = 0.0
    chemistry_target_root: str | None = None
    chemistry_target_dim: int = 15
    chemistry_clean_view: bool = True
    prototype_alignment_weight: float = 0.0
    prototype_temperature: float = 0.10
    prototype_hard_negatives: int = 8
    prototype_clean_view: bool = True
    ema_decay: float = 0.996
    variance_weight: float = 0.10
    collapse_min_std: float = 0.02
    collapse_min_rank: float = 8.0
    gradient_clip: float = 1.0
    num_workers: int = 0
    seed: int = 17
    device: str = "auto"
    resume_from: str | None = None
    initialize_from: str | None = None
    max_train_molecules: int | None = None
    acquisition: int | None = None
    excluded_acquisitions: tuple[int, ...] = ()
    log_every: int = 20
    checkpoint_every: int = 1
    snapshot_epochs: tuple[int, ...] = ()
    wandb_enabled: bool = False
    wandb_project: str = "jepanalytics"
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: str = "online"

    def __post_init__(self) -> None:
        if self.objective not in {"jepa", "mae"}:
            raise ValueError("objective must be 'jepa' or 'mae'")
        if self.alignment_weight not in {0.0, 0.05, 0.2}:
            raise ValueError("alignment_weight must be one of 0, 0.05, or 0.2")
        if self.alignment_temperature <= 0:
            raise ValueError("alignment_temperature must be positive")
        if not 2 <= self.views_per_molecule <= 5:
            raise ValueError("views_per_molecule must be between 2 and 5")
        if self.objective == "mae" and self.views_per_molecule != 2:
            raise ValueError("masked-autoencoder controls require two views per molecule")
        if self.objective == "mae" and self.encoder.tokenizer_type != "linear_patch":
            raise ValueError(
                "masked-autoencoder controls currently require linear_patch tokenization"
            )
        if self.alignment_uniformity_weight < 0:
            raise ValueError("alignment_uniformity_weight cannot be negative")
        if self.alignment_target_std <= 0:
            raise ValueError("alignment_target_std must be positive")
        if self.summary_jepa_weight < 0:
            raise ValueError("summary_jepa_weight cannot be negative")
        if self.chemistry_weight < 0:
            raise ValueError("chemistry_weight cannot be negative")
        if self.prototype_alignment_weight not in {0.0, 0.2, 1.0}:
            raise ValueError("prototype_alignment_weight must be one of 0, 0.2, or 1")
        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if self.prototype_hard_negatives < 1:
            raise ValueError("prototype_hard_negatives must be positive")
        uses_formula_targets = (
            self.chemistry_weight > 0 or self.prototype_alignment_weight > 0
        )
        if bool(self.chemistry_target_root) != uses_formula_targets:
            raise ValueError(
                "chemistry_target_root is required exactly when chemistry or prototype "
                "alignment is enabled"
            )
        if self.chemistry_target_dim < 1:
            raise ValueError("chemistry_target_dim must be positive")
        if self.chemistry_weight > 0 and self.objective != "jepa":
            raise ValueError("chemistry distillation is supported only for JEPA")
        if self.prototype_alignment_weight > 0 and self.objective != "jepa":
            raise ValueError("prototype alignment is supported only for JEPA")
        if self.prototype_alignment_weight > 0 and self.views_per_molecule <= 2:
            raise ValueError("prototype alignment requires at least three molecular views")
        if not 0.4 <= self.mask_ratio <= 0.6:
            raise ValueError("mask_ratio must remain in the preregistered 40-60% interval")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.wandb_enabled and not self.wandb_run_id:
            raise ValueError("wandb_run_id is required when W&B logging is enabled")
        if self.resume_from and self.initialize_from:
            raise ValueError("resume_from and initialize_from are mutually exclusive")
        if self.max_train_molecules is not None and self.max_train_molecules < 1:
            raise ValueError("max_train_molecules must be positive")
        if any(epoch < 1 or epoch > self.epochs for epoch in self.snapshot_epochs):
            raise ValueError("snapshot_epochs must lie between 1 and epochs")
        if len(set(self.snapshot_epochs)) != len(self.snapshot_epochs):
            raise ValueError("snapshot_epochs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_training_config(path: str | Path) -> TrainingConfig:
    raw = json.loads(Path(path).read_text())
    raw["encoder"] = EncoderConfig(**raw.get("encoder", {}))
    raw["augmentation"] = AugmentationConfig(**raw.get("augmentation", {}))
    raw["excluded_acquisitions"] = tuple(raw.get("excluded_acquisitions", ()))
    raw["wandb_tags"] = tuple(raw.get("wandb_tags", ()))
    raw["snapshot_epochs"] = tuple(raw.get("snapshot_epochs", ()))
    return TrainingConfig(**raw)


def save_config(config: TrainingConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
