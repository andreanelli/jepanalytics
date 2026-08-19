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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augment import SpectralAugmenter
from .chemistry import formula_matching_features, load_formula_targets
from .config import TrainingConfig, save_config
from .data import (
    CanonicalSpectraDataset,
    FormulaMassBatchSampler,
    MultiViewSpectrumDataset,
    PairedSpectrumDataset,
    multiview_collate,
    paired_collate,
)
from .losses import (
    alignment_diagnostics,
    covariance_regularization,
    embedding_diagnostics,
    group_centroid_regularization,
    hard_negative_prototype_alignment_loss,
    latent_vector_loss,
    masked_latent_loss,
    modality_centroid_alignment_loss,
    multi_positive_alignment_loss,
    multiview_alignment_diagnostics,
    pairwise_multiview_alignment_diagnostics,
    symmetric_multi_positive_alignment_loss,
    variance_regularization,
)
from .manifest import build_run_manifest, sha256_file, write_json_atomic
from .masking import mixed_patch_mask
from .model import (
    ChemistryPredictor,
    LatentPredictor,
    SummaryLatentPredictor,
    UniversalSpectrumEncoder,
    batch_to_encoder_kwargs,
)
from .tracking import WandbTracker


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


def jepa_token_mask(
    intensity: torch.Tensor, config: Any, mask_ratio: float
) -> torch.Tensor:
    """Mask dense signal tokens while reserving hybrid peak tokens as context."""

    dense_count = config.dense_token_count
    if dense_count == config.n_patches:
        scores = intensity.unfold(
            1, config.patch_size, config.patch_size
        ).abs().amax(dim=-1)
    else:
        scores = F.adaptive_max_pool1d(
            intensity.abs().unsqueeze(1), dense_count
        ).squeeze(1)
    dense_mask = mixed_patch_mask(scores, mask_ratio)
    if dense_count == config.n_patches:
        return dense_mask
    peak_context = torch.zeros(
        intensity.shape[0],
        config.n_patches - dense_count,
        dtype=torch.bool,
        device=intensity.device,
    )
    return torch.cat((dense_mask, peak_context), dim=1)


def dense_mask_fraction(mask: torch.Tensor, config: Any) -> float:
    return float(mask[:, : config.dense_token_count].float().mean())


TOKENIZER_STATE_PREFIXES = (
    "patch_projection.",
    "overlap_projection.",
    "multiscale_projections.",
    "peak_projection.",
    "peak_metadata_projection.",
    "token_type_embedding.",
)


def warm_start_encoder(
    encoder: UniversalSpectrumEncoder,
    state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Load all compatible backbone weights while allowing tokenizer replacement."""

    destination = encoder.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in destination and destination[key].shape == value.shape
    }
    missing, unexpected = encoder.load_state_dict(compatible, strict=False)
    invalid_missing = [
        key
        for key in missing
        if not key.startswith(TOKENIZER_STATE_PREFIXES)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "warm-start state mismatch; "
            f"missing={invalid_missing}, unexpected={list(unexpected)}"
        )
    ignored = sorted(set(state) - set(compatible))
    return {
        "loaded_keys": len(compatible),
        "ignored_keys": ignored,
        "initialized_keys": sorted(missing),
    }


class JEPAExperiment(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.online = UniversalSpectrumEncoder(config.encoder)
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.target.eval()
        self.predictor = LatentPredictor(hidden_dim=config.encoder.hidden_dim)
        self.summary_predictor = SummaryLatentPredictor(
            hidden_dim=config.encoder.hidden_dim
        )
        self.chemistry_predictor = ChemistryPredictor(
            config.encoder.hidden_dim, config.chemistry_target_dim
        )
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
        Any,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        context = self.augment(
            batch["intensity"], light=False, acquisition=batch["acquisition"]
        )
        target_intensity = self.augment(
            batch["intensity"], light=True, acquisition=batch["acquisition"]
        )
        mask = jepa_token_mask(
            context, self.config.encoder, self.config.mask_ratio
        )
        context_kwargs = {
            **batch_to_encoder_kwargs(batch),
            "intensity": context,
            "patch_mask": mask,
        }
        online = self.online(**context_kwargs)
        needs_clean_online = (
            self.config.alignment_weight > 0 and self.config.alignment_clean_view
        ) or (
            self.config.prototype_alignment_weight > 0
            and self.config.prototype_clean_view
        )
        if needs_clean_online:
            alignment_embedding = self.online(
                **{
                    **batch_to_encoder_kwargs(batch),
                    "intensity": target_intensity,
                }
            )
        else:
            alignment_embedding = online
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
            target_summary = target.general - metadata_only.general
        prediction = self.predictor(online.patches)
        summary_prediction = self.summary_predictor(online.general)
        residual_norm = target_patches[mask].norm(dim=-1).mean()
        summary_residual_norm = target_summary.norm(dim=-1).mean()
        if self.config.chemistry_weight > 0:
            if "chemistry_target" not in batch:
                raise ValueError("chemistry_weight requires chemistry targets in each batch")
            chemistry_embedding = (
                alignment_embedding
                if self.config.chemistry_clean_view
                else online
            )
            chemistry = nn.functional.smooth_l1_loss(
                self.chemistry_predictor(chemistry_embedding.general),
                batch["chemistry_target"],
            )
        else:
            chemistry = online.general.new_zeros(())
        return (
            masked_latent_loss(prediction, target_patches, mask),
            latent_vector_loss(summary_prediction, target_summary),
            chemistry,
            online,
            alignment_embedding,
            target.general,
            mask,
            residual_norm,
            summary_residual_norm,
        )

    def forward(
        self, first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        (
            first_jepa,
            first_summary_jepa,
            first_chemistry,
            _first_context_embedding,
            first_embedding,
            _first_target_general,
            first_mask,
            first_residual_norm,
            first_summary_residual_norm,
        ) = self._side(first)
        (
            second_jepa,
            second_summary_jepa,
            second_chemistry,
            _second_context_embedding,
            second_embedding,
            _second_target_general,
            second_mask,
            second_residual_norm,
            second_summary_residual_norm,
        ) = self._side(second)
        jepa = 0.5 * (first_jepa + second_jepa)
        summary_jepa = 0.5 * (first_summary_jepa + second_summary_jepa)
        chemistry = 0.5 * (first_chemistry + second_chemistry)
        joint_aligned = self.online.normalize_alignment_logits(
            torch.cat(
                (first_embedding.alignment_logits, second_embedding.alignment_logits),
                dim=0,
            ),
            update_stats=True,
        )
        first_aligned, second_aligned = joint_aligned.split(
            (first_embedding.aligned.shape[0], second_embedding.aligned.shape[0]),
            dim=0,
        )
        alignment = symmetric_multi_positive_alignment_loss(
            first_aligned,
            second_aligned,
            first["molecule_index"],
            second["molecule_index"],
            self.config.alignment_temperature,
        )
        combined = torch.cat((first_embedding.general, second_embedding.general), dim=0)
        diagnostics = embedding_diagnostics(
            combined,
            min_std=self.config.collapse_min_std,
            min_effective_rank=self.config.collapse_min_rank,
        )
        aligned_combined = joint_aligned
        aligned_diagnostics = embedding_diagnostics(
            aligned_combined,
            min_std=self.config.collapse_min_std,
            min_effective_rank=self.config.collapse_min_rank,
        )
        # The general embedding leaves a LayerNorm, so per-sample feature
        # statistics are healthy by construction and the collapse detector this
        # term used to be gated on never fired: across the whole 100k run the
        # weighted variance and covariance contribution was exactly zero.  The
        # representation every downstream probe reads therefore had no active
        # decorrelation pressure at all.  Apply it unconditionally and let
        # variance_weight control its strength.
        general_variance = variance_regularization(combined)
        general_covariance = covariance_regularization(combined)
        aligned_variance = aligned_combined.new_zeros(())
        aligned_covariance = aligned_combined.new_zeros(())
        alignment_centroid = aligned_combined.new_zeros(())
        alignment_modality_centroid = aligned_combined.new_zeros(())
        alignment_uniformity = aligned_combined.new_zeros(())
        if self.config.alignment_weight > 0:
            # A thresholded penalty let the first pilot settle exactly at the
            # collapse boundary with a large common direction. Keep the aligned
            # space spread and centered throughout training instead.
            aligned_variance = variance_regularization(
                aligned_combined, target_std=self.config.alignment_target_std
            )
            aligned_covariance = covariance_regularization(aligned_combined)
            alignment_centroid = group_centroid_regularization(
                aligned_combined,
                torch.cat((first["acquisition"], second["acquisition"])),
            )
            alignment_modality_centroid = modality_centroid_alignment_loss(
                aligned_combined,
                torch.cat((first["acquisition"], second["acquisition"])),
            )
            alignment_uniformity = (
                aligned_variance
                + aligned_covariance
                + alignment_centroid
                + alignment_modality_centroid
            )
        total = (
            jepa
            + self.config.summary_jepa_weight * summary_jepa
            + self.config.chemistry_weight * chemistry
            + self.config.alignment_weight * alignment
            + self.config.variance_weight * (general_variance + general_covariance)
            + self.config.alignment_uniformity_weight * alignment_uniformity
        )
        alignment_metrics = alignment_diagnostics(
            first_aligned,
            second_aligned,
            first["molecule_index"],
            second["molecule_index"],
        )
        metrics = {
            "loss": float(total.detach()),
            "jepa_loss": float(jepa.detach()),
            "summary_jepa_loss": float(summary_jepa.detach()),
            "chemistry_loss": float(chemistry.detach()),
            "alignment_loss": float(alignment.detach()),
            "variance_loss": float((general_variance + aligned_variance).detach()),
            "covariance_loss": float((general_covariance + aligned_covariance).detach()),
            "alignment_uniformity_loss": float(alignment_uniformity.detach()),
            "alignment_centroid_loss": float(alignment_centroid.detach()),
            "alignment_modality_centroid_loss": float(
                alignment_modality_centroid.detach()
            ),
            "embedding_std": diagnostics.mean_feature_std,
            "effective_rank": diagnostics.effective_rank,
            "aligned_embedding_std": aligned_diagnostics.mean_feature_std,
            # An L2-normalized d-dimensional embedding caps the mean
            # per-feature std at 1/sqrt(d).  Logging that reference makes
            # it obvious how much headroom alignment_target_std leaves.
            "aligned_isotropic_std": self.config.encoder.aligned_dim ** -0.5,
            "aligned_effective_rank": aligned_diagnostics.effective_rank,
            "collapsed": float(diagnostics.collapsed or aligned_diagnostics.collapsed),
            "masked_fraction": float(
                0.5
                * (
                    dense_mask_fraction(first_mask, self.config.encoder)
                    + dense_mask_fraction(second_mask, self.config.encoder)
                )
            ),
            "dense_token_count": float(self.config.encoder.dense_token_count),
            "peak_token_count": float(
                self.config.encoder.n_patches
                - self.config.encoder.dense_token_count
            ),
            "target_signal_residual_norm": float(
                0.5 * (first_residual_norm + second_residual_norm)
            ),
            "target_summary_signal_residual_norm": float(
                0.5
                * (first_summary_residual_norm + second_summary_residual_norm)
            ),
            **alignment_metrics,
        }
        pair_codes = torch.minimum(
            first["acquisition"], second["acquisition"]
        ) * 5 + torch.maximum(first["acquisition"], second["acquisition"])
        for first_acquisition in range(5):
            for second_acquisition in range(first_acquisition + 1, 5):
                metrics[
                    f"pair_fraction_{first_acquisition}_{second_acquisition}"
                ] = float(
                    (pair_codes == first_acquisition * 5 + second_acquisition)
                    .float()
                    .mean()
                )
        return total, metrics

    def forward_multiview(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        (
            jepa,
            summary_jepa,
            chemistry,
            _context_embedding,
            embedding,
            target_general,
            mask,
            residual_norm,
            summary_residual_norm,
        ) = self._side(batch)
        aligned = self.online.normalize_alignment_logits(
            embedding.alignment_logits, update_stats=True
        )
        alignment = multi_positive_alignment_loss(
            aligned,
            batch["molecule_index"],
            self.config.alignment_temperature,
        )
        if self.config.prototype_alignment_weight > 0:
            if "prototype_matching_target" not in batch:
                raise ValueError(
                    "prototype alignment requires formula/mass matching targets"
                )
            prototype, prototype_metrics = hard_negative_prototype_alignment_loss(
                embedding.general,
                target_general,
                batch["molecule_index"],
                batch["prototype_matching_target"],
                temperature=self.config.prototype_temperature,
                hard_negatives=self.config.prototype_hard_negatives,
            )
        else:
            prototype = embedding.general.new_zeros(())
            prototype_metrics = {
                "prototype_positive_cosine": 0.0,
                "prototype_hard_negative_cosine": 0.0,
                "prototype_cosine_margin": 0.0,
                "prototype_batch_top1": 0.0,
                "prototype_hard_negative_distance": 0.0,
                "prototype_negatives_per_anchor": 0.0,
            }
        general_diagnostics = embedding_diagnostics(
            embedding.general,
            min_std=self.config.collapse_min_std,
            min_effective_rank=self.config.collapse_min_rank,
        )
        aligned_diagnostics = embedding_diagnostics(
            aligned,
            min_std=self.config.collapse_min_std,
            min_effective_rank=self.config.collapse_min_rank,
        )
        # Applied unconditionally: see the two-view path for why gating this on
        # the collapse detector made it dead code.
        general_variance = variance_regularization(embedding.general)
        general_covariance = covariance_regularization(embedding.general)
        aligned_variance = variance_regularization(
            aligned, target_std=self.config.alignment_target_std
        )
        aligned_covariance = covariance_regularization(aligned)
        alignment_centroid = group_centroid_regularization(
            aligned, batch["acquisition"]
        )
        alignment_modality_centroid = modality_centroid_alignment_loss(
            aligned, batch["acquisition"]
        )
        alignment_uniformity = (
            aligned_variance
            + aligned_covariance
            + alignment_centroid
            + alignment_modality_centroid
        )
        total = (
            jepa
            + self.config.summary_jepa_weight * summary_jepa
            + self.config.chemistry_weight * chemistry
            + self.config.alignment_weight * alignment
            + self.config.prototype_alignment_weight * prototype
            + self.config.variance_weight * (general_variance + general_covariance)
            + self.config.alignment_uniformity_weight * alignment_uniformity
        )
        metrics = {
            "loss": float(total.detach()),
            "jepa_loss": float(jepa.detach()),
            "summary_jepa_loss": float(summary_jepa.detach()),
            "chemistry_loss": float(chemistry.detach()),
            "alignment_loss": float(alignment.detach()),
            "prototype_alignment_loss": float(prototype.detach()),
            "variance_loss": float((general_variance + aligned_variance).detach()),
            "covariance_loss": float(
                (general_covariance + aligned_covariance).detach()
            ),
            "alignment_uniformity_loss": float(alignment_uniformity.detach()),
            "alignment_centroid_loss": float(alignment_centroid.detach()),
            "alignment_modality_centroid_loss": float(
                alignment_modality_centroid.detach()
            ),
            "embedding_std": general_diagnostics.mean_feature_std,
            "effective_rank": general_diagnostics.effective_rank,
            "aligned_embedding_std": aligned_diagnostics.mean_feature_std,
            # An L2-normalized d-dimensional embedding caps the mean
            # per-feature std at 1/sqrt(d).  Logging that reference makes
            # it obvious how much headroom alignment_target_std leaves.
            "aligned_isotropic_std": self.config.encoder.aligned_dim ** -0.5,
            "aligned_effective_rank": aligned_diagnostics.effective_rank,
            "collapsed": float(
                general_diagnostics.collapsed or aligned_diagnostics.collapsed
            ),
            "masked_fraction": dense_mask_fraction(mask, self.config.encoder),
            "dense_token_count": float(self.config.encoder.dense_token_count),
            "peak_token_count": float(
                self.config.encoder.n_patches
                - self.config.encoder.dense_token_count
            ),
            "target_signal_residual_norm": float(residual_norm),
            "target_summary_signal_residual_norm": float(summary_residual_norm),
            **multiview_alignment_diagnostics(
                aligned, batch["molecule_index"]
            ),
            **pairwise_multiview_alignment_diagnostics(
                aligned,
                batch["molecule_index"],
                batch["acquisition"],
            ),
            **prototype_metrics,
        }
        molecule = batch["molecule_index"]
        acquisition = batch["acquisition"]
        upper = torch.triu(
            torch.ones(
                (molecule.shape[0], molecule.shape[0]),
                dtype=torch.bool,
                device=molecule.device,
            ),
            diagonal=1,
        )
        positive_pairs = (molecule[:, None] == molecule[None, :]) & upper
        pair_count = positive_pairs.float().sum().clamp_min(1.0)
        for first_acquisition in range(5):
            for second_acquisition in range(first_acquisition + 1, 5):
                family_pair = (
                    (
                        (acquisition[:, None] == first_acquisition)
                        & (acquisition[None, :] == second_acquisition)
                    )
                    | (
                        (acquisition[:, None] == second_acquisition)
                        & (acquisition[None, :] == first_acquisition)
                    )
                )
                metrics[
                    f"pair_fraction_{first_acquisition}_{second_acquisition}"
                ] = float((positive_pairs & family_pair).float().sum() / pair_count)
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
            intensity = self.augment(
                batch["intensity"], light=False, acquisition=batch["acquisition"]
            )
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


def _limit_molecules(
    dataset: CanonicalSpectraDataset,
    maximum: int | None,
    *,
    seed: int,
) -> None:
    """Deterministically retain complete molecule groups for bounded calibrations."""

    if maximum is None:
        return
    molecules = np.asarray(dataset.arrays["molecule_index"][dataset.indices])
    unique = np.unique(molecules)
    if unique.size <= maximum:
        return
    selected = np.random.default_rng(seed).choice(unique, size=maximum, replace=False)
    dataset.indices = dataset.indices[np.isin(molecules, selected)]


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
        torch.mps.set_rng_state(payload["mps_random_state"].cpu())


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
    chemistry_targets = None
    matching_targets = None
    chemistry_manifest = None
    if config.chemistry_target_root:
        chemistry_targets, chemistry_manifest = load_formula_targets(
            config.chemistry_target_root, config.data_root
        )
        if chemistry_targets.shape[1] != config.chemistry_target_dim:
            raise ValueError("chemistry target dimension does not match training config")
        if config.prototype_alignment_weight > 0:
            matching_targets = formula_matching_features(
                chemistry_targets, chemistry_manifest
            )
    dataset = CanonicalSpectraDataset(
        config.data_root,
        split="train",
        molecule_targets=chemistry_targets,
        molecule_matching_targets=matching_targets,
    )
    if dataset.manifest["n_bins"] != config.encoder.n_bins:
        raise ValueError("dataset n_bins does not match encoder configuration")
    _filter_acquisition(dataset, config.acquisition, config.excluded_acquisitions)
    _limit_molecules(dataset, config.max_train_molecules, seed=config.seed)
    if config.views_per_molecule > 2:
        sampled: PairedSpectrumDataset | MultiViewSpectrumDataset = (
            MultiViewSpectrumDataset(
                dataset,
                views_per_molecule=config.views_per_molecule,
                seed=config.seed,
            )
        )
        collate = multiview_collate
    else:
        sampled = PairedSpectrumDataset(dataset, seed=config.seed)
        collate = paired_collate
    loader_generator = torch.Generator().manual_seed(config.seed)
    matched_batch_sampler = None
    if config.prototype_alignment_weight > 0:
        if not isinstance(sampled, MultiViewSpectrumDataset) or matching_targets is None:
            raise ValueError("prototype alignment requires multi-view matching targets")
        matched_batch_sampler = FormulaMassBatchSampler(
            sampled.molecule_indices,
            matching_targets,
            batch_size=config.batch_size,
            seed=config.seed,
            drop_last=len(sampled) >= config.batch_size,
        )
        loader = DataLoader(
            sampled,
            batch_sampler=matched_batch_sampler,
            num_workers=config.num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
    else:
        loader = DataLoader(
            sampled,
            batch_size=config.batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=config.num_workers,
            collate_fn=collate,
            drop_last=len(sampled) >= config.batch_size,
            pin_memory=device.type == "cuda",
        )
    experiment: JEPAExperiment | MaskedAutoencoderExperiment
    experiment = JEPAExperiment(config) if config.objective == "jepa" else MaskedAutoencoderExperiment(config)
    initialized_from: dict[str, Any] | None = None
    if config.initialize_from:
        initialization_path = Path(config.initialize_from)
        initialization = torch.load(
            initialization_path, map_location="cpu", weights_only=False
        )
        warm_start = warm_start_encoder(
            experiment.online, initialization["online_encoder"]
        )
        if isinstance(experiment, JEPAExperiment):
            experiment.target.load_state_dict(experiment.online.state_dict())
        initialized_from = {
            "path": str(initialization_path.resolve()),
            "sha256": sha256_file(initialization_path),
            **warm_start,
        }
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
    if initialized_from is not None:
        manifest["initialized_from"] = initialized_from
    if chemistry_manifest is not None:
        manifest["chemistry_targets"] = chemistry_manifest
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
    tracker = WandbTracker.start(
        config,
        output=output,
        run_manifest=manifest,
        dataset_manifest=dataset.manifest,
        encoder_parameters=sum(parameter.numel() for parameter in experiment.online.parameters()),
        train_molecules=len(sampled),
        steps_per_epoch=len(loader),
    )
    if tracker.run is not None:
        manifest["wandb"] = json.loads(tracker.state_path.read_text())
        if config.resume_from:
            manifest["wandb"]["backfilled_history_points"] = tracker.backfill(metrics_path)
        write_json_atomic(manifest, output / "run_manifest.json")
    start_time = time.monotonic()
    steps_this_process = 0
    total_steps = config.epochs * len(loader)
    with metrics_path.open(metrics_mode) as metrics_file:
        for epoch in range(start_epoch, config.epochs):
            sampled.set_epoch(epoch)
            if matched_batch_sampler is not None:
                matched_batch_sampler.set_epoch(epoch)
            experiment.train()
            running = 0.0
            for batch_index, loaded_batch in enumerate(loader):
                step_started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                if config.views_per_molecule > 2:
                    if not isinstance(experiment, JEPAExperiment):
                        raise ValueError("multi-view training is only supported for JEPA")
                    multiview = move_batch(loaded_batch, device)
                    loss, metrics = experiment.forward_multiview(multiview)
                    molecules_this_step = int(
                        torch.unique(multiview["molecule_index"]).numel()
                    )
                    spectra_this_step = int(multiview["intensity"].shape[0])
                else:
                    first, second = loaded_batch
                    first = move_batch(first, device)
                    second = move_batch(second, device)
                    loss, metrics = experiment(first, second)
                    molecules_this_step = int(first["intensity"].shape[0])
                    spectra_this_step = molecules_this_step * 2
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {global_step}: {loss}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
                optimizer.step()
                scheduler.step()
                if isinstance(experiment, JEPAExperiment):
                    experiment.update_target()
                running += float(loss.detach())
                step_seconds = time.monotonic() - step_started
                steps_this_process += 1
                average_step_seconds = (
                    time.monotonic() - start_time
                ) / steps_this_process
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.monotonic() - start_time,
                    "step_seconds": step_seconds,
                    "molecule_pairs_per_second": molecules_this_step / step_seconds,
                    "molecules_per_second": molecules_this_step / step_seconds,
                    "spectra_per_second": spectra_this_step / step_seconds,
                    "epoch_progress": (batch_index + 1) / len(loader),
                    "global_progress": (global_step + 1) / total_steps,
                    "eta_hours": max(0, total_steps - global_step - 1)
                    * average_step_seconds
                    / 3600.0,
                    **_device_memory_metrics(device),
                    **metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                if global_step % config.log_every == 0:
                    metrics_file.flush()
                tracker.log_step(record)
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
            if (epoch + 1) in config.snapshot_epochs:
                _atomic_checkpoint(
                    checkpoint, output / f"epoch-{epoch + 1:04d}.pt"
                )
            tracker.log_epoch(
                epoch=epoch,
                global_step=max(0, global_step - 1),
                epoch_loss=epoch_loss,
                best_loss=best_loss,
            )
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
    tracker.finish(
        {
            "status": "complete",
            "best_epoch_loss": best_loss,
            "completed_epochs": config.epochs,
            "completed_steps": global_step,
            "best_checkpoint_sha256": artifacts["best_checkpoint"]["sha256"],
        }
    )
    return best_checkpoint


def load_encoder_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> UniversalSpectrumEncoder:
    payload = torch.load(path, map_location=device, weights_only=False)
    from .model import EncoderConfig

    encoder = UniversalSpectrumEncoder(config=EncoderConfig(**payload["encoder_config"]))
    missing, unexpected = encoder.load_state_dict(payload["online_encoder"], strict=False)
    allowed_missing = {
        "alignment_summary_token",
        "aligned_batch_norm.running_mean",
        "aligned_batch_norm.running_var",
        "aligned_batch_norm.num_batches_tracked",
    }
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"checkpoint state mismatch; missing={missing}, unexpected={unexpected}"
        )
    encoder.to(device)
    encoder.eval()
    return encoder
