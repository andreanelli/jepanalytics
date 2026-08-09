"""Pretraining objectives and collapse diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def masked_latent_loss(
    predicted: torch.Tensor, target: torch.Tensor, patch_mask: torch.Tensor
) -> torch.Tensor:
    if predicted.shape != target.shape or patch_mask.shape != predicted.shape[:2]:
        raise ValueError("predicted, target, and patch_mask shapes are incompatible")
    predicted = F.normalize(predicted[patch_mask], dim=-1)
    target = F.normalize(target.detach()[patch_mask], dim=-1)
    return 2.0 - 2.0 * (predicted * target).sum(dim=-1).mean()


def symmetric_alignment_loss(
    first: torch.Tensor, second: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired embeddings must have matching [batch, dimension] shapes")
    logits = F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def variance_regularization(embedding: torch.Tensor, target_std: float = 0.1) -> torch.Tensor:
    std = torch.sqrt(embedding.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(target_std - std).mean()


@dataclass(frozen=True, slots=True)
class EmbeddingDiagnostics:
    mean_feature_std: float
    effective_rank: float
    collapsed: bool


@torch.no_grad()
def embedding_diagnostics(
    embedding: torch.Tensor,
    *,
    min_std: float = 0.02,
    min_effective_rank: float = 8.0,
) -> EmbeddingDiagnostics:
    centered = embedding.float() - embedding.float().mean(dim=0, keepdim=True)
    mean_std = float(centered.std(dim=0, unbiased=False).mean().item())
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    rank = float(torch.exp(entropy).item())
    return EmbeddingDiagnostics(mean_std, rank, mean_std < min_std or rank < min_effective_rank)

