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


def covariance_regularization(embedding: torch.Tensor) -> torch.Tensor:
    """Penalize redundant feature directions without suppressing their variance.

    A variance floor prevents constant features, but it cannot detect a matrix in
    which every feature is a scaled copy of the same latent variable.  The mean
    squared off-diagonal correlation is close to one for that failure mode and
    close to ``1 / batch_size`` for independent features.  Normalizing before the
    correlation keeps this term well-scaled across acquisition families.
    """

    if embedding.ndim != 2 or embedding.shape[0] < 2:
        raise ValueError("embedding must have shape [batch >= 2, features]")
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    standardized = centered / torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    correlation = standardized.T @ standardized / embedding.shape[0]
    off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
    n_features = embedding.shape[1]
    return off_diagonal.square().sum() / max(1, n_features * (n_features - 1))


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
    # MPS does not implement SVD. Collapse monitoring is non-differentiable and
    # the matrix is only [2 * batch, hidden], so compute this diagnostic on CPU
    # without moving any model activations used by the training graph.
    diagnostic_matrix = centered.cpu() if centered.device.type == "mps" else centered
    singular = torch.linalg.svdvals(diagnostic_matrix)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    rank = float(torch.exp(entropy).item())
    return EmbeddingDiagnostics(mean_std, rank, mean_std < min_std or rank < min_effective_rank)
