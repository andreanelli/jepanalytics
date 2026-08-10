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


def symmetric_multi_positive_alignment_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    first_molecule: torch.Tensor,
    second_molecule: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric cross-view InfoNCE with every same-molecule record positive.

    With one molecule per batch row this is equivalent to the original paired
    cross-entropy. It remains correct if replicates, collision energies, or a
    distributed sampler place multiple records of one molecule in a batch:
    those records contribute to the numerator instead of becoming false
    negatives.
    """

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired embeddings must have matching [batch, dimension] shapes")
    if first_molecule.shape != (first.shape[0],) or second_molecule.shape != (
        second.shape[0],
    ):
        raise ValueError("molecule identifiers must have shape [batch]")
    logits = F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T / temperature
    positive = first_molecule[:, None] == second_molecule[None, :]
    if not torch.all(positive.any(dim=1)) or not torch.all(positive.any(dim=0)):
        raise ValueError("every alignment anchor must have a same-molecule positive")

    def direction(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = torch.logsumexp(values.masked_fill(~mask, -torch.inf), dim=1)
        denominator = torch.logsumexp(values, dim=1)
        return (denominator - numerator).mean()

    return 0.5 * (direction(logits, positive) + direction(logits.T, positive.T))


def multi_positive_alignment_loss(
    embedding: torch.Tensor,
    molecule: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Multi-view InfoNCE with all other same-molecule views as positives."""

    if embedding.ndim != 2 or molecule.shape != (embedding.shape[0],):
        raise ValueError("embedding and molecule must have shapes [batch, dim] and [batch]")
    count = embedding.shape[0]
    valid = ~torch.eye(count, dtype=torch.bool, device=embedding.device)
    positive = (molecule[:, None] == molecule[None, :]) & valid
    if not torch.all(positive.any(dim=1)):
        raise ValueError("every multi-view anchor must have another same-molecule view")
    logits = F.normalize(embedding, dim=-1) @ F.normalize(embedding, dim=-1).T
    logits = logits / temperature
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    denominator = torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1)
    return (denominator - numerator).mean()


def group_centroid_regularization(
    embedding: torch.Tensor, groups: torch.Tensor
) -> torch.Tensor:
    """Penalize dominant mean directions within every acquisition family."""

    if embedding.ndim != 2 or groups.shape != (embedding.shape[0],):
        raise ValueError("embedding and groups must have shapes [batch, dim] and [batch]")
    normalized = F.normalize(embedding, dim=-1)
    penalties = [
        normalized[groups == group].mean(dim=0).square().sum()
        for group in torch.unique(groups)
    ]
    return torch.stack(penalties).mean()


def modality_centroid_alignment_loss(
    embedding: torch.Tensor, acquisition: torch.Tensor
) -> torch.Tensor:
    """Remove separable acquisition offsets without forcing global collapse."""

    if embedding.ndim != 2 or acquisition.shape != (embedding.shape[0],):
        raise ValueError(
            "embedding and acquisition must have shapes [batch, dim] and [batch]"
        )
    normalized = F.normalize(embedding, dim=-1)
    centroids = torch.stack(
        [
            normalized[acquisition == family].mean(dim=0)
            for family in torch.unique(acquisition)
        ]
    )
    centered = centroids - centroids.mean(dim=0, keepdim=True)
    return centered.square().sum(dim=1).mean()


@torch.no_grad()
def alignment_diagnostics(
    first: torch.Tensor,
    second: torch.Tensor,
    first_molecule: torch.Tensor,
    second_molecule: torch.Tensor,
) -> dict[str, float]:
    """Batch-level signal, anisotropy, and retrieval diagnostics."""

    similarity = F.normalize(first.float(), dim=-1) @ F.normalize(
        second.float(), dim=-1
    ).T
    positive = first_molecule[:, None] == second_molecule[None, :]
    negative = ~positive
    predicted = second_molecule[similarity.argmax(dim=1)]
    combined = F.normalize(torch.cat((first.float(), second.float())), dim=-1)
    positive_mean = similarity[positive].mean()
    negative_mean = (
        similarity[negative].mean()
        if bool(negative.any())
        else similarity.new_zeros(())
    )
    return {
        "alignment_positive_cosine": float(positive_mean),
        "alignment_negative_cosine": float(negative_mean),
        "alignment_cosine_margin": float(positive_mean - negative_mean),
        "alignment_batch_top1": float((predicted == first_molecule).float().mean()),
        "alignment_centroid_norm": float(combined.mean(dim=0).norm()),
        "alignment_positives_per_anchor": float(positive.float().sum(dim=1).mean()),
    }


@torch.no_grad()
def multiview_alignment_diagnostics(
    embedding: torch.Tensor, molecule: torch.Tensor
) -> dict[str, float]:
    """Alignment diagnostics for a joint batch with two or more views."""

    normalized = F.normalize(embedding.float(), dim=-1)
    similarity = normalized @ normalized.T
    valid = ~torch.eye(embedding.shape[0], dtype=torch.bool, device=embedding.device)
    positive = (molecule[:, None] == molecule[None, :]) & valid
    negative = (molecule[:, None] != molecule[None, :]) & valid
    ranking = similarity.masked_fill(~valid, -torch.inf).argmax(dim=1)
    positive_mean = similarity[positive].mean()
    negative_mean = similarity[negative].mean()
    return {
        "alignment_positive_cosine": float(positive_mean),
        "alignment_negative_cosine": float(negative_mean),
        "alignment_cosine_margin": float(positive_mean - negative_mean),
        "alignment_batch_top1": float(
            (molecule[ranking] == molecule).float().mean()
        ),
        "alignment_centroid_norm": float(normalized.mean(dim=0).norm()),
        "alignment_positives_per_anchor": float(positive.float().sum(dim=1).mean()),
    }


@torch.no_grad()
def pairwise_multiview_alignment_diagnostics(
    embedding: torch.Tensor,
    molecule: torch.Tensor,
    acquisition: torch.Tensor,
) -> dict[str, float]:
    """Per-technique-pair margins and retrieval for W&B failure localization."""

    normalized = F.normalize(embedding.float(), dim=-1)
    metrics: dict[str, float] = {}
    families = sorted(int(value) for value in torch.unique(acquisition).tolist())
    for first_index, first_family in enumerate(families):
        first = torch.nonzero(
            acquisition == first_family, as_tuple=False
        ).flatten()
        for second_family in families[first_index + 1 :]:
            second = torch.nonzero(
                acquisition == second_family, as_tuple=False
            ).flatten()
            similarity = normalized[first] @ normalized[second].T
            positive = molecule[first, None] == molecule[None, second]
            negative = ~positive
            prefix = f"alignment_pair_{first_family}_{second_family}"
            positive_mean = similarity[positive].mean()
            negative_mean = similarity[negative].mean()
            forward_match = (
                molecule[second[similarity.argmax(dim=1)]] == molecule[first]
            ).float().mean()
            reverse_match = (
                molecule[first[similarity.argmax(dim=0)]] == molecule[second]
            ).float().mean()
            metrics[f"{prefix}_positive_cosine"] = float(positive_mean)
            metrics[f"{prefix}_cosine_margin"] = float(
                positive_mean - negative_mean
            )
            metrics[f"{prefix}_recall_at_1"] = float(
                0.5 * (forward_match + reverse_match)
            )
    return metrics


def variance_regularization(
    embedding: torch.Tensor, target_std: float = 0.1, epsilon: float = 1e-8
) -> torch.Tensor:
    std = torch.sqrt(embedding.var(dim=0, unbiased=False) + epsilon)
    return F.relu(target_std - std).mean()


def covariance_regularization(
    embedding: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
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
    standardized = F.normalize(centered, dim=0, eps=epsilon)
    correlation = standardized.T @ standardized
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
