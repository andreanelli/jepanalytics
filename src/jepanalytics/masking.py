"""Mask generation for one-dimensional JEPA pretraining."""

from __future__ import annotations

import torch


def mixed_patch_mask(
    patch_intensity: torch.Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Combine contiguous, random, and peak-centered masks in each batch.

    Parameters
    ----------
    patch_intensity:
        Absolute patch intensity with shape ``[batch, patches]``.
    mask_ratio:
        Fraction of patches hidden from the context encoder.
    """

    if patch_intensity.ndim != 2:
        raise ValueError("patch_intensity must have shape [batch, patches]")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must lie strictly between zero and one")
    batch, patches = patch_intensity.shape
    count = max(1, min(patches - 1, round(mask_ratio * patches)))
    mask = torch.zeros(batch, patches, dtype=torch.bool, device=patch_intensity.device)
    contiguous_count = count // 3
    peak_count = count // 3
    random_count = count - contiguous_count - peak_count

    for row in range(batch):
        selected: set[int] = set()
        if contiguous_count:
            start = int(
                torch.randint(
                    0,
                    max(1, patches - contiguous_count + 1),
                    (1,),
                    generator=generator,
                    device=patch_intensity.device,
                ).item()
            )
            selected.update(range(start, start + contiguous_count))

        if peak_count:
            peaks = torch.topk(patch_intensity[row], k=min(peak_count, patches)).indices.tolist()
            selected.update(int(index) for index in peaks)

        remaining = [index for index in range(patches) if index not in selected]
        if random_count and remaining:
            permutation = torch.randperm(
                len(remaining), generator=generator, device=patch_intensity.device
            )
            selected.update(remaining[int(i)] for i in permutation[:random_count].tolist())

        if len(selected) < count:
            remaining = [index for index in range(patches) if index not in selected]
            permutation = torch.randperm(
                len(remaining), generator=generator, device=patch_intensity.device
            )
            selected.update(remaining[int(i)] for i in permutation[: count - len(selected)].tolist())
        mask[row, list(selected)[:count]] = True
    return mask

