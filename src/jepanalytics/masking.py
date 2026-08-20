"""Mask generation for one-dimensional JEPA pretraining."""

from __future__ import annotations

import torch


def informative_patches(
    patch_intensity: torch.Tensor, *, relative_threshold: float = 1e-3
) -> torch.Tensor:
    """Flag patches that carry signal, relative to each row's strongest patch.

    A centroided mass spectrum occupies well under one percent of its bins, so
    most of its patches are identically empty.  Treating those as maskable
    makes the prediction target a constant, and the objective is then solved by
    emitting "empty patch" for the majority of positions.
    """

    if patch_intensity.ndim != 2:
        raise ValueError("patch_intensity must have shape [batch, patches]")
    magnitude = patch_intensity.abs()
    peak = magnitude.amax(dim=1, keepdim=True)
    return magnitude > peak * relative_threshold


def mixed_patch_mask(
    patch_intensity: torch.Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
    occupancy_aware: bool = True,
    relative_threshold: float = 1e-3,
    minimum_informative: int = 1,
) -> torch.Tensor:
    """Combine contiguous, random, and peak-centered masks in each batch.

    Parameters
    ----------
    patch_intensity:
        Absolute patch intensity with shape ``[batch, patches]``.
    mask_ratio:
        Fraction of patches hidden from the context encoder.
    occupancy_aware:
        Draw the mask from the informative patches of each row instead of from
        all patches.  For dense traces nearly every patch is informative and
        this is close to a no-op; for sparse peak lists it is the difference
        between predicting real content and predicting emptiness.
    minimum_informative:
        Rows with fewer informative patches than this fall back to masking over
        all patches, so a genuinely empty row still yields a usable mask.  A
        predicted MS/MS spectrum often carries only a handful of fragments, so
        this defaults to 1: falling back whenever a row has few peaks would
        restore the degenerate "predict emptiness" task for exactly the records
        that motivated occupancy-aware masking.
    """

    if patch_intensity.ndim != 2:
        raise ValueError("patch_intensity must have shape [batch, patches]")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must lie strictly between zero and one")
    batch, patches = patch_intensity.shape
    default_count = max(1, min(patches - 1, round(mask_ratio * patches)))
    mask = torch.zeros(batch, patches, dtype=torch.bool, device=patch_intensity.device)
    informative = (
        informative_patches(patch_intensity, relative_threshold=relative_threshold)
        if occupancy_aware
        else torch.ones_like(mask)
    )

    for row in range(batch):
        candidates = informative[row].nonzero(as_tuple=True)[0]
        if candidates.numel() < minimum_informative:
            candidates = torch.arange(patches, device=patch_intensity.device)
        available = int(candidates.numel())
        count = (
            default_count
            if available == patches
            else max(1, min(available - 1, round(mask_ratio * available)))
        )
        count = max(1, min(count, available))
        order = candidates.tolist()
        contiguous_count = count // 3
        peak_count = count // 3
        random_count = count - contiguous_count - peak_count
        selected: set[int] = set()

        if contiguous_count:
            # Contiguity is taken over the candidate sequence, so on a sparse
            # spectrum this hides a neighbouring run of peaks rather than a
            # span that is mostly empty bins.
            start = int(
                torch.randint(
                    0,
                    max(1, available - contiguous_count + 1),
                    (1,),
                    generator=generator,
                    device=patch_intensity.device,
                ).item()
            )
            selected.update(order[start : start + contiguous_count])

        if peak_count:
            values = patch_intensity[row][candidates]
            peaks = torch.topk(values, k=min(peak_count, available)).indices.tolist()
            selected.update(order[int(index)] for index in peaks)

        remaining = [index for index in order if index not in selected]
        if random_count and remaining:
            permutation = torch.randperm(
                len(remaining), generator=generator, device=patch_intensity.device
            )
            selected.update(remaining[int(i)] for i in permutation[:random_count].tolist())

        if len(selected) < count:
            remaining = [index for index in order if index not in selected]
            if remaining:
                permutation = torch.randperm(
                    len(remaining), generator=generator, device=patch_intensity.device
                )
                selected.update(
                    remaining[int(i)] for i in permutation[: count - len(selected)].tolist()
                )
        mask[row, list(selected)[:count]] = True
    return mask
