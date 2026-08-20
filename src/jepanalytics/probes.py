"""Converged probe fitting shared by every frozen-representation evaluation.

A probe must be optimized to convergence: an under-fitted probe conflates
"the representation lacks the information" with "the optimizer did not
extract it".  The previous full-batch first-order loops took at most a few
hundred gradient steps, which systematically under-reported frozen-encoder
scores relative to the mini-batch-trained supervised baseline.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def standardization_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return mean, std


def positive_weight(y: torch.Tensor, cap: float = 20.0) -> torch.Tensor:
    positives = y.sum(dim=0)
    return ((y.shape[0] - positives) / positives.clamp_min(1.0)).clamp(max=cap)


def fit_linear_probe_converged(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    weight_decay: float,
    max_iterations: int = 500,
    seed: int = 17,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, int]:
    """Fit a multilabel logistic-regression probe to convergence with L-BFGS.

    The problem is convex and small (hundreds to a few thousand rows), so a
    deterministic full-batch CPU fit is faster and more reproducible than
    tuned first-order optimization.  ``weight_decay`` is an explicit L2
    penalty on the weight matrix in the sklearn parameterization
    (per-sample loss plus ``0.5 * weight_decay * ||W||^2 / n``), so early
    stopping is unnecessary and validation data is only needed to select
    ``weight_decay`` itself.
    """

    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(train_x), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(train_y), dtype=torch.float32)
    mean, std = standardization_stats(x)
    x = (x - mean) / std
    model = nn.Linear(x.shape[1], y.shape[1])
    pos_weight = positive_weight(y)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=max_iterations,
        history_size=32,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )
    samples = x.shape[0]

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(x), y, pos_weight=pos_weight
        ) + 0.5 * weight_decay * model.weight.pow(2).sum() / samples
        loss.backward()
        return loss

    optimizer.step(closure)
    state = next(iter(optimizer.state.values()), {})
    return model, mean, std, int(state.get("n_iter", 0))
