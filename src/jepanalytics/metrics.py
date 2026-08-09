"""Dependency-light preregistered metrics and uncertainty estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = float(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="stable")
    ordered = y_true[order]
    precision = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    return float((precision * ordered).sum() / positives)


def macro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_score must have matching [samples, labels] shapes")
    values = [binary_average_precision(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])]
    return float(np.nanmean(values))


def macro_f1(y_true: np.ndarray, y_probability: np.ndarray, threshold: float = 0.5) -> float:
    predicted = y_probability >= threshold
    truth = y_true.astype(bool)
    scores = []
    for label in range(truth.shape[1]):
        tp = np.sum(predicted[:, label] & truth[:, label])
        fp = np.sum(predicted[:, label] & ~truth[:, label])
        fn = np.sum(~predicted[:, label] & truth[:, label])
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 17,
) -> BootstrapInterval:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("values must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [statistic(rng.choice(values, size=values.size, replace=True)) for _ in range(samples)]
    )
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(statistic(values)),
        lower=float(np.quantile(draws, alpha)),
        upper=float(np.quantile(draws, 1.0 - alpha)),
    )

