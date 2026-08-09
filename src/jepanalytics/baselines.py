"""Controlled local baselines and optional external encoder adapters."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch
from torch import nn


class SupervisedSpectrumCNN(nn.Module):
    """Compact supervised 1-D CNN used in the published benchmark comparison."""

    def __init__(self, n_labels: int, channels: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(channels, channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(channels * 2, n_labels)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(intensity.unsqueeze(1)).squeeze(-1))


@dataclass(frozen=True, slots=True)
class ExternalBaselineStatus:
    name: str
    available: bool
    package: str
    purpose: str


def external_baseline_status() -> list[ExternalBaselineStatus]:
    """Report optional integrations without silently substituting another model."""

    return [
        ExternalBaselineStatus(
            name="MOMENT",
            available=importlib.util.find_spec("momentfm") is not None,
            package="momentfm",
            purpose="Open time-series foundation-model embedding baseline",
        ),
        ExternalBaselineStatus(
            name="TS2Vec",
            available=importlib.util.find_spec("ts2vec") is not None,
            package="ts2vec",
            purpose="Universal contrastive time-series representation baseline",
        ),
    ]

