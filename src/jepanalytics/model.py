"""Coordinate-aware shared Transformer encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .preprocessing import ProcessedSignal, SignalProcessor
from .signal import SpectralSignal


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    n_bins: int = 4096
    patch_size: int = 32
    hidden_dim: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    aligned_dim: int = 256
    continuous_metadata_dim: int = 10
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_bins % self.patch_size:
            raise ValueError("n_bins must be divisible by patch_size")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")

    @property
    def n_patches(self) -> int:
        return self.n_bins // self.patch_size

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(slots=True)
class EmbeddingBundle:
    general: torch.Tensor
    aligned: torch.Tensor
    patches: torch.Tensor


class UniversalSpectrumEncoder(nn.Module):
    """One backbone for IR, NMR, and MS/MS signals."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        cfg = self.config
        self.patch_projection = nn.Linear(cfg.patch_size, cfg.hidden_dim)
        self.learned_position = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.summary_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.axis_embedding = nn.Embedding(3, cfg.hidden_dim)
        self.unit_embedding = nn.Embedding(3, cfg.hidden_dim)
        self.acquisition_embedding = nn.Embedding(5, cfg.hidden_dim)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(4, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.metadata_projection = nn.Sequential(
            nn.Linear(cfg.continuous_metadata_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.heads,
            dim_feedforward=int(cfg.hidden_dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.depth, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)
        self.aligned_projection = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.aligned_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.learned_position, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.summary_token, std=0.02)

    def _coordinate_features(self, metadata: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        u = (torch.arange(cfg.n_patches, device=metadata.device, dtype=metadata.dtype) + 0.5)
        u = u / cfg.n_patches
        u = u.unsqueeze(0).expand(metadata.shape[0], -1)
        start = metadata[:, 0:1]
        end = metadata[:, 1:2]
        physical_center = start + (end - start) * u
        physical_width = (end - start).expand_as(u) / cfg.n_patches
        features = torch.stack((u, u.square(), physical_center, physical_width), dim=-1)
        return self.coordinate_projection(features)

    def forward(
        self,
        intensity: torch.Tensor,
        continuous_metadata: torch.Tensor,
        axis_type: torch.Tensor,
        axis_unit: torch.Tensor,
        acquisition: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> EmbeddingBundle:
        cfg = self.config
        if intensity.ndim != 2 or intensity.shape[1] != cfg.n_bins:
            raise ValueError(f"intensity must have shape [batch, {cfg.n_bins}]")
        patches = intensity.unfold(1, cfg.patch_size, cfg.patch_size)
        tokens = self.patch_projection(patches)
        category = (
            self.axis_embedding(axis_type)
            + self.unit_embedding(axis_unit)
            + self.acquisition_embedding(acquisition)
        )
        tokens = tokens + self.learned_position + self._coordinate_features(continuous_metadata)
        tokens = tokens + category.unsqueeze(1)
        if patch_mask is not None:
            if patch_mask.shape != tokens.shape[:2]:
                raise ValueError("patch_mask must have shape [batch, patches]")
            tokens = torch.where(patch_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        summary = self.summary_token.expand(intensity.shape[0], -1, -1)
        summary = summary + category.unsqueeze(1) + self.metadata_projection(continuous_metadata).unsqueeze(1)
        encoded = self.output_norm(self.transformer(torch.cat((summary, tokens), dim=1)))
        general = encoded[:, 0]
        patch_embeddings = encoded[:, 1:]
        aligned = F.normalize(self.aligned_projection(general), dim=-1)
        return EmbeddingBundle(general=general, aligned=aligned, patches=patch_embeddings)

    @torch.inference_mode()
    def encode(
        self,
        signals: SpectralSignal | ProcessedSignal | Sequence[SpectralSignal | ProcessedSignal],
        *,
        processor: SignalProcessor | None = None,
        device: torch.device | str | None = None,
    ) -> EmbeddingBundle:
        """Encode one or more validated signals into all public representations."""

        if isinstance(signals, (SpectralSignal, ProcessedSignal)):
            signals = [signals]
        processor = processor or SignalProcessor(n_bins=self.config.n_bins)
        processed = [processor(item) if isinstance(item, SpectralSignal) else item for item in signals]
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        batch = {
            "intensity": torch.from_numpy(np.stack([item.intensity for item in processed])).to(target_device),
            "continuous_metadata": torch.from_numpy(
                np.stack([item.continuous_metadata for item in processed])
            ).to(target_device),
            "axis_type": torch.tensor([item.axis_type for item in processed], device=target_device),
            "axis_unit": torch.tensor([item.axis_unit for item in processed], device=target_device),
            "acquisition": torch.tensor([item.acquisition for item in processed], device=target_device),
        }
        was_training = self.training
        self.eval()
        result = self(**batch)
        self.train(was_training)
        return result


class LatentPredictor(nn.Module):
    def __init__(
        self, hidden_dim: int = 384, predictor_dim: int | None = None, depth: int = 3
    ) -> None:
        super().__init__()
        predictor_dim = predictor_dim or min(192, hidden_dim)
        predictor_heads = next(
            heads for heads in (6, 4, 3, 2, 1) if predictor_dim % heads == 0
        )
        self.input = nn.Linear(hidden_dim, predictor_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=predictor_heads,
            dim_feedforward=predictor_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.output = nn.Linear(predictor_dim, hidden_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.output(self.transformer(self.input(patches)))


def batch_to_encoder_kwargs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "intensity": batch["intensity"],
        "continuous_metadata": batch["continuous_metadata"],
        "axis_type": batch["axis_type"],
        "axis_unit": batch["axis_unit"],
        "acquisition": batch["acquisition"],
    }
