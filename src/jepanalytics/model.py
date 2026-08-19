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


TOKENIZER_TYPES = {
    "linear_patch",
    "overlap_conv",
    "multiscale_conv",
    "hybrid_peak_multiscale",
}


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
    tokenizer_type: str = "linear_patch"
    overlap_kernel_size: int = 63
    multiscale_kernel_sizes: tuple[int, ...] = (15, 31, 63)
    hybrid_peak_tokens: int = 32
    peak_window_size: int = 15
    peak_suppression_size: int = 15

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "multiscale_kernel_sizes", tuple(self.multiscale_kernel_sizes)
        )
        if self.n_bins % self.patch_size:
            raise ValueError("n_bins must be divisible by patch_size")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.tokenizer_type not in TOKENIZER_TYPES:
            raise ValueError(
                f"tokenizer_type must be one of {sorted(TOKENIZER_TYPES)}"
            )
        for name, value in (
            ("overlap_kernel_size", self.overlap_kernel_size),
            ("peak_window_size", self.peak_window_size),
            ("peak_suppression_size", self.peak_suppression_size),
        ):
            if value < 3 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer of at least 3")
        if not self.multiscale_kernel_sizes:
            raise ValueError("multiscale_kernel_sizes cannot be empty")
        if any(kernel < 3 or kernel % 2 == 0 for kernel in self.multiscale_kernel_sizes):
            raise ValueError("multiscale kernels must be odd integers of at least 3")
        if (
            self.tokenizer_type in {"multiscale_conv", "hybrid_peak_multiscale"}
            and self.hidden_dim % len(self.multiscale_kernel_sizes)
        ):
            raise ValueError(
                "hidden_dim must be divisible by the number of multiscale kernels"
            )
        if (
            self.tokenizer_type == "hybrid_peak_multiscale"
            and not 1 <= self.hybrid_peak_tokens < self.n_patches
        ):
            raise ValueError("hybrid_peak_tokens must be smaller than the token count")

    @property
    def n_patches(self) -> int:
        return self.n_bins // self.patch_size

    @property
    def dense_token_count(self) -> int:
        if self.tokenizer_type == "hybrid_peak_multiscale":
            return self.n_patches - self.hybrid_peak_tokens
        return self.n_patches

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(slots=True)
class EmbeddingBundle:
    general: torch.Tensor
    aligned: torch.Tensor
    patches: torch.Tensor
    alignment_logits: torch.Tensor


@dataclass(slots=True)
class TokenizedSignal:
    tokens: torch.Tensor
    normalized_coordinate: torch.Tensor
    normalized_width: torch.Tensor
    token_type: torch.Tensor


class UniversalSpectrumEncoder(nn.Module):
    """One backbone for IR, NMR, and MS/MS signals."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        cfg = self.config
        if cfg.tokenizer_type == "linear_patch":
            self.patch_projection = nn.Linear(cfg.patch_size, cfg.hidden_dim)
        elif cfg.tokenizer_type == "overlap_conv":
            self.overlap_projection = nn.Conv1d(
                1,
                cfg.hidden_dim,
                kernel_size=cfg.overlap_kernel_size,
                stride=cfg.patch_size,
                padding=cfg.overlap_kernel_size // 2,
            )
        else:
            branch_dim = cfg.hidden_dim // len(cfg.multiscale_kernel_sizes)
            self.multiscale_projections = nn.ModuleList(
                nn.Conv1d(
                    1,
                    branch_dim,
                    kernel_size=kernel,
                    stride=cfg.patch_size,
                    padding=kernel // 2,
                )
                for kernel in cfg.multiscale_kernel_sizes
            )
            if cfg.tokenizer_type == "hybrid_peak_multiscale":
                self.peak_projection = nn.Linear(cfg.peak_window_size, cfg.hidden_dim)
                self.peak_metadata_projection = nn.Sequential(
                    nn.Linear(4, cfg.hidden_dim),
                    nn.GELU(),
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                )
                self.token_type_embedding = nn.Embedding(2, cfg.hidden_dim)
        self.learned_position = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.summary_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.alignment_summary_token = nn.Parameter(
            torch.zeros(1, 1, cfg.hidden_dim)
        )
        content_attention_mask = torch.zeros(cfg.n_patches + 2, cfg.n_patches + 2)
        content_attention_mask[1:, 0] = -torch.inf
        self.register_buffer(
            "content_attention_mask", content_attention_mask, persistent=False
        )
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
        self.aligned_batch_norm = nn.BatchNorm1d(
            cfg.aligned_dim, affine=False, momentum=0.05
        )
        # Parameter-free normalization keeps old checkpoints load-compatible
        # while removing the easiest per-sample common-direction shortcut.
        self.aligned_norm = nn.LayerNorm(cfg.aligned_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.learned_position, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.summary_token, std=0.02)
        nn.init.trunc_normal_(self.alignment_summary_token, std=0.02)

    def _coordinate_features(
        self,
        metadata: torch.Tensor,
        u: torch.Tensor,
        normalized_width: torch.Tensor,
    ) -> torch.Tensor:
        start = metadata[:, 0:1]
        end = metadata[:, 1:2]
        physical_center = start + (end - start) * u
        physical_width = (end - start) * normalized_width
        features = torch.stack((u, u.square(), physical_center, physical_width), dim=-1)
        return self.coordinate_projection(features)

    def _uniform_coordinates(
        self,
        batch_size: int,
        token_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinate = (
            torch.arange(token_count, device=device, dtype=dtype) + 0.5
        ) / token_count
        coordinate = coordinate.unsqueeze(0).expand(batch_size, -1)
        width = torch.full_like(coordinate, 1.0 / token_count)
        return coordinate, width

    def _multiscale_tokens(self, intensity: torch.Tensor) -> torch.Tensor:
        branches = [
            F.gelu(projection(intensity.unsqueeze(1)))
            for projection in self.multiscale_projections
        ]
        return torch.cat(branches, dim=1).transpose(1, 2)

    def _peak_tokens(
        self, intensity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        signal = intensity.unsqueeze(1)
        baseline = F.avg_pool1d(
            signal, kernel_size=31, stride=1, padding=15
        ).squeeze(1)
        prominence = F.relu(intensity - baseline)
        local_max = F.max_pool1d(
            prominence.unsqueeze(1),
            kernel_size=cfg.peak_suppression_size,
            stride=1,
            padding=cfg.peak_suppression_size // 2,
        ).squeeze(1)
        peak_scores = torch.where(
            prominence >= local_max - torch.finfo(prominence.dtype).eps,
            prominence,
            torch.zeros_like(prominence),
        )
        indices = peak_scores.topk(cfg.hybrid_peak_tokens, dim=1).indices
        indices = indices.sort(dim=1).values
        padded = F.pad(
            intensity.unsqueeze(1),
            (cfg.peak_window_size // 2, cfg.peak_window_size // 2),
            mode="replicate",
        )
        windows = padded.unfold(-1, cfg.peak_window_size, 1).squeeze(1)
        windows = windows.gather(
            1,
            indices.unsqueeze(-1).expand(-1, -1, cfg.peak_window_size),
        )
        selected_intensity = intensity.gather(1, indices)
        selected_prominence = prominence.gather(1, indices)
        peak_metadata = torch.stack(
            (
                selected_intensity,
                selected_prominence,
                windows.mean(dim=-1),
                windows.std(dim=-1, unbiased=False),
            ),
            dim=-1,
        )
        tokens = self.peak_projection(windows) + self.peak_metadata_projection(
            peak_metadata
        )
        coordinate = (indices.to(dtype=intensity.dtype) + 0.5) / cfg.n_bins
        width = torch.full_like(coordinate, cfg.peak_window_size / cfg.n_bins)
        return tokens, coordinate, width

    def _mask_raw_input(
        self, intensity: torch.Tensor, patch_mask: torch.Tensor | None
    ) -> torch.Tensor:
        cfg = self.config
        if patch_mask is None or cfg.tokenizer_type == "linear_patch":
            return intensity
        dense_mask = patch_mask[:, : cfg.dense_token_count]
        bin_mask = F.interpolate(
            dense_mask.to(dtype=intensity.dtype).unsqueeze(1),
            size=cfg.n_bins,
            mode="nearest",
        ).squeeze(1).bool()
        return intensity.masked_fill(bin_mask, 0.0)

    def tokenize(
        self, intensity: torch.Tensor, patch_mask: torch.Tensor | None = None
    ) -> TokenizedSignal:
        """Create the fixed token budget while retaining physical coordinates."""

        cfg = self.config
        masked_intensity = self._mask_raw_input(intensity, patch_mask)
        if cfg.tokenizer_type == "linear_patch":
            patches = masked_intensity.unfold(1, cfg.patch_size, cfg.patch_size)
            tokens = self.patch_projection(patches)
            coordinate, width = self._uniform_coordinates(
                intensity.shape[0],
                cfg.n_patches,
                device=intensity.device,
                dtype=intensity.dtype,
            )
            token_type = torch.zeros_like(coordinate, dtype=torch.long)
        elif cfg.tokenizer_type == "overlap_conv":
            tokens = F.gelu(
                self.overlap_projection(masked_intensity.unsqueeze(1))
            ).transpose(1, 2)
            coordinate, width = self._uniform_coordinates(
                intensity.shape[0],
                cfg.n_patches,
                device=intensity.device,
                dtype=intensity.dtype,
            )
            token_type = torch.zeros_like(coordinate, dtype=torch.long)
        else:
            dense = self._multiscale_tokens(masked_intensity)
            dense_count = cfg.dense_token_count
            if dense.shape[1] != dense_count:
                # MPS does not implement adaptive average pooling for every
                # non-divisible input/output pair (notably 128 -> 96).  Linear
                # resampling is differentiable, coordinate ordered, and keeps
                # the hybrid's fixed token budget on Apple accelerators.
                dense = F.interpolate(
                    dense.transpose(1, 2),
                    size=dense_count,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            dense_coordinate, dense_width = self._uniform_coordinates(
                intensity.shape[0],
                dense_count,
                device=intensity.device,
                dtype=intensity.dtype,
            )
            if cfg.tokenizer_type == "multiscale_conv":
                tokens = dense
                coordinate = dense_coordinate
                width = dense_width
                token_type = torch.zeros_like(coordinate, dtype=torch.long)
            else:
                peak_tokens, peak_coordinate, peak_width = self._peak_tokens(
                    masked_intensity
                )
                tokens = torch.cat((dense, peak_tokens), dim=1)
                coordinate = torch.cat((dense_coordinate, peak_coordinate), dim=1)
                width = torch.cat((dense_width, peak_width), dim=1)
                token_type = torch.cat(
                    (
                        torch.zeros_like(dense_coordinate, dtype=torch.long),
                        torch.ones_like(peak_coordinate, dtype=torch.long),
                    ),
                    dim=1,
                )
                tokens = tokens + self.token_type_embedding(token_type)
        if tokens.shape[1] != cfg.n_patches:
            raise RuntimeError(
                f"tokenizer produced {tokens.shape[1]} tokens; expected {cfg.n_patches}"
            )
        return TokenizedSignal(tokens, coordinate, width, token_type)

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
        if patch_mask is not None and patch_mask.shape != (
            intensity.shape[0],
            cfg.n_patches,
        ):
            raise ValueError("patch_mask must have shape [batch, patches]")
        tokenized = self.tokenize(intensity, patch_mask=patch_mask)
        tokens = tokenized.tokens
        # Replace signal content first, then add index and physical-coordinate
        # features.  The previous ordering erased every masked token's location,
        # making distinct masked queries permutation-equivalent.
        if patch_mask is not None:
            tokens = torch.where(
                patch_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens
            )
        category = (
            self.axis_embedding(axis_type)
            + self.unit_embedding(axis_unit)
            + self.acquisition_embedding(acquisition)
        )
        tokens = (
            tokens
            + self.learned_position
            + self._coordinate_features(
                continuous_metadata,
                tokenized.normalized_coordinate,
                tokenized.normalized_width,
            )
        )
        summary = self.summary_token.expand(intensity.shape[0], -1, -1)
        summary = summary + category.unsqueeze(1) + self.metadata_projection(continuous_metadata).unsqueeze(1)
        alignment_summary = self.alignment_summary_token.expand(
            intensity.shape[0], -1, -1
        )
        sequence = torch.cat((summary, alignment_summary, tokens), dim=1)
        # The general token may consume all content and metadata. Content and
        # alignment tokens cannot attend back to the metadata-rich general
        # token, preventing acquisition identity from becoming the easiest
        # aligned representation while retaining one shared backbone.
        encoded = self.output_norm(
            self.transformer(
                sequence, mask=self.content_attention_mask.to(dtype=sequence.dtype)
            )
        )
        general = encoded[:, 0]
        alignment_content = encoded[:, 1]
        patch_embeddings = encoded[:, 2:]
        alignment_logits = self.aligned_projection(alignment_content)
        aligned = self.normalize_alignment_logits(
            alignment_logits, update_stats=False
        )
        return EmbeddingBundle(
            general=general,
            aligned=aligned,
            patches=patch_embeddings,
            alignment_logits=alignment_logits,
        )

    def normalize_alignment_logits(
        self, alignment_logits: torch.Tensor, *, update_stats: bool
    ) -> torch.Tensor:
        """Center a balanced joint-view batch, or use its running statistics."""

        alignment_logits = self.aligned_norm(alignment_logits)
        if self.training and update_stats:
            normalized = self.aligned_batch_norm(alignment_logits)
        else:
            normalized = F.batch_norm(
                alignment_logits,
                self.aligned_batch_norm.running_mean,
                self.aligned_batch_norm.running_var,
                training=False,
                momentum=0.0,
                eps=self.aligned_batch_norm.eps,
            )
        return F.normalize(normalized, dim=-1)

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


class SummaryLatentPredictor(nn.Module):
    """Predict the full-view EMA summary from a masked context summary."""

    def __init__(self, hidden_dim: int = 384) -> None:
        super().__init__()
        bottleneck = min(256, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_dim),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.network(summary)


class ChemistryPredictor(nn.Module):
    """Auxiliary weak-supervision head; not part of the public encoder API."""

    def __init__(self, hidden_dim: int, target_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, min(256, hidden_dim)),
            nn.GELU(),
            nn.Linear(min(256, hidden_dim), target_dim),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.network(summary)


def batch_to_encoder_kwargs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "intensity": batch["intensity"],
        "continuous_metadata": batch["continuous_metadata"],
        "axis_type": batch["axis_type"],
        "axis_unit": batch["axis_unit"],
        "acquisition": batch["acquisition"],
    }
