"""
Prediction heads for SpatialCPA-v4.

Three heads consume the transformer CLS latent of a target location:

1. :class:`ExpressionHead`  — regresses the full gene-expression vector.
2. :class:`LabelHead`        — classifies cell type and/or region.
3. :class:`OccupancyHead`    — predicts whether the location is tissue (1) or
   background (0).

Each head is a small independent module so heads can be added, removed, or
replaced (e.g. an uncertainty/variance branch, or a variational latent) without
disturbing the others.  Losses live in :mod:`spatialcpav4.losses`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Standard 2-layer GELU MLP with LayerNorm used by the heads."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class ExpressionHead(nn.Module):
    """Regress the full gene-expression vector from the latent."""

    def __init__(self, latent_dim: int, n_genes: int, hidden_dim: int, dropout: float,
                 activation: str = "softplus"):
        super().__init__()
        self.net = _mlp(latent_dim, hidden_dim, n_genes, dropout)
        # Expression is non-negative; a plain linear head can emit negatives
        # (which are unphysical and break downstream log normalization). Softplus
        # keeps the output >= 0 while staying smooth. ``"none"`` restores the
        # original linear head.
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation in (None, "none", "identity"):
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown expression activation '{activation}'")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(h))


class LabelHead(nn.Module):
    """Classify cell type and/or region from the latent.

    Either head is optional; ``forward`` returns ``None`` for a disabled branch.
    """

    def __init__(
        self,
        latent_dim: int,
        n_cell_types: Optional[int],
        n_regions: Optional[int],
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.cell_type_head = (
            _mlp(latent_dim, hidden_dim, n_cell_types, dropout)
            if n_cell_types is not None else None
        )
        self.region_head = (
            _mlp(latent_dim, hidden_dim, n_regions, dropout)
            if n_regions is not None else None
        )

    def forward(self, h: torch.Tensor):
        ct = self.cell_type_head(h) if self.cell_type_head is not None else None
        reg = self.region_head(h) if self.region_head is not None else None
        return ct, reg


class OccupancyHead(nn.Module):
    """Predict a single tissue-vs-background logit from the latent."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = _mlp(latent_dim, hidden_dim, 1, dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)  # (B,)
