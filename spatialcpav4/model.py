"""
SpatialCPA-v4 — the assembled transformer model.

Pipeline for one target location::

    neighbor features ─► TokenEmbedder ─► [CLS] + tokens ─► Transformer ─► CLS latent
                                                                             │
                          ┌──────────────────────────────┬──────────────────┤
                          ▼                              ▼                    ▼
                    ExpressionHead                  LabelHead           OccupancyHead
                   (gene expression)          (cell type / region)     (tissue prob)

The model directly predicts the middle slice from its two neighboring slices,
in contrast to the coordinate-neural-field original SpatialCPA.

Everything about widths / depth / heads / dropout / which encoders and heads are
active comes from :class:`~spatialcpav4.config.ModelConfig` and the
``n_cell_types`` / ``n_regions`` availability flags, so the architecture is fully
described by config.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .heads import DensityHead, ExpressionHead, LabelHead, OccupancyHead
from .tokens import TokenEmbedder
from .transformer import TransformerAggregator


class SpatialCPATransformer(nn.Module):
    """The ``{Slice(i-1), Slice(i+1)} -> Slice(i)`` transformer.

    Parameters
    ----------
    n_genes
        Number of genes.
    n_cell_types
        Number of cell types, or ``None`` to disable cell-type embedding/head.
    n_regions
        Number of regions, or ``None`` to disable region embedding/head.
    cfg
        The :class:`ModelConfig`.
    coord_scale
        Initial coordinate-normalisation scale (updated from data before
        training via :meth:`set_coord_scale`).
    """

    def __init__(
        self,
        n_genes: int,
        n_cell_types: Optional[int],
        n_regions: Optional[int],
        cfg: ModelConfig,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_regions = n_regions
        self.use_cell_type = n_cell_types is not None
        self.use_region = n_regions is not None

        self.token_embedder = TokenEmbedder(
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            n_regions=n_regions,
            cfg=cfg,
            coord_scale=coord_scale,
        )
        self.aggregator = TransformerAggregator(cfg)

        h = cfg.hidden_dim
        self.expression_head = ExpressionHead(
            latent_dim=h,
            n_genes=n_genes,
            hidden_dim=cfg.resolved_expression_head_hidden_dim(),
            dropout=cfg.dropout,
            activation=cfg.expression_activation,
        )
        self.label_head = LabelHead(
            latent_dim=h,
            n_cell_types=n_cell_types,
            n_regions=n_regions,
            hidden_dim=cfg.label_head_hidden_dim,
            dropout=cfg.dropout,
        )
        self.occupancy_head = OccupancyHead(
            latent_dim=h,
            hidden_dim=cfg.label_head_hidden_dim,
            dropout=cfg.dropout,
        )
        self.density_head = DensityHead(
            latent_dim=h,
            hidden_dim=cfg.label_head_hidden_dim,
            dropout=cfg.dropout,
        )

    # ---- helpers ---------------------------------------------------------- #
    def set_coord_scale(self, scale: float) -> None:
        """Set the relative-coordinate normalisation scale (call before train)."""
        self.token_embedder.set_coord_scale(scale)

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the CLS latent ``(B, hidden_dim)`` for a batch."""
        tokens = self.token_embedder(
            token_expr=batch["token_expr"],
            token_relcoord=batch["token_relcoord"],
            token_side=batch["token_side"],
            token_ct=batch.get("token_ct") if self.use_cell_type else None,
            token_reg=batch.get("token_reg") if self.use_region else None,
        )
        return self.aggregator(tokens, batch.get("token_pad_mask"))

    # ---- forward ---------------------------------------------------------- #
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run all heads.

        Returns a dict with ``latent``, ``expression``, ``occupancy_logit`` and
        (when enabled) ``cell_type_logits`` / ``region_logits``.
        """
        latent = self.encode(batch)
        ct_logits, reg_logits = self.label_head(latent)

        out: Dict[str, torch.Tensor] = {
            "latent": latent,
            "expression": self.expression_head(latent),
            "occupancy_logit": self.occupancy_head(latent),
            "density": self.density_head(latent),  # predicts log1p(intensity)
        }
        if ct_logits is not None:
            out["cell_type_logits"] = ct_logits
        if reg_logits is not None:
            out["region_logits"] = reg_logits
        return out
