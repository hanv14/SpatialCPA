"""
Token embedder for SpatialCPA-v4.

Converts the per-neighbor features produced by
:class:`~spatialcpav4.dataset.TripletTokenDataset` into a sequence of
transformer tokens of width ``hidden_dim``.

Each neighbor token is the sum of its component embeddings (the standard
"token + positional" additive recipe, generalised to several typed
components):

    token = expr_proj(expr_encoder(expr))
          + coord_encoder(relcoord)
          + side_embedding(side)
          [ + cell_type_embedding(ct) ]      # if cell types available
          [ + region_embedding(region) ]     # if regions available

Summing (rather than concatenating) keeps the width fixed and makes it easy to
switch individual components on/off depending on what annotations the dataset
provides.  A final LayerNorm + Dropout stabilises the mixed embedding.

The cell-type / region embeddings reserve index 0 as a padding/unknown slot, so
``n_cell_types + 1`` rows are allocated and indices are shifted by +1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .encoders import RelativeCoordEncoder, build_expression_encoder


class TokenEmbedder(nn.Module):
    """Build ``(B, 2k, hidden_dim)`` neighbor-token embeddings.

    Parameters
    ----------
    n_genes
        Number of genes (input width of the expression encoder).
    n_cell_types
        Number of cell types, or ``None`` to disable the cell-type component.
    n_regions
        Number of regions, or ``None`` to disable the region component.
    cfg
        The :class:`ModelConfig`.
    coord_scale
        Initial coordinate-normalisation scale.
    """

    # Reserved index used for "unknown / padding" labels.
    PAD_LABEL = 0

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int | None,
        n_regions: int | None,
        cfg: ModelConfig,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        h = cfg.hidden_dim
        self.hidden_dim = h
        self.use_cell_type = n_cell_types is not None
        self.use_region = n_regions is not None

        # Expression encoder + projection to hidden width.
        self.expr_encoder = build_expression_encoder(
            cfg.expression_encoder,
            n_genes=n_genes,
            output_dim=cfg.resolved_expression_embed_dim(),
            dropout=cfg.dropout,
        )
        self.expr_proj = (
            nn.Identity()
            if self.expr_encoder.output_dim == h
            else nn.Linear(self.expr_encoder.output_dim, h)
        )

        # Relative-coordinate encoder.
        self.coord_encoder = RelativeCoordEncoder(
            output_dim=h,
            hidden_dim=cfg.coord_hidden_dim,
            dropout=cfg.dropout,
            coord_scale=coord_scale,
        )

        # Slice-side embedding (0 = lower slice, 1 = upper slice).  Lets the
        # model treat the two flanking sections asymmetrically if useful.
        self.side_embedding = nn.Embedding(2, h)

        # Optional label embeddings (index 0 reserved for pad/unknown).
        self.cell_type_embedding = (
            nn.Embedding(n_cell_types + 1, h, padding_idx=self.PAD_LABEL)
            if self.use_cell_type else None
        )
        self.region_embedding = (
            nn.Embedding(n_regions + 1, h, padding_idx=self.PAD_LABEL)
            if self.use_region else None
        )

        self.norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(cfg.dropout)

    def set_coord_scale(self, scale: float) -> None:
        self.coord_encoder.set_coord_scale(scale)

    def forward(
        self,
        token_expr: torch.Tensor,       # (B, T, G)
        token_relcoord: torch.Tensor,   # (B, T, 4)
        token_side: torch.Tensor,       # (B, T) long
        token_ct: torch.Tensor | None = None,   # (B, T) long
        token_reg: torch.Tensor | None = None,  # (B, T) long
    ) -> torch.Tensor:
        """Return ``(B, T, hidden_dim)`` token embeddings."""
        tok = self.expr_proj(self.expr_encoder(token_expr))
        tok = tok + self.coord_encoder(token_relcoord)
        tok = tok + self.side_embedding(token_side)

        if self.use_cell_type and token_ct is not None:
            # Shift by +1 so real labels start at 1 and 0 stays "unknown".
            ct = torch.clamp(token_ct + 1, min=self.PAD_LABEL)
            tok = tok + self.cell_type_embedding(ct)

        if self.use_region and token_reg is not None:
            reg = torch.clamp(token_reg + 1, min=self.PAD_LABEL)
            tok = tok + self.region_embedding(reg)

        return self.dropout(self.norm(tok))
