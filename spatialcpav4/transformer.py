"""
Transformer aggregator for SpatialCPA-v4.

A standard Transformer *encoder* consumes the ``2k`` neighbor tokens plus a
learnable ``CLS`` token.  The CLS output is the pooled latent representation of
the target location that the prediction heads consume.

The module is deliberately thin — it wraps ``nn.TransformerEncoder`` with a CLS
token and correct padding-mask handling — so the aggregation strategy is easy to
swap later (e.g. cross-attention against a learned query, a graph transformer,
or Perceiver-style latent bottleneck).  Only :class:`TransformerAggregator`
needs to change for those extensions; the heads and training loop are agnostic
to how the latent is produced.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class TransformerAggregator(nn.Module):
    """Aggregate neighbor tokens into a single latent via a CLS token.

    Parameters
    ----------
    cfg
        The :class:`ModelConfig` (hidden dim, layers, heads, dropout, ffn width).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.hidden_dim
        if h % cfg.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({h}) must be divisible by num_heads ({cfg.num_heads})"
            )

        # Learnable CLS token, broadcast across the batch.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.resolved_dim_feedforward(),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable for small/medium depths
        )
        # enable_nested_tensor is incompatible with norm_first (pre-LN) and only
        # a speed optimisation; disable it explicitly to avoid a runtime warning.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(h)

    def forward(
        self,
        tokens: torch.Tensor,                     # (B, T, H)
        token_pad_mask: torch.Tensor | None = None,  # (B, T) bool, True = ignore
    ) -> torch.Tensor:
        """Return the CLS latent ``(B, H)``."""
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)          # (B, 1, H)
        seq = torch.cat([cls, tokens], dim=1)           # (B, 1+T, H)

        pad_mask = None
        if token_pad_mask is not None:
            # Prepend a "never masked" column for the CLS token.
            cls_col = torch.zeros(b, 1, dtype=torch.bool, device=tokens.device)
            pad_mask = torch.cat([cls_col, token_pad_mask], dim=1)  # (B, 1+T)

        out = self.encoder(seq, src_key_padding_mask=pad_mask)
        return self.norm(out[:, 0])                     # CLS position
