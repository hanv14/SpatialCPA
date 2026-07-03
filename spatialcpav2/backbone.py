"""
Spatial backbone for SpatialCPA v2.

Improvements over v1
--------------------
* Gated residual blocks (GLU-style) instead of plain ``x + MLP(x)`` — the gate
  lets the network modulate how much each block contributes at each location,
  which helps represent sharp regional boundaries without destabilising the
  smooth interior.
* Two skip re-injections of the positional features (at ~1/3 and ~2/3 depth)
  rather than a single mid-network skip, so fine coordinate detail survives all
  the way to the output.
* The trunk exposes both the final context vector ``h`` used by the heads and
  is deep enough (configurable) to model the joint tissue architecture.
"""

import torch
import torch.nn as nn


class GatedResBlock(nn.Module):
    """Residual block with a multiplicative gate:  x + g(x) * f(x)."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 2 * dim)   # -> (value, gate)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        v, g = self.fc(h).chunk(2, dim=-1)
        h = self.act(v) * torch.sigmoid(g)
        h = self.drop(self.proj(h))
        return x + h


class SpatialBackbone(nn.Module):
    """
    Positional features -> spatial-context vector h(x, y, z).

    Parameters
    ----------
    input_dim : int
        Positional-feature dimensionality.
    hidden_dim : int
        Width of hidden layers.
    output_dim : int
        Spatial-context dimensionality.
    n_layers : int
        Number of gated residual blocks.
    dropout : float
        Dropout rate.
    """

    def __init__(self, input_dim, hidden_dim=512, output_dim=256,
                 n_layers=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            GatedResBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])

        # Skip re-injection points (roughly 1/3 and 2/3 depth)
        self.skip_at = {max(1, n_layers // 3), max(2, (2 * n_layers) // 3)}
        self.skip_proj = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(len(self.skip_at))
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, feats):
        h = self.input_proj(feats)
        skip_iter = iter(self.skip_proj)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i in self.skip_at:
                h = h + next(skip_iter)(feats)
        return self.output_proj(h)
