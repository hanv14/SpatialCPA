"""
Spatial Backbone Network.

8-layer MLP with skip connection at the midpoint that produces a spatial
context vector h(x, y, z) from Fourier-encoded coordinates.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Single residual block: Linear → LayerNorm → GELU → Dropout."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class SpatialBackbone(nn.Module):
    """
    8-layer MLP backbone with skip connection at layer 4.

    Fourier features (input_dim) → project to hidden_dim → 4 layers →
    skip connection (re-inject Fourier features) → 4 layers → output (hidden_dim).

    Parameters
    ----------
    input_dim : int
        Dimensionality of Fourier features.
    hidden_dim : int
        Width of hidden layers (default 512).
    output_dim : int
        Dimensionality of spatial context vector (default 256).
    n_layers : int
        Total number of residual layers (default 8).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, input_dim, hidden_dim=512, output_dim=256,
                 n_layers=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mid = n_layers // 2

        # Project Fourier features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # First half of layers (before skip)
        self.layers_pre = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(self.mid)
        ])

        # Skip connection projection (Fourier features → hidden_dim, added)
        self.skip_proj = nn.Linear(input_dim, hidden_dim)

        # Second half of layers (after skip)
        self.layers_post = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(n_layers - self.mid)
        ])

        # Final projection to output dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, fourier_features):
        """
        Parameters
        ----------
        fourier_features : (N, input_dim) Fourier-encoded coordinates.

        Returns
        -------
        h : (N, output_dim) spatial context vector.
        """
        h = self.input_proj(fourier_features)

        for layer in self.layers_pre:
            h = layer(h)

        # Skip connection: re-inject original Fourier features
        h = h + self.skip_proj(fourier_features)

        for layer in self.layers_post:
            h = layer(h)

        return self.output_proj(h)
