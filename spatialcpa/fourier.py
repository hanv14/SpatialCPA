"""
Fourier Feature Encoding for 3D spatial coordinates.

Transforms raw (x, y, z) coordinates into high-dimensional sine/cosine
representations with separate frequency bands for xy vs z axes.
"""

import math
import torch
import torch.nn as nn
import numpy as np


class FourierFeatureEncoder(nn.Module):
    """
    Adaptive Fourier feature encoder with separate frequency bands for
    within-slice (x, y) and between-slice (z) dimensions.

    Parameters
    ----------
    n_freq_xy : int
        Number of frequency components for x and y dimensions.
    n_freq_z : int
        Number of frequency components for z dimension.
    xy_scale : float
        Characteristic spatial scale for xy (typical cell-to-cell distance in µm).
    z_scale : float
        Characteristic spatial scale for z (typical section spacing in µm).
    """

    def __init__(self, n_freq_xy=48, n_freq_z=32, xy_scale=10.0, z_scale=100.0):
        super().__init__()
        self.n_freq_xy = n_freq_xy
        self.n_freq_z = n_freq_z
        self.xy_scale = xy_scale
        self.z_scale = z_scale

        # Output dim: 2 * (2*n_freq_xy + n_freq_z) for sin+cos
        # xy: n_freq_xy freqs * 2 dims (x,y) = 2*n_freq_xy
        # z:  n_freq_z freqs * 1 dim (z)  = n_freq_z
        # total fourier = 2*(2*n_freq_xy + n_freq_z) for sin+cos
        self.output_dim = 2 * (2 * n_freq_xy + n_freq_z)

        # Create log-spaced frequency bands
        # xy: higher frequencies for fine cellular detail
        freqs_xy = 2.0 ** torch.linspace(0, math.log2(1.0 / xy_scale * 100),
                                          n_freq_xy)
        # z: lower frequencies for inter-section variation
        freqs_z = 2.0 ** torch.linspace(0, math.log2(1.0 / z_scale * 100),
                                         n_freq_z)

        self.register_buffer('freqs_xy', freqs_xy)
        self.register_buffer('freqs_z', freqs_z)

    def forward(self, coords):
        """
        Parameters
        ----------
        coords : (N, 3) tensor of (x, y, z) coordinates.

        Returns
        -------
        features : (N, output_dim) tensor of Fourier features.
        """
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]

        # Scale coordinates
        x_scaled = x / self.xy_scale
        y_scaled = y / self.xy_scale
        z_scaled = z / self.z_scale

        # Project onto frequency bands: (N, n_freq)
        proj_x = x_scaled * self.freqs_xy.unsqueeze(0)  # (N, n_freq_xy)
        proj_y = y_scaled * self.freqs_xy.unsqueeze(0)  # (N, n_freq_xy)
        proj_z = z_scaled * self.freqs_z.unsqueeze(0)   # (N, n_freq_z)

        # Concatenate all projections
        proj_all = torch.cat([proj_x, proj_y, proj_z], dim=1)  # (N, 2*n_freq_xy + n_freq_z)

        # Apply sin and cos
        features = torch.cat([torch.sin(2 * math.pi * proj_all),
                               torch.cos(2 * math.pi * proj_all)], dim=1)

        return features

    @staticmethod
    def estimate_scales(coords_np):
        """
        Estimate xy_scale and z_scale from data coordinates.

        Parameters
        ----------
        coords_np : (N, 3) numpy array of (x, y, z).

        Returns
        -------
        xy_scale, z_scale : float
        """
        xy = coords_np[:, :2]
        z_vals = np.unique(coords_np[:, 2])

        # xy_scale: median nearest-neighbor distance in 2D
        from scipy.spatial import cKDTree
        if len(xy) > 10000:
            idx = np.random.choice(len(xy), 10000, replace=False)
            xy_sub = xy[idx]
        else:
            xy_sub = xy
        tree = cKDTree(xy_sub)
        dists, _ = tree.query(xy_sub, k=2)
        xy_scale = float(np.median(dists[:, 1]))

        # z_scale: median gap between consecutive z-values
        if len(z_vals) > 1:
            z_sorted = np.sort(z_vals)
            z_gaps = np.diff(z_sorted)
            z_scale = float(np.median(z_gaps))
        else:
            z_scale = 1.0

        return max(xy_scale, 0.1), max(z_scale, 0.1)
