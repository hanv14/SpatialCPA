"""
Positional encoding for SpatialCPA v2.

Improvements over v1
--------------------
1. Physically-calibrated, multi-octave axis-aligned Fourier bands whose
   wavelengths span from the full tissue extent down to ~2x the sampling
   spacing (Nyquist). v1 used an ad-hoc ``2**linspace(0, log2(100/scale))``
   schedule that was only loosely tied to the data geometry.
2. An additional bank of anisotropic **Gaussian random Fourier features**
   over the joint (x, y, z) coordinate. Axis-aligned bands can only represent
   patterns that factorize across axes; the random bank captures oblique /
   cross-axis structure (diagonal laminae, tilted boundaries), which the
   coordinate MLP otherwise struggles to fit. This directly helps the
   spatial-structure metrics (SSIM, Moran's I).

Both banks share the same ``estimate_scales`` calibration so the encoder is
data-adaptive without any hand tuning.
"""

import math
import numpy as np
import torch
import torch.nn as nn


class FourierFeatureEncoder(nn.Module):
    """
    Calibrated multi-scale Fourier encoder with separate xy / z bands plus an
    anisotropic Gaussian random-feature bank.

    Parameters
    ----------
    n_freq_xy : int
        Axis-aligned frequency octaves for x and y.
    n_freq_z : int
        Axis-aligned frequency octaves for z.
    xy_scale : float
        Characteristic xy spacing (median cell-to-cell distance, µm).
    z_scale : float
        Characteristic z spacing (median inter-section gap, µm).
    n_rff : int
        Number of Gaussian random Fourier features over joint (x, y, z).
        Set 0 to disable.
    rff_sigma_xy, rff_sigma_z : float
        Bandwidths (in cycles per ``scale`` unit) for the random bank.
    xy_extent, z_extent : float or None
        Full spatial extent along xy / z. Used to set the lowest frequency so
        the largest wavelength covers the whole tissue. If None, defaults to
        128 * scale.
    seed : int
        RNG seed for the (fixed, non-trainable) random bank.
    """

    def __init__(self, n_freq_xy=48, n_freq_z=32, xy_scale=10.0, z_scale=100.0,
                 n_rff=96, rff_sigma_xy=1.0, rff_sigma_z=0.5,
                 xy_extent=None, z_extent=None, seed=0):
        super().__init__()
        self.n_freq_xy = n_freq_xy
        self.n_freq_z = n_freq_z
        self.xy_scale = float(xy_scale)
        self.z_scale = float(z_scale)
        self.n_rff = n_rff

        if xy_extent is None:
            xy_extent = 128.0 * self.xy_scale
        if z_extent is None:
            z_extent = 128.0 * self.z_scale

        # ── Axis-aligned bands ────────────────────────────────────────────────
        # Wavelengths from ~2*scale (Nyquist, finest resolvable) up to the full
        # extent. Frequencies (cycles per unit) are the reciprocal wavelengths,
        # geometrically spaced so each octave is represented once.
        def _band(scale, extent, n):
            lam_min = 2.0 * scale                 # finest resolvable wavelength
            lam_max = max(extent, 4.0 * scale)    # coarsest (full extent)
            lams = torch.logspace(math.log10(lam_min), math.log10(lam_max), n)
            return (1.0 / lams)                   # cycles per physical unit

        freqs_xy = _band(self.xy_scale, xy_extent, n_freq_xy)
        freqs_z = _band(self.z_scale, z_extent, n_freq_z)
        self.register_buffer("freqs_xy", freqs_xy)
        self.register_buffer("freqs_z", freqs_z)

        # ── Anisotropic Gaussian random Fourier bank ─────────────────────────
        # B ~ N(0, diag(sigma_xy, sigma_xy, sigma_z)) in cycles-per-scale units.
        if n_rff > 0:
            g = torch.Generator().manual_seed(seed)
            B = torch.randn(3, n_rff, generator=g)
            B[0] *= rff_sigma_xy / self.xy_scale
            B[1] *= rff_sigma_xy / self.xy_scale
            B[2] *= rff_sigma_z / self.z_scale
            self.register_buffer("rff_B", B)          # (3, n_rff)
        else:
            self.register_buffer("rff_B", torch.zeros(3, 0))

        # sin+cos over: xy bands (2 axes), z band (1 axis), rff bank.
        self.output_dim = 2 * (2 * n_freq_xy + n_freq_z + n_rff)

    def forward(self, coords):
        """coords: (N, 3) → (N, output_dim)."""
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]

        proj_x = x * self.freqs_xy.unsqueeze(0)   # (N, n_freq_xy)
        proj_y = y * self.freqs_xy.unsqueeze(0)
        proj_z = z * self.freqs_z.unsqueeze(0)    # (N, n_freq_z)

        parts = [proj_x, proj_y, proj_z]
        if self.rff_B.shape[1] > 0:
            proj_r = coords @ self.rff_B          # (N, n_rff)
            parts.append(proj_r)

        proj_all = torch.cat(parts, dim=1)
        ang = 2.0 * math.pi * proj_all
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    @staticmethod
    def estimate_scales(coords_np):
        """Estimate (xy_scale, z_scale) from data coordinates."""
        xy = coords_np[:, :2]
        z_vals = np.unique(coords_np[:, 2])

        from scipy.spatial import cKDTree
        if len(xy) > 10000:
            idx = np.random.choice(len(xy), 10000, replace=False)
            xy_sub = xy[idx]
        else:
            xy_sub = xy
        tree = cKDTree(xy_sub)
        dists, _ = tree.query(xy_sub, k=2)
        xy_scale = float(np.median(dists[:, 1]))

        if len(z_vals) > 1:
            z_sorted = np.sort(z_vals)
            z_gaps = np.diff(z_sorted)
            z_scale = float(np.median(z_gaps))
        else:
            z_scale = 1.0

        return max(xy_scale, 0.1), max(z_scale, 0.1)

    @staticmethod
    def estimate_extent(coords_np):
        """Estimate (xy_extent, z_extent) spans from data coordinates."""
        xy = coords_np[:, :2]
        xy_extent = float(np.median(xy.max(axis=0) - xy.min(axis=0)))
        z = coords_np[:, 2]
        z_extent = float(z.max() - z.min())
        return max(xy_extent, 1.0), max(z_extent, 1.0)
