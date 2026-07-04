"""
SpatialCPA v3 — Generative Virtual Slice Construction.

A self-contained evolution of the SpatialCPA package (the original lives in the
sibling ``spatialcpa/`` package and is left untouched). v3 keeps the continuous
3D field h(x, y, z) → (cell type, region, expression) but adds:

  * a generative Gaussian expression head (``expression_mode='gaussian'``) so
    expression can be *sampled* for normalized data, and
  * :class:`VirtualSliceGeneratorV3`, which synthesizes a complete virtual slice
    (positions, cell types, expression) at a target z from its two neighboring
    real sections ONLY — no access to the slice being generated.

The v2-style :class:`VirtualSliceGenerator` (reconstruction at known cell
positions) is also included for reference / backward comparison.
"""

from spatialcpav3.model import SpatialCPA
from spatialcpav3.trainer import SpatialCPATrainer
from spatialcpav3.inference import VirtualSliceGenerator
from spatialcpav3.virtual_slice import VirtualSliceGeneratorV3, NeighborContext
from spatialcpav3.density import (
    DensityFieldModel,
    DensityFieldTrainer,
    DensitySampler,
    KNNSelfAttention,
    estimate_bin_size,
)

__all__ = [
    "SpatialCPA",
    "SpatialCPATrainer",
    "VirtualSliceGenerator",
    "VirtualSliceGeneratorV3",
    "NeighborContext",
    "DensityFieldModel",
    "DensityFieldTrainer",
    "DensitySampler",
    "KNNSelfAttention",
    "estimate_bin_size",
]
