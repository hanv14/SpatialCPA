"""
SpatialCPA v2 — Continuous 3D Spatial Transcriptomics Prediction and Atlas.

A biology-informed coordinate neural field that learns the *joint* relationship
between 3D position, cell type, and gene expression from sparsely sampled 2D
sections, then predicts cell type + expression at any (x, y, z).

Key advances over v1:
  * Calibrated multi-scale positional encoding + anisotropic random Fourier
    features (captures oblique/cross-axis tissue structure).
  * Gated residual backbone with dual skip re-injection.
  * FiLM cell-type conditioning and posterior-marginalised expression, tying the
    cell-type and expression heads together.
  * Class-balanced training with metric-aligned expression losses
    (Pearson + gene mean/variance matching).
  * Hybrid neural + k-NN cell-type prediction and moment-calibrated expression
    inference.
"""

from spatialcpav2.model import SpatialCPAv2
from spatialcpav2.trainer import SpatialCPAv2Trainer
from spatialcpav2.inference import VirtualSliceGenerator

# Convenience aliases so downstream code can use the same names as v1.
SpatialCPA = SpatialCPAv2
SpatialCPATrainer = SpatialCPAv2Trainer

__all__ = [
    "SpatialCPAv2", "SpatialCPAv2Trainer", "VirtualSliceGenerator",
    "SpatialCPA", "SpatialCPATrainer",
]
