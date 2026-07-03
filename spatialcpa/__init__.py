"""
SpatialCPA — Continuous 3D Spatial Transcriptomics Prediction and Atlas Construction.

Learns a continuous 3D function over tissue to predict cell types, regions, and
gene expression at any (x, y, z) coordinate from sparsely sampled 2D sections.
"""

from spatialcpa.model import SpatialCPA
from spatialcpa.trainer import SpatialCPATrainer
from spatialcpa.inference import VirtualSliceGenerator
from spatialcpa.virtual_slice import VirtualSliceGeneratorV3, NeighborContext

__all__ = [
    "SpatialCPA",
    "SpatialCPATrainer",
    "VirtualSliceGenerator",
    "VirtualSliceGeneratorV3",
    "NeighborContext",
]
