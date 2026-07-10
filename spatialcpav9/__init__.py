"""
SpatialCPA-v9 — neural cross-slice flow-matching bridge for virtual-slice generation.

The first *learned* SpatialCPA generator. A conditional rectified-flow / OT-CFM
model transports the cell distribution of one flanking slice to the other in a
joint (position, expression-latent) space, conditioned on the axial gap and a
permutation-invariant summary of the neighbouring slices; integrating the learned
probability-flow ODE to the fractional depth of the target z yields the virtual
slice. It trains on the training slices only (all adjacent pairs supply OT-coupled
supervision) and generalizes to any z. If PyTorch is unavailable it falls back to
the v8 coherent optimal-transport morph.

Public API::

    from spatialcpav9 import SpatialCPAv9, SpatialCPAv9Config, Slice, SliceStack
    gen = SpatialCPAv9(stack, gene_names=genes, cell_type_names=types)
    vs = gen.generate_virtual_slice(z=target_z)
"""

from .config import (
    SpatialCPAv9Config,
    EmbeddingConfig,
    ModelConfig,
    TrainConfig,
    FlowConfig,
    TransportConfig,
    CommunicationConfig,
    AnnotationConfig,
    SynthesisConfig,
)
from .data import Slice, SliceStack
from .generator import SpatialCPAv9, VirtualSlice

__version__ = "9.0.0"

__all__ = [
    "SpatialCPAv9",
    "VirtualSlice",
    "SpatialCPAv9Config",
    "EmbeddingConfig",
    "ModelConfig",
    "TrainConfig",
    "FlowConfig",
    "TransportConfig",
    "CommunicationConfig",
    "AnnotationConfig",
    "SynthesisConfig",
    "Slice",
    "SliceStack",
]
