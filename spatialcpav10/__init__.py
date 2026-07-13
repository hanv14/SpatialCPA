"""
SpatialCPA-v10 — biologically-constrained virtual-slice generation.

Positions come from v8's diffeomorphic single-slice morph (a coherent tissue
deformation), but expression is *generated* from an explicit biological model rather
than copied: a z-continuous **cell-type gene program**, **ligand-receptor
communication** modulation from spatial neighbours, and a real residual that keeps
gene-gene structure (the "balanced hybrid"). Cell-type annotation is the organizing
first step; the niche MRF enforces which types co-localize.

    from spatialcpav10 import SpatialCPAv10, SpatialCPAv10Config, Slice, SliceStack
    gen = SpatialCPAv10(stack, gene_names=genes, cell_type_names=types)
    vs = gen.generate_virtual_slice(z=target_z)
"""

from .config import (
    SpatialCPAv10Config,
    EmbeddingConfig,
    TransportConfig,
    BridgeConfig,
    DensityConfig,
    CommunicationConfig,
    AnnotationConfig,
    SynthesisConfig,
    BiologyConfig,
)
from .data import Slice, SliceStack
from .generator import SpatialCPAv10, VirtualSlice

__version__ = "10.0.0"

__all__ = [
    "SpatialCPAv10", "VirtualSlice", "SpatialCPAv10Config",
    "EmbeddingConfig", "TransportConfig", "BridgeConfig", "DensityConfig",
    "CommunicationConfig", "AnnotationConfig", "SynthesisConfig", "BiologyConfig",
    "Slice", "SliceStack",
]
