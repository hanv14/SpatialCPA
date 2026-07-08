"""
SpatialCPA-v8 — symmetric optimal-transport bridge virtual-slice synthesis.

Training-free, pure numpy/scipy/sklearn de-novo generation of a virtual tissue
slice at an arbitrary z, from the training-only re-registered flanking slices and
a scalar target z (no held-out ``(x, y)`` or content). The headline advance over
v6 is the **bidirectional McCann barycentric bridge** (:mod:`spatialcpav8.transport`),
which builds one coherent cell sheet from *both* flanking populations — removing
v6's per-holdout choice between a coherent one-sided morph and both-slice
interpolation — plus a **density calibration** to the interpolated flanking
density field (:mod:`spatialcpav8.density`) and a niche-aware annotator.

Public API::

    from spatialcpav8 import SpatialCPAv8, SpatialCPAv8Config, Slice, SliceStack
    gen = SpatialCPAv8(stack, gene_names=genes, cell_type_names=types)
    vs = gen.generate_virtual_slice(z=target_z)
"""

from .config import (
    SpatialCPAv8Config,
    EmbeddingConfig,
    TransportConfig,
    BridgeConfig,
    DensityConfig,
    CommunicationConfig,
    AnnotationConfig,
    SynthesisConfig,
)
from .data import Slice, SliceStack
from .generator import SpatialCPAv8, VirtualSlice

__version__ = "8.0.0"

__all__ = [
    "SpatialCPAv8",
    "VirtualSlice",
    "SpatialCPAv8Config",
    "EmbeddingConfig",
    "TransportConfig",
    "BridgeConfig",
    "DensityConfig",
    "CommunicationConfig",
    "AnnotationConfig",
    "SynthesisConfig",
    "Slice",
    "SliceStack",
]
