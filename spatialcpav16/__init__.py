"""SpatialCPA-v16 — Spectral Tissue Flow.

Conditional flow matching over the *graph-spectral coefficients* of a tissue
section, with the per-gene spatial-autocorrelation profile as an explicitly
enforced target rather than an emergent property.
"""

from .config import (ExpressionConfig, FlowConfig, GeometryConfig,
                     HarmonicConfig, RuntimeConfig, SpatialCPAv16Config,
                     TypeConfig)
from .data import Slice, SliceStack, VirtualSlice
from .model import SpatialCPAv16

__version__ = "16.0.0"

__all__ = [
    "SpatialCPAv16", "SpatialCPAv16Config", "Slice", "SliceStack",
    "VirtualSlice", "HarmonicConfig", "GeometryConfig", "FlowConfig",
    "ExpressionConfig", "TypeConfig", "RuntimeConfig", "__version__",
]
