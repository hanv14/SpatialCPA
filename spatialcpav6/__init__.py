"""SpatialCPA-v6 — optimal-transport virtual-slice synthesis.

A training-free (inference-time) generator that synthesizes a held-out tissue
section from its two flanking sections and a scalar target z, by:

1. embedding cells in a foundation-model cell-state space (with a local PCA
   fallback), :mod:`spatialcpav6.embedding`;
2. placing cells via optimal-transport displacement interpolation between the
   flanking slices, :mod:`spatialcpav6.transport`;
3. annotating them with an FM cell-state prior constrained to the interpolated
   composition and cell-cell-communication (niche) architecture,
   :mod:`spatialcpav6.annotation` / :mod:`spatialcpav6.communication`;
4. transferring real expression profiles from correctly-typed training cells.

See ``README.md`` for the method and its benchmark integration.
"""

from .config import (
    SpatialCPAv6Config, EmbeddingConfig, TransportConfig,
    CommunicationConfig, AnnotationConfig, SynthesisConfig,
)
from .data import Slice, SliceStack
from .embedding import build_embedder, register_embedder, Embedder
from .generator import SpatialCPAv6, VirtualSlice

__all__ = [
    "SpatialCPAv6Config", "EmbeddingConfig", "TransportConfig",
    "CommunicationConfig", "AnnotationConfig", "SynthesisConfig",
    "Slice", "SliceStack",
    "build_embedder", "register_embedder", "Embedder",
    "SpatialCPAv6", "VirtualSlice",
]

__version__ = "6.2.0"   # 6.2.0: `adaptive` placement (morph vs interpolate per holdout)
