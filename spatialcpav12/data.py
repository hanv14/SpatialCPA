"""
Lightweight data containers for SpatialCPA-v12.

A :class:`Slice` is one aligned tissue section; a :class:`SliceStack` is an
ordered stack (by z) with a flattened global table and flanking-slice selection.
These are self-contained so v12 has no import dependency on the other SpatialCPA
packages (the versions coexist and can be benchmarked side by side). All
coordinates are physical ``(x, y, z)``; the flanking slices handed to the
synthesizer are the *training-only, re-registered* sections — the held-out slice
never enters this stack (enforced upstream by the benchmark).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


class Slice:
    """A single aligned tissue section (training-only in the benchmark)."""

    def __init__(
        self,
        expression: np.ndarray,
        coords_xy: np.ndarray,
        z_values: np.ndarray,
        cell_type_indices: Optional[np.ndarray] = None,
        section_id: str = "",
    ) -> None:
        self.expression = np.ascontiguousarray(expression, dtype=np.float32)
        self.coords_xy = np.ascontiguousarray(coords_xy, dtype=np.float32)
        self.z_values = np.ascontiguousarray(z_values, dtype=np.float32).reshape(-1)
        self.cell_type_indices = (
            None if cell_type_indices is None
            else np.ascontiguousarray(cell_type_indices, dtype=np.int64).reshape(-1)
        )
        self.section_id = str(section_id)
        self.n_spots = self.expression.shape[0]
        self.z_center = float(np.median(self.z_values)) if self.n_spots else 0.0

    def coords_3d(self) -> np.ndarray:
        return np.hstack([self.coords_xy, self.z_values.reshape(-1, 1)]).astype(np.float32)

    def median_spacing(self) -> float:
        """Median in-plane nearest-neighbor distance (tissue length scale)."""
        if self.n_spots < 2:
            return 1.0
        d, _ = cKDTree(self.coords_xy).query(self.coords_xy, k=2)
        s = float(np.median(d[:, 1]))
        return s if s > 0 else 1.0


class SliceStack:
    """Ordered stack of :class:`Slice` objects sorted by z-center."""

    def __init__(self, slices: Sequence[Slice]):
        self.slices: List[Slice] = sorted(slices, key=lambda s: s.z_center)
        self.n_slices = len(self.slices)
        if self.n_slices == 0:
            raise ValueError("SliceStack requires at least one slice")
        self.n_genes = self.slices[0].expression.shape[1]
        self.has_cell_type = all(s.cell_type_indices is not None for s in self.slices)

    def z_centers(self) -> np.ndarray:
        return np.array([s.z_center for s in self.slices], dtype=np.float64)

    def union_expression(self) -> np.ndarray:
        """All training-slice expression stacked (for fitting the embedder)."""
        return np.concatenate([s.expression for s in self.slices], axis=0)

    def n_cell_types(self) -> Optional[int]:
        if not self.has_cell_type:
            return None
        mx = max(int(s.cell_type_indices.max()) for s in self.slices if s.n_spots)
        return mx + 1

    def median_spacing(self) -> float:
        vals = [s.median_spacing() for s in self.slices if s.n_spots >= 2]
        return float(np.median(vals)) if vals else 1.0

    def pick_flanking_slices(self, z: float) -> Tuple[Slice, Slice]:
        """Nearest lower and upper training slices for a query z.

        Falls back to the two nearest by ``|Δz|`` when ``z`` is outside the
        section range (so extrapolation still yields two flanking slices).
        """
        ordered = self.slices
        below = [s for s in ordered if s.z_center <= z]
        above = [s for s in ordered if s.z_center > z]
        if below and above:
            return below[-1], above[0]
        nearest = sorted(ordered, key=lambda s: abs(s.z_center - z))[:2]
        nearest = sorted(nearest, key=lambda s: s.z_center)
        return nearest[0], nearest[-1]
