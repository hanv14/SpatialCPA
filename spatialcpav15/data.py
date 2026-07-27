"""Containers for the sectioned specimen (SpatialCPA-v15).

A :class:`Slice` is one physical section: an (n, G) expression matrix, in-plane
coordinates, per-cell z, and (optionally) integer cell-type indices.  A
:class:`SliceStack` is the ordered stack, and knows how to pick the brackets a
query z sits between.

The proposal (Phase 2.3) asks for **two or more slices per side** so the model
sees the derivative of the structure, not just its endpoints; ``brackets()``
returns exactly the ``(a-1, a, b, b+1)`` tuple that requirement implies, with
graceful duplication at the ends of the stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Slice:
    """One physical section."""

    expression: np.ndarray                     # (n, G) log-normalized
    coords_xy: np.ndarray                      # (n, 2)
    z_values: np.ndarray                       # (n,)
    cell_type_indices: Optional[np.ndarray] = None   # (n,) int
    section_id: str = ""
    counts: Optional[np.ndarray] = None        # (n, G) raw counts when available

    def __post_init__(self):
        self.expression = np.asarray(self.expression, dtype=np.float32)
        self.coords_xy = np.asarray(self.coords_xy, dtype=np.float64)
        self.z_values = np.asarray(self.z_values, dtype=np.float64)
        if self.cell_type_indices is not None:
            self.cell_type_indices = np.asarray(self.cell_type_indices, dtype=np.int64)
        if self.counts is not None:
            self.counts = np.asarray(self.counts, dtype=np.float32)

    @property
    def n_spots(self) -> int:
        return int(self.expression.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.expression.shape[1])

    @property
    def z(self) -> float:
        """The section's nominal depth (median of its cells' z)."""
        return float(np.median(self.z_values)) if self.n_spots else 0.0

    def subset(self, mask: np.ndarray) -> "Slice":
        mask = np.asarray(mask)
        return Slice(
            expression=self.expression[mask],
            coords_xy=self.coords_xy[mask],
            z_values=self.z_values[mask],
            cell_type_indices=(None if self.cell_type_indices is None
                               else self.cell_type_indices[mask]),
            section_id=self.section_id,
            counts=None if self.counts is None else self.counts[mask],
        )


@dataclass
class SliceStack:
    """An ordered stack of sections (ascending z)."""

    slices: List[Slice] = field(default_factory=list)

    def __post_init__(self):
        self.slices = sorted(self.slices, key=lambda s: s.z)

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def n_genes(self) -> int:
        return self.slices[0].n_genes if self.slices else 0

    @property
    def n_cells(self) -> int:
        return int(sum(s.n_spots for s in self.slices))

    @property
    def z_positions(self) -> np.ndarray:
        return np.array([s.z for s in self.slices], dtype=np.float64)

    @property
    def median_spacing(self) -> float:
        """Median inter-section spacing ``dz`` (the sampling period along z)."""
        zs = self.z_positions
        if len(zs) < 2:
            return 1.0
        d = np.diff(zs)
        d = d[d > 0]
        return float(np.median(d)) if d.size else 1.0

    # ── geometry helpers ─────────────────────────────────────────────────────
    def all_xy(self) -> np.ndarray:
        return np.vstack([s.coords_xy for s in self.slices])

    def all_types(self) -> Optional[np.ndarray]:
        if any(s.cell_type_indices is None for s in self.slices):
            return None
        return np.concatenate([s.cell_type_indices for s in self.slices])

    def all_expression(self) -> np.ndarray:
        return np.vstack([s.expression for s in self.slices])

    def all_counts(self) -> Optional[np.ndarray]:
        if any(s.counts is None for s in self.slices):
            return None
        return np.vstack([s.counts for s in self.slices])

    def median_nn_distance(self, max_cells: int = 4000, seed: int = 0) -> float:
        """Median in-plane nearest-neighbour distance — the natural length unit."""
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(seed)
        vals = []
        for s in self.slices:
            if s.n_spots < 4:
                continue
            xy = s.coords_xy
            if xy.shape[0] > max_cells:
                xy = xy[rng.choice(xy.shape[0], max_cells, replace=False)]
            d, _ = cKDTree(xy).query(xy, k=2)
            vals.append(np.median(d[:, 1]))
        return float(np.median(vals)) if vals else 1.0

    # ── bracket selection (Phase 2.3: two slices per side) ───────────────────
    def bracket_indices(self, z: float) -> Tuple[int, int]:
        """Inner bracket indices ``(a, b)`` with ``z_a <= z <= z_b``."""
        zs = self.z_positions
        if self.n_slices == 1:
            return 0, 0
        b = int(np.searchsorted(zs, z, side="left"))
        b = int(np.clip(b, 1, self.n_slices - 1))
        a = b - 1
        return a, b

    def brackets(self, z: float) -> Tuple[int, int, int, int]:
        """``(a-1, a, b, b+1)`` with edge duplication (Phase 2.3)."""
        a, b = self.bracket_indices(z)
        a_out = max(a - 1, 0)
        b_out = min(b + 1, self.n_slices - 1)
        return a_out, a, b, b_out

    def interp_fraction(self, z: float) -> Tuple[float, float]:
        """``(t, dz)`` — normalized position in the gap and the gap width."""
        a, b = self.bracket_indices(z)
        za, zb = self.z_positions[a], self.z_positions[b]
        dz = float(zb - za)
        if dz <= 0:
            return 0.0, float(self.median_spacing)
        return float(np.clip((z - za) / dz, -1.0, 2.0)), dz

    def distance_to_nearest_section(self, z: float) -> float:
        return float(np.min(np.abs(self.z_positions - z)))


@dataclass
class VirtualSlice:
    """A synthesized section, with the mandatory Phase 4.7 provenance channels."""

    coords: np.ndarray                      # (n, 3)
    expression: np.ndarray                  # (n, G)
    cell_type: Optional[np.ndarray] = None  # (n,) str
    dist_to_real_slice: Optional[np.ndarray] = None    # (n,) provenance ch. 1
    structural_spread: Optional[np.ndarray] = None     # (n,) provenance ch. 2
    expression_spread: Optional[np.ndarray] = None     # (n,) optional
    unresolved: Optional[np.ndarray] = None            # (n,) bool, sub-Nyquist
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.coords.shape[0])


def build_stack_from_arrays(expression: np.ndarray,
                            coords_xyz: np.ndarray,
                            sections: Sequence[str],
                            cell_type_indices: Optional[np.ndarray] = None,
                            counts: Optional[np.ndarray] = None) -> SliceStack:
    """Convenience builder: split flat arrays into a :class:`SliceStack`."""
    sections = np.asarray(sections).astype(str)
    coords_xyz = np.asarray(coords_xyz, dtype=np.float64)
    order = sorted(np.unique(sections),
                   key=lambda s: float(np.median(coords_xyz[sections == s, 2])))
    slices = []
    for sec in order:
        m = sections == sec
        slices.append(Slice(
            expression=expression[m],
            coords_xy=coords_xyz[m, :2],
            z_values=coords_xyz[m, 2],
            cell_type_indices=None if cell_type_indices is None else cell_type_indices[m],
            section_id=str(sec),
            counts=None if counts is None else counts[m],
        ))
    return SliceStack(slices)
