"""Slice containers for v16. Deliberately minimal — the method's state lives in
the harmonic bases, which are built per section in ``model.py``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Slice:
    expression: np.ndarray            # (n, G) normalized expression
    coords_xy: np.ndarray             # (n, 2)
    z: float
    cell_type_indices: np.ndarray | None = None
    section_id: str = ""

    @property
    def n_cells(self):
        return int(self.expression.shape[0])


@dataclass
class SliceStack:
    slices: list

    def __post_init__(self):
        self.slices = sorted(self.slices, key=lambda s: float(s.z))

    @property
    def n_slices(self):
        return len(self.slices)

    @property
    def z_values(self):
        return np.array([float(s.z) for s in self.slices])

    def brackets(self, z, n_side=1):
        """Indices of the ``n_side`` real sections on each side of ``z``.

        Falls back to the nearest available sections at the ends of the stack, so
        a query beyond the outermost section still gets context rather than
        failing — extrapolation is a worse estimate, not an error.
        """
        zs = self.z_values
        below = np.where(zs <= z)[0]
        above = np.where(zs > z)[0]
        lo = list(below[-n_side:]) if len(below) else []
        hi = list(above[:n_side]) if len(above) else []
        while len(lo) < n_side and len(hi) > n_side - len(lo):
            lo.append(hi.pop(0) if hi else lo[-1])
        while len(hi) < n_side and len(lo) > 0:
            hi.append(lo[-1])
        idx = sorted(set(lo + hi))
        if not idx:
            idx = list(range(min(2 * n_side, self.n_slices)))
        return idx

    def fractional_depth(self, z):
        """Where ``z`` sits between its two immediate bracketing sections, in [0, 1]."""
        zs = self.z_values
        below = np.where(zs <= z)[0]
        above = np.where(zs > z)[0]
        if not len(below):
            return 0.0, int(above[0]), int(above[min(1, len(above) - 1)])
        if not len(above):
            return 1.0, int(below[max(0, len(below) - 2)]), int(below[-1])
        i_lo, i_hi = int(below[-1]), int(above[0])
        z_lo, z_hi = zs[i_lo], zs[i_hi]
        w = 0.5 if z_hi == z_lo else float((z - z_lo) / (z_hi - z_lo))
        return float(np.clip(w, 0.0, 1.0)), i_lo, i_hi


@dataclass
class VirtualSlice:
    expression: np.ndarray
    coords: np.ndarray                # (n, 3)
    cell_type: np.ndarray | None = None
    z: float = 0.0
    diagnostics: dict = field(default_factory=dict)
