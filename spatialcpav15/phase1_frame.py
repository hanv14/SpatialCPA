"""Phase 1.4-1.6 — common 3D frame, normalized anatomical coordinates, 3D graph.

Phase 1.4 in the proposal ranks registration targets: DAPI > blockface > atlas >
label-field, and warns that the last is circular.  In this pipeline the *rigid*
inter-section registration is done upstream by the harness (training-only, per
holdout), so what remains here is:

* **1.4** a landmark-based *drift stabilization* using only landmarks derivable
  from points — the section centroid and the tissue outline.  It is applied only
  when there is measurable residual drift, and it is inverted before output so
  generated coordinates come back in the caller's frame.  We do **not** run a
  label-field registration: that is the circular option the proposal warns about
  (assume label geometry is consistent in order to align, then study how labels
  vary with z), and there is nothing here that requires it.
* **1.5** normalized anatomical coordinates — the model never sees raw microns,
  so it cannot memorize absolute geometry.
* **1.6** the 3D neighbourhood graph, storing each edge's ``dz`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import FrameConfig


# ── Phase 1.4 — landmarks from points alone ──────────────────────────────────

def section_landmarks(xy: np.ndarray) -> Dict[str, np.ndarray]:
    """Landmarks extractable from positions alone: centroid, outline extent, axes."""
    c = xy.mean(axis=0)
    Xc = xy - c
    cov = np.cov(Xc.T) if xy.shape[0] > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(-evals)
    axes = evecs[:, order]
    # Sign convention: the positive half-axis holds the larger cell mass, so the
    # axes are comparable across sections without a reference image.
    for j in range(2):
        proj = Xc @ axes[:, j]
        if (proj > 0).sum() < (proj < 0).sum():
            axes[:, j] *= -1.0
    return {"centroid": c, "axes": axes, "extent": np.sqrt(np.maximum(evals[order], 0.0))}


@dataclass
class CommonFrame:
    """The Phase 1.4/1.5 transform: per-section shift, then a global normalization."""

    shifts: Dict[str, np.ndarray]        # section_id -> in-plane shift applied
    center: np.ndarray                   # (2,) global center
    scale: float                         # global in-plane scale
    z_min: float
    z_range: float
    applied: bool
    drift: float
    median_spacing: float

    def to_normalized(self, xy: np.ndarray, z: np.ndarray,
                      section_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float64)
        if self.applied and section_id is not None and section_id in self.shifts:
            xy = xy + self.shifts[section_id]
        u = (xy - self.center) / self.scale
        w = (np.asarray(z, dtype=np.float64) - self.z_min) / self.z_range
        return u, w

    def from_normalized(self, u: np.ndarray, shift: Optional[np.ndarray] = None
                        ) -> np.ndarray:
        """Map normalized in-plane coordinates back to the caller's frame."""
        xy = np.asarray(u, dtype=np.float64) * self.scale + self.center
        if self.applied and shift is not None:
            xy = xy - shift
        return xy

    def z_to_normalized(self, z: float) -> float:
        return float((z - self.z_min) / self.z_range)


def build_common_frame(stack, cfg: FrameConfig) -> CommonFrame:
    """Phase 1.4 + 1.5 — stabilize residual drift, then normalize coordinates."""
    ids = [s.section_id for s in stack.slices]
    zs = stack.z_positions
    lms = [section_landmarks(s.coords_xy) for s in stack.slices]
    cents = np.stack([l["centroid"] for l in lms])
    spacing = stack.median_nn_distance()

    # A smooth (linear in z) centroid trend is real anatomy; deviation from it is
    # section-to-section drift.  Fit the trend robustly and measure the residual.
    if len(zs) >= 3:
        A = np.column_stack([np.ones_like(zs), zs])
        coef, *_ = np.linalg.lstsq(A, cents, rcond=None)
        trend = A @ coef
    else:
        trend = np.broadcast_to(cents.mean(axis=0), cents.shape)
    resid = cents - trend
    drift = float(np.median(np.linalg.norm(resid, axis=1)) / max(spacing, 1e-9))

    apply = (cfg.stabilize == "always"
             or (cfg.stabilize == "auto" and drift > cfg.drift_tolerance))
    shifts = {sid: (-resid[i] if apply else np.zeros(2)) for i, sid in enumerate(ids)}

    xy_all = np.vstack([s.coords_xy + shifts[s.section_id] for s in stack.slices])
    center = xy_all.mean(axis=0)
    scale = float(np.sqrt(((xy_all - center) ** 2).sum(axis=1).mean())) or 1.0
    z_min = float(zs.min())
    z_range = float(zs.max() - zs.min()) or float(stack.median_spacing) or 1.0

    return CommonFrame(shifts=shifts, center=center, scale=scale, z_min=z_min,
                       z_range=z_range, applied=bool(apply), drift=drift,
                       median_spacing=spacing)


# ── Phase 1.6 — 3D neighbourhood graph with explicit edge dz ─────────────────

@dataclass
class NeighborGraph:
    """kNN graph over the 3D stack. Every edge carries its own ``dz``."""

    idx: np.ndarray        # (n, k) neighbour indices
    dist: np.ndarray       # (n, k) 3D distance in normalized units
    dz: np.ndarray         # (n, k) |z_i - z_j| in *raw* units (proposal 1.6)
    n_cells: int


def build_neighbor_graph(u: np.ndarray, z_raw: np.ndarray, z_scale: float,
                         cfg: FrameConfig) -> NeighborGraph:
    """Build the 3D kNN graph within *and across* sections.

    ``z_scale`` converts raw z into the normalized in-plane unit so the kd-tree
    metric is not dominated by whichever axis happens to have larger numbers.
    """
    n = u.shape[0]
    k = int(min(cfg.k_neighbors_3d, max(n - 1, 1)))
    P = np.column_stack([u, np.asarray(z_raw, dtype=np.float64) * z_scale])
    d, i = cKDTree(P).query(P, k=k + 1)
    d, i = d[:, 1:], i[:, 1:]
    dz = np.abs(np.asarray(z_raw)[i] - np.asarray(z_raw)[:, None])
    return NeighborGraph(idx=i, dist=d, dz=dz, n_cells=n)
