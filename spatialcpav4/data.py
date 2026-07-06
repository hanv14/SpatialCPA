"""
Data structures and neighbor construction for SpatialCPA-v4.

The transformer version learns the mapping

    {Slice(i-1), Slice(i+1)}  ->  Slice(i)

directly.  This module turns a stack of aligned 2D sections into the concrete
training samples the model consumes:

* :class:`Slice` — a thin container for one physical section.
* :class:`SliceStack` — an ordered collection of slices plus a flattened
  "global spot table" (expression / coords / labels concatenated across all
  slices) that the dataset gathers neighbor features from.
* :func:`knn_indices` — KDTree nearest-neighbor search on physical coordinates.
* :func:`build_triplet_samples` — for every interior slice, build one sample per
  target spot (positive, occupancy=1) plus synthetic background samples
  (occupancy=0) for the occupancy head.

Neighbor indices are the expensive part, so they are cached both in memory (a
single build reuses per-slice KDTrees) and optionally on disk (a content hash of
the coordinates keys the cache file).

All distances use Euclidean distance on 3D physical coordinates (x, y, z).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# Slice                                                                        #
# --------------------------------------------------------------------------- #
class Slice:
    """A single aligned tissue section.

    Parameters
    ----------
    expression : (N, G) float array
        Per-spot gene expression (already normalised for training).
    coords_xy : (N, 2) float array
        In-plane physical coordinates.
    z_values : (N,) float array
        Per-spot z coordinate.  Usually near-constant within a section but kept
        per-spot so warped / thick sections are supported.
    cell_type_indices : (N,) int array or None
        Integer cell-type labels (``None`` if unavailable).
    region_indices : (N,) int array or None
        Integer region labels (``None`` if unavailable).
    section_id : str
        Human-readable identifier.
    """

    def __init__(
        self,
        expression: np.ndarray,
        coords_xy: np.ndarray,
        z_values: np.ndarray,
        cell_type_indices: Optional[np.ndarray] = None,
        region_indices: Optional[np.ndarray] = None,
        section_id: str = "",
    ) -> None:
        self.expression = np.ascontiguousarray(expression, dtype=np.float32)
        self.coords_xy = np.ascontiguousarray(coords_xy, dtype=np.float32)
        self.z_values = np.ascontiguousarray(z_values, dtype=np.float32).reshape(-1)
        self.cell_type_indices = (
            None if cell_type_indices is None
            else np.ascontiguousarray(cell_type_indices, dtype=np.int64).reshape(-1)
        )
        self.region_indices = (
            None if region_indices is None
            else np.ascontiguousarray(region_indices, dtype=np.int64).reshape(-1)
        )
        self.section_id = str(section_id)
        self.n_spots = self.expression.shape[0]
        self.z_center = float(np.median(self.z_values)) if self.n_spots else 0.0

    def coords_3d(self) -> np.ndarray:
        """Return (N, 3) physical coordinates."""
        return np.hstack([self.coords_xy, self.z_values.reshape(-1, 1)]).astype(np.float32)


# --------------------------------------------------------------------------- #
# SliceStack — ordered slices + flattened global spot table                    #
# --------------------------------------------------------------------------- #
class SliceStack:
    """An ordered stack of :class:`Slice` objects sorted by z-center.

    Builds a single flattened table of all spots (expression, coords, labels)
    so the dataset can gather neighbor features by a single global index.  Each
    slice knows its ``[offset, offset + n_spots)`` span in that table.
    """

    def __init__(self, slices: Sequence[Slice]):
        # Sort physically by z so "lower" / "upper" neighbors are well defined.
        self.slices: List[Slice] = sorted(slices, key=lambda s: s.z_center)
        self.n_slices = len(self.slices)
        if self.n_slices == 0:
            raise ValueError("SliceStack requires at least one slice")

        self.n_genes = self.slices[0].expression.shape[1]
        self.has_cell_type = all(s.cell_type_indices is not None for s in self.slices)
        self.has_region = all(s.region_indices is not None for s in self.slices)

        # Global offsets for each slice into the flattened table.
        self.offsets: List[int] = []
        running = 0
        for s in self.slices:
            self.offsets.append(running)
            running += s.n_spots
        self.n_spots = running

        # Flattened global arrays (read-only, shared with DataLoader workers).
        self.expr = np.concatenate([s.expression for s in self.slices], axis=0)
        self.coords = np.concatenate([s.coords_3d() for s in self.slices], axis=0)
        if self.has_cell_type:
            self.cell_type = np.concatenate(
                [s.cell_type_indices for s in self.slices], axis=0
            )
        else:
            self.cell_type = np.zeros(self.n_spots, dtype=np.int64)
        if self.has_region:
            self.region = np.concatenate(
                [s.region_indices for s in self.slices], axis=0
            )
        else:
            self.region = np.zeros(self.n_spots, dtype=np.int64)

        # Per-slice cached KDTrees (built lazily).
        self._kdtrees: List[Optional[cKDTree]] = [None] * self.n_slices

    # ---- accessors -------------------------------------------------------- #
    def global_range(self, slice_idx: int) -> Tuple[int, int]:
        start = self.offsets[slice_idx]
        return start, start + self.slices[slice_idx].n_spots

    def kdtree(self, slice_idx: int) -> cKDTree:
        """Return (and cache) the KDTree over a slice's 3D coordinates."""
        if self._kdtrees[slice_idx] is None:
            self._kdtrees[slice_idx] = cKDTree(self.slices[slice_idx].coords_3d())
        return self._kdtrees[slice_idx]

    def z_centers(self) -> np.ndarray:
        return np.array([s.z_center for s in self.slices], dtype=np.float64)

    # ---- scale estimation ------------------------------------------------- #
    def estimate_coord_scale(self, sample: int = 5000) -> float:
        """Median in-plane nearest-neighbor distance across slices.

        Used to normalise relative coordinates so the coordinate encoder sees
        O(1) inputs regardless of the physical unit system.
        """
        dists = []
        rng = np.random.default_rng(0)
        for s in self.slices:
            if s.n_spots < 2:
                continue
            tree = cKDTree(s.coords_xy)
            idx = (
                rng.choice(s.n_spots, size=min(sample, s.n_spots), replace=False)
                if s.n_spots > sample else np.arange(s.n_spots)
            )
            d, _ = tree.query(s.coords_xy[idx], k=2)
            dists.append(d[:, 1])
        if not dists:
            return 1.0
        scale = float(np.median(np.concatenate(dists)))
        return scale if scale > 0 else 1.0


# --------------------------------------------------------------------------- #
# Nearest-neighbor search                                                      #
# --------------------------------------------------------------------------- #
def knn_indices(
    query_coords: np.ndarray,
    tree: cKDTree,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """K nearest neighbors of ``query_coords`` in ``tree``.

    Returns
    -------
    idx : (Q, k) int array
        Local indices into the tree's data.  If the source slice has fewer than
        ``k`` spots the last valid neighbor is repeated (the caller receives a
        companion mask via :func:`_pad_neighbors` when needed).
    dist : (Q, k) float array
        Euclidean distances.
    """
    n_src = tree.n
    k_eff = min(k, n_src)
    dist, idx = tree.query(query_coords, k=k_eff)
    # Normalise shapes: cKDTree returns 1-D arrays when k_eff == 1.
    if k_eff == 1:
        dist = dist.reshape(-1, 1)
        idx = idx.reshape(-1, 1)
    if k_eff < k:  # pad by repeating the farthest available neighbor
        pad = k - k_eff
        idx = np.concatenate([idx, np.repeat(idx[:, -1:], pad, axis=1)], axis=1)
        dist = np.concatenate([dist, np.repeat(dist[:, -1:], pad, axis=1)], axis=1)
    return idx.astype(np.int64), dist.astype(np.float32)


# --------------------------------------------------------------------------- #
# Negative (background) sampling for the occupancy head                        #
# --------------------------------------------------------------------------- #
def sample_background_coords(
    slice_obj: Slice,
    n_samples: int,
    min_dist_factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample coordinates inside the XY bounding box but outside tissue.

    A candidate is accepted only if it is at least ``min_dist_factor *
    median_spacing`` away from every real spot, guaranteeing the label
    "background" is meaningful.  Points inherit the slice's median z.

    Returns
    -------
    coords : (M, 3) array with ``M <= n_samples`` accepted background points.
    """
    if n_samples <= 0 or slice_obj.n_spots < 2:
        return np.zeros((0, 3), dtype=np.float32)

    xy = slice_obj.coords_xy
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)
    median_spacing = float(np.median(d[:, 1]))
    reject_radius = min_dist_factor * median_spacing

    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    z_center = slice_obj.z_center

    accepted = []
    # Oversample candidates then filter; cap attempts to avoid infinite loops on
    # dense slices where little background exists.
    attempts = 0
    max_attempts = 20
    while len(accepted) < n_samples and attempts < max_attempts:
        need = n_samples - len(accepted)
        cand = rng.uniform(lo, hi, size=(need * 4, 2)).astype(np.float32)
        nn_dist, _ = tree.query(cand, k=1)
        keep = cand[nn_dist > reject_radius]
        accepted.extend(keep.tolist())
        attempts += 1

    if not accepted:
        return np.zeros((0, 3), dtype=np.float32)
    accepted = np.asarray(accepted[:n_samples], dtype=np.float32)
    z_col = np.full((accepted.shape[0], 1), z_center, dtype=np.float32)
    return np.hstack([accepted, z_col])


# --------------------------------------------------------------------------- #
# Triplet sample construction                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class TripletSamples:
    """Flat arrays describing every training sample.

    All neighbor indices are *global* indices into ``SliceStack`` tables, so the
    dataset gathers features with a single fancy-index.
    """

    lower_idx: np.ndarray      # (S, k) global indices into the lower slice
    upper_idx: np.ndarray      # (S, k) global indices into the upper slice
    target_coords: np.ndarray  # (S, 3) query coordinates
    target_spot: np.ndarray    # (S,) global index of the target spot, or -1
    occupancy: np.ndarray      # (S,) 1.0 tissue / 0.0 background
    has_target: np.ndarray     # (S,) 1.0 if expression/label supervised

    def __len__(self) -> int:
        return self.occupancy.shape[0]


def _neighbors_for_targets(
    stack: SliceStack,
    target_coords: np.ndarray,
    lower_slice: int,
    upper_slice: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Global neighbor indices from the two flanking slices for query coords."""
    lo_local, _ = knn_indices(target_coords, stack.kdtree(lower_slice), k)
    up_local, _ = knn_indices(target_coords, stack.kdtree(upper_slice), k)
    lo_global = lo_local + stack.offsets[lower_slice]
    up_global = up_local + stack.offsets[upper_slice]
    return lo_global.astype(np.int64), up_global.astype(np.int64)


def build_triplet_samples(
    stack: SliceStack,
    n_neighbors: int,
    negative_ratio: float = 0.0,
    negative_min_dist_factor: float = 2.0,
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> TripletSamples:
    """Assemble all training samples from interior slices.

    For each interior slice ``i`` (one that has both a lower and an upper
    neighbor slice), every spot becomes a positive target with neighbors drawn
    from slices ``i-1`` and ``i+1``.  Synthetic background points add occupancy
    negatives.

    Parameters mirror :class:`~spatialcpav4.config.DataConfig`.
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"triplets_{_stack_hash(stack)}_k{n_neighbors}" \
                                       f"_neg{negative_ratio}_{seed}.npz"
        if cache_path.exists():
            return _load_samples(cache_path)

    rng = np.random.default_rng(seed)
    k = n_neighbors

    lower_all, upper_all, coords_all = [], [], []
    target_spot_all, occ_all, has_target_all = [], [], []

    for i in range(1, stack.n_slices - 1):
        lower_slice, upper_slice = i - 1, i + 1
        tgt = stack.slices[i]
        g0, _ = stack.global_range(i)

        # ---- positive targets: every spot in slice i --------------------- #
        coords = tgt.coords_3d()
        lo_g, up_g = _neighbors_for_targets(stack, coords, lower_slice, upper_slice, k)
        lower_all.append(lo_g)
        upper_all.append(up_g)
        coords_all.append(coords)
        target_spot_all.append(np.arange(g0, g0 + tgt.n_spots, dtype=np.int64))
        occ_all.append(np.ones(tgt.n_spots, dtype=np.float32))
        has_target_all.append(np.ones(tgt.n_spots, dtype=np.float32))

        # ---- negative (background) targets ------------------------------- #
        n_neg = int(round(negative_ratio * tgt.n_spots))
        if n_neg > 0:
            bg = sample_background_coords(tgt, n_neg, negative_min_dist_factor, rng)
            if bg.shape[0] > 0:
                lo_g, up_g = _neighbors_for_targets(
                    stack, bg, lower_slice, upper_slice, k
                )
                lower_all.append(lo_g)
                upper_all.append(up_g)
                coords_all.append(bg)
                target_spot_all.append(np.full(bg.shape[0], -1, dtype=np.int64))
                occ_all.append(np.zeros(bg.shape[0], dtype=np.float32))
                has_target_all.append(np.zeros(bg.shape[0], dtype=np.float32))

    if not lower_all:
        raise ValueError(
            "No interior slices available: need at least 3 slices to form "
            "{i-1, i+1} -> i triplets."
        )

    samples = TripletSamples(
        lower_idx=np.concatenate(lower_all, axis=0),
        upper_idx=np.concatenate(upper_all, axis=0),
        target_coords=np.concatenate(coords_all, axis=0).astype(np.float32),
        target_spot=np.concatenate(target_spot_all, axis=0),
        occupancy=np.concatenate(occ_all, axis=0),
        has_target=np.concatenate(has_target_all, axis=0),
    )

    if cache_path is not None:
        _save_samples(cache_path, samples)
    return samples


# --------------------------------------------------------------------------- #
# Disk cache helpers                                                           #
# --------------------------------------------------------------------------- #
def _stack_hash(stack: SliceStack) -> str:
    """Stable short hash of the stack geometry (coords + section ids)."""
    h = hashlib.sha1()
    h.update(stack.coords.tobytes())
    h.update("|".join(s.section_id for s in stack.slices).encode())
    return h.hexdigest()[:12]


def _save_samples(path: Path, s: TripletSamples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        lower_idx=s.lower_idx,
        upper_idx=s.upper_idx,
        target_coords=s.target_coords,
        target_spot=s.target_spot,
        occupancy=s.occupancy,
        has_target=s.has_target,
    )


def _load_samples(path: Path) -> TripletSamples:
    d = np.load(path)
    return TripletSamples(
        lower_idx=d["lower_idx"],
        upper_idx=d["upper_idx"],
        target_coords=d["target_coords"],
        target_spot=d["target_spot"],
        occupancy=d["occupancy"],
        has_target=d["has_target"],
    )
