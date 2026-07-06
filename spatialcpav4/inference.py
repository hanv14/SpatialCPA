"""
Inference and virtual-slice generation for SpatialCPA-v4.

The :class:`Predictor` loads a trained model and predicts, for any set of query
locations flanked by a lower and an upper slice:

* gene expression,
* occupancy probability (tissue vs background),
* cell type (if the model has a cell-type head),
* region (if the model has a region head).

Two entry points:

* :meth:`Predictor.predict_slice` — predict at the spots of an existing slice
  (e.g. to reconstruct a held-out section for benchmarking).
* :meth:`Predictor.generate_virtual_slice` — synthesise a brand-new slice at an
  arbitrary z by querying a grid over the XY bounding box and keeping only
  grid points whose occupancy exceeds a threshold.

Neighbor search reuses the KDTree utilities in :mod:`spatialcpav4.data`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from .data import Slice, knn_indices
from .dataset import _duplicate_pad_mask
from .model import SpatialCPATransformer


@dataclass
class SlicePrediction:
    """Container for a predicted slice."""

    coords: np.ndarray                       # (M, 3)
    expression: np.ndarray                   # (M, G)
    occupancy_prob: np.ndarray               # (M,)
    cell_type: Optional[np.ndarray] = None   # (M,) string labels
    region: Optional[np.ndarray] = None      # (M,) string labels
    cell_type_idx: Optional[np.ndarray] = None   # (M,) int
    region_idx: Optional[np.ndarray] = None      # (M,) int


class Predictor:
    """Run a trained :class:`SpatialCPATransformer` on new locations.

    Parameters
    ----------
    model
        A trained model (already on the desired device).
    gene_names
        Gene identifiers matching the expression head's output order.
    cell_type_names, region_names
        Label vocabularies used to decode class indices to strings.
    device
        Torch device string.  Defaults to the model's device.
    n_neighbors
        Neighbors per side; must match training.
    """

    def __init__(
        self,
        model: SpatialCPATransformer,
        gene_names: Sequence[str],
        cell_type_names: Optional[Sequence[str]] = None,
        region_names: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        n_neighbors: int = 10,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(
            device if device is not None else next(model.parameters()).device.type
        )
        self.model.to(self.device)
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.region_names = list(region_names) if region_names is not None else None
        self.k = n_neighbors

    # ------------------------------------------------------------------ #
    def _build_batch_arrays(
        self,
        target_coords: np.ndarray,   # (Q, 3)
        lower: Slice,
        upper: Slice,
        lower_tree: cKDTree,
        upper_tree: cKDTree,
    ) -> Dict[str, np.ndarray]:
        """Gather neighbor-token features for a chunk of query coordinates."""
        k = self.k
        lo_idx, _ = knn_indices(target_coords, lower_tree, k)   # (Q, k)
        up_idx, _ = knn_indices(target_coords, upper_tree, k)   # (Q, k)

        lo_coords = lower.coords_3d()
        up_coords = upper.coords_3d()

        # Expression / coords per token: (Q, 2k, ...)
        tok_expr = np.concatenate(
            [lower.expression[lo_idx], upper.expression[up_idx]], axis=1
        )
        tok_coords = np.concatenate([lo_coords[lo_idx], up_coords[up_idx]], axis=1)

        delta = tok_coords - target_coords[:, None, :]
        dist = np.linalg.norm(delta, axis=2, keepdims=True)
        tok_rel = np.concatenate([delta, dist], axis=2).astype(np.float32)

        q = target_coords.shape[0]
        side = np.concatenate(
            [np.zeros((q, k), dtype=np.int64), np.ones((q, k), dtype=np.int64)], axis=1
        )

        # Cell-type / region tokens (zeros when the slice lacks labels).
        def _labels(arr_lo, arr_up):
            lo = arr_lo[lo_idx] if arr_lo is not None else np.zeros_like(lo_idx)
            up = arr_up[up_idx] if arr_up is not None else np.zeros_like(up_idx)
            return np.concatenate([lo, up], axis=1)

        tok_ct = _labels(lower.cell_type_indices, upper.cell_type_indices)
        tok_reg = _labels(lower.region_indices, upper.region_indices)

        # Padding mask from duplicate neighbors (small slices).
        glob_lo = lo_idx
        glob_up = up_idx + lower.n_spots  # offset so lower/upper indices differ
        tok_global = np.concatenate([glob_lo, glob_up], axis=1)
        pad_mask = np.stack(
            [_duplicate_pad_mask(tok_global[i], k) for i in range(q)], axis=0
        )

        return {
            "token_expr": tok_expr.astype(np.float32),
            "token_relcoord": tok_rel,
            "token_side": side,
            "token_ct": tok_ct.astype(np.int64),
            "token_reg": tok_reg.astype(np.int64),
            "token_pad_mask": pad_mask,
        }

    @torch.no_grad()
    def predict(
        self,
        target_coords: np.ndarray,
        lower: Slice,
        upper: Slice,
        batch_size: int = 4096,
    ) -> SlicePrediction:
        """Predict all outputs at ``target_coords`` flanked by two slices."""
        target_coords = np.ascontiguousarray(target_coords, dtype=np.float32)
        q = target_coords.shape[0]
        lower_tree = cKDTree(lower.coords_3d())
        upper_tree = cKDTree(upper.coords_3d())

        expr_out, occ_out = [], []
        ct_out: List[np.ndarray] = []
        reg_out: List[np.ndarray] = []

        for start in range(0, q, batch_size):
            end = min(start + batch_size, q)
            arrays = self._build_batch_arrays(
                target_coords[start:end], lower, upper, lower_tree, upper_tree
            )
            batch = {
                k: torch.from_numpy(np.ascontiguousarray(v)).to(self.device)
                for k, v in arrays.items()
            }
            out = self.model(batch)
            expr_out.append(out["expression"].float().cpu().numpy())
            occ_out.append(torch.sigmoid(out["occupancy_logit"]).float().cpu().numpy())
            if "cell_type_logits" in out:
                ct_out.append(out["cell_type_logits"].argmax(dim=-1).cpu().numpy())
            if "region_logits" in out:
                reg_out.append(out["region_logits"].argmax(dim=-1).cpu().numpy())

        expression = np.concatenate(expr_out, axis=0)
        occupancy = np.concatenate(occ_out, axis=0)

        ct_idx = np.concatenate(ct_out, axis=0) if ct_out else None
        reg_idx = np.concatenate(reg_out, axis=0) if reg_out else None
        ct_labels = (
            np.array([self.cell_type_names[i] for i in ct_idx])
            if ct_idx is not None and self.cell_type_names is not None else None
        )
        reg_labels = (
            np.array([self.region_names[i] for i in reg_idx])
            if reg_idx is not None and self.region_names is not None else None
        )

        return SlicePrediction(
            coords=target_coords,
            expression=expression,
            occupancy_prob=occupancy,
            cell_type=ct_labels,
            region=reg_labels,
            cell_type_idx=ct_idx,
            region_idx=reg_idx,
        )

    # ------------------------------------------------------------------ #
    def predict_slice(
        self,
        target_slice: Slice,
        lower: Slice,
        upper: Slice,
        batch_size: int = 4096,
    ) -> SlicePrediction:
        """Predict at the coordinates of an existing slice."""
        return self.predict(target_slice.coords_3d(), lower, upper, batch_size)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_flanking_slices(z: float, slices: Sequence[Slice]) -> Tuple[Slice, Slice]:
        """Choose the nearest lower and upper slices for a query z.

        Falls back to the two nearest slices on the available side when ``z`` is
        outside the section range (so extrapolation still produces neighbors).
        """
        ordered = sorted(slices, key=lambda s: s.z_center)
        below = [s for s in ordered if s.z_center <= z]
        above = [s for s in ordered if s.z_center > z]
        if below and above:
            return below[-1], above[0]
        # Extrapolation (z outside the section range): use the two nearest by |Δz|.
        nearest = sorted(ordered, key=lambda s: abs(s.z_center - z))[:2]
        nearest = sorted(nearest, key=lambda s: s.z_center)
        return nearest[0], nearest[-1]

    def _make_grid(
        self,
        xy_bounds: Tuple[float, float, float, float],
        n_points: int,
        grid_type: str,
        seed: int = 0,
    ) -> np.ndarray:
        """Build ``~n_points`` XY query locations inside ``(xmin,ymin,xmax,ymax)``."""
        xmin, ymin, xmax, ymax = xy_bounds
        if grid_type == "random":
            rng = np.random.default_rng(seed)
            xs = rng.uniform(xmin, xmax, size=n_points)
            ys = rng.uniform(ymin, ymax, size=n_points)
            return np.column_stack([xs, ys]).astype(np.float32)
        # Regular square lattice with ~n_points cells.
        side = max(int(round(math.sqrt(n_points))), 1)
        xs = np.linspace(xmin, xmax, side)
        ys = np.linspace(ymin, ymax, side)
        gx, gy = np.meshgrid(xs, ys)
        return np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)

    def _transfer_expression(self, coords, cell_type_idx, source_slices,
                             k=1, same_celltype=True, seed=0):
        """Copy expression onto query points from the nearest training cells.

        Pure regression collapses cell-to-cell variance (over-smoothing). As in
        SpatialZ and the original SpatialCPA's k-NN refinement, transferring real
        expression profiles from nearby training cells restores that variance.
        For each query point we take the inverse-distance-weighted expression of
        its ``k`` nearest source cells (optionally restricted to the query's
        predicted cell type). ``k=1`` copies the single nearest profile
        (maximum variance); larger ``k`` trades variance for smoothness.

        Uses only training (source) cells, so it introduces no leakage.
        """
        src_expr = np.concatenate([s.expression for s in source_slices], axis=0)
        src_xy = np.concatenate([s.coords_xy for s in source_slices], axis=0)
        if same_celltype and all(s.cell_type_indices is not None for s in source_slices):
            src_ct = np.concatenate([s.cell_type_indices for s in source_slices], axis=0)
        else:
            src_ct = None
            same_celltype = False

        out = np.zeros((coords.shape[0], src_expr.shape[1]), dtype=np.float32)
        q_xy = coords[:, :2]

        groups = ([(-1, np.arange(coords.shape[0]))] if not same_celltype
                  else [(ct, np.where(cell_type_idx == ct)[0])
                        for ct in np.unique(cell_type_idx)])
        for ct, q_idx in groups:
            if len(q_idx) == 0:
                continue
            if same_celltype:
                s_idx = np.where(src_ct == ct)[0]
                if len(s_idx) == 0:
                    s_idx = np.arange(src_expr.shape[0])  # fallback: any cell
            else:
                s_idx = np.arange(src_expr.shape[0])
            kk = min(k, len(s_idx))
            tree = cKDTree(src_xy[s_idx])
            dist, nn = tree.query(q_xy[q_idx], k=kk)
            if kk == 1:
                dist = dist[:, None]; nn = nn[:, None]
            w = 1.0 / (dist + 1e-8)
            w /= w.sum(axis=1, keepdims=True)
            src_e = src_expr[s_idx]
            out[q_idx] = np.einsum("qk,qkg->qg", w.astype(np.float32),
                                   src_e[nn])
        return out

    def generate_virtual_slice(
        self,
        z: float,
        slices: Sequence[Slice],
        xy_bounds: Optional[Tuple[float, float, float, float]] = None,
        n_grid_points: int = 1000,
        occupancy_threshold: float = 0.5,
        grid_type: str = "regular",
        batch_size: int = 4096,
        seed: int = 0,
        expression_mode: str = "regress",
        transfer_k: int = 1,
        transfer_alpha: float = 0.0,
        transfer_same_celltype: bool = True,
    ) -> SlicePrediction:
        """Synthesise a virtual slice at an arbitrary z coordinate.

        Parameters
        ----------
        z
            The z coordinate of the virtual slice.
        slices
            The reference slices (a full stack) to interpolate between.
        xy_bounds
            ``(xmin, ymin, xmax, ymax)``.  ``None`` -> bounding box of the two
            flanking slices.
        n_grid_points
            Approximate number of grid query points.
        occupancy_threshold
            Keep only grid points whose predicted occupancy exceeds this.
        grid_type
            ``"regular"`` or ``"random"``.
        batch_size
            Inference batch size.
        seed
            RNG seed for ``grid_type="random"``.

        Returns
        -------
        SlicePrediction restricted to the accepted (occupied) grid points.
        """
        lower, upper = self._pick_flanking_slices(z, slices)

        if xy_bounds is None:
            xy = np.vstack([lower.coords_xy, upper.coords_xy])
            xy_bounds = (
                float(xy[:, 0].min()), float(xy[:, 1].min()),
                float(xy[:, 0].max()), float(xy[:, 1].max()),
            )

        grid_xy = self._make_grid(xy_bounds, n_grid_points, grid_type, seed)
        coords = np.column_stack(
            [grid_xy, np.full(grid_xy.shape[0], float(z), dtype=np.float32)]
        ).astype(np.float32)

        pred = self.predict(coords, lower, upper, batch_size)

        keep = pred.occupancy_prob > occupancy_threshold
        kept_coords = pred.coords[keep]
        expression = pred.expression[keep]
        kept_ct_idx = pred.cell_type_idx[keep] if pred.cell_type_idx is not None else None

        # Expression source: regressed (smooth), transferred (real profiles from
        # nearby training cells -> preserves variance), or a blend of the two.
        if expression_mode in ("transfer", "blend") and kept_coords.shape[0] > 0:
            transferred = self._transfer_expression(
                kept_coords, kept_ct_idx, [lower, upper],
                k=transfer_k, same_celltype=transfer_same_celltype, seed=seed)
            if expression_mode == "transfer":
                expression = transferred
            else:  # blend: alpha * regressed + (1 - alpha) * transferred
                expression = transfer_alpha * expression + (1.0 - transfer_alpha) * transferred
        elif expression_mode not in ("regress", "transfer", "blend"):
            raise ValueError(f"Unknown expression_mode '{expression_mode}'")

        return SlicePrediction(
            coords=kept_coords,
            expression=expression,
            occupancy_prob=pred.occupancy_prob[keep],
            cell_type=pred.cell_type[keep] if pred.cell_type is not None else None,
            region=pred.region[keep] if pred.region is not None else None,
            cell_type_idx=kept_ct_idx,
            region_idx=pred.region_idx[keep] if pred.region_idx is not None else None,
        )
