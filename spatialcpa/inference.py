"""
Virtual Slice Generation and Inference for SpatialCPA.

The primary inference path is :func:`interpolate_flanking` /
:meth:`VirtualSliceGenerator.generate_flanking`: a held-out slice is
reconstructed at its real cell coordinates by bidirectional z-interpolation of
inverse-distance k-NN estimates from the two flanking training sections, on the
original expression scale. The neural-field prediction, k-NN spatial refinement
(:func:`knn_refine`, :func:`knn_refine_by_celltype`), and spatial smoothing
(:func:`spatial_smooth`) remain available for the learned/blended variants.
"""

import numpy as np
import torch
import anndata as ad
import pandas as pd
from scipy.spatial import cKDTree


def spatial_smooth(expression, positions, k=20, sigma_mult=1.5):
    """
    Smooth expression using Gaussian-weighted k-NN spatial averaging.

    Parameters
    ----------
    expression : (N, G) array
    positions : (N, 2) array of xy coordinates
    k : int
        Number of nearest neighbors.
    sigma_mult : float
        Sigma = sigma_mult * median nearest-neighbor distance.

    Returns
    -------
    smoothed : (N, G) array
    """
    tree = cKDTree(positions)
    dists, indices = tree.query(positions, k=k + 1)  # +1 for self

    median_dist = np.median(dists[:, 1])
    sigma = sigma_mult * median_dist

    smoothed = np.zeros_like(expression)
    for i in range(len(positions)):
        d = dists[i]
        w = np.exp(-d ** 2 / (2 * sigma ** 2))
        w /= w.sum()
        smoothed[i] = (w[:, None] * expression[indices[i]]).sum(axis=0)

    return smoothed


def knn_refine(pred_expr, pred_xy, pred_z,
               train_expr, train_xy, train_z,
               k=30, z_weight=10.0, alpha=0.3):
    """
    Refine neural network expression predictions using k-NN interpolation
    from training data.

    For each predicted cell, finds the k nearest training cells in
    weighted 3D space (z is up-weighted to prioritise cells at similar
    z-positions) and blends their expression with the NN prediction.

    Parameters
    ----------
    pred_expr : (N, G) neural network predictions.
    pred_xy : (N, 2) predicted cell xy positions.
    pred_z : (N,) predicted cell z positions.
    train_expr : (M, G) training cell expression.
    train_xy : (M, 2) training cell xy positions.
    train_z : (M,) training cell z positions.
    k : int
        Number of nearest training neighbors.
    z_weight : float
        Weight applied to z-distances (higher = prefer same z-layer).
    alpha : float
        Blend weight: alpha * nn_pred + (1-alpha) * knn_pred.

    Returns
    -------
    refined : (N, G) refined expression predictions.
    """
    # Build weighted 3D coordinates
    pred_3d = np.hstack([pred_xy, (pred_z * z_weight).reshape(-1, 1)])
    train_3d = np.hstack([train_xy, (train_z * z_weight).reshape(-1, 1)])

    tree = cKDTree(train_3d)
    dists, indices = tree.query(pred_3d, k=k)

    # Inverse distance weighting
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)

    # Weighted average of training expression
    knn_expr = np.zeros_like(pred_expr)
    for i in range(len(pred_3d)):
        knn_expr[i] = (weights[i, :, None] * train_expr[indices[i]]).sum(axis=0)

    # Blend
    return alpha * pred_expr + (1 - alpha) * knn_expr


def knn_refine_by_celltype(pred_xy, pred_z, pred_ct,
                           train_expr, train_xy, train_z, train_ct,
                           k=30, z_weight=10.0, nn_expr=None, alpha=0.0):
    """
    Cell-type-conditioned k-NN expression interpolation.

    For each predicted cell, finds the k nearest training cells OF THE SAME
    CELL TYPE in weighted 3D space, and uses their inverse-distance-weighted
    average expression. Optionally blends with neural network predictions.

    Parameters
    ----------
    pred_xy : (N, 2) predicted cell xy positions.
    pred_z : (N,) predicted cell z positions.
    pred_ct : (N,) predicted cell type indices.
    train_expr : (M, G) training cell expression.
    train_xy : (M, 2) training cell xy positions.
    train_z : (M,) training cell z positions.
    train_ct : (M,) training cell type indices.
    k : int
        Number of nearest same-type training neighbors.
    z_weight : float
        Weight applied to z-distances.
    nn_expr : (N, G) or None
        Neural network predictions to blend with k-NN.
    alpha : float
        Blend weight: alpha * nn_expr + (1-alpha) * knn_expr.
        Set to 0.0 for pure k-NN.

    Returns
    -------
    result : (N, G) expression predictions.
    """
    n_pred = len(pred_ct)
    n_genes = train_expr.shape[1]
    result = np.zeros((n_pred, n_genes), dtype=np.float32)

    unique_ct = np.unique(pred_ct)

    for ct in unique_ct:
        pred_idx = np.where(pred_ct == ct)[0]
        train_idx = np.where(train_ct == ct)[0]

        if len(train_idx) == 0:
            # Fallback: use all training cells if no same-type cells
            train_idx = np.arange(len(train_ct))

        # Build weighted 3D coordinates
        pred_3d = np.hstack([
            pred_xy[pred_idx],
            (pred_z[pred_idx] * z_weight).reshape(-1, 1)
        ])
        train_3d = np.hstack([
            train_xy[train_idx],
            (train_z[train_idx] * z_weight).reshape(-1, 1)
        ])

        k_actual = min(k, len(train_3d))
        tree = cKDTree(train_3d)
        dists, indices = tree.query(pred_3d, k=k_actual)

        # Handle k=1 case (scalar returns)
        if k_actual == 1:
            dists = dists.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # Inverse distance weighting
        weights = 1.0 / (dists + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)

        # Weighted average of same-type training expression
        train_expr_ct = train_expr[train_idx]
        for i in range(len(pred_idx)):
            result[pred_idx[i]] = (
                weights[i, :, None] * train_expr_ct[indices[i]]
            ).sum(axis=0)

    # Optionally blend with NN predictions
    if nn_expr is not None and alpha > 0:
        result = alpha * nn_expr + (1 - alpha) * result

    return result


def _idw_section(target_xy, sec_xy, sec_expr, sec_ct, k, n_cell_types):
    """Inverse-distance-weighted k-NN estimate within a single section.

    For every target cell, find its ``k`` nearest neighbours (in xy) among the
    cells of one training section and return (a) the inverse-distance-weighted
    average expression and (b) inverse-distance-weighted class votes.

    Parameters
    ----------
    target_xy : (N, 2) query positions.
    sec_xy : (M, 2) section cell positions.
    sec_expr : (M, G) section cell expression (any scale).
    sec_ct : (M,) section cell-type indices, or None.
    k : int
        Neighbours to average.
    n_cell_types : int
        Size of the vote vector.

    Returns
    -------
    expr_est : (N, G) weighted-average expression.
    votes : (N, n_cell_types) weighted class votes (zeros if sec_ct is None).
    """
    m = len(sec_xy)
    k_actual = min(k, m)
    tree = cKDTree(sec_xy)
    dists, idx = tree.query(target_xy, k=k_actual)
    if k_actual == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)

    # Weighted average expression: (N, k, G) contracted over k.
    expr_est = np.einsum("nk,nkg->ng", weights.astype(np.float32),
                         sec_expr[idx]).astype(np.float32)

    votes = np.zeros((len(target_xy), n_cell_types), dtype=np.float32)
    if sec_ct is not None:
        neigh_ct = sec_ct[idx]  # (N, k)
        rows = np.repeat(np.arange(len(target_xy)), k_actual)
        np.add.at(votes, (rows, neigh_ct.ravel()), weights.ravel())

    return expr_est, votes


def interpolate_flanking(target_xy, target_z, sections, k=12,
                         n_cell_types=1):
    """Bidirectional z-interpolation of expression from flanking sections.

    This is the core of SpatialCPA inference at a held-out slice. Each held-out
    cell sits at a known ``(x, y, z)``; the two training sections immediately
    below and above ``z`` are located, an inverse-distance k-NN estimate is
    formed independently in each, and the two estimates are linearly blended by
    the cell's fractional z-position between the sections. The prediction is
    produced on the *same scale as the section expression passed in* — feed raw
    (original-scale) expression to match ground truth for error metrics.

    Parameters
    ----------
    target_xy : (N, 2) held-out cell positions.
    target_z : (N,) held-out cell z-coordinates.
    sections : list of dict
        Each ``{"xy": (M,2), "z": float, "expr": (M,G), "ct": (M,) or None}``.
        ``z`` is the section's characteristic (median) z-coordinate.
    k : int
        Neighbours per flanking section.
    n_cell_types : int
        Number of cell-type classes for the vote vector. When the target lies
        outside the training z-range (no section on one side), the single
        nearest section carries the estimate rather than extrapolating.

    Returns
    -------
    expr : (N, G) interpolated expression.
    cell_types : (N,) predicted cell-type indices (argmax of blended votes).
    """
    order = np.argsort([s["z"] for s in sections])
    z_centers = np.array([sections[i]["z"] for i in order], dtype=np.float64)

    zt = float(np.median(target_z))
    below = np.where(z_centers <= zt + 1e-9)[0]
    above = np.where(z_centers >= zt - 1e-9)[0]

    lo = order[below[-1]] if len(below) else order[0]
    hi = order[above[0]] if len(above) else order[-1]

    z_lo = sections[lo]["z"]
    z_hi = sections[hi]["z"]

    e_lo, v_lo = _idw_section(target_xy, sections[lo]["xy"], sections[lo]["expr"],
                              sections[lo].get("ct"), k, n_cell_types)

    if lo == hi or abs(z_hi - z_lo) < 1e-9:
        # Boundary / degenerate: a single section carries the estimate.
        expr = e_lo
        votes = v_lo
    else:
        e_hi, v_hi = _idw_section(target_xy, sections[hi]["xy"],
                                  sections[hi]["expr"], sections[hi].get("ct"),
                                  k, n_cell_types)
        # Per-cell fractional position between the two flanking sections.
        w = np.clip((np.asarray(target_z, dtype=np.float64) - z_lo)
                    / (z_hi - z_lo), 0.0, 1.0).astype(np.float32)
        wc = w[:, None]
        expr = (1.0 - wc) * e_lo + wc * e_hi
        votes = (1.0 - wc) * v_lo + wc * v_hi

    cell_types = votes.argmax(axis=1) if votes.shape[1] > 0 else \
        np.zeros(len(target_xy), dtype=np.int64)
    return expr.astype(np.float32), cell_types.astype(np.int64)


class VirtualSliceGenerator:
    """
    Generate virtual tissue slices at arbitrary z-positions.

    Combines neural network predictions with k-NN spatial refinement
    from training data for high-fidelity expression reconstruction.

    Parameters
    ----------
    model : SpatialCPA
        Trained SpatialCPA model.
    cell_type_names : list of str
    gene_names : list of str
    region_names : list of str or None
    device : str
    train_sections : list of SpatialSection or None
        Training sections for k-NN refinement. If None, no refinement.
    """

    def __init__(self, model, cell_type_names, gene_names,
                 region_names=None, device='cpu', train_sections=None):
        self.model = model.to(device)
        self.model.eval()
        self.cell_type_names = cell_type_names
        self.gene_names = gene_names
        self.region_names = region_names
        self.device = device

        # Pre-build training data arrays for k-NN refinement
        self.train_expr = None
        self.train_xy = None
        self.train_z = None
        self.train_ct = None
        if train_sections is not None:
            all_expr = []
            all_xy = []
            all_z = []
            all_ct = []
            for sec in train_sections:
                all_expr.append(sec.expression)
                all_xy.append(sec.coords_xy)
                all_z.append(sec.z_values)
                all_ct.append(sec.cell_type_indices)
            self.train_expr = np.concatenate(all_expr, axis=0)
            self.train_xy = np.concatenate(all_xy, axis=0)
            self.train_z = np.concatenate(all_z, axis=0)
            self.train_ct = np.concatenate(all_ct, axis=0)
        # Keep the per-section structure for flanking z-interpolation.
        self.train_sections = train_sections

    @torch.no_grad()
    def generate_flanking(self, reference_adata, interp_expr_list=None,
                          k=12, nn_expr_blend=0.0, batch_size=4096):
        """Generate a held-out slice by bidirectional flanking interpolation.

        This is the recommended inference path for SpatialCPA on the virtual
        slice benchmark. Expression at each reference cell is estimated by
        linearly interpolating inverse-distance k-NN estimates from the two
        training sections flanking the slice in z, and cell types are assigned
        by the spatially-weighted vote of the same neighbours. Because the
        estimate is produced on the scale of the supplied section expression,
        pass raw / original-scale expression (``interp_expr_list``) so the
        prediction matches the ground truth for error-based metrics.

        Parameters
        ----------
        reference_adata : AnnData
            Reference slice supplying ``obsm['spatial']`` (or obs x/y) and
            ``obs['z']``.
        interp_expr_list : list of (M_s, G) arrays or None
            Per-section expression, aligned with ``self.train_sections``. When
            None, the sections' own ``.expression`` is used.
        k : int
            Neighbours per flanking section.
        nn_expr_blend : float
            Optional blend with the neural field's expression prediction:
            ``blend * nn + (1 - blend) * interpolation``. Off by default; the
            neural field lives on a different (normalised) scale, so blending is
            only meaningful when the model was trained on the same scale.
        batch_size : int
            Batch size for the (optional) neural expression pass.

        Returns
        -------
        virtual_adata : AnnData
        """
        if self.train_sections is None:
            raise ValueError("generate_flanking requires train_sections")

        if 'spatial' in reference_adata.obsm:
            positions = np.asarray(reference_adata.obsm['spatial'])[:, :2]
        elif 'x' in reference_adata.obs.columns:
            positions = np.column_stack([
                reference_adata.obs['x'].values.astype(np.float32),
                reference_adata.obs['y'].values.astype(np.float32),
            ])
        else:
            raise ValueError("Cannot find spatial coordinates in reference")
        positions = positions.astype(np.float32)
        z_values = reference_adata.obs['z'].values.astype(np.float32)

        if interp_expr_list is None:
            interp_expr_list = [s.expression for s in self.train_sections]

        sections = []
        for sec, expr in zip(self.train_sections, interp_expr_list):
            sections.append({
                "xy": np.asarray(sec.coords_xy, dtype=np.float32),
                "z": float(sec.z_center),
                "expr": np.asarray(expr, dtype=np.float32),
                "ct": np.asarray(sec.cell_type_indices, dtype=np.int64),
            })

        n_cell_types = len(self.cell_type_names)
        expr_all, cell_types = interpolate_flanking(
            positions, z_values, sections, k=k, n_cell_types=n_cell_types)

        if nn_expr_blend > 0:
            coords_3d = np.hstack([positions, z_values.reshape(-1, 1)]) \
                .astype(np.float32)
            nn_expr = []
            for start in range(0, len(positions), batch_size):
                end = min(start + batch_size, len(positions))
                bc = torch.tensor(coords_3d[start:end], device=self.device)
                bct = torch.tensor(cell_types[start:end], dtype=torch.long,
                                   device=self.device)
                nn_expr.append(
                    self.model.predict_expression(bc, bct).cpu().numpy())
            nn_expr = np.concatenate(nn_expr, axis=0)
            expr_all = nn_expr_blend * nn_expr + (1 - nn_expr_blend) * expr_all

        expr_all = np.clip(expr_all, 0, None).astype(np.float32)
        cell_type_labels = [self.cell_type_names[i] for i in cell_types]

        obs = pd.DataFrame({
            'cell_class': cell_type_labels,
            'cell_type_idx': cell_types,
        })
        obs.index = reference_adata.obs.index.copy()
        virtual_adata = ad.AnnData(X=expr_all, obs=obs,
                                   var=reference_adata.var.copy())
        virtual_adata.obsm['spatial'] = positions
        return virtual_adata

    @torch.no_grad()
    def generate_matching(self, reference_adata, cell_type_key='cell_class',
                          true_cell_types=None,
                          knn_k=30, knn_z_weight=10.0, knn_alpha=0.3,
                          smooth_k=0, smooth_sigma=1.5,
                          batch_size=4096):
        """
        Generate a virtual slice matching a reference slice's cell positions.

        Pipeline:
        1. Neural network predicts expression at each cell's (x, y, z)
        2. k-NN refinement blends NN prediction with local training data
        3. Spatial smoothing enforces spatial coherence

        Parameters
        ----------
        reference_adata : AnnData
            Reference slice with spatial coordinates and z values.
        cell_type_key : str
            Key for cell type labels in obs.
        true_cell_types : array of int or None
            If provided, use these cell type indices for expression conditioning.
        knn_k : int
            Number of training neighbors for k-NN refinement.
            Set to 0 to disable refinement.
        knn_z_weight : float
            Z-distance weight for k-NN search.
        knn_alpha : float
            Blend: alpha * nn + (1-alpha) * knn. Lower = more knn.
        smooth_k : int
            If > 0, apply spatial smoothing with this many neighbors.
        smooth_sigma : float
            Sigma multiplier for spatial smoothing.
        batch_size : int
            Inference batch size.

        Returns
        -------
        virtual_adata : AnnData
        """
        # Get reference positions (x, y)
        if 'spatial' in reference_adata.obsm:
            positions = np.asarray(reference_adata.obsm['spatial'])[:, :2]
        elif 'x' in reference_adata.obs.columns:
            positions = np.column_stack([
                reference_adata.obs['x'].values.astype(np.float32),
                reference_adata.obs['y'].values.astype(np.float32),
            ])
        else:
            raise ValueError("Cannot find spatial coordinates in reference")

        z_values = reference_adata.obs['z'].values.astype(np.float32)

        n = len(positions)
        coords_3d = np.hstack([
            positions,
            z_values.reshape(-1, 1),
        ]).astype(np.float32)

        # Determine cell types
        if true_cell_types is not None:
            cell_types = np.asarray(true_cell_types, dtype=np.int64)
        else:
            all_ct_probs = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_coords = torch.tensor(coords_3d[start:end],
                                            device=self.device)
                ct_probs = self.model.predict_cell_type(batch_coords)
                all_ct_probs.append(ct_probs.cpu().numpy())
            ct_probs_all = np.concatenate(all_ct_probs, axis=0)
            cell_types = ct_probs_all.argmax(axis=1)

        cell_type_labels = [self.cell_type_names[i] for i in cell_types]

        # Step 1: Neural network expression prediction
        all_expr = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_coords = torch.tensor(coords_3d[start:end],
                                        device=self.device)
            batch_ct = torch.tensor(cell_types[start:end],
                                    dtype=torch.long, device=self.device)
            expr = self.model.predict_expression(
                batch_coords, batch_ct
            ).cpu().numpy()
            all_expr.append(expr)

        expr_all = np.concatenate(all_expr, axis=0)

        # Step 2: k-NN refinement from training data
        if knn_k > 0 and self.train_expr is not None:
            if self.train_ct is not None:
                # Cell-type-conditioned k-NN: only match same cell type
                expr_all = knn_refine_by_celltype(
                    pred_xy=positions,
                    pred_z=z_values,
                    pred_ct=cell_types,
                    train_expr=self.train_expr,
                    train_xy=self.train_xy,
                    train_z=self.train_z,
                    train_ct=self.train_ct,
                    k=knn_k,
                    z_weight=knn_z_weight,
                    nn_expr=expr_all,
                    alpha=knn_alpha,
                )
            else:
                expr_all = knn_refine(
                    pred_expr=expr_all,
                    pred_xy=positions,
                    pred_z=z_values,
                    train_expr=self.train_expr,
                    train_xy=self.train_xy,
                    train_z=self.train_z,
                    k=knn_k,
                    z_weight=knn_z_weight,
                    alpha=knn_alpha,
                )

        # Step 3: Spatial smoothing
        if smooth_k > 0:
            expr_all = spatial_smooth(expr_all, positions, k=smooth_k,
                                      sigma_mult=smooth_sigma)

        # Build AnnData
        obs = pd.DataFrame({
            'cell_class': cell_type_labels,
            'cell_type_idx': cell_types,
        })
        obs.index = reference_adata.obs.index.copy()

        virtual_adata = ad.AnnData(
            X=expr_all.astype(np.float32),
            obs=obs,
            var=reference_adata.var.copy(),
        )
        virtual_adata.obsm['spatial'] = positions.astype(np.float32)

        return virtual_adata
