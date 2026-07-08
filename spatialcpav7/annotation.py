"""
Cell-type annotation for SpatialCPA-v7.

Annotation is, per the design brief, the single most important step: get the cell
types (and thus the niche) right and the expression-structure metrics follow,
because expression is transferred from correctly-typed cells. v7 builds the
per-cell cell-type prior in the foundation-model cell-state embedding, then
imposes three biological constraints and refines with the 2D + 3D communication
MRF:

1. **FM cell-state prior** — ``spatial`` (interpolated flanking type field),
   ``labelprop`` (manifold label propagation on the FM embedding, new in v7),
   or ``prototype`` / ``knn`` embedding classifiers.
2. **Composition** — the predicted proportions are pinned to the z-interpolated
   flanking mix ``p*`` (on by default in v7), the leakage-safe estimate of the
   held-out mix; targets ``celltype_composition``.
3. **Communication / niche** — labels are refined by the MRF in
   :mod:`spatialcpav7.communication` so the synthesized slice reproduces both the
   in-plane niche architecture ``M2D*`` *and* the cross-slice (z-stacking) niche
   ``M3D`` implied by the real flanking cells above/below each virtual cell
   (``celltype_nhood_agreement`` in 2D and the genuine 3D communication signal).

All statistics come from the training flanking slices + the target z; the
held-out labels are never read.
"""

from __future__ import annotations

import numpy as np

from . import communication as comm
from .data import cross_slice_neighbor_types


def spatial_logprior(coords_xy, lo_xy, lo_labels, up_xy, up_labels, n_types, t, k=10):
    """z-interpolated spatial vote of cell type from *both* flanking slices."""
    from scipy.spatial import cKDTree
    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    nq = coords_xy.shape[0]
    P = np.full((nq, n_types), 1e-6, dtype=np.float64)

    def _accum(sxy, slab, w):
        if sxy is None or len(sxy) == 0 or w <= 0:
            return
        kk = min(k, len(sxy))
        d, nn = cKDTree(np.asarray(sxy, dtype=np.float64)).query(coords_xy, k=kk)
        nn = np.atleast_2d(nn); d = np.atleast_2d(d)
        wt = w / (d + 1e-8)
        lab = np.asarray(slab).astype(int)[nn]
        for j in range(kk):
            np.add.at(P, (np.arange(nq), lab[:, j]), wt[:, j])

    _accum(lo_xy, lo_labels, 1.0 - t)
    _accum(up_xy, up_labels, t)
    P /= P.sum(axis=1, keepdims=True)
    return np.log(P)


def class_logprior(query_embed, train_embed, train_labels, n_types, cfg):
    """Per-cell log-prior over cell types from the FM embedding (prototype/knn)."""
    query_embed = np.asarray(query_embed, dtype=np.float64)
    train_embed = np.asarray(train_embed, dtype=np.float64)
    train_labels = np.asarray(train_labels).astype(int)
    nq = query_embed.shape[0]

    if cfg.classifier == "knn":
        from scipy.spatial import cKDTree
        k = min(cfg.knn_k, train_embed.shape[0])
        _, nn = cKDTree(train_embed).query(query_embed, k=k)
        nn = np.atleast_2d(nn)
        votes = train_labels[nn]
        P = np.zeros((nq, n_types))
        rows = np.repeat(np.arange(nq), votes.shape[1])
        np.add.at(P, (rows, votes.ravel()), 1.0)
        P = (P + 0.1) / (P.sum(axis=1, keepdims=True) + 0.1 * n_types)
        return np.log(P)

    centroids = np.zeros((n_types, train_embed.shape[1]))
    present = np.zeros(n_types, dtype=bool)
    for c in range(n_types):
        m = train_labels == c
        if m.any():
            centroids[c] = train_embed[m].mean(axis=0)
            present[c] = True
    d2 = np.sum((query_embed[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    scale = np.median(d2[:, present]) + 1e-9 if present.any() else 1.0
    logits = -d2 / (scale * max(cfg.prototype_temperature, 1e-6))
    logits[:, ~present] = -1e9
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return np.log(P + 1e-12)


def labelprop_logprior(query_embed, train_embed, train_labels, n_types, cfg):
    """Manifold label propagation in the FM embedding (new in v7).

    Builds a symmetric kNN affinity graph over ``[train ∪ query]`` cells in the
    foundation-model cell-state space, clamps the labelled training rows to their
    one-hot label, and diffuses labels to the query rows (Zhou et al. 2004
    label-spreading). The result is a manifold-aware posterior that respects the
    global geometry of the cell-state space rather than a single local vote — more
    robust when a query cell sits between prototypes. Falls back to prototype on
    any numerical issue.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix, diags, eye
    try:
        Q = np.asarray(query_embed, dtype=np.float64)
        Tr = np.asarray(train_embed, dtype=np.float64)
        y = np.asarray(train_labels).astype(int)
        nq, nt = Q.shape[0], Tr.shape[0]
        if nq == 0 or nt == 0:
            raise ValueError("empty set")
        Z = np.vstack([Tr, Q])
        n = Z.shape[0]
        k = min(cfg.labelprop_k, n - 1)
        d, nn = cKDTree(Z).query(Z, k=k + 1)
        d, nn = d[:, 1:], nn[:, 1:]
        sigma = np.median(d) + 1e-9
        wts = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        rows = np.repeat(np.arange(n), k)
        cols = nn.ravel()
        vals = wts.ravel()
        W = csr_matrix((vals, (rows, cols)), shape=(n, n))
        W = W.maximum(W.T)                                  # symmetrize
        deg = np.asarray(W.sum(axis=1)).ravel()
        dinv = np.zeros_like(deg); nz = deg > 0; dinv[nz] = 1.0 / np.sqrt(deg[nz])
        Dinv = diags(dinv)
        S = Dinv @ W @ Dinv                                 # normalized affinity
        Y0 = np.zeros((n, n_types))
        Y0[np.arange(nt), y] = 1.0                          # clamp training labels
        F = Y0.copy()
        a = cfg.labelprop_alpha
        for _ in range(cfg.labelprop_iter):
            F = a * (S @ F) + (1.0 - a) * Y0
        Fq = F[nt:]
        Fq = np.clip(Fq, 0, None)
        rs = Fq.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        return np.log(Fq / rs + 1e-12)
    except Exception as e:  # pragma: no cover
        print(f"[spatialcpav7.annotation] label propagation failed ({e}); "
              f"using prototype prior.")
        return class_logprior(query_embed, train_embed, train_labels, n_types,
                              _as_prototype(cfg))


class _as_prototype:
    """Adapter so ``class_logprior`` runs in prototype mode as a fallback."""
    def __init__(self, cfg):
        self.classifier = "prototype"
        self.knn_k = cfg.knn_k
        self.prototype_temperature = cfg.prototype_temperature


def constrain_composition(logprior, target_comp, iters=50):
    """Sinkhorn class bias so the soft assignment matches ``target_comp``."""
    logprior = np.asarray(logprior, dtype=np.float64)
    n, C = logprior.shape
    b = np.asarray(target_comp, dtype=np.float64)
    b = b / b.sum() if b.sum() > 0 else np.full(C, 1.0 / C)
    K = np.exp(logprior - logprior.max(axis=1, keepdims=True)) + 1e-300
    u = np.ones(n); v = np.ones(C); a = np.full(n, 1.0 / n)
    for _ in range(iters):
        u = a / (K @ v)
        v = b / (K.T @ u)
    return logprior + np.log(v + 1e-300)[None, :]


def annotate(
    query_embed, coords_xy,
    train_embed, train_labels,
    lo_xy, lo_labels, up_xy, up_labels,
    lo_expr, up_expr, gene_names, lr_pairs,
    n_types, t, init_labels, ann_cfg, comm_cfg, seed=0,
):
    """Anchored annotation: copied-type anchor + FM prior + 2D/3D communication MRF.

    Returns final integer labels of length ``len(query_embed)``.
    """
    n = query_embed.shape[0]
    init = (np.asarray(init_labels).astype(int) if init_labels is not None
            else np.zeros(n, dtype=int))

    # Soft cell-type prior (log-probabilities).
    if ann_cfg.classifier == "spatial":
        fm_logprior = spatial_logprior(
            coords_xy, lo_xy, lo_labels, up_xy, up_labels, n_types, t,
            k=comm_cfg.k_neighbors)
    elif ann_cfg.classifier == "labelprop":
        fm_logprior = labelprop_logprior(query_embed, train_embed, train_labels,
                                         n_types, ann_cfg)
    else:
        fm_logprior = class_logprior(query_embed, train_embed, train_labels,
                                     n_types, ann_cfg)

    # Strong one-hot anchor on the copied endpoint type.
    anchor = np.full((n, n_types), -1.0, dtype=np.float64)
    anchor[np.arange(n), init] = 0.0
    unary = ann_cfg.anchor_weight * anchor + ann_cfg.fm_weight * fm_logprior

    M2d, p_target = comm.target_niche_and_composition(
        lo_xy, lo_labels, up_xy, up_labels, n_types, t, k=comm_cfg.k_neighbors)
    if ann_cfg.constrain_composition:
        unary = constrain_composition(unary, p_target, ann_cfg.composition_sinkhorn_iter)

    # 3D cross-slice communication material.
    M3d = None
    H3d = None
    lr_aff = None
    if comm_cfg.enable_3d:
        M3d = comm.cross_slice_niche_matrix(
            lo_xy, lo_labels, up_xy, up_labels, n_types, k=comm_cfg.cross_k)
        H3d = cross_slice_neighbor_types(
            coords_xy, lo_xy, lo_labels, n_types, comm_cfg.cross_k, weight=1.0 - t)
        H3d += cross_slice_neighbor_types(
            coords_xy, up_xy, up_labels, n_types, comm_cfg.cross_k, weight=t)
        if comm_cfg.lr_weight > 0 and lr_pairs and lo_expr is not None:
            src_expr = np.concatenate([lo_expr, up_expr], axis=0)
            src_lab = np.concatenate([np.asarray(lo_labels).astype(int),
                                      np.asarray(up_labels).astype(int)])
            lr_aff = comm.lr_affinity_matrix(
                src_expr, src_lab, gene_names, n_types, lr_pairs,
                min_pairs=comm_cfg.lr_min_pairs)

    seed_labels = unary.argmax(axis=1)
    labels = comm.refine_labels_3d(
        coords_xy, seed_labels, unary, M2d, p_target,
        H3d, M3d, lr_aff, comm_cfg, seed=seed)
    return labels
