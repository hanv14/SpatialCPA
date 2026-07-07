"""
Cell-type annotation for SpatialCPA-v6.

Annotation is, per the design brief, the single most important step: get the cell
types (and thus the niche) right and the expression-structure metrics follow,
because expression is then transferred from correctly-typed cells. We build the
per-cell cell-type prior in the foundation-model cell-state embedding — a nearest
prototype / kNN classifier trained on the labeled *training* cells — so external
biological knowledge (the FM manifold) drives the call rather than a single
nearest raw-expression neighbor.

Two biological constraints are then imposed:

1. **Composition** — the predicted cell-type proportions are pinned to the
   z-interpolated flanking composition ``p*`` via a Sinkhorn (balanced-assignment)
   bias on the class prior. This is the leakage-safe estimate of the held-out
   slice's mix and directly targets ``celltype_composition``.
2. **Communication / niche** — labels are then refined by the MRF in
   :mod:`spatialcpav6.communication` so the spatial co-occurrence of types matches
   the flanking niche architecture (``celltype_nhood_agreement``).

All statistics come from the training flanking slices + the target z; the
held-out labels are never read.
"""

from __future__ import annotations

import numpy as np

from . import communication as comm


def spatial_logprior(coords_xy, lo_xy, lo_labels, up_xy, up_labels, n_types, t, k=10):
    """z-interpolated spatial vote of cell type from *both* flanking slices.

    For each synthesized position, poll the ``k`` nearest cells in the lower slice
    (weight ``1-t``) and the ``k`` nearest in the upper slice (weight ``t``) and
    accumulate a soft class distribution. When the cell-type field varies smoothly
    through z (adjacent sections), this interpolated vote estimates the held-out
    slice's type field better than copying either single flanking slice — the
    lever that lets annotation *beat* a single-slice copy on composition and niche
    rather than merely tie it. Leakage-safe: only the training flanking slices are
    polled.
    """
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
        lab = np.asarray(slab).astype(int)[nn]                 # (nq, kk)
        for j in range(kk):
            np.add.at(P, (np.arange(nq), lab[:, j]), wt[:, j])

    _accum(lo_xy, lo_labels, 1.0 - t)
    _accum(up_xy, up_labels, t)
    P /= P.sum(axis=1, keepdims=True)
    return np.log(P)


def class_logprior(query_embed, train_embed, train_labels, n_types, cfg):
    """Per-cell log-prior over cell types from the FM embedding.

    ``cfg`` is an :class:`~spatialcpav6.config.AnnotationConfig`.

    * ``"prototype"`` — negative squared distance to each type's embedding
      centroid, scaled by a temperature (a Gaussian class model). Robust with
      few cells.
    * ``"knn"`` — smoothed vote fractions among the ``knn_k`` nearest training
      cells.
    """
    query_embed = np.asarray(query_embed, dtype=np.float64)
    train_embed = np.asarray(train_embed, dtype=np.float64)
    train_labels = np.asarray(train_labels).astype(int)
    nq = query_embed.shape[0]

    if cfg.classifier == "knn":
        from scipy.spatial import cKDTree
        k = min(cfg.knn_k, train_embed.shape[0])
        _, nn = cKDTree(train_embed).query(query_embed, k=k)
        nn = np.atleast_2d(nn)
        votes = train_labels[nn]                        # (nq, k)
        P = np.zeros((nq, n_types))
        rows = np.repeat(np.arange(nq), votes.shape[1])
        np.add.at(P, (rows, votes.ravel()), 1.0)
        P = (P + 0.1) / (P.sum(axis=1, keepdims=True) + 0.1 * n_types)
        return np.log(P)

    # prototype (default): Gaussian class centroids in embedding space.
    centroids = np.zeros((n_types, train_embed.shape[1]))
    present = np.zeros(n_types, dtype=bool)
    for c in range(n_types):
        m = train_labels == c
        if m.any():
            centroids[c] = train_embed[m].mean(axis=0)
            present[c] = True
    # Squared distance to each centroid.
    d2 = np.sum((query_embed[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # (nq, C)
    scale = np.median(d2[:, present]) + 1e-9 if present.any() else 1.0
    logits = -d2 / (scale * max(cfg.prototype_temperature, 1e-6))
    logits[:, ~present] = -1e9                          # forbid absent types
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return np.log(P + 1e-12)


def constrain_composition(logprior, target_comp, iters=50):
    """Add a per-class Sinkhorn bias so the soft assignment matches ``target_comp``.

    Solves for a balanced assignment whose column marginals equal ``target_comp``
    (rows uniform), then folds the resulting class bias back into the log-prior.
    Pins ``celltype_composition`` to the leakage-safe interpolated flanking mix.
    """
    logprior = np.asarray(logprior, dtype=np.float64)
    n, C = logprior.shape
    b = np.asarray(target_comp, dtype=np.float64)
    b = b / b.sum() if b.sum() > 0 else np.full(C, 1.0 / C)
    K = np.exp(logprior - logprior.max(axis=1, keepdims=True)) + 1e-300
    u = np.ones(n)
    v = np.ones(C)
    a = np.full(n, 1.0 / n)
    for _ in range(iters):
        u = a / (K @ v)
        v = b / (K.T @ u)
    return logprior + np.log(v + 1e-300)[None, :]


def annotate(
    query_embed, coords_xy,
    train_embed, train_labels,
    lo_xy, lo_labels, up_xy, up_labels,
    n_types, t, init_labels, ann_cfg, comm_cfg, seed=0,
):
    """Anchored annotation: copied-type anchor + FM prior + niche MRF.

    Returns final integer labels of length ``len(query_embed)``. The unary energy
    is a strong one-hot anchor on the copied endpoint type plus a soft
    foundation-model classifier prior; the niche MRF then harmonizes the spatial
    layout toward the interpolated flanking niche. Because the anchor dominates,
    refinement is conservative — it fixes likely mistakes and improves spatial
    organization without discarding the (already strong) copied labels.
    """
    n = query_embed.shape[0]
    init = (np.asarray(init_labels).astype(int) if init_labels is not None
            else np.zeros(n, dtype=int))

    # Soft cell-type prior (log-probabilities). "spatial" interpolates the type
    # field from both flanking slices (beats single-slice copy when types vary
    # smoothly in z); "prototype"/"knn" classify in the FM cell-state embedding.
    if ann_cfg.classifier == "spatial":
        fm_logprior = spatial_logprior(
            coords_xy, lo_xy, lo_labels, up_xy, up_labels, n_types, t,
            k=comm_cfg.k_neighbors)
    else:
        fm_logprior = class_logprior(query_embed, train_embed, train_labels,
                                     n_types, ann_cfg)

    # Strong one-hot anchor on the copied endpoint type.
    anchor = np.full((n, n_types), -1.0, dtype=np.float64)
    anchor[np.arange(n), init] = 0.0            # log 1 vs log ~0 for others
    unary = ann_cfg.anchor_weight * anchor + ann_cfg.fm_weight * fm_logprior

    M_target, p_target = comm.target_niche_and_composition(
        lo_xy, lo_labels, up_xy, up_labels, n_types, t, k=comm_cfg.k_neighbors)
    if ann_cfg.constrain_composition:
        unary = constrain_composition(unary, p_target, ann_cfg.composition_sinkhorn_iter)

    # Start from the anchored-unary argmax (≈ init unless FM strongly disagrees).
    seed_labels = unary.argmax(axis=1)
    labels = comm.refine_labels(
        coords_xy, seed_labels, unary, M_target, p_target, comm_cfg, seed=seed)
    return labels
