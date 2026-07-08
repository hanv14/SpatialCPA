"""
Cell-cell communication (niche) model for SpatialCPA-v7 — in 2D *and* 3D.

Getting the cell-type *mix* right is not enough — a faithful slice must also put
the right types *next to* each other. That spatial organization is the readout of
cell-cell communication, and the benchmark scores it as
``celltype_nhood_agreement`` (agreement of the neighbourhood-enrichment matrices
``P(neighbour = j | centre = i)``).

v6 modelled this in-plane only, against the interpolated 2D niche matrix
``M2D* = (1-t)·M_lo + t·M_hi``. v7 adds the genuinely three-dimensional signal a
virtual slice actually sits in:

* **3D cross-slice niche** (``cross_slice_niche_matrix``). A virtual cell's real
  neighbours are not only the other virtual cells in its plane, but the cells of
  the flanking slices directly above and below it. From the two training slices
  we estimate ``M3D``[i, j] = P(a cell of type j sits across-z from a cell of
  type i) — how cell types *stack through z* — and reward each virtual cell the
  type that best explains the real flanking types stacked over it. This is a
  fixed per-cell potential (the flanking labels are real and fixed), so it is
  both stable and cheap.

* **Ligand-receptor flux prior** (``lr_affinity_matrix``). Two types that
  co-locate because they signal should be scored by their molecular capacity to
  do so. Using a curated ligand-receptor pair list intersected with the panel,
  ``A``[i, j] measures how strongly type i's ligands and type j's receptors are
  co-expressed. A gentle biological nudge on top of the data-driven niche
  matrices; disabled automatically when too few LR pairs match the panel.

Both refine only the label channel and use the training flanking slices + target
z only, so they are leakage-safe and cannot hurt the expression-structure metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .data import cross_slice_neighbor_types


def neighborhood_enrichment(xy, labels, n_types, k=10):
    """Row-normalized ``P(neighbour = j | centre = i)`` (matches the evaluator)."""
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    M = np.zeros((n_types, n_types), dtype=np.float64)
    if n < 2:
        return M
    kk = min(k + 1, n)
    _, nn = cKDTree(xy).query(xy, k=kk)
    nn = np.atleast_2d(nn)[:, 1:]
    neigh_t = labels[nn]
    for ti in range(n_types):
        rows = neigh_t[labels == ti]
        if rows.size:
            counts = np.bincount(rows.ravel(), minlength=n_types).astype(float)
            s = counts.sum()
            if s > 0:
                M[ti] = counts / s
    return M


def composition(labels, n_types):
    labels = np.asarray(labels).astype(int)
    if len(labels) == 0:
        return np.full(n_types, 1.0 / n_types)
    c = np.bincount(labels, minlength=n_types).astype(float)
    s = c.sum()
    return c / s if s > 0 else np.full(n_types, 1.0 / n_types)


def target_niche_and_composition(lo_xy, lo_labels, up_xy, up_labels, n_types, t, k=10):
    """z-interpolated flanking 2D niche matrix ``M2D*`` and composition ``p*``."""
    M_lo = neighborhood_enrichment(lo_xy, lo_labels, n_types, k)
    M_hi = neighborhood_enrichment(up_xy, up_labels, n_types, k)
    M = (1.0 - t) * M_lo + t * M_hi
    p_lo = composition(lo_labels, n_types)
    p_hi = composition(up_labels, n_types)
    p = (1.0 - t) * p_lo + t * p_hi
    return M, p


def cross_slice_niche_matrix(lo_xy, lo_labels, up_xy, up_labels, n_types, k=8):
    """Cross-slice (z-stacking) niche matrix ``M3D``[i, j] = P(across-z j | i).

    Built from the observed lower<->upper adjacency: for every lower cell, its
    ``k`` nearest upper cells contribute (lo_type, up_type) counts, and vice
    versa (symmetrized). Row-normalized. Encodes how types are layered through z —
    the leakage-safe estimate of the held-out slice's vertical neighbourhood,
    derived from the flanking training slices only.
    """
    lo_labels = np.asarray(lo_labels).astype(int)
    up_labels = np.asarray(up_labels).astype(int)
    M = np.zeros((n_types, n_types), dtype=np.float64)
    if len(lo_xy) == 0 or len(up_xy) == 0:
        return M

    def _accum(src_xy, src_lab, dst_xy, dst_lab):
        kk = min(k, len(dst_xy))
        _, nn = cKDTree(np.asarray(dst_xy, dtype=np.float64)).query(
            np.asarray(src_xy, dtype=np.float64), k=kk)
        nn = np.atleast_2d(nn)
        for ci in range(nn.shape[0]):
            ti = src_lab[ci]
            counts = np.bincount(dst_lab[nn[ci]], minlength=n_types)
            M[ti] += counts

    _accum(lo_xy, lo_labels, up_xy, up_labels)
    _accum(up_xy, up_labels, lo_xy, lo_labels)
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs


def lr_affinity_matrix(expr, labels, gene_names, n_types, lr_pairs, min_pairs=3):
    """Type x type ligand-receptor communication affinity ``A``[i, j].

    ``A``[i, j] = mean over matched LR pairs of (mean ligand expr in type i) ×
    (mean receptor expr in type j), z-scored across the matrix. Returns ``None``
    when fewer than ``min_pairs`` LR pairs are present in the panel (so the caller
    can disable the term rather than trust a degenerate estimate).
    """
    gidx = {str(g).upper(): i for i, g in enumerate(gene_names)}
    matched = [(gidx[l.upper()], gidx[r.upper()])
               for l, r in lr_pairs
               if l.upper() in gidx and r.upper() in gidx]
    if len(matched) < min_pairs:
        return None
    labels = np.asarray(labels).astype(int)
    expr = np.asarray(expr, dtype=np.float64)
    # mean expression per type
    mean_by_type = np.zeros((n_types, expr.shape[1]))
    for c in range(n_types):
        m = labels == c
        if m.any():
            mean_by_type[c] = expr[m].mean(axis=0)
    A = np.zeros((n_types, n_types))
    for li, ri in matched:
        A += np.outer(mean_by_type[:, li], mean_by_type[:, ri])
    A /= len(matched)
    s = A.std()
    if s > 0:
        A = (A - A.mean()) / s
    return A


def refine_labels_3d(
    xy, init_labels, class_logprior, M2d_target, p_target,
    H3d, M3d_target, lr_affinity, cfg, seed=0,
):
    """Refine labels by ICM on a niche MRF with a 2D term and a fixed 3D term.

    For each cell ``i`` choosing label ``c``, the score is::

        prior_weight    · logprior_i[c]                          (FM cell-state prior)
      + niche_weight    · Σ_j H2d_i[j] · log M2D*[c, j]           (in-plane niche, iterated)
      + niche3d_weight  · Σ_j H3d_i[j] · log M3D*[c, j]           (cross-slice niche, fixed)
      + lr_weight       · Σ_j H3d_i[j] · A[c, j]                  (LR flux, fixed, optional)
      + composition_weight · log p*[c]                            (target mix)

    ``H2d_i`` (the in-plane neighbour histogram) depends on the current labels and
    is recomputed each sweep; ``H3d_i`` (the real flanking-slice neighbour
    histogram) is fixed, so the 3D niche + LR terms are a stable per-cell unary
    potential computed once. ``cfg`` is a :class:`CommunicationConfig`.
    """
    xy = np.asarray(xy, dtype=np.float64)
    labels = np.asarray(init_labels).astype(int).copy()
    n = len(labels)
    n_types = M2d_target.shape[0]
    if n < 2 or n_types < 2 or not cfg.enabled:
        return labels

    kk = min(cfg.k_neighbors + 1, n)
    _, nn = cKDTree(xy).query(xy, k=kk)
    nn = np.atleast_2d(nn)[:, 1:]

    logM2d = np.log(np.asarray(M2d_target) + 1e-6)          # (C, C)
    logp = np.log(np.asarray(p_target) + 1e-6)              # (C,)
    logprior = np.asarray(class_logprior, dtype=np.float64)  # (n, C)

    # Fixed 3D unary: reward label c by how well M3D[c, ·] explains the real
    # flanking neighbour types stacked over cell i (H3d_i), plus LR flux.
    unary3d = np.zeros((n, n_types), dtype=np.float64)
    if cfg.enable_3d and H3d is not None and M3d_target is not None:
        H3dn = H3d / np.maximum(H3d.sum(axis=1, keepdims=True), 1e-9)
        logM3d = np.log(np.asarray(M3d_target) + 1e-6)      # (C, C)
        unary3d += cfg.niche3d_weight * (H3dn @ logM3d.T)   # (n, C)
        if cfg.lr_weight > 0 and lr_affinity is not None:
            unary3d += cfg.lr_weight * (H3dn @ np.asarray(lr_affinity).T)

    base = (cfg.prior_weight * logprior
            + cfg.composition_weight * logp[None, :]
            + unary3d)

    for _ in range(cfg.n_sweeps):
        neigh_t = labels[nn]
        H = np.zeros((n, n_types), dtype=np.float64)
        rows = np.repeat(np.arange(n), neigh_t.shape[1])
        np.add.at(H, (rows, neigh_t.ravel()), 1.0)
        H /= np.maximum(H.sum(axis=1, keepdims=True), 1.0)

        niche_score = H @ logM2d.T
        score = base + cfg.niche_weight * niche_score
        new = score.argmax(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels
