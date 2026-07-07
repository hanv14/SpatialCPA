"""
Cell-cell-communication (niche) model for SpatialCPA-v6.

Getting the cell-type *mix* right is not enough — a faithful slice must also put
the right types *next to* each other (immune cells around a vessel, neurons
layered by depth, …). That spatial organization is the readout of cell-cell
communication: which types co-locate is shaped by the ligand-receptor signaling
between them. The benchmark scores it directly as ``celltype_nhood_agreement``:
the agreement between the neighborhood-enrichment matrices
``P(neighbor = j | center = i)`` of prediction and ground truth.

We estimate the held-out slice's niche architecture as the z-interpolation of the
two flanking slices' enrichment matrices, ``M* = (1-t)·M_lo + t·M_hi`` (adjacent
sections share their niche structure), and then refine the synthesized cells'
labels — by iterated conditional modes on a Markov random field — so the
synthesized slice reproduces ``M*`` while (i) staying close to the
foundation-model cell-state prior and (ii) matching the target composition. This
is the 2D *and* 3D communication signal: ``M_lo`` and ``M_hi`` come from the
neighboring z-planes, so their interpolation encodes how the niche varies through
the volume.

Leakage-safe: ``M*`` and the target composition are built from the flanking
*training* slices only.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def neighborhood_enrichment(xy, labels, n_types, k=10):
    """Row-normalized ``P(neighbor = j | center = i)`` (matches the evaluator).

    For each cell type ``i``, the distribution over its spatial neighbors' types.
    ``k`` nearest neighbors (self excluded), conditioning on the center type so
    the matrix isolates spatial organization from composition.
    """
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    M = np.zeros((n_types, n_types), dtype=np.float64)
    if n < 2:
        return M
    kk = min(k + 1, n)
    _, nn = cKDTree(xy).query(xy, k=kk)
    nn = np.atleast_2d(nn)[:, 1:]                      # drop self
    neigh_t = labels[nn]                               # (n, k)
    for ti in range(n_types):
        rows = neigh_t[labels == ti]
        if rows.size:
            counts = np.bincount(rows.ravel(), minlength=n_types).astype(float)
            s = counts.sum()
            if s > 0:
                M[ti] = counts / s
    return M


def composition(labels, n_types):
    """Cell-type proportions as a length-``n_types`` probability vector."""
    labels = np.asarray(labels).astype(int)
    if len(labels) == 0:
        return np.full(n_types, 1.0 / n_types)
    c = np.bincount(labels, minlength=n_types).astype(float)
    s = c.sum()
    return c / s if s > 0 else np.full(n_types, 1.0 / n_types)


def target_niche_and_composition(lo_xy, lo_labels, up_xy, up_labels, n_types, t, k=10):
    """z-interpolated flanking niche matrix ``M*`` and composition ``p*``."""
    M_lo = neighborhood_enrichment(lo_xy, lo_labels, n_types, k)
    M_hi = neighborhood_enrichment(up_xy, up_labels, n_types, k)
    M = (1.0 - t) * M_lo + t * M_hi
    p_lo = composition(lo_labels, n_types)
    p_hi = composition(up_labels, n_types)
    p = (1.0 - t) * p_lo + t * p_hi
    return M, p


def refine_labels(
    xy, init_labels, class_logprior, target_nhood, target_composition,
    cfg, seed=0,
):
    """Refine cell-type labels by ICM on a niche MRF (cell-cell communication).

    Energy minimized for each cell ``i`` choosing label ``c``::

        E_i(c) =  -prior_weight       · logprior_i[c]                 (FM cell-state prior)
                  -niche_weight        · Σ_j h_i[j] · log M*[c, j]     (neighborhood match)
                  -composition_weight  · log p*[c]                     (target mix)

    where ``h_i`` is the (normalized) type histogram of cell ``i``'s current
    spatial neighbors. The niche term rewards giving cell ``i`` the label whose
    *expected* neighborhood ``M*[c]`` best explains the neighbors it actually has
    — i.e. it places each type where its communication partners already sit.
    Sweeps are run in parallel (all cells updated from the previous sweep's
    labels), which is stable and fast.

    ``cfg`` is a :class:`~spatialcpav6.config.CommunicationConfig`.
    """
    xy = np.asarray(xy, dtype=np.float64)
    labels = np.asarray(init_labels).astype(int).copy()
    n = len(labels)
    n_types = target_nhood.shape[0]
    if n < 2 or n_types < 2 or not cfg.enabled:
        return labels

    kk = min(cfg.k_neighbors + 1, n)
    _, nn = cKDTree(xy).query(xy, k=kk)
    nn = np.atleast_2d(nn)[:, 1:]                       # (n, k) neighbor indices

    logM = np.log(np.asarray(target_nhood) + 1e-6)      # (C, C): log M*[c, j]
    logp = np.log(np.asarray(target_composition) + 1e-6)  # (C,)
    logprior = np.asarray(class_logprior, dtype=np.float64)  # (n, C)
    rng = np.random.default_rng(seed)

    for _ in range(cfg.n_sweeps):
        # Neighbor type histograms under the current labels.
        neigh_t = labels[nn]                            # (n, k)
        # H[i, j] = fraction of cell i's neighbors that are type j.
        H = np.zeros((n, n_types), dtype=np.float64)
        rows = np.repeat(np.arange(n), neigh_t.shape[1])
        np.add.at(H, (rows, neigh_t.ravel()), 1.0)
        H /= np.maximum(H.sum(axis=1, keepdims=True), 1.0)

        # Score of assigning each cell each label.
        niche_score = H @ logM.T                        # (n, C): Σ_j H[i,j] logM[c,j]
        score = (cfg.prior_weight * logprior
                 + cfg.niche_weight * niche_score
                 + cfg.composition_weight * logp[None, :])
        # Softmax sample (temperature) for a smoother, less greedy update.
        if cfg.temperature > 0:
            z = score / cfg.temperature
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            # Take the argmax (ICM); temperature only sharpens the score here.
            new = p.argmax(axis=1)
        else:
            new = score.argmax(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels
