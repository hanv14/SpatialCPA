"""Tissue harmonics: the graph spectral basis v16 generates in.

Why this module is the centre of the method
-------------------------------------------
The benchmark's two spatial-autocorrelation metrics are not arbitrary summary
statistics — they are *quadratic forms in the spatial graph*. With the
row-standardized kNN weights ``benchmark`` uses, for a mean-centred per-cell
signal ``z``::

    Moran's I(z) = zᵀ S z / zᵀ z        (S = symmetrized row-standardized kNN)
    Geary's C(z) ≈ (n-1)/n · (1 - I(z))

Diagonalize ``S = Φ Λ Φᵀ`` with orthonormal ``Φ``. Writing ``ẑ = Φᵀ z`` (the
*graph Fourier coefficients* of the signal) gives the identity this whole method
is built on::

    Moran's I(z) = Σ_m λ_m ẑ_m²  /  Σ_m ẑ_m²

Moran's I is *exactly* the mean eigenvalue under the signal's normalized spectral
energy. So a generator that reproduces a gene's **normalized graph power
spectrum** reproduces its Moran's I identically — and, through the affine
relation above, its Geary's C too. Spatial autocorrelation stops being a property
one hopes emerges from a per-cell model and becomes a quantity that is written
down, predicted and checked.

Two consequences worth stating outright:

* **The two ``_pearson`` metrics are largely redundant on this weight matrix.**
  Because C is near-affine in I, a method that moves one moves the other almost
  identically. The repo's own selftest shows it: oracle 1.000/1.000,
  flanking_copy 0.975/0.976, scramble 0.271/0.282, random -0.062/-0.058 — four
  regimes, and the pair never separates by more than 0.011. One mechanism
  therefore addresses both, and a method claiming to win them separately is
  claiming something the metric cannot show.
* **Low modes are anatomy.** The eigenvectors of ``S`` at the largest λ are the
  smoothest functions on the tissue: φ₁ is essentially the dominant anatomical
  axis (for cortex, the laminar direction), and successive modes are finer
  spatial partitions. That is why the same basis also carries the depth profile
  and cell-type localization the benchmark scores — they are all low-frequency
  structure, and this basis is the natural place to model them.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


def knn_affinity(xy, k=10, symmetrize=True):
    """Row-standardized kNN adjacency, matching the evaluator's spatial weights.

    The evaluator builds ``W`` by taking each cell's ``k`` nearest neighbours
    (self dropped) and weighting each 1/k. That matrix is not symmetric — kNN is
    not a symmetric relation — but the quadratic form ``zᵀWz`` only sees its
    symmetric part, so the eigenbasis of ``S = (W + Wᵀ)/2`` is the right one and
    ``Moran's I`` is unchanged by the substitution.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    k = int(min(k, max(1, n - 1)))
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = np.atleast_2d(idx)[:, 1:]                       # drop self
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.reshape(-1)
    vals = np.full(rows.shape, 1.0 / idx.shape[1])
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return ((W + W.T) * 0.5).tocsr() if symmetrize else W


def tissue_harmonics(xy, n_modes=64, k=10, seed=0):
    """The ``n_modes`` smoothest harmonics of the tissue graph.

    Returns ``(Phi, lam)`` with ``Phi`` (n, M) orthonormal columns and ``lam``
    (M,) the corresponding eigenvalues of ``S``, sorted from smoothest
    (largest λ, lowest spatial frequency) down.

    The constant vector is an eigenvector of a row-standardized graph at λ = 1
    and carries no spatial information — every signal is centred before
    transform, so it contributes nothing and is simply kept as mode 0 for a
    complete basis.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    M = int(max(2, min(n_modes, n - 2)))
    S = knn_affinity(xy, k=k)
    if n <= M + 2:                                        # dense fallback, tiny slice
        lam, Phi = np.linalg.eigh(S.toarray())
        order = np.argsort(-lam)[:M]
        return np.asarray(Phi[:, order]), np.asarray(lam[order])
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=n)
    try:
        lam, Phi = eigsh(S, k=M, which="LA", v0=v0, tol=1e-6, maxiter=5000)
    except Exception:
        # ARPACK can fail to converge on a degenerate graph; a shifted solve on
        # the densified matrix is slow but always answers, and these slices are
        # small enough for it to be an acceptable last resort.
        lam, Phi = np.linalg.eigh(S.toarray())
        lam, Phi = lam[-M:], Phi[:, -M:]
    order = np.argsort(-lam)
    return np.asarray(Phi[:, order]), np.asarray(lam[order])


def gft(Phi, X):
    """Graph Fourier transform: per-cell signals -> spectral coefficients.

    ``X`` is (n, G) and centred here, because Moran's I is defined on the
    centred signal and the mean belongs to the marginal model, not the spatial
    one. Returns ``(Xhat (M, G), mean (G,))``.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    return Phi.T @ (X - mu), mu[0]


def igft(Phi, Xhat, mean=None):
    """Inverse transform: spectral coefficients -> per-cell signals."""
    out = Phi @ Xhat
    return out if mean is None else out + np.asarray(mean)[None, :]


def spectral_energy(Xhat, eps=1e-12):
    """Per-gene normalized spectral energy — the quantity the method matches.

    Column ``g`` of the result is the distribution of gene ``g``'s variance over
    the harmonics, summing to 1. Two signals with the same energy profile have
    the same Moran's I and Geary's C whatever else differs about them.
    """
    E = np.asarray(Xhat, dtype=np.float64) ** 2
    return E / np.maximum(E.sum(axis=0, keepdims=True), eps)


def morans_from_spectrum(Xhat, lam, eps=1e-12):
    """Predicted Moran's I per gene, straight from the spectral coefficients.

    This is the identity in the module docstring evaluated directly. It is what
    lets v16 *check* its own spatial autocorrelation before it writes anything —
    a generated section's Moran's I is known at generation time, not only after
    the evaluator runs.

    Exact only when the basis is complete; with ``M < n`` modes it is the value
    restricted to the modelled subspace, which is why the residual stage below
    exists and why its energy is accounted for rather than discarded.
    """
    E = np.asarray(Xhat, dtype=np.float64) ** 2
    tot = np.maximum(E.sum(axis=0), eps)
    return (np.asarray(lam)[:, None] * E).sum(axis=0) / tot


def morans_full(Xhat, lam, residual_var, lam_res):
    """Moran's I of the *whole* signal, modelled part plus residual bucket.

    ``morans_from_spectrum`` restricts the identity to the modelled subspace,
    which is only the answer when the basis captures essentially all the
    variance. For a gene whose variance is mostly high-frequency it is badly
    optimistic — a pure-noise gene has near-zero true Moran's I, but its
    projection onto 48 smooth modes looks smooth, so the restricted value comes
    out high.

    The complete statement carries the remainder explicitly::

        I = (Σ_m λ_m ẑ_m²  +  λ_res · V_res) / (Σ_m ẑ_m²  +  V_res)

    This is the form v16 reports, and it is what makes the method's own
    prediction comparable with the number the evaluator will compute.
    """
    E = np.asarray(Xhat, dtype=np.float64) ** 2
    num = (np.asarray(lam)[:, None] * E).sum(axis=0) + \
        np.asarray(lam_res) * np.asarray(residual_var)
    den = E.sum(axis=0) + np.asarray(residual_var)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def energy_budget(Phi, X):
    """Fraction of each gene's variance captured by the modelled harmonics.

    The complement is what the residual stage has to supply. Reporting it is how
    v16 avoids the classic low-rank failure: if the budget is 0.3, then 70 % of
    the gene's variance is high-frequency single-cell dispersion, and a method
    that emits only the smooth part would be visibly over-smoothed — inflated
    Moran's I, deflated Geary's C, and a UMAP cloud far tighter than the real
    one.
    """
    Xhat, mu = gft(Phi, X)
    total = ((np.asarray(X, dtype=np.float64) - mu[None, :]) ** 2).sum(axis=0)
    kept = (Xhat ** 2).sum(axis=0)
    return np.where(total > 0, kept / np.maximum(total, 1e-12), 0.0)
