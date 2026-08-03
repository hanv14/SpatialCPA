"""Spectral calibration: making the autocorrelation target an enforced quantity.

The generator produces a *pattern*. This module guarantees its *spatial
autocorrelation*, per gene, by construction rather than by hope.

The mechanism follows directly from the identity in ``harmonics.py``: a gene's
Moran's I is the mean harmonic eigenvalue weighted by that gene's normalized
spectral energy. So if the virtual section's realized energy profile is forced to
match a target profile, its Moran's I matches too — and, through the near-affine
relation on these weights, its Geary's C with it.

Three pieces make that operational.

**A basis-free description of a profile.** Two sections have different graphs and
therefore different eigenvalues, so a per-mode energy vector is not comparable
between them. What is comparable is energy as a function of *spatial frequency*,
so a profile is binned onto a shared normalized-eigenvalue grid. That is the
object interpolated across z.

**A residual bucket.** Only ``M`` harmonics are modelled, and for most genes the
majority of the variance is high-frequency single-cell dispersion sitting outside
them. Dropping it is the classic low-rank failure: the reconstruction comes out
smooth, Moran's I inflates, Geary's C deflates, and the UMAP cloud is far tighter
than the real one — which is exactly the ``spatial_scramble``-in-reverse failure
the benchmark is built to expose. So the profile carries an explicit residual
fraction and an effective residual eigenvalue, and the residual is *restored*,
not approximated.

**Whole-cell residual draws.** The residual is added as complete per-cell
expression vectors resampled from real training cells of the same assigned type,
never as independent per-gene noise. Independent noise would fix each gene's
variance while destroying gene-gene covariance, which the benchmark reads
directly through co-expression agreement and indirectly through the embedding
mixing — a cloud of cells with the right marginals and no covariance structure
does not overlay the real one.
"""

from __future__ import annotations

import numpy as np

from .harmonics import gft, igft

N_FREQ_BINS = 24


def _freq_bins(lam, n_bins=N_FREQ_BINS):
    """Normalized spatial-frequency bin index per harmonic.

    ``lam`` runs from smoothest (largest) down. Normalizing to [0, 1] by the
    observed range makes the grid comparable across sections whose graphs — and
    therefore whose eigenvalue ranges — differ.
    """
    lam = np.asarray(lam, dtype=np.float64)
    lo, hi = float(lam.min()), float(lam.max())
    u = (lam - lo) / max(hi - lo, 1e-12)                # 1 = smoothest
    idx = np.clip((1.0 - u) * n_bins, 0, n_bins - 1e-9).astype(int)
    return idx


def section_profile(X, Phi, lam, n_bins=N_FREQ_BINS, xy=None, k=10):
    """Per-gene spectral profile of a real section.

    Returns ``dict`` with ``energy`` (n_bins, G) summing to 1 over bins *within
    the modelled subspace*, ``kept`` (G,) the modelled fraction of total variance,
    and ``lam_res`` (G,) the effective eigenvalue of the discarded remainder.
    """
    X = np.asarray(X, dtype=np.float64)
    Xhat, mu = gft(Phi, X)
    E = Xhat ** 2                                        # (M, G)
    bins = _freq_bins(lam, n_bins)

    G = X.shape[1]
    energy = np.zeros((n_bins, G))
    np.add.at(energy, bins, E)
    modelled = energy.sum(axis=0)
    energy = energy / np.maximum(modelled, 1e-12)

    total = ((X - mu[None, :]) ** 2).sum(axis=0)
    kept = np.where(total > 0, modelled / np.maximum(total, 1e-12), 0.0)
    kept = np.clip(kept, 0.0, 1.0)

    # Effective eigenvalue of the unmodelled remainder, measured rather than
    # assumed: Moran's I of the whole signal is the energy-weighted mean
    # eigenvalue, so with the section's true I in hand the identity
    #     I · V_total = Σ_m λ_m ê_m + λ_res · V_res
    # has exactly one unknown.
    lam_res = _residual_eigenvalue(X, mu, lam, Xhat, total, modelled, xy=xy, k=k)
    return {"energy": energy, "kept": kept, "lam_res": lam_res,
            "mean": mu, "n_bins": n_bins}


def _residual_eigenvalue(X, mu, lam, Xhat, total, modelled, xy=None, k=10):
    """Solve the Moran identity for the remainder's effective eigenvalue.

    Needs the section's measured Moran's I, which needs its coordinates. When
    they are not supplied the conservative bound is used instead: the remainder
    is no smoother than the roughest fitted mode, so ``λ_min`` never overstates
    how much structure it carries.

    The solved value is bounded by the spectrum of the *whole* operator, not by
    the modelled slice of it. A row-standardized symmetrized affinity has all its
    eigenvalues in [-1, 1], and the residual lives in the part of that range the
    basis left out — typically near zero. Clipping to the modelled range instead
    (``[λ_min, λ_max]``, e.g. [0.85, 1.0] for 48 modes) forces the residual to
    look smooth and makes the whole prediction badly optimistic: a pure-noise
    gene comes out at +0.61 against a true -0.01.
    """
    lam = np.asarray(lam, dtype=np.float64)
    G = X.shape[1]
    if xy is None:
        return np.full(G, float(lam.min()))

    from .harmonics import knn_affinity
    S = knn_affinity(xy, k=k)
    Z = np.asarray(X, dtype=np.float64) - mu[None, :]
    num_total = (Z * (S @ Z)).sum(axis=0)                # zᵀSz per gene
    num_model = (lam[:, None] * (Xhat ** 2)).sum(axis=0)
    res_var = np.maximum(total - modelled, 0.0)
    out = np.full(G, 0.0)
    ok = res_var > 1e-12
    out[ok] = np.clip((num_total[ok] - num_model[ok]) / res_var[ok], -1.0, 1.0)
    return out


def blend_profiles(prof_lo, prof_hi, w):
    """The virtual section's target profile, interpolated across the gap."""
    out = {}
    for key in ("energy", "kept", "lam_res"):
        out[key] = (1.0 - w) * prof_lo[key] + w * prof_hi[key]
    out["mean"] = (1.0 - w) * prof_lo["mean"] + w * prof_hi["mean"]
    out["n_bins"] = prof_lo["n_bins"]
    return out


def apply_calibration(X_smooth, Phi, lam, target, floor=1e-9):
    """Rescale a generated section so its spectral profile matches ``target``.

    Operates per (frequency bin, gene): within a bin the *shape* the generator
    produced is kept and only the bin's total energy is set, so calibration fixes
    how much structure exists at each spatial scale without dictating where it
    is. That separation is the point — the generative model decides the pattern,
    the calibration decides its autocorrelation.

    Returns ``(X_calibrated, realized_modelled_variance)``.
    """
    X_smooth = np.asarray(X_smooth, dtype=np.float64)
    Xhat, mu = gft(Phi, X_smooth)
    bins = _freq_bins(lam, target["n_bins"])
    E = Xhat ** 2

    have = np.zeros_like(target["energy"])
    np.add.at(have, bins, E)
    have_tot = np.maximum(have.sum(axis=0, keepdims=True), floor)
    have_n = have / have_tot

    want = target["energy"]
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.sqrt(np.where(have_n > floor, want / np.maximum(have_n, floor), 0.0))
    # A bin the generator left empty cannot be scaled into existence; leave it at
    # zero rather than injecting arbitrary structure, and let the reported
    # realized profile show the shortfall.
    gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)

    Xhat_cal = Xhat * gain[bins, :]
    X_cal = igft(Phi, Xhat_cal, mu)
    realized = (Xhat_cal ** 2).sum(axis=0)
    return X_cal, realized


def scale_to_variance(X_cal, target_modelled_var, floor=1e-12):
    """Put the calibrated smooth part at the variance the target budget asks for."""
    X_cal = np.asarray(X_cal, dtype=np.float64)
    mu = X_cal.mean(axis=0, keepdims=True)
    Z = X_cal - mu
    have = (Z ** 2).sum(axis=0)
    g = np.sqrt(np.where(have > floor, target_modelled_var / np.maximum(have, floor), 0.0))
    return Z * g[None, :] + mu


def residual_draw(train_X, train_types, train_Phi_res, types_out, rng):
    """Whole-cell high-frequency residuals, resampled by cell type.

    ``train_Phi_res`` holds each training cell's residual profile — its
    expression minus the smooth part its own section's harmonics explain. Drawing
    a *row* of that matrix hands the virtual cell a real cell's joint deviation
    across all genes at once, so gene-gene covariance survives; drawing per gene
    independently would not.
    """
    out = np.zeros((len(types_out), train_X.shape[1]), dtype=np.float64)
    by_type = {}
    for t in np.unique(train_types):
        by_type[t] = np.where(train_types == t)[0]
    all_idx = np.arange(len(train_types))
    for i, t in enumerate(types_out):
        pool = by_type.get(t)
        if pool is None or len(pool) == 0:
            pool = all_idx
        out[i] = train_Phi_res[pool[rng.integers(len(pool))]]
    return out
