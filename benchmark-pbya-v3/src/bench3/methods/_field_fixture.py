"""Synthetic fixture for `_ml_learners.select_donor_by_field` — NOT a method.

Unregistered in ``config.METHODS`` and underscore-prefixed, so the runner never
sees it. It exists so the donor-selection arithmetic can be measured against a
KNOWN true field without real data, per CLAUDE.md's rule that every module ships
a synthetic fixture and tests run without GPU, data or network.

v2 — the learner and the donors share their information source.

FIXTURE v1 WAS LEAKY. It built the learner's predicted field from the held-out
section's OWN true field (`learner_pred(w["gt_field"], ...)`). The real learner
is fit on the FLANKING sections and sees the query section only through (x, y,
z, cell_type); the host's donor copies come from those same flanking sections.
So in v1 the regressor carried information the donors could not have, and every
arm that used it to correct the donors looked like a large win. That gap is an
artifact of the fixture, not a property of the method.

Here the field varies with z, the query section sits at z=0, the donor pool is
the two flanking sections at z=+-dz, and the learner's prediction is the best a
model fit on those flanks can do at the query's coordinates. The z-drift is then
a SHARED error: it is in the prediction and in the donors alike, exactly as on
real tissue. What is left for expression-side editing to fix is only the part
the two do NOT share.
"""
import numpy as np
from scipy.spatial import cKDTree
from _ml_learners import rank01

FIELD_GRID = 20


def _basis(xy, z, seed, n_basis=8, ell=0.20, zscale=0.55):
    """A smooth 3-D field sampled at in-plane ``xy`` on the plane ``z``."""
    rng = np.random.default_rng(seed)
    u = np.column_stack([xy / 100.0, np.full(len(xy), z) * zscale])
    ctr = rng.random((n_basis, 3))
    return np.exp(-((u[:, None, :] - ctr[None, :, :]) ** 2).sum(-1) / (2 * ell ** 2))


def make_world(N=5000, G=60, seed=0, hetero=3.0, n_types=18, dz=1.0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(8, G))
    base = rng.normal(scale=hetero, size=G) + 1.6

    def section(n, z, sd):
        r = np.random.default_rng(sd)
        xy = r.random((n, 2)) * 100.0
        F = _basis(xy, z, seed + 1) @ W
        F = F - F.mean(0)
        mu = np.exp(0.80 * F + base[None, :])
        return xy, np.log1p(r.poisson(mu).astype(np.float64)), np.log1p(mu), r.integers(0, n_types, n)

    q_xy, q_X, q_field, q_type = section(N, 0.0, seed + 10)
    a_xy, a_X, a_field, a_type = section(N, -dz, seed + 20)
    b_xy, b_X, b_field, b_type = section(N, +dz, seed + 30)

    tr_xy = np.vstack([a_xy, b_xy])
    Ytr = np.vstack([a_X, b_X])
    tr_type = np.concatenate([a_type, b_type])

    # The learner: fit on the flanks, evaluated at the query's coordinates. The
    # best it can do is the flanks' interpolated field -- which is NOT the query
    # section's field; the difference is the z-drift, and the donors carry it too.
    def flank_hat(xy):
        F = 0.5 * (_basis(xy, -dz, seed + 1) + _basis(xy, +dz, seed + 1)) @ W
        return np.log1p(np.exp(0.80 * (F - F.mean(0)) + base[None, :]))

    return dict(q_xy=q_xy, gt=q_X, gt_field=q_field, q_type=q_type,
                tr_xy=tr_xy, Ytr=Ytr, tr_type=tr_type,
                pred=flank_hat(q_xy), pool_pred=flank_hat(tr_xy), G=G)


def coherent_incumbent(w, seed, disp=11.0):
    """The host: copy a real donor from the FLANKS, under a smooth layout warp."""
    rng = np.random.default_rng(seed)
    warp = _basis(w["q_xy"], 0.0, seed + 41)[:, :2]
    warp = disp * (warp - warp.mean(0)) / max(warp.std(), 1e-9)
    _, nn = cKDTree(w["tr_xy"]).query(w["q_xy"] + warp, k=4)
    return w["Ytr"][nn[np.arange(len(nn)), rng.integers(0, nn.shape[1], len(nn))]]


def _knn_W(xy, k=6):
    _, nn = cKDTree(xy).query(xy, k=k + 1)
    return nn[:, 1:]


def morans_gearys(X, nn):
    N = X.shape[0]
    Xc = X - X.mean(0)
    cross = (Xc[:, None, :] * Xc[nn]).sum((0, 1))
    den = (Xc ** 2).sum(0); S0 = N * nn.shape[1]
    safe = np.where(den > 0, den, 1)
    I = np.where(den > 0, N * cross / (S0 * safe), 0.0)
    diff = ((X[:, None, :] - X[nn]) ** 2).sum((0, 1))
    C = np.where(den > 0, (N - 1) * diff / (2 * S0 * safe), 0.0)
    return I, C


def binned_field(X, xy, grid=FIELD_GRID):
    lo, hi = xy.min(0), xy.max(0)
    g = np.clip(((xy - lo) / np.where(hi - lo > 0, hi - lo, 1.0) * grid).astype(int), 0, grid - 1)
    b = g[:, 1] * grid + g[:, 0]
    cnt = np.bincount(b, minlength=grid * grid).astype(float)
    out = np.column_stack([np.bincount(b, weights=X[:, j], minlength=grid * grid)
                           for j in range(X.shape[1])])
    keep = cnt > 0
    out[keep] /= cnt[keep, None]
    return out, keep


def pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def score(emit, w, nn):
    """Ranked exactly as evaluate_paper.py:670 does, before anything is measured."""
    E, T = rank01(emit), rank01(w["gt"])
    Ig, Cg = morans_gearys(E, nn); Ir, Cr = morans_gearys(T, nn)
    fe, ke = binned_field(E, w["q_xy"]); fr, kr = binned_field(T, w["q_xy"])
    k = ke & kr
    return dict(morans_r=pearson(Ig, Ir), morans_mae=float(np.abs(Ig - Ir).mean()),
                gearys_r=pearson(Cg, Cr), field_r=pearson(fe[k], fr[k]),
                var_ratio=float(E.var(0).mean() / T.var(0).mean()))


HDR = f"{'arm':<42}{'morans_r':>9}{'moransMAE':>10}{'gearys_r':>9}{'field_r':>9}{'var/GT':>8}"
def row(n, s):
    return (f"{n:<42}{s['morans_r']:>9.4f}{s['morans_mae']:>10.4f}"
            f"{s['gearys_r']:>9.4f}{s['field_r']:>9.4f}{s['var_ratio']:>8.3f}")
