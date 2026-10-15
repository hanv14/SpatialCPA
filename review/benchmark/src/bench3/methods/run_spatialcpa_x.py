"""SpatialCPA-X — section transfer with a depth-response model, for interpolation
and extrapolation alike (method ``spatialcpa_x``).

A virtual section at depth z* — between training sections or beyond them — is a
real training section (the *donor*) carried to z* in two parts, each with its own
model fitted on the training sections only:

**Structure: which section to carry.** The transfer-fidelity model of
``run_spatialcpa_tf`` (not modified; its functions are reused): every ordered
pair of training sections (i, k) is scored by copying i onto k with the
benchmark's per-section fidelity measures, and ordinary least squares fits::

    Y[i, k] = c + a * z_i + b * z_k + beta * |z_i - z_k|

The donor is the training section with the highest predicted fidelity at z*,
``argmax_d  a * z_d + beta * |z_d - z*|``. The target term ``b * z*`` is common
to every candidate and cancels, so the rule needs no extrapolation of the model
and is the same rule inside and outside the training range. Between two
sections it chooses one of the flanks whenever ``|beta| > |a|`` (it did in every
STARmap design) and reproduces spatialcpa_tf there.

**Molecular abundance: how much of each gene at z*.** Per gene g, log mean count
per section is linear in depth, ``log m_g(z) = alpha_g + s_g * z``, fitted by
OLS over the training sections. Per-gene slopes are noisy with few sections, so
they are shrunk toward the precision-weighted mean slope by empirical Bayes:
``s_g <- s_bar + tau^2 / (tau^2 + se_g^2) * (s_g - s_bar)``, with the between-gene
variance ``tau^2 = max(0, var(s_g) - mean(se_g^2))`` estimated from the same
fit, so the amount of shrinkage is not a parameter. The donor's counts are
then carried to z* by the capture model: with ``r_g = exp(s_g * (z* - z_d))``,

* ``r_g <= 1`` (fewer molecules at z*): binomial thinning, ``x ~ Bin(x, r_g)``;
* ``r_g > 1`` (more molecules at z*): ``x + Poisson((r_g - 1) * x)``;

both give ``E[x'] = r_g * x`` and keep integer counts and every zero. Non-integer
expression (an intensity measurement) is scaled by ``r_g`` instead, and the
choice is recorded. The draws use a generator seeded from ``(seed, target index)``.

Nested baselines: ``a = 0`` copies the nearest section; ``s = 0`` leaves counts as
they are (``spatialcpa_tf``). No network and no tuned constant: the four
transfer coefficients and the per-gene depth slopes are fitted, the shrinkage
is estimated, and the fallbacks (fewer than three sections) are recorded in
``uns/method_params``.
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2_io  # noqa: E402
import run_spatialcpa_tf as _TF  # noqa: E402  (not modified)

METHOD_NAME = "spatialcpa_x"


# ── structure: the donor ─────────────────────────────────────────────────────

def choose_donor_any(coef, z, z_t):
    """``(donor, scores)``: the training section maximising ``a z_d + beta |z_d - z_t|``
    for coefficients ``(c, a, b, beta)``; ties go to the nearest section (then the
    shallower one)."""
    _c, a, _b, beta = (float(v) for v in coef)
    scores = {d: a * zd + beta * abs(zd - z_t) for d, zd in z.items()}
    best = max(scores.values())
    tied = [d for d, s in scores.items() if s == best]
    donor = min(tied, key=lambda d: (abs(z[d] - z_t), z[d]))
    return donor, scores


def nearest_section(z, z_t):
    return min(z, key=lambda d: (abs(z[d] - z_t), z[d]))


# ── molecular abundance: the depth response ──────────────────────────────────

def depth_slopes(section_means, z):
    """Per-gene depth slopes of log mean count, empirical-Bayes shrunk.

    ``section_means``: {section: (G,) mean count per cell}. Genes not detected in
    every training section get slope 0 (left as they are). Returns
    ``(slopes (G,), info)``. With two sections there are no standard errors: the
    slope of the log total is used for every gene; with one, slopes are 0."""
    ids = list(section_means)
    M = np.array([section_means[s] for s in ids], dtype=float)          # (n, G)
    G = M.shape[1]
    ok = np.all(M > 0, axis=0)
    slopes = np.zeros(G)
    info = {"n_sections": len(ids), "n_genes_fitted": int(ok.sum()), "mode": None}
    if len(ids) < 2 or not ok.any():
        info["mode"] = "none (fewer than two sections)"
        return slopes, info
    zz = np.array([z[s] for s in ids], dtype=float)
    A = np.column_stack([np.ones_like(zz), zz])
    if len(ids) == 2:
        tot = np.log(M[:, ok].sum(axis=1))
        g = float(np.linalg.lstsq(A, tot, rcond=None)[0][1])
        slopes[ok] = g
        info.update(mode="global (two sections)", global_slope=g)
        return slopes, info
    Y = np.log(M[:, ok])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    s = coef[1]
    resid = Y - A @ coef
    se2 = (resid ** 2).sum(axis=0) / (len(ids) - 2) / ((zz - zz.mean()) ** 2).sum()
    se2 = np.maximum(se2, np.finfo(float).tiny)
    sbar = float(np.average(s, weights=1.0 / se2))
    tau2 = float(max(0.0, np.var(s, ddof=1) - se2.mean())) if len(s) > 1 else 0.0
    slopes[ok] = sbar + tau2 / (tau2 + se2) * (s - sbar)
    info.update(mode="empirical Bayes", mean_slope=sbar, tau2=tau2,
                mean_shrinkage=float(np.mean(se2 / (tau2 + se2))))
    return slopes, info


def transfer_counts(X, r, generator, integer):
    """Carry counts ``X`` (n, G) to a capture ratio ``r`` (G,): binomial thinning
    where ``r <= 1``, ``x + Poisson((r - 1) x)`` where ``r > 1`` (integer counts),
    or ``X * r`` (non-integer). Returns float32 (n, G)."""
    X = np.asarray(X, dtype=np.float64)
    r = np.broadcast_to(np.asarray(r, dtype=np.float64), X.shape)
    if not integer:
        return (X * r).astype(np.float32)
    xi = np.rint(X).astype(np.int64)
    out = xi.copy()
    down = r <= 1.0
    out[down] = generator.binomial(xi[down], r[down])
    up = ~down
    out[up] = xi[up] + generator.poisson((r[up] - 1.0) * xi[up])
    return out.astype(np.float32)


# ── the method ───────────────────────────────────────────────────────────────

def _fidelity_model(adata, X, sections, coords, types, ids, z):
    """The transfer-fidelity model of spatialcpa_tf on the training pairs."""
    ep = _TF._load_scorer()
    pp = dict(adata.uns.get("paper_protocol") or {})
    markers = [str(g) for g in pp.get("marker_genes", [])] or ep.MARKER_GENES
    layers = ([str(g) for g in pp.get("layer_superficial", [])] or ep.LAYER_SUPERFICIAL,
              [str(g) for g in pp.get("layer_deep", [])] or ep.LAYER_DEEP)
    genes = np.asarray(adata.var_names, dtype=str)
    order = np.argsort(genes, kind="stable")
    sorted_genes = [str(g) for g in genes[order]]
    pairs = [(i, k) for i in ids for k in ids if i != k]
    rows = []
    for i, k in pairs:
        mi, mk = sections == i, sections == k
        rows.append(_TF.section_metrics(
            ep, X[mi][:, order], coords[mi, :2], None if types is None else types[mi],
            X[mk][:, order], coords[mk, :2], None if types is None else types[mk],
            sorted_genes, markers, layers))
    Y, used = _TF.fidelity_composite(rows) if rows else (np.zeros(0), [])
    model = _TF.fit_fidelity_model(pairs, z, Y) if pairs and used else None
    if model is not None and (model["dof"] < 1 or model["rank"] < 4):
        return None, used, len(pairs), f"not identifiable (rank {model['rank']}, dof {model['dof']})"
    return model, used, len(pairs), None


def run_method(adata, targets, seed):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float32)
    integer = bool(np.all(X == np.rint(X)) and np.all(X >= 0))
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    types = (adata.obs["cell_type"].values.astype(str) if "cell_type" in adata.obs.columns
             else None)
    ids = sorted(set(sections.tolist()),
                 key=lambda s: float(np.median(coords[sections == s, 2])))
    z = {s: float(np.median(coords[sections == s, 2])) for s in ids}
    masks = {s: sections == s for s in ids}
    print(f"  training sections: {', '.join(f'{s}@{z[s]:.1f}' for s in ids)}")

    model, used, n_pairs, why = _fidelity_model(adata, X, sections, coords, types, ids, z)
    rec = {"metrics_used": used, "n_pairs": n_pairs, "sections": z}
    if model is not None:
        V = model["s2"] * np.linalg.pinv(model["X"].T @ model["X"])
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        names = ("c", "a", "b", "beta")
        rec["model"] = {"coef": dict(zip(names, (float(v) for v in model["coef"]))),
                        "se": dict(zip(names, (float(v) for v in se))),
                        "dof": model["dof"], "r2": model["r2"]}
        print(f"  structure: a = {model['coef'][1]:+.4f} ± {se[1]:.4f}, beta = "
              f"{model['coef'][3]:+.4f} ± {se[3]:.4f} per µm (dof {model['dof']}, "
              f"R² {model['r2']:.3f}, {n_pairs} pairs)")
    else:
        rec["model"] = None
        rec["model_reason"] = why or "no scorable pairs"
        print(f"  structure: no fidelity model ({rec['model_reason']}); donor = nearest")

    means = {s: X[masks[s]].mean(axis=0).astype(np.float64) for s in ids}
    slopes, dinfo = depth_slopes(means, z)
    dinfo["count_model"] = ("binomial thinning / Poisson addition" if integer
                            else "scaled (non-integer expression)")
    rec["depth_response"] = {**dinfo, "slopes": dict(zip(map(str, adata.var_names),
                                                         map(float, slopes)))}
    print(f"  abundance: {dinfo['mode']}; {dinfo['count_model']}"
          + (f"; mean slope {dinfo['mean_slope']:+.4f}/µm, tau² {dinfo['tau2']:.2e}, "
             f"mean shrinkage {dinfo['mean_shrinkage']:.2f}" if "tau2" in dinfo else ""))

    lo_z, hi_z = min(z.values()), max(z.values())
    results, decisions = {}, []
    for j, (sec, zt) in enumerate(targets):
        if model is not None:
            donor, scores = choose_donor_any(model["coef"], z, zt)
            reason = "model"
        else:
            donor, scores, reason = nearest_section(z, zt), {}, "nearest (no model)"
        r = np.exp(slopes * (zt - z[donor]))
        g = np.random.default_rng([int(seed), j])
        m = masks[donor]
        n = int(m.sum())
        Xo = transfer_counts(X[m], r, g, integer)
        results[sec] = {"X": sp.csr_matrix(Xo),
                        "coords": np.column_stack([coords[m, :2], np.full(n, zt)]),
                        "cell_type": types[m] if types is not None else np.array(["NA"] * n)}
        mode = "extrapolation" if (zt < lo_z or zt > hi_z) else "interpolation"
        decisions.append({"section": sec, "z": zt, "mode": mode, "donor": donor,
                          "reason": reason, "nearest": nearest_section(z, zt),
                          "scores": {k: float(v) for k, v in scores.items()},
                          "median_capture_ratio": float(np.median(r))})
        print(f"  {sec}: z={zt:.2f} ({mode}) donor {donor} ({reason}; nearest "
              f"{nearest_section(z, zt)}); median capture ratio {np.median(r):.3f}; {n} cells")
    rec["decisions"] = decisions
    return results, rec


def main():
    p = _v2_io.add_v2_args(argparse.ArgumentParser(
        description="SpatialCPA-X: section transfer with a depth-response model"))
    args = p.parse_args()
    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    if adata.n_obs == 0:
        print("ERROR: no training cells", file=sys.stderr)
        return 1
    t0 = time.time()
    results, rec = run_method(adata, targets, args.seed)
    wall = time.time() - t0
    params = {"seed": args.seed, "spatialcpa_x": rec, "generation_only": True}
    _v2_io.write_prediction_h5(results, list(adata.var_names), target_sections, params, wall,
                               args.output, METHOD_NAME)
    print(f"  wall time {wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
