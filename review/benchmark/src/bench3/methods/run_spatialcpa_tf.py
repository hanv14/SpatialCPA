"""SpatialCPA-TF — virtual sections by transfer-fidelity donor selection
(method ``spatialcpa_tf``).

A virtual section at depth z* is a real training section — the *donor* —
transferred whole to z*: its cells' in-plane positions, raw counts and cell
types, verbatim. Everything the method decides is which donor, and it decides
that from a model of how well one real section reproduces another, fitted on
the training sections only:

1. **Transfer-fidelity matrix.** For every ordered pair of training sections
   (i, k), section i is copied onto section k and scored against it with the
   benchmark's own per-section fidelity measures (``FIDELITY_METRICS``, the
   functions ``evaluate_paper`` uses, called the same way). Each measure is put
   on a common scale by its standard deviation over the training pairs and the
   signed sum is the pair's fidelity ``Y[i, k]``. Deterministic: no random draws.
2. **Fidelity model.** Ordinary least squares on the training pairs::

       Y[i, k] = c + a * z_i + b * z_k + beta * |z_i - z_k| + e

   ``a`` is the donor-depth gradient (sections at some depths transfer better —
   acquisition quality that varies through a volume), ``b`` the target-depth
   gradient (some depths are easier to reproduce) and ``beta`` the decay of
   fidelity with the gap. Four coefficients; nothing is tuned.
3. **Donor choice.** For a target z* bracketed by training sections at
   z_lo < z* < z_hi, the predicted fidelity advantage of the upper flank is::

       gain = a * (z_hi - z_lo) + beta * (|z_hi - z*| - |z_lo - z*|)

   (the target terms cancel). The donor is the flank with the larger predicted
   fidelity — the Bayes decision under the fitted model. ``a = 0`` reduces to
   copying the nearest section, so that baseline is the nested special case; a
   tie, an unestimable contrast (too few sections) or a one-sided target gives
   the nearest section, and the reason is recorded.

No neural network, no generation randomness, no post-hoc mixing: the output
cells are real cells, so their joint expression, spatial autocorrelation and
neighbourhoods are those of real tissue. The fitted coefficients, their standard
errors and every decision are written to ``uns/method_params``.

Uses only the training-only input (``_v2_io.guard_no_holdout``) and the scorer
module ``bench3.evaluate_paper``, which is not modified.
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

METHOD_NAME = "spatialcpa_tf"

# The fidelity measures, with the sign that makes larger better (the eight
# per-section ``evaluate_paper`` metrics flow-cv validated on).
FIDELITY_METRICS = (
    ("embedding_mixing_pca", +1),
    ("morans_pearson", +1),
    ("morans_mae", -1),
    ("marker_depth_r", +1),
    ("marker_field_r", +1),
    ("celltype_localization", +1),
    ("rare_celltype_localization", +1),
    ("gene_detection_spearman", +1),
)


# ── fidelity of one section copied onto another ──────────────────────────────

def _load_scorer():
    """``bench3.evaluate_paper``, the benchmark's scorer (not modified)."""
    src = str(Path(__file__).resolve().parents[2])
    if src not in sys.path:
        sys.path.insert(0, src)
    from bench3 import evaluate_paper as ep
    return ep


def section_metrics(ep, pred_X, pred_xy, pred_types, gt_X, gt_xy, gt_types,
                    gene_names, markers, layers):
    """One section's ``evaluate_paper`` metrics (UMAP off), from arrays: the same
    calls, arguments and order as the scorer's per-section loop. ``pred_X`` and
    ``gt_X`` are raw expression with columns in ``gene_names`` (sorted) order."""
    k, grid, seed = ep.SPATIAL_K, ep.FIELD_GRID, ep.RANDOM_SEED
    pR, gR = ep._rank_normalize(pred_X), ep._rank_normalize(gt_X)
    cols, _basis = ep.scoring_columns(gene_names, markers, gt_xy, gR, spatial_k=k)
    pred_xy_al, _ainfo = ep.align_by_expression(
        pred_xy, pR[:, cols], gt_xy, gR[:, cols], grid=grid, seed=seed)
    m = {}
    m.update(ep.spatial_autocorrelation_metrics(pred_xy, pR, gt_xy, gR, k=k))
    m.update(ep.embedding_continuity(pR, gR, seed=seed, use_umap=False))
    m.update(ep.marker_metrics(pred_xy_al, pR, gt_xy, gR, gene_names,
                               markers=markers, grid=grid,
                               depth_bins=ep.DEPTH_BINS, k=k, layers=layers))
    m.update(ep.gene_detection_metrics(pred_X, gt_X))
    if gt_types is not None:
        m.update(ep.celltype_localization(pred_xy_al, pred_types, gt_xy, gt_types,
                                          seed=seed))
    return m


def fidelity_composite(rows, metrics=FIDELITY_METRICS):
    """``rows``: one metric dict per pair. Returns ``(Y, used)``: the signed sum of
    each metric divided by its standard deviation over the rows, and the metrics
    used — a metric missing or non-finite in any row, or constant, is left out."""
    Y, used = np.zeros(len(rows)), []
    for key, sign in metrics:
        v = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        if not np.all(np.isfinite(v)):
            continue
        sd = float(v.std())
        if sd <= 0.0:
            continue
        Y += sign * v / sd
        used.append(key)
    return Y, used


# ── the fidelity model and the donor decision ────────────────────────────────

def _design(pairs, z):
    """Rows ``[1, z_i, z_k, |z_i - z_k|]`` for ordered pairs (i, k)."""
    return np.array([[1.0, z[i], z[k], abs(z[i] - z[k])] for i, k in pairs], dtype=float)


def fit_fidelity_model(pairs, z, Y):
    """OLS of ``Y[i, k] = c + a z_i + b z_k + beta |z_i - z_k|``. Returns a dict
    with the coefficients ``coef`` = (c, a, b, beta), the design ``X``, residual
    variance ``s2``, residual degrees of freedom ``dof``, ``rank`` and ``r2``."""
    X = _design(pairs, z)
    Y = np.asarray(Y, dtype=float)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    rank = int(np.linalg.matrix_rank(X))
    dof = len(Y) - rank
    resid = Y - X @ coef
    s2 = float(resid @ resid / dof) if dof > 0 else float("nan")
    tss = float(((Y - Y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")
    return {"coef": coef, "X": X, "s2": s2, "dof": dof, "rank": rank, "r2": r2}


def choose_donor(model, z_lo, z_hi, z_t):
    """The flank with the larger predicted fidelity at ``z_t`` (``z_lo < z_t <
    z_hi``). Returns ``(use_upper, info)``; the nearest flank (lower on an exact
    midpoint, as v18's rule) when the gain is zero or its contrast is not
    estimable from the training pairs."""
    near_upper = (z_t - z_lo) > (z_hi - z_t)
    c = np.array([0.0, z_hi - z_lo, 0.0, abs(z_hi - z_t) - abs(z_lo - z_t)])
    X = model["X"]
    in_row_space = np.allclose(X.T @ np.linalg.pinv(X.T) @ c, c,
                               atol=1e-8 * (1.0 + np.abs(c).max()))
    estimable = model["dof"] > 0 and in_row_space
    if not estimable:
        return near_upper, {"gain": None, "se": None, "reason": "not estimable: nearest"}
    gain = float(c @ model["coef"])
    se = float(np.sqrt(model["s2"] * c @ np.linalg.pinv(X.T @ X) @ c))
    if gain == 0.0:
        return near_upper, {"gain": gain, "se": se, "reason": "tie: nearest"}
    return gain > 0.0, {"gain": gain, "se": se, "reason": "model"}


# ── the method ───────────────────────────────────────────────────────────────

def run_method(adata, targets):
    ep = _load_scorer()
    pp = dict(adata.uns.get("paper_protocol") or {})
    markers = [str(g) for g in pp.get("marker_genes", [])] or ep.MARKER_GENES
    layers = ([str(g) for g in pp.get("layer_superficial", [])] or ep.LAYER_SUPERFICIAL,
              [str(g) for g in pp.get("layer_deep", [])] or ep.LAYER_DEEP)

    genes = np.asarray(adata.var_names, dtype=str)
    order = np.argsort(genes, kind="stable")
    sorted_genes = [str(g) for g in genes[order]]
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float32)
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    types = (adata.obs["cell_type"].values.astype(str) if "cell_type" in adata.obs.columns
             else None)
    ids = sorted(set(sections.tolist()),
                 key=lambda s: float(np.median(coords[sections == s, 2])))
    z = {s: float(np.median(coords[sections == s, 2])) for s in ids}
    masks = {s: sections == s for s in ids}
    print(f"  training sections: {', '.join(f'{s}@{z[s]:.1f}' for s in ids)}")

    # 1. transfer-fidelity matrix over ordered training pairs
    pairs = [(i, k) for i in ids for k in ids if i != k]
    rows = []
    for i, k in pairs:
        mi, mk = masks[i], masks[k]
        rows.append(section_metrics(
            ep, X[mi][:, order], coords[mi, :2], None if types is None else types[mi],
            X[mk][:, order], coords[mk, :2], None if types is None else types[mk],
            sorted_genes, markers, layers))
    Y, used = fidelity_composite(rows)
    print(f"  {len(pairs)} ordered pairs scored on {len(used)} fidelity measures")

    # 2. fidelity model
    model = fit_fidelity_model(pairs, z, Y) if pairs and used else None
    if model is not None:
        c_, a_, b_, beta_ = (float(v) for v in model["coef"])
        V = (model["s2"] * np.linalg.pinv(model["X"].T @ model["X"])
             if model["dof"] > 0 else np.full((4, 4), np.nan))
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        print(f"  fidelity model: a (donor depth) = {a_:+.4f} ± {se[1]:.4f}, "
              f"b (target depth) = {b_:+.4f} ± {se[2]:.4f}, beta (gap) = {beta_:+.4f} ± "
              f"{se[3]:.4f} per µm; dof {model['dof']}, R² {model['r2']:.3f}")

    # 3. donor per target, then transfer
    results, decisions = {}, []
    for sec, zt in targets:
        below = [s for s in ids if z[s] <= zt]
        above = [s for s in ids if z[s] > zt]
        if below and above:
            lo, hi = max(below, key=z.get), min(above, key=z.get)
            if model is None:
                use_upper = (zt - z[lo]) > (z[hi] - zt)
                info = {"gain": None, "se": None, "reason": "no model: nearest"}
            else:
                use_upper, info = choose_donor(model, z[lo], z[hi], zt)
            donor = hi if use_upper else lo
        else:                                            # outside the training range
            lo = hi = donor = (max(below, key=z.get) if below else min(above, key=z.get))
            info = {"gain": None, "se": None, "reason": "one-sided: nearest"}
        m = masks[donor]
        n = int(m.sum())
        results[sec] = {"X": sp.csr_matrix(X[m]),
                        "coords": np.column_stack([coords[m, :2], np.full(n, zt)]),
                        "cell_type": types[m] if types is not None else np.array(["NA"] * n)}
        decisions.append({"section": sec, "z": zt, "lower": lo, "upper": hi, "donor": donor,
                          **info})
        g = "—" if info["gain"] is None else f"{info['gain']:+.2f} ± {info['se']:.2f}"
        print(f"  {sec}: z={zt:.2f} flanks {lo}/{hi}; upper-flank gain {g} -> donor {donor} "
              f"({info['reason']}; {n} cells)")

    rec = {"metrics_used": used, "n_pairs": len(pairs),
           "sections": {s: z[s] for s in ids}, "decisions": decisions}
    if model is not None:
        rec["model"] = {"coef": dict(zip(("c", "a", "b", "beta"),
                                         (float(v) for v in model["coef"]))),
                        "se": dict(zip(("c", "a", "b", "beta"), (float(v) for v in se))),
                        "dof": model["dof"], "r2": model["r2"]}
    return results, rec


def main():
    p = _v2_io.add_v2_args(argparse.ArgumentParser(
        description="SpatialCPA-TF: transfer-fidelity donor selection"))
    args = p.parse_args()
    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    if adata.obs["section"].nunique() < 1 or adata.n_obs == 0:
        print("ERROR: no training cells", file=sys.stderr)
        return 1
    t0 = time.time()
    results, rec = run_method(adata, targets)
    wall = time.time() - t0
    params = {"seed": args.seed, "deterministic": True, "transfer_fidelity": rec,
              "generation_only": True}
    _v2_io.write_prediction_h5(results, list(adata.var_names), target_sections, params, wall,
                               args.output, METHOD_NAME)
    print(f"  wall time {wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
