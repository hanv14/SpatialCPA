"""Is a method's uncertainty map calibrated? (evaluation-only, review benchmark)

Reads ``prediction.h5`` + ``uncertainty.npz`` (written by
``methods/run_spatialcpav18_uq.py``) and the dataset file, and writes
``uncertainty_metrics.json``. No method or ``prepare_dataset`` is rerun; the
pinned scorer is not touched.

Per held-out section, the prediction is posed onto the GT exactly as
``evaluate_paper`` poses it (``align_by_expression`` on the marker panel). Both
are binned on one ``FIELD_GRID`` x ``FIELD_GRID`` lattice; in every patch with at
least ``MIN_PATCH_CELLS`` cells on both sides:

* the **error** is the L2 distance between the mean rank-normalized expression
  of the predicted and of the real cells in the patch (scale-fair, as every
  paper metric is);
* each **uncertainty score** is the mean over the patch's predicted cells:
  ``u_flow`` (the flow's sample spread), ``u_untrained`` (an untrained flow's),
  ``d_flank`` (disagreement of the two flanking sections), plus ``sampling`` =
  sqrt(1/n_pred + 1/n_gt), the error a patch has from its cell count alone.

For each score: Spearman rho with the error; the partial rho with the
``sampling`` baseline's ranks regressed out of both (edge patches are small and
noisy — a score that only finds them is not informative); AUROC for the worst
quarter of patches; and the mean error per score quintile (a calibrated score
rises monotonically). Per section and pooled (weighted by patch count).

Usage:
    python -m src.bench3.evaluate_uncertainty                     # what's missing
    python -m src.bench3.evaluate_uncertainty --results-dir ../reproduced/uq --force
    python -m src.bench3.evaluate_uncertainty --grid 10    # declared sensitivity: coarser patches
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr

from ._v2bridge import load_prediction  # noqa: F401  (also puts src/benchmark on sys.path)
from .align import align_by_expression, scoring_columns
from .config import (DATASET_NAME, FIELD_GRID, MARKER_GENES, RANDOM_SEED, RESULTS_DIR,
                     SPATIAL_K, dataset_path)

from benchmark.evaluate_generation import _rank_normalize  # noqa: E402

SIDECAR = "uncertainty.npz"
OUT_NAME = "uncertainty_metrics.json"
SCORES = ("u_flow", "u_untrained", "d_flank", "sampling")
MIN_PATCH_CELLS = 5
WORST_FRAC = 0.25
N_QUANTILES = 5


def _dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def patch_table(pred_xy, pred_R, gt_xy, gt_R, scores, grid=FIELD_GRID,
                min_cells=MIN_PATCH_CELLS):
    """Per-patch error and mean scores on one lattice over both clouds.
    ``scores``: {name: (n_pred,)}. Returns ``(err (P,), {name: (P,)})``."""
    allxy = np.vstack([pred_xy, gt_xy])
    lo, hi = allxy.min(0), allxy.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)

    def pid(xy):
        b = np.clip(((xy - lo) / span * grid).astype(int), 0, grid - 1)
        return b[:, 1] * grid + b[:, 0]

    pp, gp = pid(pred_xy), pid(gt_xy)
    err, out = [], {k: [] for k in list(scores) + ["sampling"]}
    for p in np.intersect1d(pp, gp):
        mp, mg = pp == p, gp == p
        npc, ngc = int(mp.sum()), int(mg.sum())
        if npc < min_cells or ngc < min_cells:
            continue
        err.append(float(np.linalg.norm(pred_R[mp].mean(0) - gt_R[mg].mean(0))))
        for k, v in scores.items():
            out[k].append(float(np.mean(v[mp])))
        out["sampling"].append(float(np.sqrt(1.0 / npc + 1.0 / ngc)))
    return np.asarray(err), {k: np.asarray(v) for k, v in out.items()}


def _auroc(score, positive):
    """Probability a positive patch outranks a negative one (ties count half)."""
    pos, neg = positive.sum(), (~positive).sum()
    if pos == 0 or neg == 0:
        return np.nan
    r = rankdata(score)
    return float((r[positive].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _residual_ranks(y, x):
    ry, rx = rankdata(y), rankdata(x)
    A = np.column_stack([np.ones_like(rx), rx])
    return ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]


def score_metrics(err, score, control):
    """rho, partial rho (``control`` regressed out), AUROC on the worst
    ``WORST_FRAC`` patches, and the mean error per score quantile."""
    if len(err) < 10 or np.std(score) == 0:
        return {"rho": None, "rho_partial": None, "auroc_worst": None, "quantile_err": None}
    rho = float(spearmanr(score, err).correlation)
    rp = (float(spearmanr(_residual_ranks(score, control),
                          _residual_ranks(err, control)).correlation)
          if np.std(control) > 0 else rho)
    worst = err >= np.quantile(err, 1.0 - WORST_FRAC)
    edges = np.quantile(score, np.linspace(0, 1, N_QUANTILES + 1))
    q = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, N_QUANTILES - 1)
    qerr = [float(err[q == i].mean()) if (q == i).any() else None for i in range(N_QUANTILES)]
    return {"rho": rho, "rho_partial": rp, "auroc_worst": _auroc(score, worst),
            "quantile_err": qerr}


def evaluate_uncertainty(prediction_path, h5ad_path, output_path=None, grid=FIELD_GRID):
    import anndata as ad
    pred = load_prediction(prediction_path)
    side_path = Path(prediction_path).with_name(SIDECAR)
    holdout = [str(s) for s in pred["holdout_sections"]]
    out = {"method": pred["method_name"], "holdout_sections": holdout,
           "dataset_file": str(h5ad_path),
           "eval": "uncertainty calibration (evaluate_uncertainty.py)", "grid": int(grid)}
    if not side_path.exists():
        out.update(applicable=False, reason=f"no {SIDECAR} beside the prediction")
        return _finish(out, output_path)
    side = np.load(side_path)
    adata = ad.read_h5ad(h5ad_path)
    pp = dict(adata.uns.get("paper_protocol") or {})
    markers = [str(g) for g in pp.get("marker_genes", [])] or list(MARKER_GENES)
    genes = sorted(set(map(str, pred["gene_names"])) & set(map(str, adata.var_names)))
    pgi = [list(map(str, pred["gene_names"])).index(g) for g in genes]
    ggi = [list(map(str, adata.var_names)).index(g) for g in genes]
    secs = adata.obs["section"].astype(str).values
    xyz = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    per, n_patches = {}, []
    for sec in holdout:
        keys = [f"{sec}::{k}" for k in SCORES[:3]]
        pm = pred["section"] == sec
        if not all(k in side for k in keys) or pm.sum() == 0:
            per[sec] = {"error": "no uncertainty for this section"}
            continue
        scores = {k.split("::")[1]: np.asarray(side[k], dtype=np.float64) for k in keys}
        if any(len(v) != pm.sum() for v in scores.values()):
            raise ValueError(f"{SIDECAR} does not match the prediction's cells for {sec}")
        pX = _dense(pred["X"][pm][:, pgi]).astype(np.float64)
        g = secs == sec
        gX = _dense(adata.X[g][:, ggi]).astype(np.float64)
        pxy = np.column_stack([pred["x"][pm], pred["y"][pm]]).astype(np.float64)
        gxy = xyz[g, :2]
        pR, gR = _rank_normalize(pX), _rank_normalize(gX)
        cols, _ = scoring_columns(genes, markers, gxy, gR, spatial_k=SPATIAL_K)
        pxy_al, _ = align_by_expression(pxy, pR[:, cols], gxy, gR[:, cols],
                                        grid=FIELD_GRID, seed=RANDOM_SEED)
        err, tab = patch_table(pxy_al, pR, gxy, gR, scores, grid=grid)
        per[sec] = {"n_patches": int(len(err)),
                    **{k: score_metrics(err, tab[k], tab["sampling"]) for k in SCORES}}
        n_patches.append(len(err))

    usable = [s for s in per if "error" not in per[s]]
    w = np.asarray(n_patches, dtype=float)
    for k in SCORES:
        for stat in ("rho", "rho_partial", "auroc_worst"):
            vals = np.array([per[s][k][stat] if per[s][k][stat] is not None else np.nan
                             for s in usable], dtype=float)
            ok = np.isfinite(vals)
            out[f"{k}_{stat}"] = float(np.average(vals[ok], weights=w[ok])) if ok.any() else None
    out.update(applicable=True, per_section=per)
    return _finish(out, output_path)


def _finish(out, output_path):
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    out = clean(out)
    if output_path:
        Path(output_path).write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description="Uncertainty calibration for every prediction")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--grid", type=int, default=FIELD_GRID,
                    help=f"patch lattice (default FIELD_GRID={FIELD_GRID}); any other value "
                         f"is a sensitivity analysis written to uncertainty_metrics_g<grid>.json")
    args = ap.parse_args()
    root = Path(args.results_dir)
    done = skipped = failed = 0
    for side in sorted(root.rglob(SIDECAR)):
        pred_path = side.with_name("prediction.h5")
        rel = pred_path.relative_to(root).parts
        out_path = side.with_name(OUT_NAME if args.grid == FIELD_GRID
                                  else OUT_NAME.replace(".json", f"_g{args.grid}.json"))
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        gt = dataset_path("/".join(rel[1:-2]) or DATASET_NAME)
        print(f"Uncertainty calibration for {'/'.join(rel[:-1])} ...")
        try:
            m = evaluate_uncertainty(pred_path, gt, out_path, grid=args.grid)
            print("  " + "  ".join(f"{k}: rho={m.get(k + '_rho')} partial="
                                    f"{m.get(k + '_rho_partial')}" for k in SCORES))
            done += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\nScored {done}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
