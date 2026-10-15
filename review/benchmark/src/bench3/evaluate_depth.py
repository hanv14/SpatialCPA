"""Raw-scale abundance metrics for generated sections (benchmark-pbya-v3, review).

An evaluation-side add-on, like ``evaluate_vascular``: it reads each run's
``prediction.h5`` and the dataset file it was scored against and writes
``depth_metrics.json`` beside ``metrics.json``. The pinned scorer
(``evaluate_paper.py``) is not touched.

Why it exists: ``evaluate_paper`` rank-normalizes every gene, so it is
deliberately blind to the *level* of expression — a section emitted at the wrong
molecular depth (all counts too high or too low, gene by gene) scores the same
as one at the right depth. In thick-tissue imaging capture falls with depth, so
for any method that emits counts, the level is part of what a reconstructed
section gets right or wrong. Per held-out section, on raw counts over the genes
shared by the prediction and the GT:

``lib_logratio``             log(median per-cell total, prediction / GT); 0 is right.
``lib_abs_logratio``         its absolute value.
``gene_mean_abs_logratio``   median over genes of |log(mean count, prediction / GT)|
                             (genes with a zero mean on either side are left out).
``gene_mean_log_r``          Pearson r of the per-gene log mean counts (profile
                             shape, blind to a common scale factor).

Pooled values are averages weighted by the GT cell count, as ``evaluate_paper``.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ._v2bridge import load_prediction
from .config import DATASET_NAME, RESULTS_DIR, dataset_path, resolve_dataset_arg

OUT_NAME = "depth_metrics.json"
POOLED = ("lib_logratio", "lib_abs_logratio", "gene_mean_abs_logratio", "gene_mean_log_r")
MIN_CELLS = 30


def _dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def section_metrics(pred_X, gt_X):
    """The four abundance metrics of one section (raw (n, G) matrices, same genes)."""
    pred_X = np.clip(np.asarray(pred_X, dtype=np.float64), 0.0, None)
    gt_X = np.clip(np.asarray(gt_X, dtype=np.float64), 0.0, None)
    lp, lg = float(np.median(pred_X.sum(1))), float(np.median(gt_X.sum(1)))
    m = {"lib_logratio": float(np.log(lp / lg)) if lp > 0 and lg > 0 else np.nan}
    m["lib_abs_logratio"] = abs(m["lib_logratio"])
    mp, mg = pred_X.mean(0), gt_X.mean(0)
    ok = (mp > 0) & (mg > 0)
    lr = np.log(mp[ok] / mg[ok])
    m["gene_mean_abs_logratio"] = float(np.median(np.abs(lr))) if ok.any() else np.nan
    m["gene_mean_log_r"] = (float(np.corrcoef(np.log(mp[ok]), np.log(mg[ok]))[0, 1])
                            if ok.sum() > 2 else np.nan)
    m["n_genes"] = int(ok.sum())
    return m


def evaluate_depth(prediction_path, h5ad_path, output_path=None):
    import anndata as ad
    pred = load_prediction(prediction_path)
    holdout = [str(s) for s in pred["holdout_sections"]]
    adata = ad.read_h5ad(h5ad_path)
    out = {"method": pred["method_name"],
           "dataset": str(adata.uns.get("dataset_name", DATASET_NAME)),
           "dataset_file": str(h5ad_path), "holdout_sections": holdout,
           "eval": "raw-scale abundance (evaluate_depth.py)"}
    genes = sorted(set(map(str, pred["gene_names"])) & set(map(str, adata.var_names)))
    pg, gg = list(map(str, pred["gene_names"])), list(map(str, adata.var_names))
    pgi, ggi = [pg.index(g) for g in genes], [gg.index(g) for g in genes]
    pX = _dense(pred["X"][:, pgi])
    secs = adata.obs["section"].astype(str).values
    per, weights = {}, []
    for sec in holdout:
        pm, gm = pred["section"] == sec, secs == sec
        if pm.sum() < MIN_CELLS or gm.sum() < MIN_CELLS:
            per[sec] = {"error": f"too few cells (pred={int(pm.sum())}, gt={int(gm.sum())})"}
            continue
        per[sec] = section_metrics(pX[pm], _dense(adata.X[gm][:, ggi]))
        weights.append(int(gm.sum()))
    usable = [s for s in per if "error" not in per[s]]
    w = np.asarray(weights, dtype=float)
    for key in POOLED:
        vals = np.array([per[s][key] for s in usable], dtype=float)
        ok = np.isfinite(vals)
        out[key] = float(np.average(vals[ok], weights=w[ok])) if ok.any() else None
    out["per_section"] = per
    if output_path:
        Path(output_path).write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description="Raw-scale abundance metrics for every prediction")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--dataset", default=None,
                    help="ground-truth h5ad (name or path); default: resolved per "
                         "prediction from its results path")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--force", action="store_true", help=f"recompute even when {OUT_NAME} exists")
    args = ap.parse_args()
    override = resolve_dataset_arg(args.dataset) if args.dataset else None
    root = Path(args.results_dir)
    done = skipped = failed = 0
    for pred_path in sorted(root.rglob("prediction.h5")):
        rel = pred_path.relative_to(root).parts
        method = rel[0]
        if method.startswith("_") or method == "summary":
            continue
        if args.methods and method not in args.methods:
            continue
        ds_name = "/".join(rel[1:-2]) or DATASET_NAME
        gt_path = override if override is not None else dataset_path(ds_name)
        out_path = pred_path.parent / OUT_NAME
        if not Path(gt_path).exists() or (out_path.exists() and not args.force):
            skipped += 1
            continue
        try:
            evaluate_depth(pred_path, gt_path, out_path)
            done += 1
        except Exception:
            traceback.print_exc()
            failed += 1
    print(f"depth metrics: {done} written, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
