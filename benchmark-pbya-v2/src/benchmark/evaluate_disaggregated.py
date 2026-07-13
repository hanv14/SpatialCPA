"""Disaggregated evaluation: per-gene and per-cell metrics WITHOUT reduction.

``evaluate.py`` and ``evaluate_generation.py`` compute rich per-gene arrays and
per-cell matches, but only write the *reduced* summaries (median / mean per
held-out slice) into ``metrics.json``. That throws away the distribution the
reduction was taken over, so you cannot ask "which genes did the method get
right?" or "where in the slice is the error concentrated?" from the standard
outputs.

This module re-runs the identical computations (reusing the functions in
``evaluate`` / ``evaluate_generation`` — nothing is re-implemented differently)
and emits the **full tables** as long-format CSVs, one row per gene / per cell,
so every value that would otherwise be collapsed to a mean or median is kept.

It reads the *same* ``prediction.h5`` + ground-truth ``h5ad`` the standard
evaluators read and adds no new inputs. It does not modify or import-time touch
any other script.

Outputs (per method x dataset x holdout)
----------------------------------------
* ``per_gene_matched.csv`` — one row per gene, over the cell-matched pairs
  (exactly the pairs ``evaluate.py`` reduces to ``pearson_median`` etc.):
  ``pearson``, ``spearman``, ``rmse``, ``mae``. Cell-matched, correspondence
  dependent — kept for reference, like the matched block in ``evaluate.py``.
* ``per_gene_generation.csv`` — one row per (held-out section, gene), the
  correspondence-free per-gene quantities ``evaluate_generation.py`` reduces to
  ``gen_gene_mean_pearson`` / ``gen_gene_var_pearson`` / ``gen_morans_agreement``
  / ``gen_field_pearson``: per-gene mean & variance (pred and GT, log-normalized),
  per-gene Moran's I (pred and GT, rank-normalized), and per-gene binned field
  Pearson. These are the PRIMARY (scale-fair) quantities for generation.
* ``per_cell.csv`` — one row per held-out GT cell (per section): nearest
  predicted cell and distance (disaggregates ``matching_rate``), cell-type of
  GT and nearest prediction with an agreement flag (disaggregates
  ``celltype_accuracy``), and for matched cells the across-gene expression
  Pearson and MAE (log-normalized).

Usage
-----
    # one prediction
    python -m src.benchmark.evaluate_disaggregated \
        --prediction results/spatialz/cosmx_nsclc_3d/holdout_S3/prediction.h5 \
        --ground-truth <data.h5ad> --output-dir <dir>

    # whole results tree (writes the three CSVs next to each prediction.h5,
    # plus concatenated masters under results/summary/disaggregated/)
    python -m src.benchmark.evaluate_disaggregated --all
    python -m src.benchmark.evaluate_disaggregated --all --methods spatialz feast
    python -m src.benchmark.evaluate_disaggregated --all --datasets cosmx_nsclc_3d --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .config import (
    DATASETS,
    METHODS,
    NN_MATCH_THRESHOLD_UM,
    RESULTS_DIR,
    SUMMARY_DIR,
)
from .evaluate import (
    align_predictions_inplace,
    load_ground_truth,
    load_prediction,
    match_cells,
    _expression_error,
    _gene_correlations,
)
from .evaluate_generation import _morans_i, _normalize_counts, _rank_normalize
from .leakage_guard import align_prediction_to_gt


# --------------------------------------------------------------------------- #
# Small helpers reusing the established math                                   #
# --------------------------------------------------------------------------- #
def _common_gene_index(pred, gt):
    """Return (common_gene_names, pred_gene_idx, gt_gene_idx) — same as evaluate."""
    common = np.intersect1d(pred["gene_names"], gt.var_names.values)
    if len(common) == 0:
        return common, None, None
    pgi = np.array([np.where(pred["gene_names"] == g)[0][0] for g in common])
    ggi = np.array([np.where(gt.var_names.values == g)[0][0] for g in common])
    return common, pgi, ggi


def _dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def _per_gene_field_pearson(pred_xy, pred_X, gt_xy, gt_X, grid=20):
    """Per-gene Pearson between the two binned spatial mean-fields.

    The per-gene counterpart of ``evaluate_generation.field_metrics`` (which
    returns only the median over genes). Occupied-in-both bins only.
    """
    all_xy = np.vstack([pred_xy, gt_xy])
    xe = np.linspace(all_xy[:, 0].min(), all_xy[:, 0].max(), grid + 1)
    ye = np.linspace(all_xy[:, 1].min(), all_xy[:, 1].max(), grid + 1)

    def binned_means(xy, X):
        xb = np.clip(np.digitize(xy[:, 0], xe) - 1, 0, grid - 1)
        yb = np.clip(np.digitize(xy[:, 1], ye) - 1, 0, grid - 1)
        flat = yb * grid + xb
        n_bins = grid * grid
        sums = np.zeros((n_bins, X.shape[1]))
        cnts = np.zeros(n_bins)
        np.add.at(sums, flat, X)
        np.add.at(cnts, flat, 1.0)
        occ = cnts > 0
        means = np.zeros_like(sums)
        means[occ] = sums[occ] / cnts[occ, None]
        return means, occ

    pm, po = binned_means(pred_xy, pred_X)
    gm, go = binned_means(gt_xy, gt_X)
    both = po & go
    n_genes = pred_X.shape[1]
    out = np.full(n_genes, np.nan)
    if both.sum() < 4:
        return out
    pm, gm = pm[both], gm[both]
    for g in range(n_genes):
        if pm[:, g].std() > 0 and gm[:, g].std() > 0:
            out[g] = np.corrcoef(pm[:, g], gm[:, g])[0, 1]
    return out


def _rowwise_pearson(P, T):
    """Per-row (per-cell) Pearson across columns (genes). Returns (n,) array."""
    Pc = P - P.mean(axis=1, keepdims=True)
    Tc = T - T.mean(axis=1, keepdims=True)
    num = (Pc * Tc).sum(axis=1)
    den = np.sqrt((Pc ** 2).sum(axis=1) * (Tc ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r


# --------------------------------------------------------------------------- #
# Core: build the three disaggregated tables for one prediction               #
# --------------------------------------------------------------------------- #
def disaggregate(prediction_path, h5ad_path, grid=20, moran_k=10,
                 ids=None):
    """Return {per_gene_matched, per_gene_generation, per_cell} DataFrames.

    ``ids`` is an optional dict of identifier columns (method/dataset/holdout_id)
    prepended to every row; ``method`` defaults to the name stored in the
    prediction file.
    """
    pred = load_prediction(prediction_path)
    holdout_sections = [str(s) for s in pred["holdout_sections"]]
    gt = load_ground_truth(h5ad_path, holdout_sections)

    ids = dict(ids) if ids else {}
    ids.setdefault("method", pred["method_name"])
    ids.setdefault("dataset", None)
    ids.setdefault("holdout_id", None)

    common, pgi, ggi = _common_gene_index(pred, gt)
    empty = pd.DataFrame()
    if len(common) == 0:
        return {"per_gene_matched": empty, "per_gene_generation": empty,
                "per_cell": empty}

    # ---- (A) cell-matched per-gene, reproducing evaluate.py exactly ---------
    # Align predicted coords onto GT (global, in-place) then NN-match — the same
    # two steps evaluate.evaluate() performs before _gene_correlations.
    pred_matched = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in pred.items()}
    align_predictions_inplace(pred_matched, gt)
    pred_idx, gt_idx = match_cells(pred_matched, gt)

    per_gene_matched_rows = []
    if len(pred_idx) > 0:
        pred_Xm = _dense(pred_matched["X"][pred_idx][:, pgi])
        gt_Xm = _dense(gt.X[gt_idx][:, ggi])
        corr = _gene_correlations(pred_Xm, gt_Xm)
        err = _expression_error(pred_Xm, gt_Xm)
        for gi, gene in enumerate(common):
            per_gene_matched_rows.append({
                **ids,
                "gene": str(gene),
                "n_matched_cells": int(len(pred_idx)),
                "pearson": _f(corr["pearson_per_gene"][gi]),
                "spearman": _f(corr["spearman_per_gene"][gi]),
                "rmse": _f(err["rmse_per_gene"][gi]),
                "mae": _f(err["mae_per_gene"][gi]),
            })
    per_gene_matched = pd.DataFrame(per_gene_matched_rows)

    # ---- (B) correspondence-free per-(section, gene) + (C) per-cell ---------
    pred_X_all = _dense(pred["X"][:, pgi])
    gt_sections = gt.obs["section"].values.astype(str)
    gt_spatial = gt.obsm["spatial"]
    has_ct = "cell_type" in gt.obs.columns

    per_gene_gen_rows = []
    per_cell_rows = []
    for sec in holdout_sections:
        pm = pred["section"] == sec
        gm = gt_sections == sec
        if pm.sum() < 5 or gm.sum() < 5:
            continue

        pred_X = pred_X_all[pm]
        gt_X = _dense(gt.X[gm][:, ggi])

        # scale-fair rank-normalized (primary) + log-normalized (mean/var).
        pR, gR = _rank_normalize(pred_X), _rank_normalize(gt_X)
        pL, gL = _normalize_counts(pred_X), _normalize_counts(gt_X)

        gt_xy = gt_spatial[gm, :2]
        pred_xy = np.column_stack([pred["x"][pm], pred["y"][pm]])
        pred_xy_al = align_prediction_to_gt(pred_xy, gt_xy, with_scale=True)

        mi_pred = _morans_i(pred_xy, pR, k=moran_k)
        mi_gt = _morans_i(gt_xy, gR, k=moran_k)
        field_r = _per_gene_field_pearson(pred_xy_al, pR, gt_xy, gR, grid=grid)

        pmean, gmean = pL.mean(axis=0), gL.mean(axis=0)
        pvar, gvar = pL.var(axis=0), gL.var(axis=0)

        for gi, gene in enumerate(common):
            per_gene_gen_rows.append({
                **ids,
                "section": str(sec),
                "gene": str(gene),
                "gene_mean_pred": _f(pmean[gi]),
                "gene_mean_gt": _f(gmean[gi]),
                "gene_var_pred": _f(pvar[gi]),
                "gene_var_gt": _f(gvar[gi]),
                "morans_i_pred": _f(mi_pred[gi]),
                "morans_i_gt": _f(mi_gt[gi]),
                "field_pearson": _f(field_r[gi]),
            })

        # ---- per-cell: every GT cell -> nearest predicted cell -------------
        tree = cKDTree(pred_xy_al)
        dists, nn_idx = tree.query(gt_xy, k=1)
        matched = dists <= NN_MATCH_THRESHOLD_UM

        gt_ct = (gt.obs["cell_type"].values[gm].astype(str)
                 if has_ct else np.array([""] * gm.sum()))
        pred_ct_sec = pred["cell_type"][pm].astype(str)

        # across-gene expression similarity for matched cells (log-normalized).
        cell_pearson = np.full(gm.sum(), np.nan)
        cell_mae = np.full(gm.sum(), np.nan)
        if matched.any():
            gt_L_sec = gL[matched]
            pred_L_sec = pL[nn_idx[matched]]
            cell_pearson[matched] = _rowwise_pearson(pred_L_sec, gt_L_sec)
            cell_mae[matched] = np.abs(pred_L_sec - gt_L_sec).mean(axis=1)

        gt_local = np.where(gm)[0]
        for j in range(gm.sum()):
            m = bool(matched[j])
            per_cell_rows.append({
                **ids,
                "section": str(sec),
                "gt_index": int(gt_local[j]),
                "gt_x": float(gt_xy[j, 0]),
                "gt_y": float(gt_xy[j, 1]),
                "nn_pred_index_in_section": int(nn_idx[j]),
                "nn_pred_x": float(pred_xy_al[nn_idx[j], 0]),
                "nn_pred_y": float(pred_xy_al[nn_idx[j], 1]),
                "nn_dist_um": float(dists[j]),
                "matched": m,
                "gt_cell_type": str(gt_ct[j]),
                "nn_pred_cell_type": str(pred_ct_sec[nn_idx[j]]) if m else "",
                "celltype_correct": (bool(gt_ct[j] == pred_ct_sec[nn_idx[j]])
                                     if m and has_ct else None),
                "cell_pearson": _f(cell_pearson[j]),
                "cell_mae": _f(cell_mae[j]),
            })

    return {
        "per_gene_matched": per_gene_matched,
        "per_gene_generation": pd.DataFrame(per_gene_gen_rows),
        "per_cell": pd.DataFrame(per_cell_rows),
    }


def _f(v):
    """NaN -> None for clean CSV/JSON; else Python float."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


# --------------------------------------------------------------------------- #
# Results-tree walker (mirrors aggregate_results / evaluate_all)              #
# --------------------------------------------------------------------------- #
def _iter_predictions(results_dir, methods=None, datasets=None):
    """Yield (pred_path, gt_h5ad, out_dir, ids) for each prediction.h5.

    Path layout: results/{method}/{dataset...}/{holdout_id}/prediction.h5
    (same parsing as aggregate_results / evaluate_all).
    """
    results_dir = Path(results_dir)
    for pred_file in sorted(results_dir.rglob("prediction.h5")):
        parts = pred_file.relative_to(results_dir).parts
        if len(parts) < 3:
            continue
        method = parts[0]
        if method not in METHODS:
            continue
        if methods and method not in methods:
            continue
        holdout_id = parts[-2]
        dataset = "/".join(parts[1:-2])
        if datasets and dataset not in datasets:
            continue
        if dataset not in DATASETS:
            print(f"  SKIP {method}/{dataset}/{holdout_id}: dataset not in registry")
            continue
        ids = {"method": method, "dataset": dataset, "holdout_id": holdout_id}
        yield pred_file, str(DATASETS[dataset]["path"]), pred_file.parent, ids


_TABLES = ("per_gene_matched", "per_gene_generation", "per_cell")


def run_all(results_dir=None, methods=None, datasets=None, force=False,
            grid=20, moran_k=10, summary_dir=None):
    """Disaggregate every prediction in the tree; write per-run + master CSVs.

    Master CSVs default to ``<results_dir>/summary/disaggregated/`` so a custom
    ``--results-dir`` keeps its summaries within its own tree (in the normal case
    ``results_dir == RESULTS_DIR`` this is exactly ``SUMMARY_DIR/disaggregated``).
    """
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    if summary_dir:
        summary_dir = Path(summary_dir)
    elif results_dir == RESULTS_DIR:
        summary_dir = SUMMARY_DIR / "disaggregated"
    else:
        summary_dir = results_dir / "summary" / "disaggregated"

    preds = list(_iter_predictions(results_dir, methods=methods, datasets=datasets))
    print(f"Found {len(preds)} prediction files")

    masters = {t: [] for t in _TABLES}
    n_ok = n_skip = n_stale = n_fail = 0
    for pred_path, gt_path, out_dir, ids in preds:
        rel = pred_path.relative_to(results_dir).parent
        csvs = [out_dir / f"{t}.csv" for t in _TABLES]
        exist = all(c.exists() for c in csvs)
        # "up to date" = all three CSVs exist AND none is older than the
        # prediction. A regenerated prediction.h5 (newer mtime) is treated as
        # not-done so new results are never scored with a stale cache.
        fresh = exist and min(c.stat().st_mtime for c in csvs) >= pred_path.stat().st_mtime
        if fresh and not force:
            n_skip += 1
            # still fold the existing (up-to-date) per-run CSVs into the masters
            for t in _TABLES:
                try:
                    masters[t].append(pd.read_csv(out_dir / f"{t}.csv"))
                except Exception:
                    pass
            continue
        if exist and not force:
            n_stale += 1
            print(f"  Re-evaluating {rel} (prediction newer than cached CSVs)...")
        else:
            print(f"  Disaggregating {rel}...")
        try:
            tables = disaggregate(str(pred_path), gt_path, grid=grid,
                                  moran_k=moran_k, ids=ids)
            for t in _TABLES:
                df = tables[t]
                df.to_csv(out_dir / f"{t}.csv", index=False)
                if len(df):
                    masters[t].append(df)
            pgm = tables["per_gene_matched"]
            n_genes = len(pgm)
            n_cells = len(tables["per_cell"])
            print(f"    {n_genes} matched-gene rows, {n_cells} cell rows")
            n_ok += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            n_fail += 1

    summary_dir.mkdir(parents=True, exist_ok=True)
    for t in _TABLES:
        if masters[t]:
            alldf = pd.concat(masters[t], ignore_index=True)
            path = summary_dir / f"{t}.csv"
            alldf.to_csv(path, index=False)
            print(f"Wrote {len(alldf)} rows -> {path}")
    print(f"\nDone: {n_ok} evaluated ({n_stale} of them re-evaluated as stale), "
          f"{n_skip} skipped (up to date), {n_fail} failed")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Disaggregated per-gene / per-cell evaluation (no reduction)")
    ap.add_argument("--prediction", help="Path to a single prediction.h5")
    ap.add_argument("--ground-truth", help="Path to the dataset's data.h5ad")
    ap.add_argument("--output-dir",
                    help="Directory for the three CSVs (single-prediction mode; "
                         "defaults to the prediction's own directory)")
    ap.add_argument("--all", action="store_true",
                    help="Walk the whole results tree instead of one prediction")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--methods", nargs="+", help="(--all) restrict to these methods")
    ap.add_argument("--datasets", nargs="+", help="(--all) restrict to these datasets")
    ap.add_argument("--force", action="store_true",
                    help="(--all) recompute every prediction, even ones whose "
                         "CSVs are already up to date (by default only new or "
                         "regenerated predictions are (re)evaluated)")
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--moran-k", type=int, default=10)
    args = ap.parse_args()

    if args.all:
        run_all(results_dir=args.results_dir, methods=args.methods,
                datasets=args.datasets, force=args.force,
                grid=args.grid, moran_k=args.moran_k)
        return

    if not args.prediction or not args.ground_truth:
        ap.error("single-prediction mode needs --prediction and --ground-truth "
                 "(or use --all for the whole results tree)")

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.prediction).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = disaggregate(args.prediction, args.ground_truth,
                          grid=args.grid, moran_k=args.moran_k)
    for t in _TABLES:
        df = tables[t]
        path = out_dir / f"{t}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {len(df):>7d} rows -> {path}")


if __name__ == "__main__":
    main()
