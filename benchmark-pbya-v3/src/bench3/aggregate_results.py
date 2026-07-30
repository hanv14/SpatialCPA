"""Collect every metrics.json in the v3 results tree into tidy CSVs.

Writes three tables under ``results/summary/``:

``all_metrics.csv``
    one row per (method, holdout), all metric columns.
``per_section_metrics.csv``
    one row per (method, holdout, held-out section) — the disaggregated view,
    which matters here because sections 2, 4 and 6 sit at different depths in
    the volume and are not equally easy.
``summary_by_method.csv``
    one row per method, averaged over holdouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DATASET_NAME, METHOD_ORDER, METHODS, METRIC_NAMES, RESULTS_DIR, SUMMARY_DIR


def _method_sort_key(m):
    return (METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER), m)


def aggregate(results_dir=None):
    """Return (per-run DataFrame, per-section DataFrame)."""
    results_dir = Path(results_dir or RESULTS_DIR)
    rows, section_rows = [], []

    for metrics_file in sorted(results_dir.rglob("metrics.json")):
        parts = metrics_file.relative_to(results_dir).parts
        if len(parts) < 3 or parts[0] in ("summary",) or parts[0].startswith("_"):
            continue
        method = parts[0]
        holdout_id = parts[-2]
        dataset = "/".join(parts[1:-2]) or DATASET_NAME

        metrics = json.loads(metrics_file.read_text())
        per_section = metrics.get("per_section", {}) or {}

        row = {
            "method": method,
            "dataset": dataset,
            "holdout_id": holdout_id,
            "family": METHODS.get(method, {}).get("family"),
            "n_holdout_cells_gt": metrics.get("n_holdout_cells_gt"),
            "n_predicted_cells": metrics.get("n_predicted_cells"),
            "n_common_genes": metrics.get("n_common_genes"),
            "error": metrics.get("error"),
        }
        res_file = metrics_file.parent / "resources.json"
        if res_file.exists():
            res = json.loads(res_file.read_text())
            row["wall_time_s"] = res.get("wall_time_s")
            row["peak_rss_mb"] = res.get("peak_rss_mb")
            row["peak_gpu_mb"] = res.get("peak_gpu_mb")
        for m in METRIC_NAMES:
            row[m] = metrics.get(m)
        rows.append(row)

        for section, sm in per_section.items():
            if not isinstance(sm, dict):
                continue
            srow = {"method": method, "dataset": dataset,
                    "holdout_id": holdout_id, "section": section}
            srow.update({(k if k.startswith("n_") else f"paper_{k}"): v
                         for k, v in sm.items()})
            section_rows.append(srow)

    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(section_rows)
    for frame in (df, sdf):
        if len(frame):
            frame.sort_values(
                by=["method", "holdout_id"], key=lambda c: (
                    c.map(_method_sort_key) if c.name == "method" else c),
                inplace=True, ignore_index=True)
    return df, sdf


def summarize(df):
    """Average each metric per (dataset, method) across holdouts.

    Grouping includes the dataset on purpose. Averaging a method's STARmap and
    ExSeq scores into one number would report a quantity that describes neither
    volume — and the paper's protocol is defined on STARmap alone, so a pooled
    row could not be quoted as the reproduction either.
    """
    if len(df) == 0:
        return pd.DataFrame()
    metric_cols = [m for m in METRIC_NAMES if m in df.columns]
    agg = {m: "mean" for m in metric_cols}
    agg["holdout_id"] = "count"
    if "wall_time_s" in df.columns:
        agg["wall_time_s"] = "sum"
    keys = ["dataset", "method", "family"] if "dataset" in df.columns \
        else ["method", "family"]
    out = df.groupby(keys, dropna=False).agg(agg).reset_index()
    out.rename(columns={"holdout_id": "n_runs"}, inplace=True)
    sort_cols = [c for c in ("dataset", "method") if c in out.columns]
    return out.sort_values(
        sort_cols,
        key=lambda c: c.map(_method_sort_key) if c.name == "method" else c,
        ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Aggregate v3 results")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out-dir", default=str(SUMMARY_DIR))
    args = ap.parse_args()

    df, sdf = aggregate(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print(f"No metrics.json found under {args.results_dir}. "
              f"Run `python -m src.bench3.run_all` first.")
        return

    df.to_csv(out_dir / "all_metrics.csv", index=False)
    print(f"Wrote {len(df)} rows to {out_dir / 'all_metrics.csv'}")
    if len(sdf):
        sdf.to_csv(out_dir / "per_section_metrics.csv", index=False)
        print(f"Wrote {len(sdf)} rows to {out_dir / 'per_section_metrics.csv'}")
    summary = summarize(df)
    summary.to_csv(out_dir / "summary_by_method.csv", index=False)
    print(f"Wrote {len(summary)} rows to {out_dir / 'summary_by_method.csv'}")

    headline = ["paper_umap_mixing", "paper_morans_pearson", "paper_gearys_pearson",
                "paper_marker_depth_r", "paper_celltype_localization"]
    cols = [c for c in headline if c in summary.columns]
    if cols:
        if "dataset" in summary.columns:
            for ds, sub in summary.groupby("dataset"):
                print(f"\n── headline paper metrics — {ds} (mean over holdouts) ──")
                print(sub.set_index("method")[cols].round(3).to_string())
        else:
            print("\n── headline paper metrics (mean over holdouts) ──")
            print(summary.set_index("method")[cols].round(3).to_string())


if __name__ == "__main__":
    main()
