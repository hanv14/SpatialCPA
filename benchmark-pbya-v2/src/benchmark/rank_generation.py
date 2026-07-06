"""Rank methods on the PRIMARY generation metrics (benchmark-pbya-v2).

Reads all ``metrics.json`` under the results tree (via
:func:`aggregate_results.aggregate`) and prints, per dataset, a method × metric
table averaged over holdout sections, plus a composite ranking. Only the
correspondence-free, scale-fair primary metrics are used for ranking — the
cell-matched metrics are not valid for de-novo generation (see README).

Usage:
    python -m src.benchmark.rank_generation                 # all datasets
    python -m src.benchmark.rank_generation --dataset cosmx_nsclc_3d
    python -m src.benchmark.rank_generation --csv ranking.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate_results import aggregate
from .config import RESULTS_DIR, SUMMARY_DIR

# Primary generation metrics and their direction (+1 = higher better, -1 = lower).
PRIMARY = [
    ("gen_coexpression_agreement", +1),
    ("gen_morans_agreement", +1),
    ("gen_sinkhorn", -1),               # OT distance: lower = better
    ("gen_celltype_composition", +1),
    ("gen_gene_var_pearson", +1),       # variance preservation (over-smoothing guard)
]


def _fmt(v):
    return "   NA " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"


def rank(df, datasets=None):
    """Return per-(dataset, method) mean of primary metrics + composite rank."""
    metric_cols = [m for m, _ in PRIMARY if m in df.columns]
    if not metric_cols:
        return pd.DataFrame()

    grouped = (df.groupby(["dataset", "method"])
                 .agg({**{m: "mean" for m in metric_cols}, "holdout_id": "count"})
                 .rename(columns={"holdout_id": "n_holdouts"})
                 .reset_index())
    if datasets:
        grouped = grouped[grouped["dataset"].isin(datasets)]

    # Composite rank per dataset: average of per-metric ranks (1 = best).
    out = []
    for ds, g in grouped.groupby("dataset"):
        g = g.copy()
        rank_cols = []
        for m, direction in PRIMARY:
            if m not in g.columns:
                continue
            # ascending=True ranks smallest as 1; flip for "higher is better".
            r = g[m].rank(ascending=(direction < 0), method="min", na_option="bottom")
            g[f"__rank_{m}"] = r
            rank_cols.append(f"__rank_{m}")
        g["composite_rank"] = g[rank_cols].mean(axis=1) if rank_cols else np.nan
        g = g.sort_values("composite_rank")
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else grouped


def print_tables(ranked):
    metric_cols = [m for m, _ in PRIMARY if m in ranked.columns]
    for ds, g in ranked.groupby("dataset"):
        print(f"\n══ {ds} ══  (↑ better, except gen_sinkhorn ↓; ranked by composite)")
        header = f"  {'method':22s} " + " ".join(f"{m.replace('gen_',''):>16s}" for m in metric_cols) + f"  {'n':>3s} {'rank':>5s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, r in g.sort_values("composite_rank").iterrows():
            cells = " ".join(f"{_fmt(r[m]):>16s}" for m in metric_cols)
            print(f"  {r['method']:22s} {cells}  {int(r['n_holdouts']):>3d} {r['composite_rank']:>5.2f}")

    # Overall (mean composite rank across datasets).
    if "composite_rank" in ranked.columns and ranked["dataset"].nunique() > 1:
        overall = (ranked.groupby("method")["composite_rank"].mean()
                          .sort_values())
        print("\n══ overall (mean composite rank across datasets, lower = better) ══")
        for method, v in overall.items():
            print(f"  {method:22s} {v:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Rank methods on primary generation metrics")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--csv", default=str(SUMMARY_DIR / "generation_ranking.csv"))
    args = ap.parse_args()

    df = aggregate(args.results_dir)
    if len(df) == 0:
        print("No metrics.json found under", args.results_dir)
        return
    ranked = rank(df, datasets=args.dataset)
    if len(ranked) == 0:
        print("No primary generation metrics found (run evaluate_generation first).")
        return
    print_tables(ranked)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    keep = ["dataset", "method", "n_holdouts", "composite_rank"] + \
           [m for m, _ in PRIMARY if m in ranked.columns]
    ranked[keep].to_csv(args.csv, index=False)
    print(f"\nWrote ranking to {args.csv}")


if __name__ == "__main__":
    main()
