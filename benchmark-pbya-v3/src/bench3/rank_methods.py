"""Rank the methods on the paper's validation criteria.

The four things the SpatialZ paper validated map onto four metric groups. Each
group is scored, methods are ranked within each group, and the composite rank is
the mean of the group ranks — so a method cannot win by being excellent at one
criterion and hopeless at the rest, and no single group dominates just because it
happens to contribute more individual numbers.

    continuity     paper_umap_mixing            ↑   real vs reconstructed overlap
                   paper_umap_centroid_dist     ↓
    autocorr       paper_morans_pearson         ↑   Moran's I agreement across genes
                   paper_gearys_pearson         ↑   Geary's C agreement across genes
                   paper_morans_mae             ↓   ... and at the right level
                   paper_gearys_mae             ↓       (catches over-smoothing)
    markers        paper_marker_depth_r         ↑   laminar profile of Flt1/Pcp4/Cux2
                   paper_marker_field_r         ↑   2-D pattern of the same markers
                   paper_marker_morans_mae      ↓
    localization   paper_celltype_localization  ↑   cell types in the right place
                   paper_gene_mean_spearman     ↑   expression similarity

``--include-gen`` adds v2's correspondence-free generation metrics as a fifth
group, which is useful for checking that the v3 ranking and the v2 sweep tell the
same story.

Usage:
    python -m src.bench3.rank_methods
    python -m src.bench3.rank_methods --design loo --include-gen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate_results import aggregate
from .config import DATASET_SPECS, METHOD_ORDER, RESULTS_DIR, SUMMARY_DIR

# group -> [(metric, direction)] with +1 = higher better, -1 = lower better.
GROUPS = {
    "continuity": [
        ("paper_umap_mixing", +1),
        ("paper_umap_centroid_dist", -1),
    ],
    "autocorr": [
        ("paper_morans_pearson", +1),
        ("paper_gearys_pearson", +1),
        ("paper_morans_mae", -1),
        ("paper_gearys_mae", -1),
    ],
    "markers": [
        ("paper_marker_depth_r", +1),
        ("paper_marker_field_r", +1),
        ("paper_marker_morans_mae", -1),
    ],
    "localization": [
        ("paper_celltype_localization", +1),
        ("paper_gene_mean_spearman", +1),
    ],
}

GEN_GROUP = [
    ("gen_coexpression_agreement", +1),
    ("gen_morans_agreement", +1),
    ("gen_sinkhorn", -1),
    ("gen_celltype_nhood_agreement", +1),
    ("gen_gene_var_pearson", +1),
]

HEADLINE = [
    "paper_umap_mixing", "paper_morans_pearson", "paper_gearys_pearson",
    "paper_marker_depth_r", "paper_marker_field_r",
    "paper_celltype_localization", "paper_cell_count_ratio",
]

# Reported alongside the scored metrics but deliberately NOT ranked: the number
# of synthesized cells is emergent in generation-only mode, so it is a diagnostic
# ("did the method produce a plausible amount of tissue?"), not a quality score.
DIAGNOSTICS = ["paper_cell_count_ratio"]


def rank(df, include_gen=False):
    """Per-method group scores + composite rank (1 = best), for ONE dataset.

    Ranking is only meaningful within a dataset: the criteria are scored against
    that volume's ground truth and its marker panel, and the composite is a rank
    among the methods run on it. ``rank_all`` applies this per dataset; pooling
    them would rank methods on an average of two different tasks.
    """
    groups = dict(GROUPS)
    if include_gen:
        groups["generation"] = GEN_GROUP

    scored_cols = sorted({m for ms in groups.values() for m, _ in ms
                          if m in df.columns})
    if not scored_cols:
        return pd.DataFrame()
    metric_cols = scored_cols + [c for c in DIAGNOSTICS if c in df.columns]

    g = (df.groupby("method")
           .agg({**{m: "mean" for m in metric_cols}, "holdout_id": "count"})
           .rename(columns={"holdout_id": "n_runs"})
           .reset_index())

    group_rank_cols = []
    for gname, metrics in groups.items():
        present = [(m, d) for m, d in metrics
                   if m in g.columns and g[m].notna().any()]
        if not present:
            # Every metric in this criterion is missing (e.g. UMAP was skipped);
            # ranking on it would hand out meaningless ties, so leave it out of
            # the composite entirely rather than diluting it.
            continue
        # Rank within each metric (1 = best), then average across the group so
        # every criterion carries equal weight regardless of metric count.
        ranks = []
        for m, direction in present:
            ranks.append(g[m].rank(ascending=(direction < 0), method="min",
                                   na_option="bottom"))
        col = f"rank_{gname}"
        g[col] = pd.concat(ranks, axis=1).mean(axis=1)
        group_rank_cols.append(col)

    g["composite_rank"] = g[group_rank_cols].mean(axis=1) if group_rank_cols else np.nan
    g["__order"] = g["method"].map(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER))
    return g.sort_values(["composite_rank", "__order"]).drop(columns="__order")


def rank_all(df, include_gen=False):
    """Rank each dataset separately; returns one frame with a ``dataset`` column."""
    if "dataset" not in df.columns:
        return rank(df, include_gen=include_gen)
    out = []
    for ds, sub in df.groupby("dataset"):
        r = rank(sub, include_gen=include_gen)
        if len(r):
            r.insert(0, "dataset", ds)
            out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _fmt(v, width=9):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'NA':>{width}}"
    return f"{v:>+{width}.3f}"


def print_tables(ranked):
    metric_cols = [m for m in HEADLINE if m in ranked.columns]
    print("\n══ headline paper metrics (mean over holdouts) ══")
    header = f"  {'method':<20s}" + "".join(
        f"{m.replace('paper_', ''):>22s}" for m in metric_cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in ranked.iterrows():
        print(f"  {r['method']:<20s}" + "".join(
            _fmt(r.get(m), 22) for m in metric_cols))

    rank_cols = [c for c in ranked.columns if c.startswith("rank_")]
    print("\n══ ranking by criterion (1 = best) ══")
    header = (f"  {'method':<20s}" + "".join(f"{c.replace('rank_', ''):>14s}"
                                             for c in rank_cols)
              + f"{'composite':>12s}{'n':>4s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in ranked.iterrows():
        cells = "".join(f"{r[c]:>14.2f}" for c in rank_cols)
        print(f"  {r['method']:<20s}{cells}{r['composite_rank']:>12.2f}"
              f"{int(r['n_runs']):>4d}")

    if "paper_cell_count_ratio" in ranked.columns:
        print("\n  (cell_count_ratio is diagnostic, not scored: the number of cells "
              "is\n   emergent in generation-only mode — 1.0 means the method "
              "synthesized as\n   many cells as the held-out section really had.)")


def main():
    ap = argparse.ArgumentParser(description="Rank methods on the paper criteria")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--design", default=None, choices=[None, "paper", "loo"],
                    help="restrict to holdouts of this design")
    ap.add_argument("--dataset-name", default=None,
                    help="rank only this dataset (default: each one separately)")
    ap.add_argument("--include-gen", action="store_true",
                    help="add v2's generation metrics as a fifth criterion")
    ap.add_argument("--csv", default=str(Path(SUMMARY_DIR) / "ranking.csv"))
    args = ap.parse_args()

    df, _ = aggregate(args.results_dir)
    if len(df) == 0:
        print(f"No metrics.json found under {args.results_dir}.")
        return
    if args.design == "paper":
        df = df[df["holdout_id"].str.startswith("paper")]
    elif args.design == "loo":
        df = df[df["holdout_id"].str.startswith("loo")]
    if len(df) == 0:
        print(f"No runs for design={args.design}.")
        return

    if args.dataset_name:
        df = df[df["dataset"] == args.dataset_name]
        if len(df) == 0:
            print(f"No runs for dataset={args.dataset_name!r}.")
            return

    ranked = rank_all(df, include_gen=args.include_gen)
    if len(ranked) == 0:
        print("No paper metrics found — run evaluate_all first.")
        return

    # One table per dataset. A composite rank is a rank *among the methods run on
    # that volume*; concatenating two datasets' ranks would compare positions in
    # different races.
    if "dataset" in ranked.columns:
        for ds, sub in ranked.groupby("dataset"):
            kind = DATASET_SPECS.get(ds, {}).get("kind")
            tag = f"  [{kind}]" if kind else ""
            print(f"\n{'=' * 72}\n  dataset: {ds}{tag}\n{'=' * 72}")
            print_tables(sub)
    else:
        print_tables(ranked)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.csv, index=False)
    print(f"\nWrote ranking to {args.csv}")
    if "dataset" in ranked.columns and ranked["dataset"].nunique() > 1:
        print("Note: ranks are within a dataset. The STARmap row is the paper "
              "reproduction; other datasets are protocol analogues (config."
              "DATASET_SPECS['kind']). Do not average their composites.")


if __name__ == "__main__":
    main()
