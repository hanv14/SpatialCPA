"""Gather all metrics.json files into a single summary CSV."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DATASETS, RESULTS_DIR, METRIC_NAMES, SUMMARY_DIR

# Methods excluded from aggregation
EXCLUDED_METHODS = {"stode"}


def aggregate(results_dir=None):
    """Walk results tree, collect metrics.json files into a DataFrame."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    results_dir = Path(results_dir)

    rows = []
    for metrics_file in sorted(results_dir.rglob("metrics.json")):
        # Parse path: results/{method}/{dataset}/{holdout_id}/metrics.json
        parts = metrics_file.relative_to(results_dir).parts
        if len(parts) < 3:
            continue

        method = parts[0]
        if method == "summary" or method in EXCLUDED_METHODS:
            continue
        # Skip paper reproduction directories
        if method.endswith("_paper_repro") or method.endswith("_paper_audit"):
            continue
        # dataset may contain / (e.g., merfish_hypothalamus/animal_1)
        holdout_id = parts[-2]
        dataset = "/".join(parts[1:-2])

        with open(metrics_file) as f:
            metrics = json.load(f)

        row = {
            "method": method,
            "dataset": dataset,
            "holdout_id": holdout_id,
        }

        # Add tier and technology from dataset registry
        ds_info = DATASETS.get(dataset, {})
        row["tier"] = ds_info.get("tier")
        row["technology"] = ds_info.get("technology")

        # Add resource info if available
        resources_file = metrics_file.parent / "resources.json"
        if resources_file.exists():
            with open(resources_file) as f:
                res = json.load(f)
            row["wall_time_s"] = res.get("wall_time_s")
            row["peak_rss_mb"] = res.get("peak_rss_mb")
            row["peak_gpu_mb"] = res.get("peak_gpu_mb")

        # Add all metric columns
        for m in METRIC_NAMES:
            row[m] = metrics.get(m)

        row["n_holdout_cells_gt"] = metrics.get("n_holdout_cells_gt")
        row["n_predicted_cells"] = metrics.get("n_predicted_cells")
        row["n_matched_cells"] = metrics.get("n_matched_cells")
        row["n_common_genes"] = metrics.get("n_common_genes")
        row["error"] = metrics.get("error")

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def summarize(df):
    """Aggregate per-holdout results into per method-dataset means."""
    if len(df) == 0:
        return pd.DataFrame()

    metric_cols = [m for m in METRIC_NAMES if m in df.columns]
    group_cols = ["method", "dataset", "tier", "technology"]
    agg_dict = {m: "mean" for m in metric_cols}
    agg_dict["holdout_id"] = "count"
    agg_dict["wall_time_s"] = "sum"

    summary = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    summary.rename(columns={"holdout_id": "n_holdouts"}, inplace=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--output", default=str(SUMMARY_DIR / "all_metrics.csv"))
    args = parser.parse_args()

    df = aggregate(args.results_dir)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")

    # Write summary CSV
    summary = summarize(df)
    summary_path = Path(args.output).parent / "summary_by_method_dataset.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {len(summary)} summary rows to {summary_path}")

    # Print summary
    if len(df) > 0:
        print("\n── Summary by method ──")
        agg = df.groupby("method")[["pearson_median", "ssim_median",
                                     "density_pearson", "matching_rate"]].median()
        print(agg.to_string())


if __name__ == "__main__":
    main()
