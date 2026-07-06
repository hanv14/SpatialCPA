"""Run the full benchmarking campaign: all methods × all available datasets.

This script bypasses conda env isolation and runs methods directly in the
current environment, since all dependencies are installed.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np

from .config import DATASETS, RESULTS_DIR
from .evaluate import evaluate
from .holdout import generate_holdouts
from .resource_monitor import ResourceMonitor, write_resources


def run_feast_holdout(adata, gene_names, holdout_config, output_dir, seed=42):
    """Run FEAST on a single holdout."""
    from .methods import run_feast

    pred_path = output_dir / "prediction.h5"
    train_adata, target_z = run_feast.prepare_input(adata, holdout_config["holdout_sections"])

    t0 = time.time()
    results = run_feast.run_method(train_adata, target_z, seed=seed)
    wall_time = time.time() - t0

    method_params = {"seed": seed, "paste2_s": 0.7, "paste2_alpha": 0.1}
    run_feast.format_output(results, gene_names, holdout_config["holdout_sections"],
                            method_params, wall_time, str(pred_path))
    return pred_path, wall_time


def run_spatialz_holdout(adata, gene_names, holdout_config, output_dir, seed=42):
    """Run SpatialZ on a single holdout."""
    from .methods import run_spatialz

    pred_path = output_dir / "prediction.h5"
    train_adata, target_z = run_spatialz.prepare_input(adata, holdout_config["holdout_sections"])

    t0 = time.time()
    results = run_spatialz.run_method(train_adata, target_z, seed=seed,
                                       syn_mode="default", device="auto")
    wall_time = time.time() - t0

    method_params = {"seed": seed, "syn_mode": "default"}
    run_spatialz.format_output(results, gene_names, holdout_config["holdout_sections"],
                                method_params, wall_time, str(pred_path))
    return pred_path, wall_time


def run_isost_holdout(adata, gene_names, holdout_config, output_dir, seed=42):
    """Run isoST on a single holdout."""
    from .methods import run_isost

    pred_path = output_dir / "prediction.h5"
    train_adata, target_z = run_isost.prepare_input(adata, holdout_config["holdout_sections"])

    t0 = time.time()
    results = run_isost.run_method(train_adata, target_z, seed=seed,
                                    epochs=[100, 100, 100], device="cuda:0")
    wall_time = time.time() - t0

    method_params = {"seed": seed, "epochs": [100, 100, 100]}
    run_isost.format_output(results, gene_names, holdout_config["holdout_sections"],
                            method_params, wall_time, str(pred_path))
    return pred_path, wall_time


METHOD_RUNNERS = {
    "feast": run_feast_holdout,
    "spatialz": run_spatialz_holdout,
    "isost": run_isost_holdout,
}


def run_campaign(methods=None, datasets=None, strategy="leave_one_out",
                 k_values=None, skip_existing=True, seed=42):
    """Run full benchmarking campaign."""

    if methods is None:
        methods = ["feast", "spatialz", "isost"]
    if datasets is None:
        datasets = list(DATASETS.keys())
    if k_values is None:
        k_values = [2]

    for dataset_name in datasets:
        info = DATASETS[dataset_name]
        h5ad_path = str(info["path"])

        if not info["path"].exists():
            print(f"\n[SKIP] {dataset_name}: file not found")
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        adata = ad.read_h5ad(h5ad_path)
        gene_names = adata.var_names.tolist()
        n_sections = len(adata.obs["section"].unique())
        print(f"  {adata.n_obs:,} cells, {adata.n_vars} genes, {n_sections} sections")

        # Skip datasets with too few sections
        if n_sections < 3:
            print(f"  SKIP: only {n_sections} sections (need >= 3 for LOO)")
            continue

        # Generate holdouts
        holdouts = []
        if "leave_one_out" in strategy if isinstance(strategy, list) else strategy == "leave_one_out":
            holdouts.extend(generate_holdouts(h5ad_path, strategy="leave_one_out"))
        if "leave_k_out" in strategy if isinstance(strategy, list) else strategy == "leave_k_out":
            for k in k_values:
                holdouts.extend(generate_holdouts(h5ad_path, strategy="leave_k_out", k=k))

        print(f"  {len(holdouts)} holdout configs")

        for method in methods:
            if method not in METHOD_RUNNERS:
                print(f"  [SKIP] Unknown method: {method}")
                continue

            runner = METHOD_RUNNERS[method]
            print(f"\n  --- {method} on {dataset_name} ---")

            for i, holdout in enumerate(holdouts):
                hid = holdout["holdout_id"]
                out_dir = RESULTS_DIR / method / dataset_name / hid
                metrics_path = out_dir / "metrics.json"

                if skip_existing and metrics_path.exists():
                    print(f"  [{i+1}/{len(holdouts)}] SKIP {hid} (exists)")
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"  [{i+1}/{len(holdouts)}] {hid}...", end=" ", flush=True)

                try:
                    pred_path, wall_time = runner(
                        adata, gene_names, holdout, out_dir, seed=seed
                    )

                    if pred_path.exists():
                        metrics = evaluate(str(pred_path), h5ad_path,
                                           str(metrics_path))
                        pearson = metrics.get("pearson_median", "N/A")
                        ssim = metrics.get("ssim_median", "N/A")
                        print(f"pearson={pearson:.3f} ssim={ssim:.3f} ({wall_time:.0f}s)"
                              if isinstance(pearson, float) else f"error ({wall_time:.0f}s)")
                    else:
                        print(f"no prediction ({wall_time:.0f}s)")

                except Exception as e:
                    print(f"FAILED: {e}")
                    traceback.print_exc()
                    # Write error to metrics file
                    with open(metrics_path, "w") as f:
                        json.dump({"error": str(e)}, f)

        del adata

    # Aggregate
    print(f"\n{'='*70}")
    print("Aggregating results...")
    from .aggregate_results import aggregate
    from .config import SUMMARY_DIR
    df = aggregate()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_DIR / "all_metrics.csv", index=False)
    print(f"Wrote {len(df)} results to {SUMMARY_DIR / 'all_metrics.csv'}")

    if len(df) > 0:
        print("\n── Summary by method ──")
        summary = df.groupby("method")[["pearson_median", "ssim_median",
                                         "density_pearson"]].median()
        print(summary.to_string())

    # Plots
    print("\nGenerating plots...")
    from .plot_results import generate_all_plots
    generate_all_plots()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full benchmark campaign")
    parser.add_argument("--methods", nargs="+", default=["feast", "spatialz", "isost"])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to run (default: all)")
    parser.add_argument("--strategy", default="leave_one_out")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even if results exist")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_campaign(
        methods=args.methods,
        datasets=args.datasets,
        strategy=args.strategy,
        skip_existing=not args.no_skip,
        seed=args.seed,
    )
