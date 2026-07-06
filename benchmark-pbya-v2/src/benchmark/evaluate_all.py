"""Evaluate all prediction.h5 files against their ground-truth datasets.

Walks the results tree, finds prediction.h5 files, and runs evaluate.py
on each one. Useful for:
  - Re-evaluating after changing metrics
  - Evaluating predictions produced with --no-eval
  - Regenerating metrics.json after code changes

Usage:
    python -m src.benchmark.evaluate_all                    # evaluate all missing
    python -m src.benchmark.evaluate_all --force            # re-evaluate everything
    python -m src.benchmark.evaluate_all --methods feast    # specific methods
    python -m src.benchmark.evaluate_all --datasets cosmx_nsclc_3d  # specific datasets
"""

import argparse
import json
import time
from pathlib import Path

from .config import DATASETS, METHODS, RESULTS_DIR
from .evaluate import evaluate


def find_predictions(results_dir=None, methods=None, datasets=None):
    """Find all prediction.h5 files in the results tree.

    Yields (prediction_path, input_h5ad_path, metrics_path) tuples.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    results_dir = Path(results_dir)

    for pred_file in sorted(results_dir.rglob("prediction.h5")):
        parts = pred_file.relative_to(results_dir).parts
        if len(parts) < 3:
            continue

        method = parts[0]
        # Skip non-method directories
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

        input_path = str(DATASETS[dataset]["path"])
        metrics_path = pred_file.parent / "metrics.json"

        yield pred_file, input_path, metrics_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate all predictions")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--methods", nargs="+", help="Only evaluate these methods")
    parser.add_argument("--datasets", nargs="+", help="Only evaluate these datasets")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if metrics.json exists")
    args = parser.parse_args()

    predictions = list(find_predictions(
        args.results_dir, methods=args.methods, datasets=args.datasets))
    print(f"Found {len(predictions)} prediction files")

    n_eval = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()

    for pred_path, input_path, metrics_path in predictions:
        rel = pred_path.relative_to(RESULTS_DIR).parent
        if metrics_path.exists() and not args.force:
            n_skip += 1
            continue

        print(f"  Evaluating {rel}...")
        try:
            metrics = evaluate(str(pred_path), input_path, str(metrics_path))
            pearson = metrics.get("pearson_median", "N/A")
            print(f"    Pearson median: {pearson}")
            n_eval += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            n_fail += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s: {n_eval} evaluated, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
