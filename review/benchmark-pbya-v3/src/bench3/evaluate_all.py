"""(Re-)evaluate every prediction under the v3 results tree.

Prediction and evaluation are decoupled: a metric definition can change without
re-running the (expensive) methods, and a method failure never loses evaluation
state.

Usage:
    python -m src.bench3.evaluate_all              # evaluate what's missing
    python -m src.bench3.evaluate_all --force      # re-evaluate everything
    python -m src.bench3.evaluate_all --methods spatialz --force
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .config import DATASET_NAME, RESULTS_DIR, dataset_path, resolve_dataset_arg
from .run_benchmark import evaluate_prediction


def main():
    ap = argparse.ArgumentParser(description="Evaluate all v3 predictions")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--dataset", default=None,
                    help="ground-truth h5ad to score against. Default: resolve it "
                         "per prediction from the dataset in its results path, so a "
                         "multi-dataset tree is never scored against one volume. "
                         "Accepts a dataset NAME or a path")
    ap.add_argument("--dataset-name", default=None,
                    help="restrict to predictions of this dataset")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict to these methods")
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate even when metrics.json exists")
    ap.add_argument("--no-umap", action="store_true")
    args = ap.parse_args()

    override_gt = resolve_dataset_arg(args.dataset) if args.dataset else None
    results_dir = Path(args.results_dir)
    n_done = n_skip = n_fail = 0
    for pred_path in sorted(results_dir.rglob("prediction.h5")):
        rel = pred_path.relative_to(results_dir).parts
        method = rel[0]
        if method.startswith("_") or method == "summary":
            continue
        if args.methods and method not in args.methods:
            continue
        ds_name = "/".join(rel[1:-2]) or DATASET_NAME
        if args.dataset_name and ds_name != args.dataset_name:
            continue
        gt_path = override_gt if override_gt is not None else dataset_path(ds_name)
        if not Path(gt_path).exists():
            print(f"SKIP {'/'.join(rel[:-1])}: ground truth for {ds_name!r} not "
                  f"found at {gt_path}")
            n_skip += 1
            continue
        metrics_path = pred_path.parent / "metrics.json"
        if metrics_path.exists() and not args.force:
            n_skip += 1
            continue
        label = "/".join(rel[:-1])
        print(f"Evaluating {label} (gt={Path(gt_path).name}) ...")
        try:
            m = evaluate_prediction(pred_path, dataset_path=str(gt_path),
                                    metrics_path=metrics_path,
                                    use_umap=not args.no_umap)
            print(f"  umap_mixing={m.get('paper_umap_mixing')} "
                  f"marker_depth_r={m.get('paper_marker_depth_r')}")
            n_done += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            n_fail += 1

    print(f"\nEvaluated {n_done}, skipped {n_skip} (already had metrics.json), "
          f"failed {n_fail}")


if __name__ == "__main__":
    main()
