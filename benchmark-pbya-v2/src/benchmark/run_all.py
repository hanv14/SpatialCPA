"""Orchestrate a full benchmarking campaign.

Usage examples:
    # Tier 1 datasets, all available methods, LOO only
    python -m src.benchmark.run_all --tier 1 --strategy leave_one_out

    # Specific method + dataset
    python -m src.benchmark.run_all --methods feast --datasets cosmx_nsclc_3d

    # Full campaign (all tiers, both strategies)
    python -m src.benchmark.run_all --tier 1 2 --strategy leave_one_out leave_k_out --k 2 3
"""

import argparse
import json
import time
from pathlib import Path

from .config import DATASETS, METHODS, RESULTS_DIR
from .holdout import generate_holdouts
from .run_benchmark import run_single


def build_campaign(methods, datasets, strategies, k_values, exclude_boundary=True):
    """Build list of (method, dataset, holdout_config) tuples."""
    campaign = []
    for dataset in datasets:
        h5ad_path = str(DATASETS[dataset]["path"])
        for strategy in strategies:
            if strategy == "leave_one_out":
                holdouts = generate_holdouts(h5ad_path, strategy="leave_one_out",
                                             exclude_boundary=exclude_boundary)
            elif strategy == "leave_k_out":
                holdouts = []
                for k in k_values:
                    holdouts.extend(generate_holdouts(
                        h5ad_path, strategy="leave_k_out", k=k,
                        exclude_boundary=exclude_boundary,
                    ))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            for holdout in holdouts:
                for method in methods:
                    campaign.append((method, dataset, holdout))
    return campaign


def main():
    parser = argparse.ArgumentParser(description="Run full benchmarking campaign")
    parser.add_argument("--methods", nargs="+",
                        help="Methods to run (default: all available)")
    parser.add_argument("--datasets", nargs="+",
                        help="Datasets to use (default: all in specified tiers)")
    parser.add_argument("--tier", nargs="+", type=int, default=[1],
                        help="Dataset tiers to include (default: 1)")
    parser.add_argument("--strategy", nargs="+",
                        default=["leave_one_out"],
                        choices=["leave_one_out", "leave_k_out"])
    parser.add_argument("--k", nargs="+", type=int, default=[2],
                        help="Values of k for leave-k-out")
    parser.add_argument("--include-boundary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if prediction.h5 already exists")
    parser.add_argument("--no-eval", action="store_true",
                        help="Produce predictions only; run evaluate_all.py separately")
    args = parser.parse_args()

    # Resolve methods
    if args.methods:
        methods = args.methods
    else:
        methods = [m for m, info in METHODS.items() if info["available"]]

    # Resolve datasets
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [d for d, info in DATASETS.items() if info["tier"] in args.tier]

    # Validate
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method: {m}")
    for d in datasets:
        if d not in DATASETS:
            raise ValueError(f"Unknown dataset: {d}")

    print(f"Campaign: {len(methods)} methods x {len(datasets)} datasets")
    print(f"  Methods: {methods}")
    print(f"  Datasets: {datasets}")
    print(f"  Strategies: {args.strategy}")

    campaign = build_campaign(
        methods, datasets, args.strategy, args.k,
        exclude_boundary=not args.include_boundary,
    )
    print(f"  Total runs: {len(campaign)}")

    results_log = []
    t0 = time.time()
    for i, (method, dataset, holdout) in enumerate(campaign, 1):
        holdout_id = holdout["holdout_id"]
        prediction_path = RESULTS_DIR / method / dataset / holdout_id / "prediction.h5"

        if args.skip_existing and prediction_path.exists():
            print(f"[{i}/{len(campaign)}] SKIP (exists): {method} | {dataset} | {holdout_id}")
            continue

        print(f"\n[{i}/{len(campaign)}] Running: {method} | {dataset} | {holdout_id}")
        result = run_single(method, dataset, holdout, dry_run=args.dry_run,
                            run_eval=not args.no_eval)
        result["method"] = method
        result["dataset"] = dataset
        result["holdout_id"] = holdout_id
        results_log.append(result)

    elapsed = time.time() - t0
    n_success = sum(1 for r in results_log if r.get("success"))
    n_fail = sum(1 for r in results_log if not r.get("success") and not r.get("dry_run"))
    print(f"\nCampaign complete in {elapsed:.0f}s: {n_success} succeeded, {n_fail} failed")

    # Write run log
    log_path = RESULTS_DIR / "summary" / "run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
