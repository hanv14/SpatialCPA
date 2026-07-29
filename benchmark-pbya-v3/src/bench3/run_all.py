"""Run the full v3 campaign: every method through the paper design.

By default this is the seven-method comparison the folder exists for —
spatialcpav8/v11/v14/v15, SpatialZ, FEAST and isoST — each reconstructing
sections 2, 4 and 6 of the STARmap visual cortex from sections 1, 3, 5 and 7.

The training-only input is built once and shared by every method, so the
comparison is apples-to-apples by construction.

Usage:
    python -m src.bench3.run_all                              # all 7 methods
    python -m src.bench3.run_all --methods spatialz feast     # a subset
    python -m src.bench3.run_all --dry-run                    # print the plan
    python -m src.bench3.run_all --design loo --skip-existing
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import (
    DATASET_NAME, DATASET_PATH, METHOD_ORDER, METHODS, RANDOM_SEED,
    REGISTRATION, RESULTS_DIR, SUMMARY_DIR,
)
from .design import build_designs, describe
from .preflight import probe_method, report
from .run_benchmark import build_input, run_single


def resolve_methods(requested):
    if not requested:
        return [m for m in METHOD_ORDER if METHODS[m].get("available", False)]
    unknown = [m for m in requested if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s): {unknown}; known: {sorted(METHODS)}")
    return list(requested)


def main():
    ap = argparse.ArgumentParser(description="Run the full benchmark-pbya-v3 campaign")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="methods to run (default: all available, in METHOD_ORDER)")
    ap.add_argument("--design", default="paper", choices=["paper", "loo"])
    ap.add_argument("--dataset", default=str(DATASET_PATH))
    ap.add_argument("--registration", default=REGISTRATION,
                    choices=["none", "rigid", "paste"])
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a run whose prediction.h5 already exists")
    ap.add_argument("--no-eval", action="store_true",
                    help="predictions only; evaluate later with evaluate_all")
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the per-method environment check (see preflight.py). "
                         "The check costs seconds and catches a broken or stale "
                         "method package before hours of campaign are spent on it.")
    ap.add_argument("--continue-on-error", action="store_true", default=True,
                    help="keep going when a method fails (default)")
    ap.add_argument("extra_args", nargs="*",
                    help="extra args forwarded to every method wrapper")
    args = ap.parse_args()

    if not Path(args.dataset).exists():
        raise SystemExit(
            f"dataset not found: {args.dataset}\n"
            f"Build it first:  python -m src.bench3.prepare_starmap")

    methods = resolve_methods(args.methods)
    configs = build_designs(args.dataset, design=args.design)

    print(f"benchmark-pbya-v3 campaign — design={args.design}, dataset={DATASET_NAME}")
    print(describe(configs))
    print(f"\n  methods ({len(methods)}): {', '.join(methods)}")
    print(f"  total runs: {len(methods) * len(configs)}")
    if args.extra_args:
        print(f"  wrapper extra args: {' '.join(args.extra_args)}")

    # Environment check first: a method whose package is missing or broken will
    # fail every holdout, so find that out in seconds rather than after the run.
    # Methods that fail are dropped and named again at the end; a method that
    # merely resolves outside the checkout is reported but still run, because
    # that is a legitimate setup, not an error.
    skipped_preflight = []
    if not args.dry_run and not args.no_preflight:
        print("\nPreflight (skip with --no-preflight)...")
        results = [probe_method(m) for m in methods]
        report(results)
        skipped_preflight = [r["method"] for r in results if not r["ok"]]
        if skipped_preflight:
            methods = [m for m in methods if m not in skipped_preflight]
            if not methods:
                raise SystemExit(
                    "\nEvery method failed preflight — nothing to run. Fix the "
                    "environments above, or re-run with --no-preflight to try "
                    "anyway.")
            print(f"\nSkipping {len(skipped_preflight)} method(s) that failed "
                  f"preflight: {', '.join(skipped_preflight)}")
            print(f"Continuing with: {', '.join(methods)}")

    # Build every shared input up front so a method failure never leaves a
    # half-built input, and so the cost is paid once and visibly.
    if not args.dry_run:
        print("\nBuilding shared training-only inputs...")
        for cfg in configs:
            build_input(cfg, dataset_path=args.dataset,
                        registration=args.registration)

    log, t0 = [], time.time()
    total = len(methods) * len(configs)
    i = 0
    for cfg in configs:
        for method in methods:
            i += 1
            holdout_id = cfg["holdout_id"]
            pred = (Path(RESULTS_DIR) / method / DATASET_NAME / holdout_id
                    / "prediction.h5")
            if args.skip_existing and pred.exists():
                print(f"\n[{i}/{total}] SKIP (exists): {method} | {holdout_id}")
                continue
            print(f"\n[{i}/{total}] {method} | {holdout_id}")
            try:
                res = run_single(method, cfg, dataset_path=args.dataset,
                                 extra_args=args.extra_args,
                                 dry_run=args.dry_run,
                                 run_eval=not args.no_eval,
                                 registration=args.registration,
                                 use_umap=not args.no_umap, seed=args.seed)
            except Exception as e:
                if not args.continue_on_error:
                    raise
                print(f"  ERROR: {e}")
                res = {"success": False, "error": str(e)}
            res.update({"method": method, "dataset": DATASET_NAME,
                        "holdout_id": holdout_id, "design": args.design})
            log.append(res)

    n_ok = sum(1 for r in log if r.get("success"))
    n_bad = sum(1 for r in log if not r.get("success") and not r.get("dry_run"))
    print(f"\nCampaign finished in {time.time() - t0:.0f}s: "
          f"{n_ok} succeeded, {n_bad} failed")
    if n_bad:
        for r in log:
            if not r.get("success") and not r.get("dry_run"):
                print(f"  FAILED {r['method']} | {r['holdout_id']}: "
                      f"{r.get('error')}  (log: {r.get('log_path', 'n/a')})")
    if skipped_preflight:
        print(f"  NOT RUN (failed preflight): {', '.join(skipped_preflight)}\n"
              f"    diagnose with: python -m src.bench3.preflight --methods "
              f"{' '.join(skipped_preflight)}")

    if not args.dry_run:
        log_path = Path(SUMMARY_DIR) / f"run_log_{args.design}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log, indent=2))
        print(f"Run log: {log_path}")
        print("\nNext:\n"
              "  python -m src.bench3.aggregate_results\n"
              "  python -m src.bench3.rank_methods\n"
              "  python -m src.bench3.plot_paper_figures")


if __name__ == "__main__":
    main()
