"""Run one method x one dataset x one holdout configuration.

Spawns the method wrapper in its conda environment and monitors resources.
Produces prediction.h5 only — evaluation is a separate step (evaluate.py).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import DATASETS, METHODS, PROJECT_ROOT, RESULTS_DIR, RANDOM_SEED
from .resource_monitor import ResourceMonitor, write_resources


def run_single(method, dataset, holdout_config, extra_args=None, dry_run=False,
               input_override=None, run_eval=True):
    """Run a single method/dataset/holdout combination.

    Parameters
    ----------
    method : str
        Method name (key in METHODS).
    dataset : str
        Dataset name (key in DATASETS).
    holdout_config : dict
        From holdout.py: holdout_id, holdout_sections, remaining_sections, holdout_z.
    extra_args : list[str]
        Additional CLI args passed to the method wrapper.
    dry_run : bool
        If True, print command but don't execute.
    input_override : str, optional
        Override the input h5ad path (e.g. for subset files).
    run_eval : bool
        If True, run evaluation after prediction (default True for backward compat).

    Returns
    -------
    dict with keys: success, prediction_path, resources_path, error
    """
    method_info = METHODS[method]
    dataset_info = DATASETS[dataset]

    holdout_id = holdout_config["holdout_id"]
    # Build output directory
    out_dir = RESULTS_DIR / method / dataset / holdout_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = out_dir / "prediction.h5"
    metrics_path = out_dir / "metrics.json"
    resources_path = out_dir / "resources.json"
    log_path = out_dir / "method_log.txt"

    wrapper_path = PROJECT_ROOT / method_info["wrapper"]
    input_path = input_override if input_override else str(dataset_info["path"])
    conda_env = method_info["conda_env"]

    # Build command
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", str(wrapper_path),
        "--input", input_path,
        "--holdout-sections", *[str(s) for s in holdout_config["holdout_sections"]],
        "--output", str(prediction_path),
        "--seed", str(RANDOM_SEED),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[run_benchmark] {method} | {dataset} | {holdout_id}")
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        print("  (dry run — skipping)")
        return {"success": False, "dry_run": True}

    # Run with resource monitoring
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        monitor = ResourceMonitor(proc.pid)
        monitor.start()
        returncode = proc.wait()
        resources = monitor.stop()

    write_resources(resources, resources_path)

    if returncode != 0:
        print(f"  FAILED (exit code {returncode}). See {log_path}")
        return {
            "success": False,
            "error": f"exit code {returncode}",
            "log_path": str(log_path),
            "resources_path": str(resources_path),
        }

    if not prediction_path.exists():
        print(f"  FAILED: prediction.h5 not produced")
        return {"success": False, "error": "no prediction.h5"}

    result = {
        "success": True,
        "resources_path": str(resources_path),
        "prediction_path": str(prediction_path),
    }

    # Optionally run evaluation
    if run_eval:
        from .evaluate import evaluate
        print(f"  Evaluating...")
        try:
            metrics = evaluate(str(prediction_path), input_path, str(metrics_path))
            print(f"  Done. Pearson median: {metrics.get('pearson_median', 'N/A')}")
            result["metrics_path"] = str(metrics_path)
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            result["eval_error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run one benchmark configuration")
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--holdout-json", required=True,
                        help="Path to JSON file with one holdout config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation (produce prediction.h5 only)")
    parser.add_argument("extra_args", nargs="*",
                        help="Extra args forwarded to method wrapper")
    args = parser.parse_args()

    with open(args.holdout_json) as f:
        holdout_config = json.load(f)

    result = run_single(
        args.method, args.dataset, holdout_config,
        extra_args=args.extra_args, dry_run=args.dry_run,
        run_eval=not args.no_eval,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
