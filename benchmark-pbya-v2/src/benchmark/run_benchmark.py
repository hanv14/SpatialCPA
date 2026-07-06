"""Run one method x one dataset x one holdout — leakage-hardened (v2).

Difference from v1: this orchestrator enforces leakage safety centrally, so a
method wrapper *cannot* see the held-out slice even if it wanted to.

Per (dataset, holdout) it:
  1. Loads the full dataset.
  2. Splits off the held-out section(s) (kept aside for evaluation only).
  3. Re-registers the TRAINING slices into a common frame (training-only, so the
     frame never depends on the held-out slice) — policy per dataset.
  4. Writes a training-only ``train_registered.h5ad`` (built once per holdout,
     reused across methods for a fair comparison).
  5. Invokes the method wrapper in GENERATION-ONLY mode, passing only the
     training-only input + a scalar target z + the section label (for output
     tagging / evaluation join). The held-out (x, y) are never passed.
  6. Evaluates the synthesized slice against the held-out ground truth, with a
     rigid prediction→GT alignment (the training frame differs from the GT frame
     by a rigid transform after training-only re-registration).

See ``leakage_guard.py`` and ``README.md`` for the full policy.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import (
    DATASETS, METHODS, PROJECT_ROOT, RESULTS_DIR, RANDOM_SEED, registration_for,
)
from .resource_monitor import ResourceMonitor, write_resources


# ── Leakage-safe input construction (shared across methods) ───────────────────

def build_registered_input(dataset, holdout_config, registration=None,
                           force=False):
    """Build (or reuse) the training-only, re-registered input for a holdout.

    Returns
    -------
    dict with keys: input_path (str), targets (list[(section, z)]), registration.
    """
    import anndata as ad
    from .leakage_guard import split_holdout, reregister_training

    dataset_info = DATASETS[dataset]
    holdout_id = holdout_config["holdout_id"]
    holdout_sections = [str(s) for s in holdout_config["holdout_sections"]]

    reg = registration if registration is not None else registration_for(dataset)

    cache_dir = RESULTS_DIR / "_v2_inputs" / dataset / holdout_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    input_path = cache_dir / "train_registered.h5ad"
    meta_path = cache_dir / "targets.json"

    if input_path.exists() and meta_path.exists() and not force:
        with open(meta_path) as f:
            meta = json.load(f)
        return {"input_path": str(input_path), "targets": meta["targets"],
                "registration": meta["registration"]}

    adata = ad.read_h5ad(str(dataset_info["path"]))

    # Target z per held-out section (computed BEFORE removing them — a scalar
    # position, which every interpolation method is entitled to know).
    coords_all = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    secs_all = adata.obs["section"].values.astype(str)
    targets = []
    for sec in holdout_sections:
        m = secs_all == sec
        if not m.any():
            continue
        targets.append([sec, float(np.median(coords_all[m, 2]))])

    # Split + training-only re-registration.
    train_adata, _held = split_holdout(adata, holdout_sections)
    train_adata = reregister_training(train_adata, method=reg, verbose=True)

    # Persist. Stringify uns transforms for h5 safety.
    if "v2_registration" in train_adata.uns:
        train_adata.uns["v2_registration"] = json.loads(
            json.dumps(train_adata.uns["v2_registration"], default=str)
        )
    train_adata.write_h5ad(str(input_path))

    with open(meta_path, "w") as f:
        json.dump({"targets": targets, "registration": reg,
                   "holdout_sections": holdout_sections}, f, indent=2)

    return {"input_path": str(input_path), "targets": targets, "registration": reg}


def run_single(method, dataset, holdout_config, extra_args=None, dry_run=False,
               run_eval=True, registration=None):
    """Run a single method/dataset/holdout combination (generation-only)."""
    method_info = METHODS[method]
    dataset_info = DATASETS[dataset]

    if not method_info.get("available", False):
        reason = method_info.get("disabled_reason", "disabled in v2")
        print(f"[run_benchmark_v2] SKIP {method}: {reason}")
        return {"success": False, "error": f"method disabled: {reason}"}

    holdout_id = holdout_config["holdout_id"]
    out_dir = RESULTS_DIR / method / dataset / holdout_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = out_dir / "prediction.h5"
    metrics_path = out_dir / "metrics.json"
    resources_path = out_dir / "resources.json"
    log_path = out_dir / "method_log.txt"

    # ── Build the leakage-safe, training-only, re-registered input ────────────
    print(f"[run_benchmark_v2] {method} | {dataset} | {holdout_id}")
    reg_info = build_registered_input(dataset, holdout_config, registration=registration)
    input_path = reg_info["input_path"]
    targets = reg_info["targets"]
    if not targets:
        return {"success": False, "error": "no target sections found"}

    target_sections = [t[0] for t in targets]
    target_zs = [str(t[1]) for t in targets]
    print(f"  registration={reg_info['registration']} | "
          f"targets={[(s, round(float(z), 2)) for s, z in targets]}")

    wrapper_path = PROJECT_ROOT / method_info["wrapper"]
    conda_env = method_info["conda_env"]

    # GENERATION-ONLY interface: training-only input + scalar target z + label.
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", str(wrapper_path),
        "--input", input_path,
        "--target-section", *target_sections,
        "--target-z", *target_zs,
        "--output", str(prediction_path),
        "--seed", str(RANDOM_SEED),
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        print("  (dry run — skipping)")
        return {"success": False, "dry_run": True}

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                cwd=str(PROJECT_ROOT))
        monitor = ResourceMonitor(proc.pid)
        monitor.start()
        returncode = proc.wait()
        resources = monitor.stop()
    write_resources(resources, resources_path)

    if returncode != 0:
        print(f"  FAILED (exit code {returncode}). See {log_path}")
        return {"success": False, "error": f"exit code {returncode}",
                "log_path": str(log_path), "resources_path": str(resources_path)}
    if not prediction_path.exists():
        print(f"  FAILED: prediction.h5 not produced")
        return {"success": False, "error": "no prediction.h5"}

    result = {"success": True, "resources_path": str(resources_path),
              "prediction_path": str(prediction_path)}

    if run_eval:
        from .evaluate import evaluate
        print(f"  Evaluating (rigid-aligned)...")
        try:
            metrics = evaluate(str(prediction_path), str(dataset_info["path"]),
                               str(metrics_path))
            print(f"  Done. Pearson median: {metrics.get('pearson_median', 'N/A')}")
            result["metrics_path"] = str(metrics_path)
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            result["eval_error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run one v2 benchmark configuration")
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--holdout-json", required=True,
                        help="Path to JSON file with one holdout config")
    parser.add_argument("--registration", default=None,
                        choices=["rigid", "paste", "none"],
                        help="override the per-dataset re-registration policy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("extra_args", nargs="*",
                        help="Extra args forwarded to method wrapper")
    args = parser.parse_args()

    with open(args.holdout_json) as f:
        holdout_config = json.load(f)

    result = run_single(
        args.method, args.dataset, holdout_config,
        extra_args=args.extra_args, dry_run=args.dry_run,
        run_eval=not args.no_eval, registration=args.registration,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
