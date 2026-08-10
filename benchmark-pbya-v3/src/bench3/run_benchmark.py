"""Run one method against one holdout design — benchmark-pbya-v3.

Per (design, method) this orchestrator:

  1. Splits the held-out sections off the paper dataset (evaluation only).
  2. Re-registers the *training* slices into a common frame — training-only, so
     the frame never depended on a held-out slice. For STARmap this is the
     identity: the volume is one 3-D imaging block whose z-planes are inherently
     co-registered, and re-registering would only distort (``config.REGISTRATION``).
  3. Writes ``train_registered.h5ad`` — built **once per holdout and reused by
     every method**, so the comparison is apples-to-apples.
  4. Invokes the method wrapper (v2's, unchanged) in generation-only mode with
     the training-only input plus one scalar target z per held-out section. The
     held-out (x, y) are never passed; the number of synthesized cells is
     emergent.
  5. Evaluates the result three ways and merges them into one ``metrics.json``:
       * ``evaluate_paper``       — the SpatialZ paper's validation strategy (primary)
       * ``evaluate_generation``  — v2's correspondence-free generation metrics
       * ``evaluate``             — v2's cell-matched metrics, reference only

Usage:
    python -m src.bench3.run_benchmark --method spatialz
    python -m src.bench3.run_benchmark --method spatialcpav15_gen --design loo
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .assets import sanitize_weight_args
from ._v2bridge import (
    ResourceMonitor, evaluate, evaluate_generation, reregister_training,
    split_holdout, write_resources,
)
from .config import (
    DATASET_NAME, DATASET_PATH, INPUTS_CACHE, METHODS, RANDOM_SEED,
    REGISTRATION, RESULTS_DIR, dataset_not_found_message, resolve_dataset_arg,
    spec,
)
from .design import build_designs
from .prepare_dataset import h5ad_safe
from .evaluate_paper import evaluate_paper


# ── Which dataset is this? ───────────────────────────────────────────────────

def dataset_meta(path):
    """``(name, registration)`` for a built dataset, read from the file itself.

    v3 handles more than one dataset, and the results tree, the log lines and the
    registration policy all have to follow the file actually being run rather than
    a module constant. ``prepare_dataset`` writes ``uns['dataset_name']``; a file
    predating that, or an unregistered name, falls back to the STARmap defaults.
    """
    import anndata as ad

    name = DATASET_NAME
    try:
        adata = ad.read_h5ad(str(path), backed="r")
        try:
            name = str(adata.uns.get("dataset_name", DATASET_NAME))
        finally:
            adata.file.close()
    except Exception:
        pass
    try:
        return name, spec(name).get("registration", REGISTRATION)
    except ValueError:
        return name, REGISTRATION


# ── Leakage-safe input, shared across methods ────────────────────────────────

def build_input(holdout_config, dataset_path=DATASET_PATH,
                registration=REGISTRATION, force=False, verbose=True,
                dataset_name=DATASET_NAME):
    """Build (or reuse) the training-only, re-registered input for a holdout.

    Returns ``{"input_path": str, "targets": [(section, z), ...], "registration": str}``.
    """
    import anndata as ad

    holdout_id = holdout_config["holdout_id"]
    holdout_sections = [str(s) for s in holdout_config["holdout_sections"]]

    cache_dir = Path(INPUTS_CACHE) / dataset_name / holdout_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    input_path = cache_dir / "train_registered.h5ad"
    meta_path = cache_dir / "targets.json"

    # A cache hit is only valid if the dataset it was cut from has not changed
    # since. ``train_registered.h5ad`` is a *copy* — same uns, same cells minus
    # the held-out sections — so rebuilding the dataset leaves every cached input
    # stale, and nothing downstream can tell. That is not hypothetical: a build
    # fix that corrected uns had to be applied twice, because the second run
    # still read a pre-fix cache and failed identically:
    #     IORegistryError: ... encoding_type='null' ... key 'cell_subsample'
    # Comparing mtimes makes the cache self-invalidating, so a dataset rebuild is
    # enough on its own.
    stale = (input_path.exists() and Path(dataset_path).exists()
             and input_path.stat().st_mtime < Path(dataset_path).stat().st_mtime)
    if stale and verbose:
        print(f"  input cache is older than {Path(dataset_path).name} — rebuilding")
    if input_path.exists() and meta_path.exists() and not force and not stale:
        meta = json.loads(meta_path.read_text())
        return {"input_path": str(input_path), "targets": meta["targets"],
                "registration": meta["registration"]}

    adata = ad.read_h5ad(str(dataset_path))

    # Target z per held-out section: a scalar position, computed before the
    # sections are removed. A position is what any interpolation method is
    # entitled to know; the held-out geometry and content are not.
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    secs = adata.obs["section"].values.astype(str)
    targets = []
    for sec in holdout_sections:
        m = secs == sec
        if m.any():
            targets.append([sec, float(np.median(coords[m, 2]))])

    train_adata, _held = split_holdout(adata, holdout_sections)
    train_adata = reregister_training(train_adata, method=registration,
                                      verbose=verbose)

    if "v2_registration" in train_adata.uns:   # stringify for h5 safety
        train_adata.uns["v2_registration"] = json.loads(
            json.dumps(train_adata.uns["v2_registration"], default=str))

    # Sanitize uns HERE, not only where the dataset is built.
    #
    # This file is what the methods actually read, and they read it under the
    # OLDER anndata pinned in their conda envs, which cannot decode
    # encoding_type='null' or a non-string mapping key. Fixing only
    # prepare_dataset left a gap: a dataset built before that fix still carries
    # the bad uns, and since the input is a straight copy the poison propagates —
    # the mtime guard above cannot help either, because the cache is not *older*
    # than its source, it is faithful to a bad one. Cleaning at this boundary
    # makes the method input valid whatever the dataset happens to hold, so no
    # dataset rebuild is required to unblock a run.
    train_adata.uns = h5ad_safe(dict(train_adata.uns))
    train_adata.write_h5ad(str(input_path))
    meta_path.write_text(json.dumps(
        {"targets": targets, "registration": registration,
         "holdout_sections": holdout_sections}, indent=2))

    if verbose:
        kept = sorted({str(s) for s in train_adata.obs["section"].values})
        print(f"  input: {train_adata.n_obs} cells from {kept} "
              f"(registration={registration})")
    return {"input_path": str(input_path), "targets": targets,
            "registration": registration}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_prediction(prediction_path, dataset_path=DATASET_PATH,
                        metrics_path=None, use_umap=True):
    """Run all three evaluators and merge into one metrics dict."""
    metrics = {}

    # PRIMARY — the paper's validation strategy.
    paper = evaluate_paper(str(prediction_path), str(dataset_path),
                           output_path=None, use_umap=use_umap)
    per_section = paper.pop("per_section", {})
    metrics.update(paper)

    # v2's correspondence-free generation metrics, so v3 rows can be read next to
    # the v2 sweep.
    gen = evaluate_generation(str(prediction_path), str(dataset_path),
                              output_path=None)
    for k, v in gen.items():
        if k in ("method", "holdout_sections", "n_holdout_cells_gt",
                 "n_predicted_cells", "n_common_genes", "eval"):
            continue
        metrics[k if k.startswith("gen_") else f"gen_{k}"] = v

    # v2's cell-matched metrics — reference only (no correspondence exists).
    try:
        matched = evaluate(str(prediction_path), str(dataset_path), output_path=None)
        for k, v in matched.items():
            metrics.setdefault(k, v)
    except Exception as e:
        metrics["matched_eval_error"] = str(e)

    metrics["per_section"] = per_section
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


# ── One run ───────────────────────────────────────────────────────────────────

def run_single(method, holdout_config, dataset_path=DATASET_PATH,
               extra_args=None, dry_run=False, run_eval=True,
               registration=None, use_umap=True, seed=RANDOM_SEED):
    info = METHODS.get(method)
    if info is None:
        raise ValueError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    if not info.get("available", False):
        reason = info.get("disabled_reason", "disabled")
        print(f"[v3] SKIP {method}: {reason}")
        return {"success": False, "error": f"method disabled: {reason}"}

    holdout_id = holdout_config["holdout_id"]
    dataset_name, dataset_registration = dataset_meta(dataset_path)
    if registration is None:
        registration = dataset_registration
    out_dir = Path(RESULTS_DIR) / method / dataset_name / holdout_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "prediction.h5"
    metrics_path = out_dir / "metrics.json"
    resources_path = out_dir / "resources.json"
    log_path = out_dir / "method_log.txt"

    print(f"[v3] {method} | {dataset_name} | {holdout_id}")
    reg_info = build_input(holdout_config, dataset_path=dataset_path,
                           registration=registration, dataset_name=dataset_name)
    targets = reg_info["targets"]
    if not targets:
        return {"success": False, "error": "no target sections found"}
    print(f"  targets: {[(s, round(float(z), 2)) for s, z in targets]}")

    wrapper = Path(info["wrapper"])
    if not wrapper.exists():
        return {"success": False,
                "error": f"wrapper not found: {wrapper} (is benchmark-pbya-v2 present?)"}

    cmd = [
        "conda", "run", "-n", info["conda_env"], "--no-capture-output",
        "python", str(wrapper),
        "--input", reg_info["input_path"],
        "--target-section", *[s for s, _ in targets],
        "--target-z", *[str(z) for _, z in targets],
        "--output", str(prediction_path),
        "--seed", str(seed),
    ]
    # Method-pinned wrapper configuration (see METHODS in config.py), then the
    # caller's extras last so an explicit command-line flag overrides the pin.
    cmd.extend(str(a) for a in info.get("wrapper_args", ()))
    if extra_args:
        cmd.extend(extra_args)
    if dry_run:
        print(f"  cmd: {' '.join(cmd)}")
        print("  (dry run — skipping)")
        return {"success": False, "dry_run": True}

    # Pretrained-weights arguments: hand the method a checkpoint this torch can
    # actually load (same weights — see assets.py). Done before the command is
    # logged so the log shows what really ran.
    cmd = sanitize_weight_args(cmd, info.get("sanitize_weight_args", ()),
                               info["conda_env"])
    print(f"  cmd: {' '.join(cmd)}")

    # cwd is v2's project root so the wrapper's relative imports and package
    # discovery behave exactly as they do under v2.
    cwd = wrapper.resolve().parents[3]
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                cwd=str(cwd))
        monitor = ResourceMonitor(proc.pid)
        monitor.start()
        returncode = proc.wait()
        resources = monitor.stop()
    write_resources(resources, resources_path)

    if returncode != 0:
        print(f"  FAILED (exit {returncode}). See {log_path}")
        return {"success": False, "error": f"exit code {returncode}",
                "log_path": str(log_path)}
    if not prediction_path.exists():
        print("  FAILED: no prediction.h5 produced")
        return {"success": False, "error": "no prediction.h5",
                "log_path": str(log_path)}

    # A method can exit 0 having silently substituted a fallback for itself. Such a
    # slice is not the method under test, so it must not be scored: move it aside
    # (evaluate_all walks the results tree, and would otherwise pick it up later)
    # and fail the run.
    degraded = _degraded_marker(log_path, info.get("invalid_log_markers", ()))
    if degraded:
        quarantined = prediction_path.with_suffix(".h5.degraded")
        prediction_path.replace(quarantined)
        print(f"  FAILED: the method degraded to a fallback — its log contains "
              f"{degraded!r}, so this is not {method}. Prediction moved to "
              f"{quarantined.name} and left unscored; see {log_path}")
        return {"success": False, "error": f"degraded to fallback ({degraded!r})",
                "log_path": str(log_path), "quarantined": str(quarantined)}

    result = {"success": True, "prediction_path": str(prediction_path),
              "resources_path": str(resources_path)}
    if not run_eval:
        return result

    print("  evaluating (paper + generation + matched)...")
    try:
        m = evaluate_prediction(prediction_path, dataset_path=dataset_path,
                                metrics_path=metrics_path, use_umap=use_umap)
        print(f"    [paper] umap_mixing={_f(m.get('paper_umap_mixing'))} "
              f"morans_r={_f(m.get('paper_morans_pearson'))} "
              f"gearys_r={_f(m.get('paper_gearys_pearson'))} "
              f"marker_depth_r={_f(m.get('paper_marker_depth_r'))} "
              f"localization={_f(m.get('paper_celltype_localization'))}")
        print(f"    [gen]   coexpr={_f(m.get('gen_coexpression_agreement'))} "
              f"morans_agree={_f(m.get('gen_morans_agreement'))} "
              f"sinkhorn={_f(m.get('gen_sinkhorn'))}")
        result["metrics_path"] = str(metrics_path)
    except Exception as e:
        print(f"  evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        result["eval_error"] = str(e)
    return result


def _degraded_marker(log_path, markers):
    """First ``markers`` entry present in the method log, or None."""
    if not markers or not Path(log_path).exists():
        return None
    text = Path(log_path).read_text(errors="replace")
    return next((m for m in markers if m in text), None)


def _f(v):
    return "NA" if v is None else f"{float(v):+.3f}"


def main():
    ap = argparse.ArgumentParser(
        description="Run one v3 method against the STARmap paper design")
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--design", default="paper", choices=["paper", "loo", "wide"])
    ap.add_argument("--holdout-block", type=int, default=None,
                    help="consecutive sections held out per run for --design wide "
                         "(default 3; clamped to n-2, keeping the first and last "
                         "sections as input). Ignored by other designs.")
    ap.add_argument("--holdout-id", default=None,
                    help="run only this holdout (relevant for --design loo)")
    ap.add_argument("--dataset", default=None,
                    help="dataset NAME or a path to a built data.h5ad")
    ap.add_argument("--registration", default=None,
                    choices=["none", "rigid", "paste"],
                    help="training-only re-registration policy (default: the "
                         "dataset's own, from config.DATASET_SPECS — 'none' for a "
                         "single co-registered imaging volume)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("extra_args", nargs="*",
                    help="extra args forwarded to the method wrapper")
    args = ap.parse_args()

    resolved = resolve_dataset_arg(args.dataset)
    if not resolved.exists():
        raise SystemExit(dataset_not_found_message(args.dataset, resolved))
    args.dataset = str(resolved)
    configs = build_designs(args.dataset, design=args.design,
                            holdout_block=args.holdout_block)
    if args.holdout_id:
        configs = [c for c in configs if c["holdout_id"] == args.holdout_id]
        if not configs:
            raise SystemExit(f"no holdout with id {args.holdout_id!r}")

    for cfg in configs:
        res = run_single(args.method, cfg, dataset_path=args.dataset,
                         extra_args=args.extra_args, dry_run=args.dry_run,
                         run_eval=not args.no_eval,
                         registration=args.registration,
                         use_umap=not args.no_umap, seed=args.seed)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
