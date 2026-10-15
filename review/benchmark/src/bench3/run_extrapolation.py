"""Extrapolation campaign: hold out the sections at one end of the stack.

Every ``design.py`` design keeps the first and last sections as input, so no
held-out section is ever outside the training range. This campaign does the
opposite, without changing ``design.py``: ``extrap_hi_k`` holds out the k
deepest sections and ``extrap_lo_k`` the k shallowest, so targets lie one to k
section spacings beyond the training range. Each run goes through
``run_benchmark.run_single`` (same training-only input, leakage guard, wrapper
contract and pinned scorer as every other design) and is then scored by
``evaluate_depth`` as well.

    python -m src.bench3.run_extrapolation --methods spatialcpa_x spatialcpa_tf \
        --dataset starmap_visual_cortex --side both --depth 3 --seed 1

Results go to ``$BENCH_V3_RESULTS/<method>/<dataset>/extrap_{hi,lo}_<k>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import RANDOM_SEED, RESULTS_DIR, resolve_dataset_arg
from .design import sorted_sections
from .evaluate_depth import OUT_NAME as DEPTH_OUT
from .evaluate_depth import evaluate_depth
from .run_benchmark import build_input, dataset_meta, run_single


def extrapolation_designs(adata, side="both", depth=3):
    """Hold-out configs with the k sections at one end held out (k = 1..depth).
    At least two sections always remain as input."""
    labels, z_by = sorted_sections(adata)
    if len(labels) - depth < 2:
        raise ValueError(f"{len(labels)} sections cannot hold out {depth} at one end "
                         f"and keep two as input")
    out = []
    for k in range(1, depth + 1):
        for s, held in (("hi", labels[-k:]), ("lo", labels[:k])):
            if side not in ("both", s):
                continue
            out.append({"holdout_id": f"extrap_{s}_{k}", "design": "extrapolation",
                        "holdout_sections": list(held),
                        "remaining_sections": [x for x in labels if x not in held],
                        "holdout_z": {x: z_by[x] for x in held}})
    return out


def main():
    import anndata as ad
    ap = argparse.ArgumentParser(description="Extrapolation campaign (end-of-stack hold-outs)")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--side", choices=["hi", "lo", "both"], default="both")
    ap.add_argument("--depth", type=int, default=3,
                    help="hold out 1..depth sections at the chosen end(s)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("extra_args", nargs="*")
    args = ap.parse_args()

    path = resolve_dataset_arg(args.dataset)
    if not path.exists():
        raise SystemExit(f"dataset not found: {path}")
    dataset_name, registration = dataset_meta(str(path))
    configs = extrapolation_designs(ad.read_h5ad(path), args.side, args.depth)
    print(f"extrapolation campaign — dataset={dataset_name}, {len(configs)} designs, "
          f"methods {', '.join(args.methods)}")
    for cfg in configs:
        print(f"  {cfg['holdout_id']}: held out {cfg['holdout_sections']}")
    bad = 0
    for cfg in configs:
        build_input(cfg, dataset_path=str(path), registration=registration,
                    dataset_name=dataset_name)
        for method in args.methods:
            out_dir = Path(RESULTS_DIR) / method / dataset_name / cfg["holdout_id"]
            if args.skip_existing and (out_dir / "prediction.h5").exists():
                print(f"SKIP (exists): {method} | {cfg['holdout_id']}")
            else:
                res = run_single(method, cfg, dataset_path=str(path),
                                 extra_args=args.extra_args, registration=registration,
                                 use_umap=not args.no_umap, seed=args.seed)
                if not res.get("success"):
                    print(f"  FAILED {method} | {cfg['holdout_id']}: {res.get('error')}")
                    bad += 1
                    continue
            if (out_dir / "prediction.h5").exists():
                d = evaluate_depth(out_dir / "prediction.h5", str(path), out_dir / DEPTH_OUT)
                print(f"    [depth] lib_logratio={d['lib_logratio']} "
                      f"gene_mean_abs_logratio={d['gene_mean_abs_logratio']}")
    print(f"extrapolation campaign finished: {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
