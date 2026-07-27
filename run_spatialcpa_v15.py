#!/usr/bin/env python
"""SpatialCPA-v15 — one-command run of the full proposed pipeline + benchmark.

    python run_spatialcpa_v15.py

With no arguments this:

1. builds a real evaluation block from the bundled STARmap 3D cortex dataset
   (``data/starmap/...``) — a contiguous 11-plane block, leiden clusters as cell
   types, three interior sections held out one at a time;
2. runs, for every holdout, the **real** ``benchmark-pbya-v2`` leakage-safe
   pipeline — split off the held-out section, re-register the training slices
   without it, hand the method only a training-only file and a scalar target z;
3. runs **SpatialCPA-v15** at its production defaults — on the **GPU when one is
   available**, under a co-operative memory policy so other processes can use the
   same card (see ``--device`` / ``--gpu-memory-fraction`` and
   ``spatialcpav15/runtime.py``) — (the complete pipeline of
   ``spatialcpa_v15_idea.md``: type embedding -> structure completion -> marked
   point process -> latent-diffusion expression -> provenance), and
   **SpatialCPA-v8** at its production defaults;
4. scores both with the real ``benchmark.evaluate_generation`` correspondence-free
   generation metrics and prints a per-metric WIN / LOSE / TIE table plus a
   paired significance test across holdouts;
5. writes everything to ``results_v15/``.

Nothing is hard-coded or fabricated: every number printed comes from the
benchmark evaluator run on the prediction files, which are kept for inspection.

Other datasets
--------------
    python run_spatialcpa_v15.py --data <any>.h5ad --holdouts S2,S3,S4
    python run_spatialcpa_v15.py --synthetic events     # non-monotone z events
    python run_spatialcpa_v15.py --synthetic drift      # drifting niches
    python run_spatialcpa_v15.py --synthetic imc --registration rigid   # IMC-like

The dataset needs ``obs['section']``, ``obsm['spatial']`` (x, y, z) and, ideally,
``obs['cell_type']``; without labels v15 falls back to Phase 1.3 joint typing.

To run inside the full benchmark campaign instead (needs the processed datasets
and conda envs, which are not bundled here)::

    cd benchmark-pbya-v2
    python -m src.benchmark.run_benchmark --method spatialcpav15_gen \\
        --dataset starmap_visual_cortex --holdout-json one_holdout.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
BENCH = REPO / "benchmark-pbya-v2"
sys.path.insert(0, str(BENCH / "src"))

import numpy as np                                    # noqa: E402
import anndata as ad                                  # noqa: E402

from benchmark import leakage_guard as lg             # noqa: E402
from benchmark.evaluate_generation import evaluate_generation   # noqa: E402

GEN_METRICS = [
    "coexpression_agreement", "morans_agreement", "sinkhorn",
    "celltype_composition", "celltype_nhood_agreement",
    "gene_mean_pearson", "gene_var_pearson",
    "field_pearson", "field_ssim", "density_pearson",
]
LOWER_IS_BETTER = {"sinkhorn"}
WRAPPERS = {
    "v15": BENCH / "src/benchmark/methods/run_spatialcpav15.py",
    "v8": BENCH / "src/benchmark/methods/run_spatialcpav8.py",
}
STARMAP = REPO / "data/starmap/STARmap_Wang2018three_data_3D_data.h5ad"


# ── dataset preparation ──────────────────────────────────────────────────────

def build_starmap_block(out: Path, z_lo: int = 20, z_hi: int = 30) -> Path:
    """Contiguous z-block of the bundled STARmap 3D cortex, one section per plane."""
    if out.exists():
        return out
    if not STARMAP.exists():
        raise SystemExit(f"STARmap dataset not found at {STARMAP}. "
                         f"Pass --data <file.h5ad> or use --synthetic.")
    a = ad.read_h5ad(str(STARMAP))
    z = a.obs["z"].values.astype(float)
    s = a[(z >= z_lo) & (z <= z_hi)].copy()
    zc = s.obs["z"].values.astype(float)
    s.obs["section"] = [f"z{int(v)}" for v in zc]
    s.obs["cell_type"] = s.obs["leiden"].astype(str).values
    s.obsm = {"spatial": np.column_stack([s.obs["x"], s.obs["y"], zc]).astype(float)}
    s.obsp = {}
    s.uns = {"expression_type": "raw_counts"}
    out.parent.mkdir(parents=True, exist_ok=True)
    s.write_h5ad(str(out))
    return out


def build_synthetic(kind: str, out: Path) -> Path:
    if out.exists():
        return out
    sys.path.insert(0, str(REPO / "spatialcpav15" / "validation"))
    mod = __import__(f"make_synth_{kind}")
    out.parent.mkdir(parents=True, exist_ok=True)
    mod.make().write_h5ad(str(out))
    return out


# ── one method x one holdout, through the real leakage-safe pipeline ─────────

def run_one(data_path: Path, holdout: str, method: str, out_dir: Path,
            registration: str, extra=()) -> dict:
    """One method x one holdout, through the real leakage-safe pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    full = ad.read_h5ad(str(data_path))
    train, _held = lg.split_holdout(full, [holdout])
    train = lg.reregister_training(train, method=registration)
    train_path = out_dir / f"train_registered_{holdout}.h5ad"
    train.write_h5ad(str(train_path))
    lg.assert_no_leakage(train, [holdout])

    secs = full.obs["section"].values.astype(str)
    z = float(np.median(full.obsm["spatial"][secs == holdout, 2]))

    pred = out_dir / f"prediction_{method}_{holdout}.h5"
    log = out_dir / f"log_{method}_{holdout}.txt"
    cmd = [sys.executable, str(WRAPPERS[method]),
           "--input", str(train_path),
           "--target-section", holdout, "--target-z", str(z),
           "--output", str(pred), "--seed", "42",
           *(extra if method == "v15" else ())]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    log.write_text(proc.stdout + "\n===STDERR===\n" + proc.stderr)
    if not pred.exists():
        print(proc.stdout[-3000:])
        print(proc.stderr[-2000:])
        raise SystemExit(f"{method} produced no prediction for holdout {holdout} "
                         f"(see {log})")
    m = evaluate_generation(str(pred), str(data_path),
                            output_path=str(out_dir / f"metrics_{method}_{holdout}.json"))
    m["_wall_seconds"] = wall
    return m


# ── reporting ────────────────────────────────────────────────────────────────

def paired_test(a: list, b: list, lower_better: bool) -> float:
    """Two-sided paired t-test p-value over holdouts (n is small; report it as-is)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    d = (b[ok] - a[ok]) if lower_better else (a[ok] - b[ok])   # >0 means v15 better
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if abs(d.mean()) > 0 else 1.0
    t = d.mean() / (sd / np.sqrt(ok.sum()))
    try:
        from scipy import stats
        return float(2 * (1 - stats.t.cdf(abs(t), ok.sum() - 1)))
    except Exception:
        return float("nan")


def report(agg: dict, holdouts: list, out_dir: Path, tol: float = 0.01) -> dict:
    print(f"\n{'metric':30s} {'v15':>10s} {'v8':>10s} {'delta':>9s} "
          f"{'p(paired)':>10s}  verdict")
    print("-" * 84)
    summary, wins, losses, ties = {}, 0, 0, 0
    for k in GEN_METRICS:
        av, bv = agg["v15"][k], agg["v8"][k]
        if not av or not bv:
            print(f"{k:30s} {'n/a':>10} {'n/a':>10} {'':>9} {'':>10}  n/a")
            continue
        a, b = float(np.mean(av)), float(np.mean(bv))
        lower = k in LOWER_IS_BETTER
        delta = (b - a) if lower else (a - b)          # >0 means v15 better
        rel = delta / max(abs(b), 1e-9)
        p = paired_test(av, bv, lower)
        if rel > tol:
            verdict, wins = "WIN", wins + 1
        elif rel < -tol:
            verdict, losses = "lose", losses + 1
        else:
            verdict, ties = "tie (non-inferior)", ties + 1
        summary[k] = {"v15": a, "v8": b, "delta": delta, "relative": rel,
                      "p_paired": p, "verdict": verdict,
                      "v15_per_holdout": av, "v8_per_holdout": bv}
        print(f"{k:30s} {a:10.4f} {b:10.4f} {delta:+9.4f} {p:10.3f}  {verdict}")
    print("-" * 84)
    print(f"v15 vs v8 over {len(holdouts)} holdouts "
          f"({', '.join(holdouts)}): WIN={wins} TIE={ties} LOSE={losses}")
    (out_dir / "summary.json").write_text(json.dumps(
        {"holdouts": holdouts, "wins": wins, "ties": ties, "losses": losses,
         "metrics": summary}, indent=2))
    print(f"\nFull results, predictions and logs: {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Run the full SpatialCPA-v15 pipeline and benchmark it against v8",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--data", default=None,
                    help="h5ad with obs['section'] + obsm['spatial'] "
                         "(default: a block of the bundled STARmap 3D cortex)")
    ap.add_argument("--synthetic", choices=["events", "drift", "imc"], default=None,
                    help="use a synthetic stack instead of the STARmap block "
                         "('imc' is the continuous-intensity / rigid-registration path)")
    ap.add_argument("--holdouts", default=None,
                    help="comma-separated held-out sections (default: 3 interior ones)")
    ap.add_argument("--methods", default="v15,v8",
                    help="which methods to run (default both, for the comparison)")
    ap.add_argument("--registration", default="none",
                    choices=["none", "rigid", "paste"],
                    help="training-only re-registration policy; 'none' matches the "
                         "benchmark's volumetric policy for STARmap")
    ap.add_argument("--output", default=str(REPO / "results_v15"))
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="relative margin within which a difference counts as a tie")
    # GPU: v15 uses one automatically when present. These pass through to the
    # wrapper; v8 is training-free numpy/scipy and ignores them.
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="where v15's three trained models run (default: GPU if present)")
    ap.add_argument("--gpu-memory-fraction", type=float, default=None,
                    help="cap v15 at this fraction of the GPU so a co-tenant process "
                         "always has the rest (e.g. 0.5); default uncapped")
    args = ap.parse_args()

    v15_extra = ["--device", args.device]
    if args.gpu_memory_fraction is not None:
        v15_extra += ["--gpu-memory-fraction", str(args.gpu_memory_fraction)]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data:
        data_path = Path(args.data)
    elif args.synthetic:
        data_path = build_synthetic(args.synthetic,
                                    out_dir / f"synth_{args.synthetic}.h5ad")
    else:
        data_path = build_starmap_block(out_dir / "starmap_block.h5ad")

    full = ad.read_h5ad(str(data_path))
    secs = full.obs["section"].values.astype(str)
    order = sorted(np.unique(secs),
                   key=lambda s: float(np.median(full.obsm["spatial"][secs == s, 2])))
    if args.holdouts:
        holdouts = args.holdouts.split(",")
    else:                                   # three interior sections
        mid = len(order) // 2
        holdouts = order[max(mid - 1, 1):min(mid + 2, len(order) - 1)]

    zs = np.array([np.median(full.obsm["spatial"][secs == s, 2]) for s in order])
    dz = float(np.median(np.diff(zs))) if len(zs) > 1 else 1.0
    print(f"dataset : {data_path}")
    print(f"sections: {len(order)} ({order[0]} .. {order[-1]}), dz = {dz:g}")
    print(f"holdouts: {holdouts}   (R = dz / output spacing = 1 at native spacing)")
    print(f"methods : {args.methods}   registration: {args.registration}\n")

    methods = [m.strip() for m in args.methods.split(",")]
    agg = {m: {k: [] for k in GEN_METRICS} for m in methods}
    for h in holdouts:
        for m in methods:
            print(f"[{m} | holdout {h}] running...", flush=True)
            res = run_one(data_path, h, m, out_dir, args.registration,
                          extra=v15_extra)
            for k in GEN_METRICS:
                if res.get(k) is not None:
                    agg[m][k].append(float(res[k]))
            print("   " + "  ".join(
                f"{k}={res[k]:.3f}" for k in GEN_METRICS if res.get(k) is not None)
                + f"   [{res['_wall_seconds']:.0f}s]")

    if set(methods) >= {"v15", "v8"}:
        report(agg, holdouts, out_dir, tol=args.tolerance)
    else:
        print(json.dumps({m: {k: float(np.mean(v)) for k, v in d.items() if v}
                          for m, d in agg.items()}, indent=2))


if __name__ == "__main__":
    main()
