"""Compare SpatialCPA-v15 (default settings) against SpatialCPA-v8 (default settings)
through the **real** ``benchmark-pbya-v2`` generation evaluator.

Only the *input data* may be synthetic; the leakage-safe holdout/re-registration, the
method wrappers and the correspondence-free generation evaluator are the real benchmark
code.  No leaderboard numbers are fabricated here.

v8 is the strongest prior SpatialCPA generator (training-free entropic-OT bridges /
diffeomorphic morphs of an anchor slice).  v15 shares none of that machinery — it is the
learned four-phase pipeline of ``spatialcpa_v15_idea.md``.  This script tallies, per
correspondence-free generation metric, whether v15 beats or matches v8.

Usage
-----
    python spatialcpav15/validation/compare_v15_v8.py <data.h5ad> <holdout[,holdout...]>
    # e.g.
    python spatialcpav15/validation/compare_v15_v8.py starmap_block.h5ad z24,z25,z26
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmark-pbya-v2"
sys.path.insert(0, str(BENCH / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import anndata as ad  # noqa: E402
from benchmark import leakage_guard as lg  # noqa: E402
from benchmark.evaluate_generation import evaluate_generation  # noqa: E402

GEN = ["coexpression_agreement", "morans_agreement", "sinkhorn",
       "celltype_composition", "celltype_nhood_agreement",
       "gene_mean_pearson", "gene_var_pearson",
       "field_pearson", "field_ssim", "density_pearson"]
LOWER_BETTER = {"sinkhorn"}
WRAPPERS = {
    "v15": str(BENCH / "src/benchmark/methods/run_spatialcpav15.py"),
    "v8": str(BENCH / "src/benchmark/methods/run_spatialcpav8.py"),
}


def run(full, holdout, method, extra=(), reg="none", keep=None):
    """Run one wrapper on one holdout through the real leakage-safe pipeline."""
    wd = Path(keep) if keep else Path(tempfile.mkdtemp())
    wd.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(full)
    tr, _ = lg.split_holdout(a, [holdout])
    tr = lg.reregister_training(tr, method=reg)
    tp = wd / f"train_{holdout}.h5ad"
    tr.write_h5ad(tp)
    z = float(np.median(a.obsm["spatial"][a.obs["section"].values == holdout, 2]))
    pp = wd / f"pred_{method}_{holdout}.h5"
    cmd = [sys.executable, WRAPPERS[method], "--input", str(tp),
           "--target-section", holdout, "--target-z", str(z),
           "--output", str(pp), "--seed", "42"] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not pp.exists():
        print(r.stdout[-4000:])
        print("STDERR", r.stderr[-3000:])
        raise SystemExit(f"{method} produced no prediction for {holdout}")
    g = evaluate_generation(str(pp), str(full))
    return {k: g.get(k) for k in GEN}, r.stdout


def main():
    full = sys.argv[1]
    holdouts = sys.argv[2].split(",") if len(sys.argv) > 2 else ["S3"]
    verbose = "-v" in sys.argv
    agg = {m: {k: [] for k in GEN} for m in ("v15", "v8")}
    for holdout in holdouts:
        for m in ("v15", "v8"):
            res, log = run(full, holdout, m)
            if verbose:
                print(log)
            for k in GEN:
                if res[k] is not None:
                    agg[m][k].append(res[k])
            print(f"  [{m} | {holdout}] "
                  + " ".join(f"{k.split('_')[0]}={res[k]:.3f}"
                             for k in GEN if res[k] is not None))

    print(f"\n{'metric':28s} {'v15':>12s} {'v8':>12s}  result   "
          f"(mean over {len(holdouts)} holdouts)")
    w = l = t = 0
    for k in GEN:
        av, bv = agg["v15"][k], agg["v8"][k]
        if not av or not bv:
            print(f"{k:28s} {'n/a':>12} {'n/a':>12}  n/a")
            continue
        a, b = float(np.mean(av)), float(np.mean(bv))
        better = (a < b) if k in LOWER_BETTER else (a > b)
        worse = (a > b) if k in LOWER_BETTER else (a < b)
        res = ("WIN" if (better and abs(a - b) > 1e-4)
               else ("lose" if (worse and abs(a - b) > 1e-4) else "tie"))
        w += res == "WIN"
        l += res == "lose"
        t += res == "tie"
        print(f"{k:28s} {a:12.4f} {b:12.4f}  {res}")
    print(f"\nv15 vs v8: WIN={w} LOSE={l} TIE={t}")


if __name__ == "__main__":
    main()
