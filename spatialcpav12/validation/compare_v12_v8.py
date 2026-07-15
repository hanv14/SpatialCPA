"""Compare SpatialCPA-v12 (default generative decode) against SpatialCPA-v8 (default)
through the REAL ``benchmark-pbya-v2`` generation evaluators on a synthetic dataset.

v8 is the strongest prior SpatialCPA generator; v12 is a *generative neural-field*
method that shares none of v8's machinery (no OT / diffeomorphic morph / OT fusion /
niche MRF). This tallies, per correspondence-free generation metric, whether v12 beats
or matches v8. Only the input data is synthetic — the evaluators and method wrappers are
the real ones. No real-leaderboard numbers are reported or fabricated.

Usage
-----
    python make_synth_distinct.py   dis.h5ad && python compare_v12_v8.py dis.h5ad S3
    python make_synth_volumetric.py vol.h5ad && python compare_v12_v8.py vol.h5ad S3
"""
import sys, os, subprocess, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmark-pbya-v2"
sys.path.insert(0, str(BENCH / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import anndata as ad  # noqa: E402
from benchmark import leakage_guard as lg  # noqa: E402
from benchmark.evaluate_generation import evaluate_generation  # noqa: E402

GEN = ["coexpression_agreement", "morans_agreement", "sinkhorn", "celltype_composition",
       "celltype_nhood_agreement", "gene_mean_pearson", "gene_var_pearson",
       "field_pearson", "field_ssim", "density_pearson"]
LOWER_BETTER = {"sinkhorn"}
WRAPPERS = {
    "v12": str(BENCH / "src/benchmark/methods/run_spatialcpav12.py"),
    "v8":  str(BENCH / "src/benchmark/methods/run_spatialcpav8.py"),
}


def run(full, holdout, method, extra):
    wd = Path(tempfile.mkdtemp())
    a = ad.read_h5ad(full)
    tr, _ = lg.split_holdout(a, [holdout])
    tr = lg.reregister_training(tr, method="none")
    tp = wd / "t.h5ad"; tr.write_h5ad(tp)
    z = float(np.median(a.obsm["spatial"][a.obs["section"].values == holdout, 2]))
    pp = wd / "p.h5"
    cmd = [sys.executable, WRAPPERS[method], "--input", str(tp),
           "--target-section", holdout, "--target-z", str(z),
           "--output", str(pp), "--seed", "42"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not pp.exists():
        print(r.stdout[-2000:]); print("STDERR", r.stderr[-2000:]); raise SystemExit(1)
    g = evaluate_generation(str(pp), str(full))
    return {k: g.get(k) for k in GEN}


def main():
    full = sys.argv[1]
    holdout = sys.argv[2] if len(sys.argv) > 2 else "S3"
    epochs = sys.argv[3] if len(sys.argv) > 3 else "300"
    v12 = run(full, holdout, "v12", ["--epochs", epochs])
    v8 = run(full, holdout, "v8", [])
    print(f"\n{'metric':28s} {'v12':>12s} {'v8':>12s}  result")
    w = l = t = 0
    for k in GEN:
        a, b = v12[k], v8[k]
        if a is None or b is None:
            print(f"{k:28s} {str(a):>12} {str(b):>12}  n/a"); continue
        better = (a < b) if k in LOWER_BETTER else (a > b)
        worse = (a > b) if k in LOWER_BETTER else (a < b)
        res = "WIN" if (better and abs(a - b) > 1e-4) else ("lose" if (worse and abs(a - b) > 1e-4) else "tie")
        w += res == "WIN"; l += res == "lose"; t += res == "tie"
        print(f"{k:28s} {a:12.4f} {b:12.4f}  {res}")
    print(f"\nv12 vs v8: WIN={w} LOSE={l} TIE={t}")


if __name__ == "__main__":
    main()
