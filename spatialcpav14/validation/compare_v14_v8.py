"""Compare SpatialCPA-v14 (H3D-FLA, default settings) against SpatialCPA-v8 (default)
through the REAL ``benchmark-pbya-v2`` generation evaluators on a synthetic dataset.

v8 is the strongest prior SpatialCPA generator; v14 is the flow-matching H3D-FLA method
(conditional flow matching in a joint molecular-morphological latent space with 3D
positional-attention context, gap-aware + z-marginalized training, biology-informed
regularizers, real-profile grounding) — it shares **none** of v8's machinery (no OT /
barycentric morph / OT fusion / niche MRF) and nothing from v13 (no cell-sentence LM /
retrieval softmax). This tallies, per correspondence-free generation metric, whether v14
beats or matches v8. Only the input data is synthetic — the evaluators and method
wrappers are the real ones. No real-leaderboard numbers are reported or fabricated.

Usage
-----
    python make_synth_distinct.py   dis.h5ad && python compare_v14_v8.py dis.h5ad S3
    python make_synth_volumetric.py vol.h5ad && python compare_v14_v8.py vol.h5ad S3
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
    "v14": str(BENCH / "src/benchmark/methods/run_spatialcpav14.py"),
    "v8":  str(BENCH / "src/benchmark/methods/run_spatialcpav8.py"),
}


def run(full, holdout, method, extra, reg="none"):
    wd = Path(tempfile.mkdtemp())
    a = ad.read_h5ad(full)
    tr, _ = lg.split_holdout(a, [holdout])
    tr = lg.reregister_training(tr, method=reg)
    tp = wd / "t.h5ad"; tr.write_h5ad(tp)
    z = float(np.median(a.obsm["spatial"][a.obs["section"].values == holdout, 2]))
    pp = wd / "p.h5"
    cmd = [sys.executable, WRAPPERS[method], "--input", str(tp),
           "--target-section", holdout, "--target-z", str(z),
           "--output", str(pp), "--seed", "42"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not pp.exists():
        print(r.stdout[-3000:]); print("STDERR", r.stderr[-2000:]); raise SystemExit(1)
    g = evaluate_generation(str(pp), str(full))
    return {k: g.get(k) for k in GEN}


def main():
    full = sys.argv[1]
    holdouts = sys.argv[2].split(",") if len(sys.argv) > 2 else ["S3"]
    epochs = sys.argv[3] if len(sys.argv) > 3 else "160"
    agg = {m: {k: [] for k in GEN} for m in ("v14", "v8")}
    for holdout in holdouts:
        v14 = run(full, holdout, "v14", ["--epochs", epochs])
        v8 = run(full, holdout, "v8", [])
        for k in GEN:
            if v14[k] is not None:
                agg["v14"][k].append(v14[k])
            if v8[k] is not None:
                agg["v8"][k].append(v8[k])
    print(f"\n{'metric':28s} {'v14':>12s} {'v8':>12s}  result   (mean over {len(holdouts)} holdouts)")
    w = l = t = 0
    for k in GEN:
        av = agg["v14"][k]; bv = agg["v8"][k]
        if not av or not bv:
            print(f"{k:28s} {'n/a':>12} {'n/a':>12}  n/a"); continue
        a, b = float(np.mean(av)), float(np.mean(bv))
        better = (a < b) if k in LOWER_BETTER else (a > b)
        worse = (a > b) if k in LOWER_BETTER else (a < b)
        res = "WIN" if (better and abs(a - b) > 1e-4) else ("lose" if (worse and abs(a - b) > 1e-4) else "tie")
        w += res == "WIN"; l += res == "lose"; t += res == "tie"
        print(f"{k:28s} {a:12.4f} {b:12.4f}  {res}")
    print(f"\nv14 vs v8: WIN={w} LOSE={l} TIE={t}")


if __name__ == "__main__":
    main()
