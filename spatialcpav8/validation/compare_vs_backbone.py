"""Compare the v8 default (smooth morph) against a single-slice-copy baseline.

``--placement backbone`` copies the single nearest training slice — the archetype
of SpatialZ, the strongest simple baseline. This script runs both through the real
``benchmark-pbya-v2`` evaluators on a synthetic dataset and tallies, per metric,
whether the smooth morph beats that copy. It is the synthetic proxy for "does v8
beat SpatialZ"; on the real datasets the actual SpatialZ is weaker than a clean
copy, so the real margin is expected to be larger.

Usage
-----
    python make_synth_volumetric.py vol.h5ad && python compare_vs_backbone.py vol.h5ad S3
    python make_synth_distinct.py   dis.h5ad && python compare_vs_backbone.py dis.h5ad S3
"""
import sys
import os
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmark-pbya-v2"
sys.path.insert(0, str(BENCH / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import anndata as ad  # noqa: E402
from benchmark import leakage_guard as lg  # noqa: E402
from benchmark.evaluate import evaluate  # noqa: E402
from benchmark.evaluate_generation import evaluate_generation  # noqa: E402

GEN = ["coexpression_agreement", "morans_agreement", "sinkhorn",
       "celltype_composition", "celltype_nhood_agreement", "gene_mean_pearson",
       "gene_var_pearson", "field_pearson", "field_ssim", "density_pearson",
       "morans_i_pred_median"]
CM = ["pearson_median", "celltype_accuracy", "celltype_f1_macro",
      "density_pearson", "morans_i_median", "dice_density"]
LOWER_BETTER = {"gen_sinkhorn"}
WRAPPER = str(BENCH / "src/benchmark/methods/run_spatialcpav8.py")


def run(full, holdout, placement):
    wd = Path(tempfile.mkdtemp())
    a = ad.read_h5ad(full)
    tr, _ = lg.split_holdout(a, [holdout])
    tr = lg.reregister_training(tr, method="none")
    tp = wd / "t.h5ad"
    tr.write_h5ad(tp)
    z = float(np.median(a.obsm["spatial"][a.obs["section"].values == holdout, 2]))
    pp = wd / "p.h5"
    r = subprocess.run([sys.executable, WRAPPER, "--input", str(tp),
                        "--target-section", holdout, "--target-z", str(z),
                        "--output", str(pp), "--seed", "42",
                        "--placement", placement], capture_output=True, text=True)
    if not pp.exists():
        print(r.stdout[-1200:], r.stderr[-1200:]); raise SystemExit(1)
    g = evaluate_generation(str(pp), str(full))
    c = evaluate(str(pp), str(full))
    out = {f"gen_{k}": g.get(k) for k in GEN}
    out.update({f"cm_{k}": c.get(k) for k in CM})
    return out


def main():
    full = sys.argv[1]
    holdout = sys.argv[2] if len(sys.argv) > 2 else "S3"
    v8 = run(full, holdout, "smooth_morph")     # v8 default
    bb = run(full, holdout, "backbone")         # SpatialZ-like single-slice copy
    print(f"{'metric':30s} {'v8(smooth)':>12s} {'backbone':>12s}  result")
    w = l = t = 0
    for k in v8:
        a, b = v8[k], bb[k]
        if a is None or b is None:
            continue
        better = (a < b) if k in LOWER_BETTER else (a > b)
        worse = (a > b) if k in LOWER_BETTER else (a < b)
        res = "win" if (better and abs(a - b) > 1e-4) else ("LOSE" if (worse and abs(a - b) > 1e-4) else "tie")
        w += res == "win"; l += res == "LOSE"; t += res == "tie"
        print(f"{k:30s} {a:12.4f} {b:12.4f}  {res}")
    print(f"\nsmooth_morph vs backbone(SpatialZ-proxy): WIN={w} LOSE={l} TIE={t}")


if __name__ == "__main__":
    main()
