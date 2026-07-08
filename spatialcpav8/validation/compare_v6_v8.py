"""Reproduce the v6-vs-v8 ablation on a synthetic multi-slice dataset.

This exercises the *actual* benchmark-pbya-v2 evaluators
(``evaluate.py`` + ``evaluate_generation.py``) and the real method wrappers, so
the numbers are computed by the same code path the leaderboard uses — only the
data is synthetic (the processed benchmark datasets are not bundled in the repo).

Usage
-----
    # near-identical / volumetric regime (STARmap-like):
    python make_synth_volumetric.py vol.h5ad
    python compare_v6_v8.py vol.h5ad S3

    # distinct-tissue regime (IMC-like):
    python make_synth_distinct.py distinct.h5ad
    python compare_v6_v8.py distinct.h5ad S3

Requires numpy/scipy/scikit-learn/scikit-image/anndata/scanpy/h5py (the
``bench_spatialcpa`` environment).
"""
import sys
import os
import subprocess
import tempfile
from pathlib import Path

# repo root = three levels up from this file (spatialcpav8/validation/<file>).
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


def run_one(full, wrapper, holdout):
    wd = Path(tempfile.mkdtemp())
    adata = ad.read_h5ad(full)
    train, _ = lg.split_holdout(adata, [holdout])
    train = lg.reregister_training(train, method="none")
    tp = wd / "train.h5ad"
    train.write_h5ad(tp)
    z = float(np.median(adata.obsm["spatial"][adata.obs["section"].values == holdout, 2]))
    pp = wd / "pred.h5"
    r = subprocess.run(
        [sys.executable, wrapper, "--input", str(tp), "--target-section", holdout,
         "--target-z", str(z), "--output", str(pp), "--seed", "42"],
        capture_output=True, text=True)
    if not pp.exists():
        print(r.stdout[-1500:]); print(r.stderr[-1500:]); raise SystemExit(1)
    g = evaluate_generation(str(pp), str(full))
    c = evaluate(str(pp), str(full))
    out = {f"gen_{k}": g.get(k) for k in GEN}
    out.update({f"cm_{k}": c.get(k) for k in CM})
    return out


def main():
    full = sys.argv[1]
    holdout = sys.argv[2] if len(sys.argv) > 2 else "S3"
    v6 = run_one(full, str(BENCH / "src/benchmark/methods/run_spatialcpav6.py"), holdout)
    v8 = run_one(full, str(BENCH / "src/benchmark/methods/run_spatialcpav8.py"), holdout)
    print(f"{'metric':32s} {'v6':>8s} {'v8':>8s}  winner")
    tally = {"v8": 0, "v6": 0, "tie": 0}
    for k in v6:
        a, b = v6[k], v8[k]
        if a is None or b is None:
            continue
        if k in LOWER_BETTER:
            w = "v8" if b < a - 1e-4 else ("v6" if b > a + 1e-4 else "tie")
        else:
            w = "v8" if b > a + 1e-4 else ("v6" if b < a - 1e-4 else "tie")
        tally[w] += 1
        print(f"{k:32s} {a:8.3f} {b:8.3f}  {w}")
    print("TALLY", tally)


if __name__ == "__main__":
    main()
