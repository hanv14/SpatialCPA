"""Grid search STAGATE hyperparameters to reproduce SpatialZ paper ARI/NMI.

The SpatialZ paper (Lin et al. 2025, Nature Methods) reports STAGATE ARI/NMI
for spatial domain clustering but does NOT specify STAGATE hyperparameters
(rad_cutoff, resolution, n_epochs, hidden_dims) anywhere:
  - Not in the paper text or Methods section
  - Not in GitHub (github.com/senlin-lin/SpatialZ)
  - Not in Zenodo (10.5281/zenodo.17416727)
  - Not in the tutorial docs (spatialz-tutorial.readthedocs.io)
  - Not in supplementary materials (MOESM1-6)

This script grid-searches rad_cutoff and Leiden resolution to find the
combination that best matches the paper's reported values:
  Paper STAGATE sparse: ARI=0.4924, NMI=0.5905 (from MOESM5.xlsx)

Protocol: per-section STAGATE (not joint), STAGATE_pyG v1.0.0, 1000 epochs

Usage:
    conda run -n bench_spatialz python src/benchmark/run_spatialz_stagate_gridsearch.py
"""

import sys
import json
import warnings
import gc
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "spatialz" / "SpatialZ_code"))

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse import issparse
import STAGATE_pyG

DEMO_DIR = Path(__file__).resolve().parents[2] / "tools" / "spatialz" / "SpatialZ_code" / "data"
OUT_PATH = Path(__file__).resolve().parents[2] / "results" / "spatialz_paper_audit" / "stagate_gridsearch.json"
ORDERED = ["0.04", "0.09", "0.14", "0.19", "0.24"]
GT_KEY = "Region"
PAPER_ARI = 0.4924
PAPER_NMI = 0.5905


def load_data():
    adatas = {}
    for name, sid in [("4", "0.04"), ("9", "0.09"), ("14", "0.14"), ("19", "0.19"), ("24", "0.24")]:
        adatas[sid] = ad.read_h5ad(str(DEMO_DIR / f"merfish_{name}_paste.h5ad"))
    return adatas


def main():
    adatas = load_data()
    n_regions = adatas["0.04"].obs[GT_KEY].nunique()
    print(f"Ground truth regions: {n_regions}")

    results = {}
    for rad in [30, 50, 80]:
        for res in [0.3, 0.5, 0.8, 1.0]:
            aris, nmis = [], []
            for run in range(5):
                seed = 42 + run
                torch.cuda.empty_cache()
                gc.collect()
                for sid in ORDERED:
                    gt_adata = adatas[sid]
                    gt_labels = gt_adata.obs[GT_KEY].values.astype(str)
                    temp = gt_adata.copy()
                    if issparse(temp.X):
                        temp.X = temp.X.toarray()
                    sc.pp.normalize_total(temp, target_sum=1e4)
                    sc.pp.log1p(temp)
                    STAGATE_pyG.Cal_Spatial_Net(temp, rad_cutoff=rad)
                    temp = STAGATE_pyG.train_STAGATE(
                        temp, device="cuda:0", random_seed=seed, n_epochs=1000)
                    sc.pp.neighbors(temp, use_rep="STAGATE")
                    sc.tl.leiden(temp, resolution=res)
                    pred = temp.obs["leiden"].values
                    aris.append(adjusted_rand_score(gt_labels, pred))
                    nmis.append(normalized_mutual_info_score(gt_labels, pred))
                    n_clusters = len(set(pred))
                    del temp

            med_ari = float(np.median(aris))
            med_nmi = float(np.median(nmis))
            key = f"rad{rad}_res{res}"
            results[key] = {"ari": med_ari, "nmi": med_nmi, "n_clusters": n_clusters}
            print(f"rad={rad:>3} res={res:.1f}: ARI={med_ari:.4f} NMI={med_nmi:.4f} "
                  f"(clusters≈{n_clusters}, target={n_regions})")

    best = min(results.keys(),
               key=lambda k: abs(results[k]["ari"] - PAPER_ARI) + abs(results[k]["nmi"] - PAPER_NMI))
    print(f"\nBest: {best} → ARI={results[best]['ari']:.4f} NMI={results[best]['nmi']:.4f}")
    print(f"Paper:        ARI={PAPER_ARI:.4f} NMI={PAPER_NMI:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
