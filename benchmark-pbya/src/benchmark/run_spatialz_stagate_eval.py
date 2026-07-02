"""Reproduce SpatialZ paper's STAGATE ARI/NMI evaluation (Extended Data Fig 1j-k).

Protocol (from paper Supp Fig 22-23):
  1. Build 'sparse' dataset: 5 real MERFISH sections, all sharing PASTE-aligned 2D coords
  2. Build 'dense' dataset: 5 real + 12 virtual SpatialZ sections (17 total)
  3. Run STAGATE on EACH dataset as a single spatial graph
     (cells across all slices are connected in 2D via Cal_Spatial_Net)
  4. Cluster with Leiden
  5. Evaluate ARI/NMI per real section against Region ground truth
  6. Repeat 10 times per trial, 10 trials (paper: 50 values = 5 sections × 10 trials)

Usage:
    conda run -n bench_spatialz python src/benchmark/run_spatialz_stagate_eval.py
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
DENSE_DIR = Path(__file__).resolve().parents[2] / "results" / "spatialz_paper_audit" / "dense_output"
OUT_PATH = Path(__file__).resolve().parents[2] / "results" / "spatialz_paper_audit" / "spatialz_merfish_ari_nmi_v3.json"
GT_KEY = "Region"
ORDERED = ["0.04", "0.09", "0.14", "0.19", "0.24"]

# Paper values from MOESM5.xlsx (supplementary source data)
PAPER_VALUES = {
    "stagate_sparse_ari_median": 0.4924,
    "stagate_sparse_nmi_median": 0.5905,
    "stagate_dense_ari_median": 0.5226,
    "stagate_dense_nmi_median": 0.5999,
}


def load_demo_data():
    adatas = {}
    for name, sid in [("4", "0.04"), ("9", "0.09"), ("14", "0.14"), ("19", "0.19"), ("24", "0.24")]:
        adatas[sid] = ad.read_h5ad(str(DEMO_DIR / f"merfish_{name}_paste.h5ad"))
    return adatas


def build_sparse(adatas):
    """5 real sections, shared 2D coordinates."""
    slices = [adatas[s].copy() for s in ORDERED]
    combined = ad.concat(slices, join="outer")
    if issparse(combined.X):
        combined.X = combined.X.toarray()
    combined.obsm["spatial"] = np.vstack([a.obsm["spatial"] for a in slices])

    # Track section boundaries
    section_idx = []
    offset = 0
    for sid in ORDERED:
        n = adatas[sid].n_obs
        section_idx.append((sid, offset, offset + n))
        offset += n
    return combined, section_idx


def build_dense():
    """All 17 slices (5 real + 12 virtual), shared 2D coordinates."""
    dense_files = sorted(DENSE_DIR.glob("*.h5ad"))
    slices = [ad.read_h5ad(str(f)) for f in dense_files]
    combined = ad.concat(slices, join="outer")
    if issparse(combined.X):
        combined.X = combined.X.toarray()
    combined.obsm["spatial"] = np.vstack([a.obsm["spatial"] for a in slices])

    # Track real section boundaries
    real_section_idx = []
    offset = 0
    for f, a in zip(dense_files, slices):
        n = a.n_obs
        if "_raw" in f.stem:
            sid = f.stem.replace("_raw", "")
            real_section_idx.append((sid, offset, offset + n))
        offset += n
    return combined, real_section_idx


def run_stagate(adata, seed=42, rad_cutoff=80, n_epochs=1000,
                resolution=0.3, device="cuda:0"):
    """Run STAGATE per-section.

    Best parameters from grid search (run_spatialz_stagate_gridsearch.py):
        rad_cutoff=80, resolution=0.3, n_epochs=1000
        → ARI=0.424, NMI=0.575 (paper: ARI=0.492, NMI=0.591)
    Paper does not specify STAGATE hyperparameters anywhere.
    """
    temp = adata.copy()
    sc.pp.normalize_total(temp, target_sum=1e4)
    sc.pp.log1p(temp)
    STAGATE_pyG.Cal_Spatial_Net(temp, rad_cutoff=rad_cutoff)
    temp = STAGATE_pyG.train_STAGATE(temp, device=device, random_seed=seed, n_epochs=n_epochs)
    sc.pp.neighbors(temp, use_rep="STAGATE")
    sc.tl.leiden(temp, resolution=resolution)
    return temp


def evaluate(adata_clustered, section_idx, adatas_gt):
    aris, nmis = [], []
    for sid, start, end in section_idx:
        gt = adatas_gt[sid].obs[GT_KEY].values.astype(str)
        pred = adata_clustered.obs["leiden"].values[start:end]
        aris.append(adjusted_rand_score(gt, pred))
        nmis.append(normalized_mutual_info_score(gt, pred))
    return aris, nmis


def main():
    print("Loading demo data...")
    adatas = load_demo_data()

    print("Building sparse (5 sections) and dense (17 sections)...")
    sparse, sparse_idx = build_sparse(adatas)
    dense, dense_idx = build_dense()
    print(f"  Sparse: {sparse.n_obs} cells, Dense: {dense.n_obs} cells")

    sparse_aris, sparse_nmis = [], []
    dense_aris, dense_nmis = [], []

    n_runs = 10
    for run in range(n_runs):
        seed = 42 + run
        torch.cuda.empty_cache()
        gc.collect()

        # Sparse
        result = run_stagate(sparse, seed=seed)
        aris, nmis = evaluate(result, sparse_idx, adatas)
        sparse_aris.extend(aris)
        sparse_nmis.extend(nmis)
        del result
        torch.cuda.empty_cache()
        gc.collect()

        # Dense
        result = run_stagate(dense, seed=seed)
        aris, nmis = evaluate(result, dense_idx, adatas)
        dense_aris.extend(aris)
        dense_nmis.extend(nmis)
        del result
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  Run {run + 1}/{n_runs}: "
              f"sARI={np.median(sparse_aris):.4f} sNMI={np.median(sparse_nmis):.4f} "
              f"dARI={np.median(dense_aris):.4f} dNMI={np.median(dense_nmis):.4f}")

    print(f"\nFinal (n={len(sparse_aris)}):")
    print(f"  Sparse: ARI={np.median(sparse_aris):.4f}, NMI={np.median(sparse_nmis):.4f}")
    print(f"  Dense:  ARI={np.median(dense_aris):.4f}, NMI={np.median(dense_nmis):.4f}")
    print(f"  Paper:  sARI={PAPER_VALUES['stagate_sparse_ari_median']:.4f} "
          f"sNMI={PAPER_VALUES['stagate_sparse_nmi_median']:.4f} | "
          f"dARI={PAPER_VALUES['stagate_dense_ari_median']:.4f} "
          f"dNMI={PAPER_VALUES['stagate_dense_nmi_median']:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "stagate_sparse_ari": sparse_aris,
            "stagate_sparse_nmi": sparse_nmis,
            "stagate_dense_ari": dense_aris,
            "stagate_dense_nmi": dense_nmis,
        }, f, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
