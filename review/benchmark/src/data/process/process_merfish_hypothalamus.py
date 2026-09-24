#!/usr/bin/env python
"""Process MERFISH hypothalamus to standardized h5ad.

Uses the full Dryad/Zenodo CSV (1M cells, 36 animals). Selects all 11 naive
animals as separate specimens. Animals 1, 2, 7 have full 12-section AP coverage;
others have 5-6 sections covering anterior or posterior half.

Raw data: Moffitt2018_MERFISH_hypothalamus_full.csv
  - 1,027,848 cells × 161 genes (including 5 Blank controls)
  - 36 animals (naive, parenting, mating, aggression conditions)
  - Centroid_X/Y in µm, Bregma in mm

Output: data/processed/merfish_hypothalamus/animal_{N}/data.h5ad (N=1-11)
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "merfish_hypothalamus"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME

METADATA = {
    "technology": "MERFISH",
    "species": "mouse",
    "tissue": "hypothalamus (preoptic)",
    "expression_type": "normalized",
    "section_spacing_um": 50.0,
    "source": "Dryad 10.5061/dryad.8t8s248",
}

CSV_FILE = "Moffitt2018_MERFISH_hypothalamus_full.csv"

# Metadata columns in the CSV (everything else is a gene)
META_COLS = [
    "Cell_ID", "Animal_ID", "Animal_sex", "Behavior",
    "Bregma", "Centroid_X", "Centroid_Y",
    "Cell_class", "Neuron_cluster_ID",
]

# All 11 naive animals (behavior="Naive")
# Animals 1, 2, 7: full 12-section AP coverage (-0.29 to +0.26)
# Animals 3, 5, 6: 6 anterior sections (+0.01 to +0.26)
# Animals 8, 9, 10, 11: 6 posterior sections (-0.29 to -0.04)
# Animal 4: 5 posterior sections (-0.29 to -0.04, missing -0.24)
NAIVE_ANIMALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def check_raw_data():
    """Verify that required raw files exist."""
    csv_path = RAW_DIR / CSV_FILE
    if not csv_path.exists():
        print(f"ERROR: Missing {csv_path}")
        print(f"Run: python src/data/download/download_{DATASET_NAME}.py")
        sys.exit(1)
    print(f"Raw data check passed: {csv_path.name} ({csv_path.stat().st_size / 1e6:.0f} MB)")


def ensure_sparse_csr(X):
    """Convert expression matrix to CSR sparse format."""
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def build_spatial_3d(coords_2d, z):
    """Add z column to 2D spatial coordinates."""
    n = coords_2d.shape[0]
    z_col = np.full((n, 1), z, dtype=np.float64) if np.isscalar(z) else z.reshape(-1, 1)
    return np.hstack([np.array(coords_2d, dtype=np.float64), z_col])


def verify(adata):
    """Verify the processed AnnData meets standards."""
    assert sp.issparse(adata.X) and adata.X.format == "csr"
    assert "spatial" in adata.obsm and adata.obsm["spatial"].shape == (adata.n_obs, 3)
    assert "section" in adata.obs.columns and "cell_type" in adata.obs.columns
    print(f"  Verified: {adata.n_obs} cells × {adata.n_vars} genes")
    print(
        f"  Spatial range: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
        f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
        f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]"
    )
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")


def process():
    """Load full CSV and produce per-animal h5ad files."""
    print(f"Processing {DATASET_NAME}...")
    check_raw_data()

    print("  Loading full CSV (~1 GB)...")
    df = pd.read_csv(RAW_DIR / CSV_FILE)
    print(f"  Loaded: {len(df):,} cells × {len(df.columns)} columns")

    # Identify gene columns (everything not in META_COLS)
    gene_cols = [c for c in df.columns if c not in META_COLS]
    # Drop Blank control columns
    blank_cols = [c for c in gene_cols if c.startswith("Blank")]
    gene_cols = [c for c in gene_cols if not c.startswith("Blank")]
    # Drop all-NaN columns (e.g. Fos — measured separately via cfos imaging)
    all_nan_cols = [c for c in gene_cols if df[c].isna().all()]
    if all_nan_cols:
        gene_cols = [c for c in gene_cols if c not in all_nan_cols]
        print(f"  Dropped all-NaN gene columns: {all_nan_cols}")
    print(f"  Genes: {len(gene_cols)} (dropped {len(blank_cols)} Blank + {len(all_nan_cols)} all-NaN)")

    # Filter to full-coverage naive animals
    df_sel = df[df["Animal_ID"].isin(NAIVE_ANIMALS)].copy()
    print(f"  Selected animals {NAIVE_ANIMALS}: {len(df_sel):,} cells")

    # Drop gene columns that are all-NaN in selected animals
    # (e.g. Fos — cfos imaging only available for behavior-experiment animals)
    all_nan_in_sel = [c for c in gene_cols if df_sel[c].isna().all()]
    if all_nan_in_sel:
        gene_cols = [c for c in gene_cols if c not in all_nan_in_sel]
        print(f"  Dropped {len(all_nan_in_sel)} all-NaN genes in selected animals: {all_nan_in_sel}")
        print(f"  Genes after filtering: {len(gene_cols)}")

    for animal_id in NAIVE_ANIMALS:
        specimen = f"animal_{animal_id}"
        print(f"\n  Processing {specimen}...")
        adf = df_sel[df_sel["Animal_ID"] == animal_id].copy()

        # Sort by Bregma for consistent ordering
        adf = adf.sort_values("Bregma")
        bregmas = sorted(adf["Bregma"].unique())
        print(f"    {len(adf):,} cells, {len(bregmas)} sections")
        print(f"    Bregma: {[f'{b:+.2f}' for b in bregmas]}")

        # Build expression matrix
        X = ensure_sparse_csr(adf[gene_cols].values.astype(np.float32))

        # Build spatial coordinates
        # x, y: Centroid_X/Y already in µm
        # z: Bregma in mm → µm
        coords_x = adf["Centroid_X"].values.astype(np.float64)
        coords_y = adf["Centroid_Y"].values.astype(np.float64)
        coords_z = adf["Bregma"].values.astype(np.float64) * 1000.0  # mm → µm
        spatial = np.column_stack([coords_x, coords_y, coords_z])

        # Build section labels from Bregma
        section_labels = [f"bregma_{b:+.2f}" for b in adf["Bregma"].values]

        # Build obs
        obs = pd.DataFrame(index=adf["Cell_ID"].astype(str).values)
        obs["cell_type"] = adf["Cell_class"].values.astype(str)
        obs["section"] = section_labels
        obs["animal_id"] = str(animal_id)
        obs["animal_sex"] = adf["Animal_sex"].values.astype(str)
        obs["behavior"] = adf["Behavior"].values.astype(str)
        obs["bregma_mm"] = adf["Bregma"].values
        obs["neuron_cluster_id"] = adf["Neuron_cluster_ID"].values

        # Build var
        var = pd.DataFrame(index=gene_cols)
        var.index.name = "gene"

        # Create AnnData
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.obsm["spatial"] = spatial
        adata.uns["spatial_metadata"] = METADATA
        adata.uns["expression_type"] = METADATA["expression_type"]
        adata.uns["dataset_name"] = DATASET_NAME

        verify(adata)

        # Save
        out_path = OUT_DIR / specimen / "data.h5ad"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"    Saving to {out_path}...")
        adata.write_h5ad(out_path)
        print(f"    Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    process()
