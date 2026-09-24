#!/usr/bin/env python
"""Process Deep-STARmap dataset to standardized h5ad."""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "deep_starmap"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_PATH = PROJECT_ROOT / "data" / "processed" / f"{DATASET_NAME}.h5ad"
METADATA = {
    "technology": "Deep-STARmap",
    "species": "mouse",
    "tissue": "brain (thick blocks)",
    "n_sections": None,
    "section_spacing_um": 0.70,
    "coordinate_units": "micrometers",
    "expression_type": "raw_counts",
    "source": "Zenodo 10.5281/zenodo.16783354",
}

# Voxel size from Sui et al. 2025 (Nature Methods): Leica TCS SP8, 25x NA 0.95
VOXEL_XY_UM = 0.32  # µm per pixel in x,y
VOXEL_Z_UM = 0.70   # µm per z-step


def check_raw_data():
    files = [
        "Brain_Deep_STARmap_expression_matrix.csv",
        "Brain_Deep_STARmap_spatial.csv",
        "Brain_Deep_STARmap_X_umap.csv",
    ]
    missing = [f for f in files if not (RAW_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing files in {RAW_DIR}:")
        for f in missing:
            print(f"  - {f}")
        print(f"Run: python src/data/download/download_{DATASET_NAME}.py")
        sys.exit(1)


def ensure_sparse_csr(X):
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def build_spatial_3d(coords_2d, z=0.0):
    n = coords_2d.shape[0]
    z_col = np.full((n, 1), z, dtype=np.float64)
    return np.hstack([np.array(coords_2d, dtype=np.float64), z_col])


def verify(adata):
    assert sp.issparse(adata.X) and adata.X.format == "csr"
    assert "spatial" in adata.obsm and adata.obsm["spatial"].shape == (adata.n_obs, 3)
    assert "section" in adata.obs.columns and "cell_type" in adata.obs.columns
    print(f"  Verified: {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"  Spatial: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
        f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
        f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]"
    )
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")


def process():
    print(f"Processing {DATASET_NAME}...")
    check_raw_data()

    # Load expression matrix (no index column — rows are positional)
    print("  Loading expression matrix (this may take a while for 404 MB)...")
    expr = pd.read_csv(RAW_DIR / "Brain_Deep_STARmap_expression_matrix.csv")
    print(f"  Expression shape: {expr.shape}")

    # Load spatial coordinates (columns: x, y, z, Harmony_labels, FUSEmap_main_level, FUSEmap_sub_level)
    # No index column — x is a data column, not an index
    print("  Loading spatial coordinates...")
    spatial_df = pd.read_csv(RAW_DIR / "Brain_Deep_STARmap_spatial.csv")
    print(f"  Spatial shape: {spatial_df.shape}")
    print(f"  Spatial columns: {list(spatial_df.columns)}")

    # Load UMAP coordinates (has unnamed index column)
    print("  Loading UMAP coordinates...")
    umap_df = pd.read_csv(RAW_DIR / "Brain_Deep_STARmap_X_umap.csv", index_col=0)
    print(f"  UMAP shape: {umap_df.shape}")

    # All three files have the same number of rows, aligned positionally
    assert expr.shape[0] == spatial_df.shape[0] == umap_df.shape[0], (
        f"Row count mismatch: expr={expr.shape[0]}, spatial={spatial_df.shape[0]}, umap={umap_df.shape[0]}"
    )
    n_cells = expr.shape[0]
    print(f"  All files have {n_cells} cells (positionally aligned)")

    # Build sparse expression matrix
    X = ensure_sparse_csr(expr.values.astype(np.float32))

    # Extract 3D spatial coordinates (raw values are voxel indices)
    # Convert to physical µm: voxel size 0.32 × 0.32 × 0.70 µm (Sui et al. 2025)
    spatial = np.column_stack([
        spatial_df["x"].values * VOXEL_XY_UM,
        spatial_df["y"].values * VOXEL_XY_UM,
        spatial_df["z"].values * VOXEL_Z_UM,
    ]).astype(np.float64)

    # Extract cell type annotations from spatial file
    cell_types = spatial_df["FUSEmap_sub_level"].values.astype(str)

    # Use z coordinate as section identifier
    sections = spatial_df["z"].values.astype(str)

    # UMAP embedding
    umap_coords = umap_df.values.astype(np.float64)

    # Build obs with positional index
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    obs = pd.DataFrame(index=cell_ids)
    obs["cell_type"] = cell_types
    obs["section"] = sections
    obs["Harmony_labels"] = spatial_df["Harmony_labels"].values.astype(str)
    obs["FUSEmap_main_level"] = spatial_df["FUSEmap_main_level"].values.astype(str)

    # Build var
    var = pd.DataFrame(index=expr.columns)
    var.index.name = "gene"

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    adata.obsm["X_umap"] = umap_coords
    adata.uns["dataset"] = METADATA
    adata.uns["expression_type"] = METADATA["expression_type"]
    adata.uns["dataset_name"] = DATASET_NAME

    adata.obs["section"] = adata.obs["section"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)

    verify(adata)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to {OUT_PATH}...")
    adata.write_h5ad(OUT_PATH)
    print(f"  Done. File size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")

    return adata


if __name__ == "__main__":
    process()
