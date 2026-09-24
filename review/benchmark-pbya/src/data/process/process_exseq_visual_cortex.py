#!/usr/bin/env python
"""Process ExSeq visual cortex dataset to standardized h5ad."""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "exseq_visual_cortex"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_PATH = PROJECT_ROOT / "data" / "processed" / f"{DATASET_NAME}.h5ad"

CSV_FILE = "spacejam2_cellxgene.csv"

METADATA = {
    "technology": "ExSeq",
    "species": "mouse",
    "tissue": "visual cortex",
    "expression_type": "raw_counts",
    "section_spacing_um": None,
    "source": "Boyden Lab / spacejam2",
}


def check_raw_data():
    """Verify that the required raw data file exists."""
    csv_path = RAW_DIR / CSV_FILE
    if not csv_path.exists():
        print(f"ERROR: Required file not found: {csv_path}")
        sys.exit(1)
    print(f"  Found: {csv_path.name} ({csv_path.stat().st_size / 1e6:.2f} MB)")
    return csv_path


def ensure_sparse_csr(X):
    """Convert expression matrix to CSR sparse format."""
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def build_spatial_3d(x_um, y_um, z_um):
    """Build (n, 3) spatial array from coordinate arrays in micrometers."""
    return np.column_stack([
        np.array(x_um, dtype=np.float64),
        np.array(y_um, dtype=np.float64),
        np.array(z_um, dtype=np.float64),
    ])


def verify(adata):
    """Verify output meets standardized format requirements."""
    assert sp.issparse(adata.X) and adata.X.format == "csr", \
        "X must be CSR sparse"
    assert "spatial" in adata.obsm and adata.obsm["spatial"].shape == (adata.n_obs, 3), \
        "obsm['spatial'] must be (n_obs, 3)"
    assert "section" in adata.obs.columns, "obs must have 'section' column"
    assert "cell_type" in adata.obs.columns, "obs must have 'cell_type' column"
    print(f"  Verified: {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"  Spatial: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
        f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
        f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]"
    )
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")


def process():
    """Main processing pipeline."""
    print(f"Processing {DATASET_NAME}...")

    csv_path = check_raw_data()

    print(f"  Reading {CSV_FILE}...")
    df = pd.read_csv(csv_path, index_col=0)
    print(f"  Loaded {df.shape[0]} cells x {df.shape[1]} columns")

    # Separate coordinate columns from gene expression columns
    coord_cols = ["x_um", "y_um", "z_um"]
    for col in coord_cols:
        if col not in df.columns:
            print(f"ERROR: Expected coordinate column '{col}' not found in CSV.")
            print(f"  Available columns: {list(df.columns)}")
            sys.exit(1)

    gene_cols = [c for c in df.columns if c not in coord_cols]
    print(f"  Gene columns: {len(gene_cols)}")

    # Build CSR sparse expression matrix from gene columns
    X = ensure_sparse_csr(df[gene_cols].values.astype(np.float32))

    # Build (n, 3) spatial array from coordinate columns
    spatial = build_spatial_3d(df["x_um"], df["y_um"], df["z_um"])

    # Load cell type annotations from SpaceTx EDV results (same spacejam2 cells,
    # row-aligned — verified by coordinate matching within float precision)
    cell_types = "unknown"
    edv_path = RAW_DIR / "results_adata.h5ad"
    if edv_path.exists():
        edv = ad.read_h5ad(edv_path)
        ann_col = "edv_predictions_|_merged_cluster_smFISH"
        if len(edv) == len(df) and ann_col in edv.obs.columns:
            cell_types = edv.obs[ann_col].values.astype(str)
            n_typed = sum(1 for ct in cell_types if ct != "unknown")
            print(f"  Cell types from EDV results: {n_typed}/{len(df)} cells ({edv.obs[ann_col].nunique()} types)")

    obs = pd.DataFrame(
        {
            "section": "visual_cortex",
            "cell_type": cell_types,
        },
        index=df.index.astype(str),
    )

    # Build var DataFrame
    var = pd.DataFrame(index=gene_cols)
    var.index.name = None

    # Assemble AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    adata.uns["spatial_metadata"] = METADATA
    adata.uns["expression_type"] = METADATA["expression_type"]
    adata.uns["dataset_name"] = DATASET_NAME

    verify(adata)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUT_PATH)
    print(f"\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print("Done.")


def main():
    process()


if __name__ == "__main__":
    main()
