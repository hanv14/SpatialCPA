#!/usr/bin/env python
"""Process Vizgen MERFISH mouse brain to standardized h5ad.

Expects Vizgen-format CSVs: cell_by_gene.csv (expression) and cell_metadata.csv
(spatial coordinates). These files require free registration at Vizgen's website.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "vizgen_merfish_brain"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "vizgen_merfish_brain.h5ad"

# Section z-spacing in microns
SECTION_Z_SPACING = 10.0


def check_raw_data():
    """Check that required Vizgen files exist."""
    gene_path = RAW_DIR / "cell_by_gene.csv"
    meta_path = RAW_DIR / "cell_metadata.csv"
    if not gene_path.exists() or not meta_path.exists():
        print(f"Vizgen MERFISH data not available.")
        print(f"This dataset requires free registration at: https://info.vizgen.com/mouse-brain-data")
        print(f"Expected files in {RAW_DIR}:")
        print(f"  - cell_by_gene.csv")
        print(f"  - cell_metadata.csv")
        print()
        print("Skipping processing (no data).")
        sys.exit(0)
    return gene_path, meta_path


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
    """Process Vizgen MERFISH data from CSVs to h5ad."""
    gene_path, meta_path = check_raw_data()

    if OUT_PATH.exists():
        print(f"  SKIP (exists): {OUT_PATH}")
        return

    # Load expression matrix
    print(f"  Loading expression: {gene_path.name} ({gene_path.stat().st_size / 1e9:.1f} GB)")
    expr = pd.read_csv(gene_path, index_col=0)
    print(f"  Loaded: {expr.shape[0]} cells x {expr.shape[1]} genes")

    # Load metadata
    print(f"  Loading metadata: {meta_path.name} ({meta_path.stat().st_size / 1e6:.0f} MB)")
    meta = pd.read_csv(meta_path, index_col=0)
    print(f"  Metadata: {meta.shape[0]} rows x {meta.shape[1]} columns")
    print(f"  Metadata columns: {sorted(meta.columns.tolist())}")

    # Align indices
    common_idx = expr.index.intersection(meta.index)
    print(f"  Common cells: {len(common_idx)} / {expr.shape[0]}")
    expr = expr.loc[common_idx]
    meta = meta.loc[common_idx]

    # Build AnnData
    X = sp.csr_matrix(expr.values.astype(np.float32))
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=common_idx), var=pd.DataFrame(index=expr.columns))

    # Spatial coordinates: center_x, center_y + section-derived z
    x_col = "center_x" if "center_x" in meta.columns else "x"
    y_col = "center_y" if "center_y" in meta.columns else "y"
    x_vals = meta[x_col].values.astype(np.float64)
    y_vals = meta[y_col].values.astype(np.float64)

    # Section / slice for z coordinate
    sec_col = None
    for candidate in ["fov", "slice_id", "slice", "section", "z"]:
        if candidate in meta.columns:
            sec_col = candidate
            break

    if sec_col is not None:
        sections = meta[sec_col].astype(str)
        unique_sections = sorted(sections.unique())
        section_to_z = {s: i * SECTION_Z_SPACING for i, s in enumerate(unique_sections)}
        z_vals = sections.map(section_to_z).values.astype(np.float64)
        adata.obs["section"] = sections.values
    else:
        z_vals = np.zeros(len(meta), dtype=np.float64)
        adata.obs["section"] = "0"

    adata.obsm["spatial"] = np.column_stack([x_vals, y_vals, z_vals])

    # Cell type — Vizgen data may not have cell type annotations
    ct_col = None
    for candidate in ["cell_type", "celltype", "cluster", "annotation", "class"]:
        if candidate in meta.columns:
            ct_col = candidate
            break
    if ct_col is not None:
        adata.obs["cell_type"] = meta[ct_col].astype(str).values
    else:
        adata.obs["cell_type"] = "unknown"

    # Dataset-level metadata
    adata.uns["technology"] = "MERFISH"
    adata.uns["species"] = "mouse"
    adata.uns["tissue"] = "brain (coronal)"
    adata.uns["expression_type"] = "counts"
    adata.uns["source"] = "info.vizgen.com/mouse-brain-data"
    adata.uns["dataset_name"] = DATASET_NAME
    adata.uns["n_sections"] = int(adata.obs["section"].nunique())

    verify(adata)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Writing: {OUT_PATH}")
    adata.write_h5ad(OUT_PATH)
    print(f"  Saved: {OUT_PATH}")


def main():
    print(f"Processing {DATASET_NAME}")
    print(f"  Raw dir: {RAW_DIR}")

    if not RAW_DIR.exists():
        print(f"Raw directory not found: {RAW_DIR}")
        print("Skipping processing (no data).")
        sys.exit(0)

    process()

    print(f"\nDone. Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
