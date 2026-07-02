#!/usr/bin/env python
"""Process Array-seq mouse kidney 3D dataset to standardized h5ad.

Source: GSE266244 (GSM8243015_3D_KI_Raw_Z_aligned_annotated.h5ad)
Output: data/processed/arrayseq_kidney/data.h5ad
Description: 8 serial kidney sections at 100 µm spacing from Array-seq technology.
  The source h5ad has X,Y in mm and z_plane (1-8) for section identity.
  Expression is log1p-normalized (non-integer values 0.51-4.32).
  Subregion annotations available (Cortex-CT, Cortex-DCT, Cortex-G, Cortex-PCT,
  Fat, ISOM, Medulla, OSOM, Urothelium).
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "arrayseq_kidney"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME

SECTION_SPACING_UM = 100.0  # 100 µm between serial sections (isoST paper)

METADATA = {
    "technology": "Array-seq",
    "species": "mouse",
    "tissue": "kidney",
    "n_sections": 8,
    "section_spacing_um": SECTION_SPACING_UM,
    "expression_type": "log1p_normalized",
    "coordinate_units": "micrometers",
    "source": "GEO GSE266244",
    "paper": "isoST",
}


def check_raw_data():
    """Verify that required raw files exist."""
    h5ad_path = RAW_DIR / "GSM8243015_3D_KI_Raw_Z_aligned_annotated.h5ad"
    if not h5ad_path.exists():
        print(f"ERROR: Missing raw file: {h5ad_path}")
        print("Run download_arrayseq_kidney.py first.")
        sys.exit(1)
    print(f"Raw data check passed: {h5ad_path.name} ({h5ad_path.stat().st_size:,} bytes)")
    return h5ad_path


def ensure_sparse_csr(X):
    """Convert expression matrix to CSR sparse format."""
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def build_spatial_3d(coords_2d, z=0.0):
    """Add z column to 2D spatial coordinates."""
    n = coords_2d.shape[0]
    z_col = np.full((n, 1), z, dtype=np.float64)
    return np.hstack([np.array(coords_2d, dtype=np.float64), z_col])


def verify(adata):
    """Verify the processed AnnData meets standards."""
    assert sp.issparse(adata.X) and adata.X.format == "csr", "X must be CSR sparse"
    assert "spatial" in adata.obsm and adata.obsm["spatial"].shape == (adata.n_obs, 3), "obsm['spatial'] must be (n, 3)"
    assert "section" in adata.obs.columns, "obs must have 'section' column"
    assert "cell_type" in adata.obs.columns, "obs must have 'cell_type' column"

    sections = sorted(adata.obs["section"].unique())
    print(f"  Verified: {adata.n_obs} spots x {adata.n_vars} genes, {len(sections)} sections")
    print(
        f"  Spatial range: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
        f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
        f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]"
    )
    print(f"  Sections: {sections}")
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")
    ct_counts = adata.obs["cell_type"].value_counts()
    print(f"  Cell types: {len(ct_counts)} unique")
    print(f"  Top cell types:\n{ct_counts.head(10).to_string()}")


def process():
    """Main processing pipeline."""
    print(f"Processing {DATASET_NAME}...")

    h5ad_path = check_raw_data()

    # Load the pre-aligned 3D kidney h5ad
    print(f"  Loading {h5ad_path.name}...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.n_obs} spots x {adata.n_vars} genes")

    # Ensure CSR sparse
    adata.X = ensure_sparse_csr(adata.X)

    # Check expression values
    sample_vals = adata.X.data[:min(10000, len(adata.X.data))]
    is_integer = np.allclose(sample_vals, np.round(sample_vals))
    print(f"  Expression: min={sample_vals.min():.2f}, max={sample_vals.max():.2f}, "
          f"mean={sample_vals.mean():.2f}, integer={is_integer}")
    # Values are non-integer (0.51-4.32 range), log1p normalized
    expr_type = "log1p_normalized"

    # Spatial coordinates: X, Y are in mm, z_plane gives section (1-8)
    # Convert X, Y from mm to µm
    x_mm = adata.obs["X"].values.astype(np.float64)
    y_mm = adata.obs["Y"].values.astype(np.float64)
    x_um = x_mm * 1000.0  # mm -> µm
    y_um = y_mm * 1000.0

    # z from z_plane (1-8) at 100 µm spacing
    z_plane = adata.obs["z_plane"].values.astype(np.float64)
    z_um = (z_plane - 1.0) * SECTION_SPACING_UM  # section 1 -> z=0, section 8 -> z=700

    spatial_3d = np.column_stack([x_um, y_um, z_um])
    adata.obsm["spatial"] = spatial_3d

    print(f"  Spatial range: x=[{x_um.min():.1f}, {x_um.max():.1f}] µm, "
          f"y=[{y_um.min():.1f}, {y_um.max():.1f}] µm, "
          f"z=[{z_um.min():.1f}, {z_um.max():.1f}] µm")

    # Section column: use z_plane as section identifier
    adata.obs["section"] = [f"section_{int(z)}" for z in z_plane]

    # Cell type: use Subregion annotations (kidney structural regions)
    adata.obs["cell_type"] = adata.obs["Subregion"].astype(str)
    print(f"  Subregion annotations: {adata.obs['cell_type'].nunique()} unique")

    # Gene names
    adata.var_names_make_unique()

    # Metadata
    adata.uns["spatial_metadata"] = METADATA.copy()
    adata.uns["expression_type"] = expr_type
    adata.uns["dataset_name"] = DATASET_NAME

    # Verify
    verify(adata)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "data.h5ad"
    print(f"  Saving to {out_path}...")
    adata.write_h5ad(out_path)
    print(f"  Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")

    return adata


def main():
    process()


if __name__ == "__main__":
    main()
