#!/usr/bin/env python
"""Process HuBMAP 3D IMC spleen data to standardized h5ad.

NOTE: This dataset only contains imaging data (OME-TIFF image stack),
no cell-level expression tables. Processing is skipped unless
segmentation/quantification CSVs are found.
"""
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "hubmap_imc_spleen"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME

def ensure_sparse_csr(X):
    if sp.issparse(X): return X.tocsr()
    return sp.csr_matrix(X)

def build_spatial_3d(coords_2d, z=0.0):
    n = coords_2d.shape[0]
    z_col = np.full((n, 1), z, dtype=np.float64)
    return np.hstack([np.array(coords_2d, dtype=np.float64), z_col])

def verify(adata):
    assert sp.issparse(adata.X) and adata.X.format == 'csr'
    assert 'spatial' in adata.obsm and adata.obsm['spatial'].shape == (adata.n_obs, 3)
    assert 'section' in adata.obs.columns and 'cell_type' in adata.obs.columns
    print(f"  Verified: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  Spatial: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
          f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
          f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]")
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")


def process():
    """Check for cell-level expression data; skip if only imaging."""
    if not RAW_DIR.exists():
        print(f"SKIP: {RAW_DIR} does not exist.")
        print("  Run: python src/data/download/download_hubmap_imc_spleen.py")
        sys.exit(0)

    print(f"Scanning {RAW_DIR} for cell-level expression data ...")

    # Search for any CSV/TSV files that might contain quantification
    csv_files = sorted(RAW_DIR.rglob("*.csv"))
    tsv_files = sorted(RAW_DIR.rglob("*.tsv"))
    h5ad_files = sorted(RAW_DIR.rglob("*.h5ad"))
    txt_files = sorted(RAW_DIR.rglob("*.txt"))

    all_data_files = csv_files + tsv_files + h5ad_files
    print(f"  Found: {len(csv_files)} CSV, {len(tsv_files)} TSV, "
          f"{len(h5ad_files)} h5ad, {len(txt_files)} TXT files")

    # Check for image files
    tiff_files = sorted(RAW_DIR.rglob("*.tiff")) + sorted(RAW_DIR.rglob("*.tif"))
    ome_files = sorted(RAW_DIR.rglob("*.ome.tiff")) + sorted(RAW_DIR.rglob("*.ome.tif"))
    print(f"  Found: {len(tiff_files)} TIFF/TIF files ({len(ome_files)} OME-TIFF)")

    if h5ad_files:
        # If we find an h5ad, try to load it
        print(f"  Found h5ad: {h5ad_files[0]}")
        adata = ad.read_h5ad(h5ad_files[0])
        print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
        # Process it (would need standardization)
        # For now, this is unexpected - print info and continue
        adata.X = ensure_sparse_csr(adata.X)
        # ... would need full processing here
        return

    if csv_files or tsv_files:
        # Check if any CSV contains cell-level data
        for f in (csv_files + tsv_files)[:5]:
            sep = '\t' if f.suffix == '.tsv' else ','
            try:
                df = pd.read_csv(f, sep=sep, nrows=5)
                print(f"    {f.name}: {len(df.columns)} cols - {list(df.columns[:5])}")
                if len(df.columns) > 5:
                    print(f"      (cell-level data may be available)")
            except Exception:
                pass
        # If we get here with CSV data, further manual inspection is needed
        print("\n  NOTE: Found CSV/TSV files. Manual inspection needed to determine")
        print("  if these contain cell-level expression data.")

    # Default: imaging-only dataset
    print("\nSKIP: HuBMAP IMC spleen dataset contains only imaging data (3D IMC image stack).")
    print("  No cell-level expression tables available.")
    print("  Source: HuBMAP d3130f4a")
    print("  Technology: 3D IMC | Species: human | Tissue: spleen")
    print("")
    print("  To process this dataset, cell segmentation and quantification must be")
    print("  performed on the OME-TIFF image stack first (e.g., using CellProfiler,")
    print("  Mesmer, or similar tools).")
    sys.exit(0)


if __name__ == "__main__":
    process()
