#!/usr/bin/env python
"""Process Allen Zhuang MERFISH mouse brain to standardized h5ad.

Processes 4 regional parcellations (Zhuang-ABCA-1 through 4) separately,
each with its own per-region metadata CSV for CCF coordinates and cell types.
Outputs one h5ad per region.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "allen_zhuang_merfish"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME

REGIONS = ["Zhuang-ABCA-1", "Zhuang-ABCA-2", "Zhuang-ABCA-3", "Zhuang-ABCA-4"]


def check_raw_data():
    """Check that all required raw files exist."""
    missing = []
    for region in REGIONS:
        h5ad = RAW_DIR / f"{region}-log2.h5ad"
        meta = RAW_DIR / f"{region}-cell_metadata.csv"
        if not h5ad.exists():
            missing.append(h5ad.name)
        if not meta.exists():
            missing.append(meta.name)
    if missing:
        print(f"ERROR: Missing files in {RAW_DIR}:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)
    print(f"  All {len(REGIONS) * 2} input files present.")


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


def find_ccf_columns(df):
    """Find CCF coordinate columns in metadata DataFrame."""
    candidates = [
        ("x_ccf", "y_ccf", "z_ccf"),
        ("ccf_x", "ccf_y", "ccf_z"),
        ("x_reconstructed", "y_reconstructed", "z_reconstructed"),
        ("x", "y", "z"),
        ("X", "Y", "Z"),
    ]
    for x, y, z in candidates:
        if x in df.columns and y in df.columns and z in df.columns:
            return x, y, z
    cols = df.columns.tolist()
    x_cols = [c for c in cols if "x" in c.lower() and ("ccf" in c.lower() or "coord" in c.lower())]
    y_cols = [c for c in cols if "y" in c.lower() and ("ccf" in c.lower() or "coord" in c.lower())]
    z_cols = [c for c in cols if "z" in c.lower() and ("ccf" in c.lower() or "coord" in c.lower())]
    if x_cols and y_cols and z_cols:
        return x_cols[0], y_cols[0], z_cols[0]
    raise ValueError(
        f"No CCF coordinate columns found. Available columns: {sorted(df.columns.tolist())}"
    )


def find_cell_type_column(df):
    """Find cell type column in metadata DataFrame."""
    candidates = [
        "cell_type", "celltype", "CellType", "cluster", "cluster_alias",
        "cluster_label", "class", "class_label", "subclass", "subclass_label",
        "supertype", "supertype_label", "annotation",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"No cell type column found. Available columns: {sorted(df.columns.tolist())}"
    )


def find_section_column(df):
    """Find section column in metadata DataFrame."""
    candidates = [
        "section", "section_id", "slice", "slice_id",
        "brain_section_label", "z_reconstructed",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def process_region(region):
    """Process a single Zhuang-ABCA region."""
    out_path = OUT_DIR / f"{region}.h5ad"
    if out_path.exists():
        print(f"  SKIP (exists): {out_path.name}")
        return

    h5ad_path = RAW_DIR / f"{region}-log2.h5ad"
    meta_path = RAW_DIR / f"{region}-cell_metadata.csv"

    print(f"\n  --- {region} ---")
    print(f"  Loading h5ad: {h5ad_path.name} ({h5ad_path.stat().st_size / 1e9:.1f} GB)")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    print(f"  Loading metadata: {meta_path.name} ({meta_path.stat().st_size / 1e6:.0f} MB)")
    meta = pd.read_csv(meta_path, index_col=0)
    print(f"  Metadata: {meta.shape[0]} rows x {meta.shape[1]} columns")

    # Align — intersect cell indices
    adata.obs.index = adata.obs.index.astype(str)
    meta.index = meta.index.astype(str)
    common_idx = adata.obs.index.intersection(meta.index)
    print(f"  Common cells: {len(common_idx)} / {adata.n_obs} ({100*len(common_idx)/adata.n_obs:.0f}%)")

    if len(common_idx) == 0:
        print(f"  ERROR: No matching cells for {region}. Skipping.")
        return

    adata = adata[common_idx].copy()
    meta_aligned = meta.loc[common_idx]

    # Standardize expression matrix
    adata.X = ensure_sparse_csr(adata.X)

    # Swap var_names: Ensembl IDs -> gene symbols
    if "gene_symbol" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var.index.values
        new_names = adata.var["gene_symbol"].fillna(adata.var["ensembl_id"])
        adata.var.index = new_names.values
        adata.var_names_make_unique()
        n_ens = sum(1 for n in adata.var_names if n.startswith("ENSMUSG"))
        print(f"  Gene names: swapped to symbols ({n_ens} Ensembl IDs remaining)")

    # Extract CCF 3D coordinates (in mm → convert to µm)
    x_col, y_col, z_col = find_ccf_columns(meta_aligned)
    print(f"  Using CCF columns: {x_col}, {y_col}, {z_col}")
    spatial_3d = np.column_stack([
        meta_aligned[x_col].values.astype(np.float64) * 1000.0,  # mm → µm
        meta_aligned[y_col].values.astype(np.float64) * 1000.0,
        meta_aligned[z_col].values.astype(np.float64) * 1000.0,
    ])
    adata.obsm["spatial"] = spatial_3d

    # Cell type
    ct_col = find_cell_type_column(meta_aligned)
    print(f"  Using cell type column: {ct_col}")
    adata.obs["cell_type"] = meta_aligned[ct_col].astype(str).values

    # Section
    sec_col = find_section_column(meta_aligned)
    if sec_col is not None:
        adata.obs["section"] = meta_aligned[sec_col].astype(str).values
    else:
        z_vals = spatial_3d[:, 2]
        adata.obs["section"] = np.round(z_vals, 1).astype(str)

    # Copy all remaining metadata columns from CSV
    skip_cols = {x_col, y_col, z_col, ct_col, sec_col} - {None}
    for col in meta_aligned.columns:
        if col not in skip_cols and col not in adata.obs.columns:
            adata.obs[col] = meta_aligned[col].values

    # Dataset-level metadata
    adata.uns["technology"] = "MERFISH"
    adata.uns["species"] = "mouse"
    adata.uns["tissue"] = "whole brain"
    adata.uns["expression_type"] = "log2_normalized"
    adata.uns["source"] = f"Allen Brain Cell Atlas ({region})"
    adata.uns["dataset_name"] = DATASET_NAME
    adata.uns["region"] = region
    adata.uns["n_sections"] = int(adata.obs["section"].nunique())

    verify(adata)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"  Saved: {out_path}")


def process():
    """Process all 4 regions."""
    for region in REGIONS:
        process_region(region)


def main():
    print(f"Processing {DATASET_NAME}")
    print(f"  Raw dir: {RAW_DIR}")

    if not RAW_DIR.exists():
        print(f"ERROR: Raw directory not found: {RAW_DIR}")
        sys.exit(1)

    check_raw_data()
    process()

    print(f"\nDone. Output in {OUT_DIR}")


if __name__ == "__main__":
    main()
