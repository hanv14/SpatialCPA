#!/usr/bin/env python
"""Process Allen MERFISH mouse brain to standardized h5ad."""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "allen_merfish_brain"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "allen_merfish_brain.h5ad"


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
    # Common naming conventions for Allen CCF coordinates
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
    # Also try partial matches
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
        "cell_type", "celltype", "CellType", "cluster", "cluster_label",
        "class", "class_label", "subclass", "subclass_label",
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


def main():
    print(f"Processing {DATASET_NAME}")
    print(f"  Raw dir: {RAW_DIR}")

    if not RAW_DIR.exists():
        print(f"ERROR: Raw directory not found: {RAW_DIR}")
        sys.exit(1)

    if OUT_PATH.exists():
        print(f"  SKIP (exists): {OUT_PATH}")
        return

    # Find input files
    h5ad_path = RAW_DIR / "C57BL6J-638850-raw.h5ad"
    meta_path = RAW_DIR / "cell_metadata_with_cluster_annotation.csv"
    gene_path = RAW_DIR / "gene.csv"

    for p in [h5ad_path, meta_path]:
        if not p.exists():
            print(f"ERROR: Required file not found: {p}")
            sys.exit(1)

    # Load h5ad -- this is 7.6 GB, load fully for X conversion
    print(f"  Loading h5ad: {h5ad_path.name} ({h5ad_path.stat().st_size / 1e9:.1f} GB)")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # Load cell metadata CSV
    print(f"  Loading metadata: {meta_path.name} ({meta_path.stat().st_size / 1e9:.1f} GB)")
    meta = pd.read_csv(meta_path, index_col=0)
    print(f"  Metadata: {meta.shape[0]} rows x {meta.shape[1]} columns")
    print(f"  Metadata columns: {sorted(meta.columns.tolist())[:20]}...")

    # Align metadata to adata obs index
    common_idx = adata.obs.index.intersection(meta.index)
    print(f"  Common cells: {len(common_idx)} / {adata.n_obs}")
    if len(common_idx) < adata.n_obs * 0.5:
        print("  WARNING: Less than 50% of cells matched. Trying string conversion...")
        meta.index = meta.index.astype(str)
        adata.obs.index = adata.obs.index.astype(str)
        common_idx = adata.obs.index.intersection(meta.index)
        print(f"  Common cells after str conversion: {len(common_idx)} / {adata.n_obs}")

    # Subset to matched cells
    adata = adata[common_idx].copy()
    meta = meta.loc[common_idx]

    # Standardize expression matrix
    adata.X = ensure_sparse_csr(adata.X)

    # Extract CCF 3D coordinates (native 3D from Allen CCF, in mm → convert to µm)
    x_col, y_col, z_col = find_ccf_columns(meta)
    print(f"  Using CCF columns: {x_col}, {y_col}, {z_col}")
    spatial_3d = np.column_stack([
        meta[x_col].values.astype(np.float64) * 1000.0,  # mm → µm
        meta[y_col].values.astype(np.float64) * 1000.0,
        meta[z_col].values.astype(np.float64) * 1000.0,
    ])
    adata.obsm["spatial"] = spatial_3d

    # Extract cell type
    ct_col = find_cell_type_column(meta)
    print(f"  Using cell type column: {ct_col}")
    adata.obs["cell_type"] = meta[ct_col].astype(str).values

    # Extract section info
    sec_col = find_section_column(meta)
    if sec_col is not None:
        print(f"  Using section column: {sec_col}")
        adata.obs["section"] = meta[sec_col].astype(str).values
    else:
        # Derive section from z coordinate (binned)
        z_vals = spatial_3d[:, 2]
        z_unique = np.unique(np.round(z_vals, 1))
        print(f"  Deriving sections from z coordinate ({len(z_unique)} unique z values)")
        adata.obs["section"] = np.round(z_vals, 1).astype(str)

    # Copy all remaining metadata columns from CSV
    skip_cols = {x_col, y_col, z_col, ct_col, sec_col} - {None}
    for col in meta.columns:
        if col not in skip_cols and col not in adata.obs.columns:
            adata.obs[col] = meta[col].values

    # Dataset-level metadata
    adata.uns["technology"] = "MERFISH"
    adata.uns["species"] = "mouse"
    adata.uns["tissue"] = "whole brain"
    adata.uns["expression_type"] = "raw_counts"
    adata.uns["source"] = "MERFISH-C57BL6J-638850"
    adata.uns["dataset_name"] = DATASET_NAME
    adata.uns["n_sections"] = 59

    # Load gene metadata if available
    if gene_path.exists():
        gene_meta = pd.read_csv(gene_path, index_col=0)
        print(f"  Gene metadata: {gene_meta.shape[0]} genes x {gene_meta.shape[1]} columns")
        for col in gene_meta.columns:
            if col not in adata.var.columns:
                common_genes = adata.var.index.intersection(gene_meta.index)
                if len(common_genes) > 0:
                    adata.var.loc[common_genes, col] = gene_meta.loc[common_genes, col]

    # Swap var_names: Ensembl IDs -> gene symbols
    if "gene_symbol" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var.index.values
        new_names = adata.var["gene_symbol"].fillna(adata.var["ensembl_id"])
        adata.var.index = new_names.values
        adata.var_names_make_unique()
        n_ens = sum(1 for n in adata.var_names if n.startswith("ENSMUSG"))
        print(f"  Gene names: swapped to symbols ({n_ens} Ensembl IDs remaining)")

    verify(adata)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Writing: {OUT_PATH}")
    adata.write_h5ad(OUT_PATH)
    print(f"  Saved: {OUT_PATH}")

    print(f"\nDone. Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
