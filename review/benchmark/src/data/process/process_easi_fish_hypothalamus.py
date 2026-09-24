#!/usr/bin/env python
"""Process EASI-FISH hypothalamus dataset to standardized h5ad.

Splits into 3 separate h5ad files (LHA1, LHA2, LHA3) — independent tissue samples
with overlapping z-ranges, not one consecutive stack.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "easi_fish_hypothalamus"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME
METADATA = {
    "technology": "EASI-FISH",
    "species": "mouse",
    "tissue": "hypothalamus (LHA)",
    "n_sections": None,
    "section_spacing_um": None,
    "coordinate_units": "micrometers",
    "expression_type": "normalized",
    "source": "Figshare 13749154",
}

SAMPLES = ["LHA1", "LHA2", "LHA3"]


def check_raw_data():
    files = [
        "EASI_FISH_gene_count.csv",
        "EASI_FISH_metadata.csv",
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


def build_spatial_3d(coords_x, coords_y, coords_z):
    return np.column_stack([
        np.asarray(coords_x, dtype=np.float64),
        np.asarray(coords_y, dtype=np.float64),
        np.asarray(coords_z, dtype=np.float64),
    ])


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

    # Load gene count matrix
    print("  Loading gene count matrix...")
    counts = pd.read_csv(RAW_DIR / "EASI_FISH_gene_count.csv", index_col=0)
    print(f"  Counts shape: {counts.shape}")

    # Load metadata with coordinates
    print("  Loading metadata...")
    meta = pd.read_csv(RAW_DIR / "EASI_FISH_metadata.csv", index_col=0)
    print(f"  Metadata shape: {meta.shape}")
    print(f"  Metadata columns: {list(meta.columns)}")

    # Align cells
    common_cells = counts.index.intersection(meta.index)
    if len(common_cells) == 0:
        counts.index = counts.index.astype(str)
        meta.index = meta.index.astype(str)
        common_cells = counts.index.intersection(meta.index)
    print(f"  Common cells: {len(common_cells)}")

    counts = counts.loc[common_cells]
    meta = meta.loc[common_cells]

    # Replace -1.0 sentinel values with 0 (missing/undetected)
    vals = counts.values.astype(np.float32)
    n_neg = (vals < 0).sum()
    if n_neg > 0:
        print(f"  Replacing {n_neg} negative sentinel values with 0")
        vals[vals < 0] = 0.0

    # Extract sample prefix from cell IDs (e.g., LHA1_1 -> LHA1)
    cell_ids = pd.Series(common_cells.astype(str), index=common_cells)
    sample_prefix = cell_ids.str.extract(r'^([A-Z]+\d+)_', expand=False)
    print(f"  Sample prefixes found: {sorted(sample_prefix.unique().tolist())}")
    print(f"  Cells per sample:")
    for s in sorted(sample_prefix.unique()):
        print(f"    {s}: {(sample_prefix == s).sum()}")

    # Coordinate columns
    print(f"  Coordinate columns: x=x, y=y, z=z")

    # Remove old single-file output if it exists
    old_single = PROJECT_ROOT / "data" / "processed" / f"{DATASET_NAME}.h5ad"
    if old_single.exists():
        print(f"  Removing old single-file output: {old_single}")
        old_single.unlink()

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each sample separately
    gene_names = counts.columns
    for sample in SAMPLES:
        print(f"\n  Processing sample {sample}...")
        mask = sample_prefix == sample
        if mask.sum() == 0:
            print(f"  WARNING: No cells found for {sample}, skipping")
            continue

        sample_cells = common_cells[mask]
        sample_meta = meta.loc[sample_cells]
        sample_vals = vals[mask.values]

        # Build sparse matrix
        X = ensure_sparse_csr(sample_vals)

        # Build 3D spatial coordinates
        spatial = build_spatial_3d(
            sample_meta["x"].values,
            sample_meta["y"].values,
            sample_meta["z"].values,
        )

        # Build obs — cell_type from metadata, section = sample name
        obs = pd.DataFrame(index=sample_cells)
        obs["cell_type"] = sample_meta["cell_type"].values.astype(str)
        obs["section"] = sample

        # Copy remaining metadata columns
        skip_cols = {"x", "y", "z", "cell_type"}
        for c in sample_meta.columns:
            if c not in skip_cols:
                obs[c] = sample_meta[c].values

        # Build var
        var = pd.DataFrame(index=gene_names)
        var.index.name = "gene"

        # Create AnnData
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.obsm["spatial"] = spatial
        adata.uns["dataset"] = METADATA
        adata.uns["expression_type"] = METADATA["expression_type"]
        adata.uns["dataset_name"] = DATASET_NAME

        adata.obs["section"] = adata.obs["section"].astype(str)
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)

        verify(adata)

        # Save
        out_path = OUT_DIR / f"{sample}.h5ad"
        print(f"  Saving to {out_path}...")
        adata.write_h5ad(out_path)
        print(f"  Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")

    print(f"\nAll samples processed. Output directory: {OUT_DIR}")


if __name__ == "__main__":
    process()
