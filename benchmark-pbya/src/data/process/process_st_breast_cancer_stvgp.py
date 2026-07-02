#!/usr/bin/env python
"""Process ST breast cancer data (Stahl et al., 2016) for stVGP paper reproduction.

Source: spatialresearch.org (DOI: 10.1126/science.aaf2403)
  - 4 serial cryosections of human breast cancer tissue
  - TSV count matrices: rows=spots (named "XxY"), columns=genes
  - Original Spatial Transcriptomics platform: 100 µm spot diameter, 200 µm center-to-center
Output: data/processed/st_breast_cancer_stvgp/data.h5ad
Description: 4 consecutive 10 µm sections, ~250-264 spots each, ~14,800 genes.
  Array coordinates converted to µm using 200 µm inter-spot spacing.
  z-spacing: 10 µm between consecutive serial sections (standard cryosection thickness).

  stVGP paper key genes: FN1, COL3A1, LUM, COL1A1, SPARC, PRSS23
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "st_breast_cancer_stvgp"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME

LAYERS = [
    {"file": "Layer1_BC_count_matrix-1.tsv", "section": "Layer1"},
    {"file": "Layer2_BC_count_matrix-1.tsv", "section": "Layer2"},
    {"file": "Layer3_BC_count_matrix-1.tsv", "section": "Layer3"},
    {"file": "Layer4_BC_count_matrix-1.tsv", "section": "Layer4"},
]

# ST array: 200 µm center-to-center spot spacing
# Array coordinates are integer grid positions
ST_SPOT_SPACING_UM = 200.0

# Section thickness: 10 µm cryosections (Stahl et al. 2016 Methods)
SECTION_THICKNESS_UM = 10.0

# Physical z-positions for consecutive serial sections
SECTION_Z_UM = {
    "Layer1": 0.0,
    "Layer2": 10.0,
    "Layer3": 20.0,
    "Layer4": 30.0,
}

METADATA = {
    "technology": "Spatial Transcriptomics",
    "species": "human",
    "tissue": "breast cancer",
    "n_sections": 4,
    "section_thickness_um": 10.0,
    "spot_spacing_um": 200.0,
    "expression_type": "raw_counts",
    "coordinate_units": "micrometers",
    "source": "spatialresearch.org",
    "doi": "10.1126/science.aaf2403",
    "paper": "Stahl et al., 2016, Science",
    "benchmark_paper": "stVGP (Wang et al., 2026, Advanced Science)",
    "key_genes": ["FN1", "COL3A1", "LUM", "COL1A1", "SPARC", "PRSS23"],
}


def check_raw_data():
    """Verify that required raw files exist."""
    missing = []
    for layer in LAYERS:
        fpath = RAW_DIR / layer["file"]
        if not fpath.exists():
            missing.append(layer["file"])

    if missing:
        print(f"ERROR: Missing raw files in {RAW_DIR}:")
        for f in missing:
            print(f"  - {f}")
        print("Run download_st_breast_cancer_stvgp.py first.")
        sys.exit(1)
    print("Raw data check passed.")


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
    assert "spatial" in adata.obsm and adata.obsm["spatial"].shape == (adata.n_obs, 3), \
        "obsm['spatial'] must be (n, 3)"
    assert "section" in adata.obs.columns, "obs must have 'section' column"
    assert "cell_type" in adata.obs.columns, "obs must have 'cell_type' column"

    sections = adata.obs["section"].unique()
    print(f"  Verified: {adata.n_obs} spots x {adata.n_vars} genes, {len(sections)} sections")
    for sec in sorted(sections):
        n_sec = (adata.obs["section"] == sec).sum()
        print(f"    {sec}: {n_sec} spots")

    print(
        f"  Spatial range: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
        f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
        f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]"
    )
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")

    # Check key stVGP genes
    key_genes = ["FN1", "COL3A1", "LUM", "COL1A1", "SPARC", "PRSS23"]
    found = [g for g in key_genes if g in adata.var_names]
    missing = [g for g in key_genes if g not in adata.var_names]
    print(f"  stVGP key genes present: {len(found)}/{len(key_genes)} — {found}")
    if missing:
        print(f"  stVGP key genes MISSING: {missing}")

    # Check expression values are integers (raw counts)
    sample_vals = adata.X.data[:min(10000, len(adata.X.data))]
    is_integer = np.allclose(sample_vals, np.round(sample_vals))
    print(f"  Integer counts: {is_integer}")


def process():
    """Main processing pipeline."""
    print(f"Processing {DATASET_NAME}...")

    check_raw_data()

    adatas = []
    for layer_info in LAYERS:
        section = layer_info["section"]
        fpath = RAW_DIR / layer_info["file"]
        print(f"\n  Loading {section} from {layer_info['file']}...")

        # Read TSV: first column is spot name ("XxY"), rest are gene counts
        df = pd.read_csv(fpath, sep="\t", index_col=0)
        print(f"    Raw shape: {df.shape[0]} spots x {df.shape[1]} genes")

        # Parse spatial coordinates from spot names (format: "XxY", float values)
        spot_names = pd.Series(df.index.astype(str))
        parts = spot_names.str.split("x")
        x_array = parts.str[0].astype(float).values
        y_array = parts.str[1].astype(float).values

        # Convert array coordinates to micrometers
        x_um = x_array * ST_SPOT_SPACING_UM
        y_um = y_array * ST_SPOT_SPACING_UM
        coords_2d = np.column_stack([x_um, y_um])

        # Build 3D coordinates
        z = SECTION_Z_UM[section]
        spatial_3d = build_spatial_3d(coords_2d, z=z)

        # Create AnnData
        X = df.values.astype(np.float32)
        adata = ad.AnnData(
            X=ensure_sparse_csr(X),
            obs=pd.DataFrame(index=[f"{section}_{s}" for s in df.index.astype(str)]),
            var=pd.DataFrame(index=df.columns),
        )
        adata.obsm["spatial"] = spatial_3d
        adata.obs["section"] = section
        adata.obs["cell_type"] = "unknown"  # No cell type annotations in this dataset

        print(f"    {section}: {adata.n_obs} spots, {adata.n_vars} genes, z={z:.0f} µm")
        print(f"    x range: [{x_um.min():.0f}, {x_um.max():.0f}] µm")
        print(f"    y range: [{y_um.min():.0f}, {y_um.max():.0f}] µm")

        adatas.append(adata)

    # Concatenate all sections
    print(f"\n  Concatenating {len(adatas)} sections...")
    adata_combined = ad.concat(adatas, join="outer", fill_value=0)
    adata_combined.X = ensure_sparse_csr(adata_combined.X)

    # Reassemble spatial coordinates (concat may lose obsm)
    spatial_all = np.vstack([a.obsm["spatial"] for a in adatas])
    adata_combined.obsm["spatial"] = spatial_all

    # Make var names unique
    adata_combined.var_names_make_unique()

    # Check expression type
    sample_vals = adata_combined.X.data[:min(10000, len(adata_combined.X.data))]
    is_integer = np.allclose(sample_vals, np.round(sample_vals))
    expr_type = "raw_counts" if is_integer else "normalized"

    # Metadata
    adata_combined.uns["spatial_metadata"] = METADATA.copy()
    adata_combined.uns["expression_type"] = expr_type
    adata_combined.uns["dataset_name"] = DATASET_NAME

    # Verify
    verify(adata_combined)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "data.h5ad"
    print(f"\n  Saving to {out_path}...")
    adata_combined.write_h5ad(out_path)
    print(f"  Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")

    return adata_combined


def main():
    process()


if __name__ == "__main__":
    main()
