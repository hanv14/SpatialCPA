#!/usr/bin/env python
"""Process DLPFC Visium data (stVGP paper) to standardized h5ad.

Source: spatialLIBD (Maynard et al., 2021)
  - 10x h5 files from AWS S3
  - Spatial coordinates from GitHub (tissue_positions_list)
  - Layer annotations from barcode_level_layer_map.tsv
Output: data/processed/visium_dlpfc_stvgp/data.h5ad
Description: 4 Visium sections (151673-151676) from human DLPFC, donor Br8100.
  Two pairs of serial sections: (151673, 151674) and (151675, 151676), ~300 µm apart.
  Within each pair, sections are 10 µm adjacent serial sections.
  Layer annotations: L1-L6 + WM (white matter).
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "visium_dlpfc_stvgp"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME

SAMPLES = ["151673", "151674", "151675", "151676"]

# Section spacing: 151673-151674 are adjacent (10 µm), 151675-151676 are adjacent (10 µm),
# and the two pairs are ~300 µm apart. All from donor Br8100.
# Physical z-positions (µm):
SECTION_Z_UM = {
    "151673": 0.0,
    "151674": 10.0,
    "151675": 310.0,   # 300 µm gap between pairs
    "151676": 320.0,
}

VISIUM_SPOT_DIAMETER_UM = 55.0

METADATA = {
    "technology": "Visium",
    "species": "human",
    "tissue": "dorsolateral prefrontal cortex (DLPFC)",
    "donor": "Br8100",
    "n_sections": 4,
    "section_thickness_um": 10.0,
    "expression_type": "raw_counts",
    "coordinate_units": "micrometers",
    "source": "spatialLIBD (Maynard et al., 2021)",
    "paper": "stVGP",
}


def check_raw_data():
    """Verify that required raw files exist."""
    missing = []
    for sample in SAMPLES:
        for fname in [
            f"{sample}_filtered_feature_bc_matrix.h5",
            f"{sample}_tissue_positions_list.csv",
            f"{sample}_scalefactors_json.json",
        ]:
            if not (RAW_DIR / fname).exists():
                missing.append(fname)

    layer_file = RAW_DIR / "barcode_level_layer_map.tsv"
    if not layer_file.exists():
        missing.append("barcode_level_layer_map.tsv")

    if missing:
        print(f"ERROR: Missing raw files in {RAW_DIR}:")
        for f in missing:
            print(f"  - {f}")
        print("Run download_visium_dlpfc_stvgp.py first.")
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
    assert "spatial" in adata.obsm and adata.obsm["spatial"].shape == (adata.n_obs, 3), "obsm['spatial'] must be (n, 3)"
    assert "section" in adata.obs.columns, "obs must have 'section' column"
    assert "cell_type" in adata.obs.columns, "obs must have 'cell_type' column"

    sections = adata.obs["section"].unique()
    print(f"  Verified: {adata.n_obs} spots x {adata.n_vars} genes, {len(sections)} sections")
    print(
        f"  Spatial range: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
        f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
        f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]"
    )
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")

    # Layer annotation coverage
    n_annotated = (adata.obs["cell_type"] != "unknown").sum()
    print(f"  Layer annotations: {n_annotated}/{adata.n_obs} spots ({n_annotated/adata.n_obs:.1%})")
    if n_annotated > 0:
        print(f"  Layer distribution:\n{adata.obs['cell_type'].value_counts().to_string()}")


def process():
    """Main processing pipeline."""
    print(f"Processing {DATASET_NAME}...")

    check_raw_data()

    # Load layer annotations
    print("  Loading layer annotations...")
    layer_map = pd.read_csv(
        RAW_DIR / "barcode_level_layer_map.tsv",
        sep="\t",
        header=None,
        names=["barcode", "sample", "layer"],
    )
    # Filter to our samples
    layer_map = layer_map[layer_map["sample"].isin([int(s) for s in SAMPLES])]
    # Create lookup: (sample, barcode) -> layer
    layer_lookup = {}
    for _, row in layer_map.iterrows():
        layer_lookup[(str(int(row["sample"])), row["barcode"])] = row["layer"]
    print(f"    {len(layer_lookup)} barcode-layer mappings for our 4 samples")

    adatas = []
    for sample in SAMPLES:
        print(f"\n  Processing sample {sample}...")

        # Load scalefactors
        with open(RAW_DIR / f"{sample}_scalefactors_json.json") as f:
            scalefactors = json.load(f)
        spot_diam_px = scalefactors["spot_diameter_fullres"]
        um_per_px = VISIUM_SPOT_DIAMETER_UM / spot_diam_px
        print(f"    Scalefactors: spot_diameter={spot_diam_px:.1f}px, scale={um_per_px:.4f} µm/px")

        # Load expression matrix
        h5_path = RAW_DIR / f"{sample}_filtered_feature_bc_matrix.h5"
        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()
        print(f"    Loaded: {adata.n_obs} spots x {adata.n_vars} genes")

        # Load tissue positions
        # Format: barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
        pos_df = pd.read_csv(
            RAW_DIR / f"{sample}_tissue_positions_list.csv",
            header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
        )
        pos_df = pos_df.set_index("barcode")

        # Filter to in-tissue spots that are in the expression matrix
        common_barcodes = adata.obs.index.intersection(pos_df.index)
        pos_df = pos_df.loc[common_barcodes]
        adata = adata[common_barcodes].copy()

        # Filter to in-tissue only
        in_tissue = pos_df["in_tissue"] == 1
        adata = adata[in_tissue.values].copy()
        pos_df = pos_df[in_tissue]
        print(f"    In-tissue spots: {adata.n_obs}")

        # Convert pixel coordinates to µm
        coords_2d = np.column_stack([
            pos_df["pxl_col"].values.astype(np.float64) * um_per_px,
            pos_df["pxl_row"].values.astype(np.float64) * um_per_px,
        ])

        # Build 3D coordinates
        z = SECTION_Z_UM[sample]
        spatial_3d = build_spatial_3d(coords_2d, z=z)
        adata.obsm["spatial"] = spatial_3d

        # Section and layer annotations
        adata.obs["section"] = sample
        layers = []
        for bc in adata.obs.index:
            layer = layer_lookup.get((sample, bc), "unknown")
            layers.append(layer)
        adata.obs["cell_type"] = layers

        n_annotated = sum(1 for l in layers if l != "unknown")
        print(f"    Layer annotations: {n_annotated}/{adata.n_obs} ({n_annotated/adata.n_obs:.1%})")
        print(f"    z = {z:.0f} µm")

        # Prefix obs names for global uniqueness
        adata.obs_names = [f"{sample}_{bc}" for bc in adata.obs_names]

        adatas.append(adata)

    # Concatenate
    print(f"\n  Concatenating {len(adatas)} sections...")
    adata_combined = ad.concat(adatas, join="outer", fill_value=0)
    adata_combined.X = ensure_sparse_csr(adata_combined.X)

    # Reassemble spatial coordinates (concat may lose obsm)
    spatial_all = np.vstack([a.obsm["spatial"] for a in adatas])
    adata_combined.obsm["spatial"] = spatial_all

    # Check expression type
    sample_vals = adata_combined.X.data[:min(10000, len(adata_combined.X.data))]
    is_integer = np.allclose(sample_vals, np.round(sample_vals))
    expr_type = "raw_counts" if is_integer else "normalized"

    # Gene names
    adata_combined.var_names_make_unique()

    # Metadata
    adata_combined.uns["spatial_metadata"] = METADATA.copy()
    adata_combined.uns["expression_type"] = expr_type
    adata_combined.uns["dataset_name"] = DATASET_NAME

    # Verify
    verify(adata_combined)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "data.h5ad"
    print(f"  Saving to {out_path}...")
    adata_combined.write_h5ad(out_path)
    print(f"  Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")

    return adata_combined


def main():
    process()


if __name__ == "__main__":
    main()
