#!/usr/bin/env python
"""Process Visium spinal cord (isoST paper) to standardized h5ad.

Source: GSE234774 expression from spatial_3d MTX + original Visium tissue_positions_list.csv
Output: data/processed/visium_spinal_cord_isost/data.h5ad
Description: Mouse mid-thoracic spinal cord Visium serial sections.
  48 sections total: 16 physical sections x 3 conditions (uninjured, 7days, 2months post-injury).
  4 slides (075, 077, 078, 103) x 4 capture areas (A-D) = 16 section positions.
  x,y from original Visium tissue_positions_list.csv (pixel coords -> µm).
  z = section_number * 10 µm (standard Visium fresh-frozen section thickness).
  NOTE: We do NOT use the spatial_3d coordinates (those are isoST-reconstructed per-cell 3D
  positions, which would be circular for evaluating interpolation methods).
"""
import gzip
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "visium_spinal_cord_isost"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME

METADATA = {
    "technology": "Visium",
    "species": "mouse",
    "tissue": "spinal cord (mid-thoracic)",
    "expression_type": "raw_counts",
    "coordinate_units": "micrometers",
    "source": "GEO GSE234774",
    "paper": "isoST (Li et al. 2025)",
    "n_sections_per_condition": 16,
    "conditions": ["uninjured", "7days", "2months"],
}

# Section spacing: 10 µm (standard Visium fresh-frozen cryosection thickness)
SECTION_SPACING_UM = 10.0


def check_raw_data():
    """Verify that required raw files exist."""
    required = [
        "GSE234774_spatial_3d_barcodes.txt.gz",
        "GSE234774_spatial_3d_features.txt.gz",
        "GSE234774_spatial_3d_filtered_spatial_3d.mtx.gz",
        "GSE234774_spatial_3d_meta.txt.gz",
    ]
    missing = [f for f in required if not (RAW_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing raw files in {RAW_DIR}:")
        for f in missing:
            print(f"  - {f}")
        print("Run download_visium_spinal_cord_isost.py first.")
        sys.exit(1)
    # Check tissue_positions files exist
    tp_files = list(RAW_DIR.glob("GSM*_tissue_positions_list.csv.gz"))
    if not tp_files:
        print("ERROR: No tissue_positions_list.csv.gz files found")
        sys.exit(1)
    print(f"Raw data check passed ({len(tp_files)} tissue_positions files).")


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


def _read_gz_text(path):
    """Read a gzipped text file, return lines."""
    with gzip.open(path, "rt") as f:
        return f.read().splitlines()


def _load_tissue_positions():
    """Load all tissue_positions_list.csv files and build barcode->xy lookup.

    Returns dict: barcode -> (x_um, y_um, slide_spot)
    where x_um, y_um are pixel coords converted to µm.
    """
    import json

    tp_files = sorted(RAW_DIR.glob("GSM*_tissue_positions_list.csv.gz"))
    barcode_coords = {}

    for tp_file in tp_files:
        # Extract slide_spot from filename: GSM7474504_075_A_tissue_positions_list.csv.gz
        fname = tp_file.name
        parts = fname.split("_")
        slide_spot = f"{parts[1]}_{parts[2]}"  # e.g. "075_A"

        # Load scalefactors for this capture area
        sf_file = list(RAW_DIR.glob(f"*_{slide_spot}_scalefactors_json.json.gz"))
        if sf_file:
            with gzip.open(sf_file[0], "rt") as f:
                sf = json.load(f)
            spot_diameter_px = sf["spot_diameter_fullres"]
            um_per_px = 55.0 / spot_diameter_px  # Visium spots are 55 µm
        else:
            um_per_px = 55.0 / 94.4  # fallback from typical Visium

        # Load tissue positions: barcode, in_tissue, row, col, pxl_row, pxl_col
        with gzip.open(tp_file, "rt") as f:
            tp = pd.read_csv(f, header=None,
                             names=["barcode", "in_tissue", "row", "col", "pxl_row", "pxl_col"])

        for _, r in tp[tp["in_tissue"] == 1].iterrows():
            # Prefix barcode with slide_spot to match meta barcodes
            full_bc = f"{slide_spot}-{r['barcode']}"
            x_um = r["pxl_col"] * um_per_px
            y_um = r["pxl_row"] * um_per_px
            barcode_coords[full_bc] = (x_um, y_um, slide_spot)

    print(f"  Loaded {len(barcode_coords)} in-tissue spots from {len(tp_files)} capture areas")
    return barcode_coords


def process():
    """Main processing pipeline."""
    print(f"Processing {DATASET_NAME}...")

    check_raw_data()

    # Load barcodes (from 3D expression matrix — same cells, we just replace coordinates)
    print("  Loading barcodes...")
    barcodes = _read_gz_text(RAW_DIR / "GSE234774_spatial_3d_barcodes.txt.gz")
    print(f"    {len(barcodes)} barcodes")

    # Load features
    print("  Loading features...")
    features_lines = _read_gz_text(RAW_DIR / "GSE234774_spatial_3d_features.txt.gz")
    features = []
    feature_ids = []
    for line in features_lines:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            feature_ids.append(parts[0])
            features.append(parts[1])
        else:
            features.append(parts[0])
            feature_ids.append(parts[0])
    print(f"    {len(features)} features")

    # Load expression matrix
    print("  Loading expression matrix...")
    mtx_path = RAW_DIR / "GSE234774_spatial_3d_filtered_spatial_3d.mtx.gz"
    X = sio.mmread(mtx_path).T.tocsr()
    print(f"    Matrix shape: {X.shape}")
    assert X.shape[0] == len(barcodes), f"Barcode count mismatch: {X.shape[0]} vs {len(barcodes)}"
    assert X.shape[1] == len(features), f"Feature count mismatch: {X.shape[1]} vs {len(features)}"

    # Load 3D metadata (for section assignments — id, group, sections columns)
    print("  Loading metadata...")
    meta = pd.read_csv(RAW_DIR / "GSE234774_spatial_3d_meta.txt.gz", sep="\t", index_col=0)
    meta.index = barcodes  # index is all "SeuratProject", replace with barcodes

    # Load original Visium tissue_positions for xy coordinates
    print("  Loading tissue_positions_list.csv files...")
    barcode_coords = _load_tissue_positions()

    # Match barcodes to tissue_positions xy + assign flat z from section number
    x_arr = np.zeros(len(barcodes), dtype=np.float64)
    y_arr = np.zeros(len(barcodes), dtype=np.float64)
    z_arr = np.zeros(len(barcodes), dtype=np.float64)
    matched = 0

    for i, bc in enumerate(barcodes):
        section_num = int(meta.iloc[i]["sections"])  # 1-16
        z_arr[i] = section_num * SECTION_SPACING_UM

        if bc in barcode_coords:
            x_arr[i], y_arr[i], _ = barcode_coords[bc]
            matched += 1
        else:
            # Fallback: use 3D meta x,y (will be wrong scale but better than 0)
            x_arr[i] = meta.iloc[i]["x"]
            y_arr[i] = meta.iloc[i]["y"]

    print(f"  Matched {matched}/{len(barcodes)} barcodes to tissue_positions "
          f"({100*matched/len(barcodes):.1f}%)")

    spatial_3d = np.column_stack([x_arr, y_arr, z_arr])

    # Check expression type
    sample_vals = X.data[:min(10000, len(X.data))]
    is_integer = np.allclose(sample_vals, np.round(sample_vals))
    expr_type = "raw_counts" if is_integer else "normalized"
    print(f"  Expression type: {expr_type}")

    # Create AnnData
    adata = ad.AnnData(
        X=ensure_sparse_csr(X),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=features),
    )
    if feature_ids != features:
        adata.var["ensembl_id"] = feature_ids

    adata.obsm["spatial"] = spatial_3d

    # Section = unique section ID (slide_spot_condition_replicate)
    adata.obs["section"] = meta["id"].values
    adata.obs["condition"] = meta["group"].values
    adata.obs["slide"] = meta["slide"].values.astype(str)
    adata.obs["spot"] = meta["spot"].values
    adata.obs["section_number"] = meta["sections"].values.astype(int)
    adata.obs["cell_type"] = "unknown"

    adata.var_names_make_unique()

    # Save combined file
    print(f"\n  Saving combined file...")
    adata.uns["spatial_metadata"] = METADATA.copy()
    adata.uns["spatial_metadata"]["n_sections"] = adata.obs["section"].nunique()
    adata.uns["expression_type"] = expr_type
    adata.uns["dataset_name"] = DATASET_NAME

    verify(adata)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = OUT_DIR / "data.h5ad"
    adata.write_h5ad(combined_path)
    print(f"  Combined file size: {combined_path.stat().st_size / 1e6:.1f} MB")

    return adata


def main():
    process()


if __name__ == "__main__":
    main()
