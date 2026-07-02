#!/usr/bin/env python
"""Process CosMx SMI NSCLC 3D data to standardized h5ad.

Raw data: data/raw/cosmx_nsclc_3d/
  - preprocessed_objects.zip: cosmx.h5ad (~340K cells × 960 genes) with
    expression, cell type annotations, and STIM-normalized coords
  - cosmx_flat_files.zip: per-section metadata with physical pixel coords
    (CenterX_global_px, CenterY_global_px)

The preprocessed h5ad has STIM-registered coordinates in arbitrary [0, 1000]
units. We replace these with physical µm coordinates from the flat files:
  - x,y: global pixel coords × 0.18 µm/pixel (CosMx SMI standard)
  - z: section_order × 30 µm (every 6th 5-µm cryosection)

Output: data/processed/cosmx_nsclc_3d.h5ad
"""
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "cosmx_nsclc_3d"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_PATH = PROJECT_ROOT / "data" / "processed" / f"{DATASET_NAME}.h5ad"

PREPROCESSED_ZIP = "preprocessed_objects.zip"
H5AD_INNER = "preprocessed_objects/cosmx.h5ad"
FLAT_FILES_ZIP = "cosmx_flat_files.zip"

# CosMx SMI pixel size: 0.18 µm × 0.18 µm (Nanostring standard, 20x objective)
PIXEL_SIZE_UM = 0.18

# Section spacing: sections 4, 10, 16, 22, 28, 34 (every 6th 5-µm cryosection)
# = 30 µm between consecutive imaged sections (Pentimalli et al. 2025, Cell Systems)
SECTION_SPACING_UM = 30.0

# Ordered sections (ascending physical z)
SECTIONS_ORDERED = [4, 10, 16, 22, 28, 34]

# Run IDs per section (from flat file naming)
SECTION_RUN = {4: "Run1129", 10: "Run1129", 16: "Run1129",
               22: "Run1137", 28: "Run1137", 34: "Run1137"}


def check_raw_data():
    """Verify that required zip archives are present."""
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data directory not found: {RAW_DIR}")
        print(f"  Run: python src/data/download/download_{DATASET_NAME}.py")
        sys.exit(1)

    for zname in [PREPROCESSED_ZIP, FLAT_FILES_ZIP]:
        zpath = RAW_DIR / zname
        if not zpath.exists():
            print(f"ERROR: Required archive not found: {zpath}")
            sys.exit(1)
        print(f"  Found: {zname} ({zpath.stat().st_size / 1e6:.0f} MB)")


def ensure_sparse_csr(X):
    """Convert X to CSR sparse matrix if not already."""
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def build_spatial_3d(coords_2d, z=0.0):
    """Append a z column to 2-D coordinate array."""
    n = coords_2d.shape[0]
    z_col = np.full((n, 1), z, dtype=np.float64)
    return np.hstack([np.array(coords_2d, dtype=np.float64), z_col])


def verify(adata):
    """Assert standardized format requirements."""
    assert sp.issparse(adata.X) and adata.X.format == "csr", \
        f"X must be CSR sparse, got: {type(adata.X)}"
    assert "spatial" in adata.obsm, "obsm['spatial'] missing"
    assert adata.obsm["spatial"].shape == (adata.n_obs, 3), \
        f"obsm['spatial'] shape {adata.obsm['spatial'].shape} != ({adata.n_obs}, 3)"
    assert "section" in adata.obs.columns, "obs['section'] missing"
    assert "cell_type" in adata.obs.columns, "obs['cell_type'] missing"
    print(f"  Verified: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    sp_arr = adata.obsm["spatial"]
    print(
        f"  Spatial: x=[{sp_arr[:,0].min():.1f}, {sp_arr[:,0].max():.1f}], "
        f"y=[{sp_arr[:,1].min():.1f}, {sp_arr[:,1].max():.1f}], "
        f"z=[{sp_arr[:,2].min():.1f}, {sp_arr[:,2].max():.1f}]"
    )
    density = adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])
    print(f"  Sparsity: {1 - density:.1%}")
    print(f"  Sections: {adata.obs['section'].nunique()} unique")
    print(f"  Cell types: {adata.obs['cell_type'].nunique()} unique")


def _load_flat_file_coords(flat_zip_path):
    """Load per-cell physical coordinates from the flat files ZIP.

    Returns a dict mapping cell key (section_num, fov, cell_ID) -> (x_px, y_px).
    """
    coords = {}
    with zipfile.ZipFile(flat_zip_path, "r") as zf:
        for sec_num in SECTIONS_ORDERED:
            run = SECTION_RUN[sec_num]
            meta_member = (
                f"cosmx_flat_files/Section_{sec_num:02d}/"
                f"{run}_Section_{sec_num:02d}_metadata_file.csv"
            )
            print(f"  Loading flat file coords: Section_{sec_num:02d}...")
            with zf.open(meta_member) as f:
                df = pd.read_csv(f, usecols=["fov", "cell_ID",
                                              "CenterX_global_px",
                                              "CenterY_global_px"])
            for _, row in df.iterrows():
                key = (sec_num, int(row["fov"]), int(row["cell_ID"]))
                coords[key] = (row["CenterX_global_px"],
                               row["CenterY_global_px"])
            print(f"    {len(df)} cells loaded")
    return coords


def _parse_obs_name(name):
    """Parse obs_name like 'section_4_fov_1_ID_5' -> (section_num, fov, cell_ID)."""
    m = re.match(r"section_(\d+)_fov_(\d+)_ID_(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def process():
    """Extract, standardize, verify and save the CosMx NSCLC 3D dataset."""
    print(f"Processing {DATASET_NAME} ...")
    check_raw_data()

    # --- 1. Load preprocessed h5ad (expression + cell types) ----------------
    zip_path = RAW_DIR / PREPROCESSED_ZIP
    print(f"  Opening archive: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        h5ad_members = [n for n in names if n.endswith(".h5ad")]
        if not h5ad_members:
            print("ERROR: No .h5ad file found inside the zip archive.")
            sys.exit(1)
        target = H5AD_INNER if H5AD_INNER in h5ad_members else h5ad_members[0]
        print(f"  Extracting: {target}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            zf.extract(target, path=tmp_dir)
            h5ad_path = Path(tmp_dir) / target
            print(f"  Reading h5ad ({h5ad_path.stat().st_size / 1e6:.1f} MB) ...")
            adata = ad.read_h5ad(h5ad_path)

    print(f"  Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # --- 2. Rename 'celltypes' -> 'cell_type' ------------------------------
    if "celltypes" in adata.obs.columns:
        adata.obs.rename(columns={"celltypes": "cell_type"}, inplace=True)
    elif "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "unknown"

    # --- 3. Ensure 'section' column ----------------------------------------
    if "section" not in adata.obs.columns:
        adata.obs["section"] = "section_0"
    adata.obs["section"] = adata.obs["section"].astype(str)

    # --- 4. Load physical coordinates from flat files ----------------------
    flat_zip_path = RAW_DIR / FLAT_FILES_ZIP
    print("  Loading physical coordinates from flat files...")
    flat_coords = _load_flat_file_coords(flat_zip_path)
    print(f"  Total flat file coordinates: {len(flat_coords):,}")

    # --- 5. Build physical spatial array -----------------------------------
    # Map section number -> z in µm
    sec_to_z = {sec: float(i) * SECTION_SPACING_UM
                for i, sec in enumerate(SECTIONS_ORDERED)}

    spatial = np.zeros((adata.n_obs, 3), dtype=np.float64)
    n_matched = 0
    n_unmatched = 0

    # Per-section: center x,y (remove arbitrary stage offsets), convert to µm
    # First pass: collect coordinates per section
    section_cells = {}  # sec_num -> list of (obs_idx, x_px, y_px)
    for idx, obs_name in enumerate(adata.obs_names):
        parsed = _parse_obs_name(obs_name)
        if parsed is None:
            n_unmatched += 1
            continue
        sec_num, fov, cell_id = parsed
        key = (sec_num, fov, cell_id)
        if key in flat_coords:
            x_px, y_px = flat_coords[key]
            if sec_num not in section_cells:
                section_cells[sec_num] = []
            section_cells[sec_num].append((idx, x_px, y_px))
            n_matched += 1
        else:
            n_unmatched += 1

    print(f"  Matched: {n_matched:,} / {adata.n_obs:,} cells "
          f"({n_unmatched} unmatched)")

    # Second pass: per-section centering and µm conversion
    for sec_num, cells in section_cells.items():
        indices = [c[0] for c in cells]
        xs = np.array([c[1] for c in cells])
        ys = np.array([c[2] for c in cells])

        # Center each section (remove stage offset)
        xs_centered = xs - xs.min()
        ys_centered = ys - ys.min()

        # Convert pixels to µm
        spatial[indices, 0] = xs_centered * PIXEL_SIZE_UM
        spatial[indices, 1] = ys_centered * PIXEL_SIZE_UM
        spatial[indices, 2] = sec_to_z[sec_num]

        extent_x = (xs.max() - xs.min()) * PIXEL_SIZE_UM
        extent_y = (ys.max() - ys.min()) * PIXEL_SIZE_UM
        print(f"  Section {sec_num:02d}: {len(cells):,} cells, "
              f"extent {extent_x:.0f} x {extent_y:.0f} µm, "
              f"z={sec_to_z[sec_num]:.0f} µm")

    adata.obsm["spatial"] = spatial

    # Drop cells without coordinates (if any)
    if n_unmatched > 0:
        has_coords = np.any(spatial != 0, axis=1)
        if not has_coords.all():
            n_drop = (~has_coords).sum()
            print(f"  Dropping {n_drop} cells without coordinates")
            adata = adata[has_coords].copy()
            adata.X = ensure_sparse_csr(adata.X)

    # --- 6. Ensure X is CSR sparse -----------------------------------------
    adata.X = ensure_sparse_csr(adata.X)

    # --- 7. Attach standardized metadata -----------------------------------
    adata.uns["technology"] = "CosMx SMI"
    adata.uns["species"] = "human"
    adata.uns["tissue"] = "lung cancer (NSCLC)"
    adata.uns["n_sections"] = int(adata.obs["section"].nunique())
    adata.uns["expression_type"] = "raw_counts"
    adata.uns["coordinate_units"] = "micrometers"
    adata.uns["source"] = "Nanostring / MDC Berlin lung-3d-browser"
    adata.uns["dataset_name"] = DATASET_NAME

    # --- 8. Verify and save ------------------------------------------------
    verify(adata)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {OUT_PATH} ...")
    adata.write_h5ad(OUT_PATH)
    print(f"Done: {OUT_PATH}")


def main():
    process()


if __name__ == "__main__":
    main()
