#!/usr/bin/env python
"""Process Visium mouse brain (cell2location) to standardized h5ad."""
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
DATASET_NAME = "visium_mouse_brain_cell2location"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME

METADATA = {
    "technology": "Visium",
    "species": "mouse",
    "tissue": "brain (coronal)",
    "expression_type": "raw_counts",
    "section_spacing_um": 210.0,  # alternating 10 µm Visium + 200 µm snRNA-seq thick sections
    "coordinate_units": "micrometers",
    "source": "ArrayExpress E-MTAB-11114",
}

# Visium spot diameter in µm (for pixel-to-µm conversion)
VISIUM_SPOT_DIAMETER_UM = 55.0

# Section spacing: alternating 10 µm (Visium) + 200 µm (snRNA-seq) = 210 µm center-to-center
SECTION_SPACING_UM = 210.0

# Slide → mouse mapping (from Visium_mouse.csv in ZIP)
# Slide C05717-020 (positions C1,D1,E1) = Mouse 1, 3 serial coronal sections
# Slide C05717-021 (positions B1,C1) = Mouse 2, 2 serial coronal sections
SLIDE_TO_MOUSE = {
    "C05717-020": "mouse_1",
    "C05717-021": "mouse_2",
}

# Section → slide mapping (from Visium_mouse.csv)
SECTION_TO_SLIDE = {
    "ST8059048": "C05717-020",
    "ST8059049": "C05717-020",
    "ST8059050": "C05717-020",
    "ST8059051": "C05717-021",
    "ST8059052": "C05717-021",
}

H5AD_FILE = "all_cells_20200625.h5ad"
ZIP_FILE = "mouse_brain_visium_wo_cloupe_data.zip"


def check_raw_data():
    """Verify that required raw files exist."""
    missing = []
    for fname in [H5AD_FILE, ZIP_FILE]:
        if not (RAW_DIR / fname).exists():
            missing.append(fname)
    if missing:
        print(f"ERROR: Missing raw files in {RAW_DIR}:")
        for f in missing:
            print(f"  - {f}")
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
    return np.hstack([coords_2d, z_col])


def _get_visium_um_per_pixel(adata):
    """Extract µm/pixel from Visium scalefactors stored by scanpy.read_visium()."""
    if "spatial" in adata.uns:
        for lib_id, lib_data in adata.uns["spatial"].items():
            if isinstance(lib_data, dict) and "scalefactors" in lib_data:
                spot_diam_px = lib_data["scalefactors"].get("spot_diameter_fullres")
                if spot_diam_px:
                    return VISIUM_SPOT_DIAMETER_UM / spot_diam_px
    return None


def _extract_spatial_coords(adata):
    """Extract 2D spatial coordinates from an AnnData object."""
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.shape[1] >= 2:
            return coords[:, :2]

    for x_col, y_col in [
        ("x", "y"),
        ("X", "Y"),
        ("x_coord", "y_coord"),
        ("array_row", "array_col"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ]:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            return np.column_stack(
                [adata.obs[x_col].values.astype(float),
                 adata.obs[y_col].values.astype(float)]
            )

    raise ValueError(
        f"Could not find spatial coordinates. "
        f"obsm keys: {list(adata.obsm.keys())}, "
        f"obs columns: {list(adata.obs.columns)}"
    )


def _has_spatial_coords(adata):
    """Check whether an AnnData object has spatial coordinates."""
    if "spatial" in adata.obsm:
        return True
    for x_col, y_col in [
        ("x", "y"), ("X", "Y"), ("x_coord", "y_coord"),
        ("array_row", "array_col"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ]:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            return True
    return False


def _find_section_column(adata):
    """Find a column in obs that indicates section identity."""
    for col in ["sample", "section", "slide", "batch", "library_id", "sample_id"]:
        if col in adata.obs.columns:
            return col
    return None


def _load_visium_from_zip(zip_path, extract_dir):
    """Extract the ZIP and load Visium h5ad/h5 files from it."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    extract_path = Path(extract_dir)

    # Look for h5ad files
    h5ad_files = sorted(extract_path.rglob("*.h5ad"))
    if h5ad_files:
        print(f"Found {len(h5ad_files)} h5ad files in ZIP.")
        adatas = []
        for i, f in enumerate(h5ad_files):
            print(f"  Loading: {f.name}")
            a = ad.read_h5ad(f)
            adatas.append((f.stem, a))
        return adatas

    # Look for filtered_feature_bc_matrix.h5 (spaceranger output)
    h5_files = sorted(extract_path.rglob("filtered_feature_bc_matrix.h5"))
    if h5_files:
        print(f"Found {len(h5_files)} spaceranger h5 files in ZIP.")
        adatas = []
        for i, f in enumerate(h5_files):
            section_name = f.parent.name
            if section_name == "outs":
                section_name = f.parents[1].name
            print(f"  Loading: {section_name}")
            import scanpy as sc
            # Check for spatial directory alongside the h5
            spatial_dir = f.parent / "spatial"
            if spatial_dir.exists():
                a = sc.read_visium(f.parent)
            else:
                a = sc.read_10x_h5(f)
            adatas.append((section_name, a))
        return adatas

    # Look for any h5 files
    h5_files = sorted(extract_path.rglob("*.h5"))
    if h5_files:
        print(f"Found {len(h5_files)} h5 files in ZIP.")
        adatas = []
        for i, f in enumerate(h5_files):
            print(f"  Loading: {f.name}")
            import scanpy as sc
            a = sc.read_10x_h5(f)
            adatas.append((f.stem, a))
        return adatas

    raise FileNotFoundError(
        f"No h5ad or h5 files found in extracted ZIP. "
        f"Contents: {[str(p.relative_to(extract_path)) for p in extract_path.rglob('*') if p.is_file()][:20]}"
    )


def verify(adata):
    """Verify the processed AnnData meets standards."""
    assert sp.issparse(adata.X) and adata.X.format == "csr"
    assert "spatial" in adata.obsm and adata.obsm["spatial"].shape == (adata.n_obs, 3)
    assert "section" in adata.obs.columns and "cell_type" in adata.obs.columns
    print(f"  Verified: {adata.n_obs} cells × {adata.n_vars} genes")
    print(
        f"  Spatial range: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
        f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
        f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]"
    )
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")


def process():
    """Load Visium mouse brain data from the cell2location dataset."""
    h5ad_path = RAW_DIR / H5AD_FILE
    zip_path = RAW_DIR / ZIP_FILE

    # First try: check if the h5ad has spatial data
    print(f"Checking {H5AD_FILE} for spatial coordinates...")
    adata_h5ad = ad.read_h5ad(h5ad_path, backed="r")
    if _has_spatial_coords(adata_h5ad):
        print("  h5ad file has spatial coordinates, loading fully...")
        adata_h5ad.file.close()
        adata_h5ad = ad.read_h5ad(h5ad_path)
        section_col = _find_section_column(adata_h5ad)
        if section_col:
            sections = adata_h5ad.obs[section_col].unique()
            print(f"  Found {len(sections)} sections in column '{section_col}'")
            adata_h5ad.obs["section"] = adata_h5ad.obs[section_col].astype(str)
        else:
            adata_h5ad.obs["section"] = "section_0"

        coords_2d = _extract_spatial_coords(adata_h5ad)

        # Convert pixels → µm using Visium scalefactors
        um_per_px = _get_visium_um_per_pixel(adata_h5ad)
        if um_per_px:
            print(f"  Converting x,y from pixels to µm (scale={um_per_px:.4f} µm/px)")
            coords_2d = coords_2d * um_per_px

        # Assign z per section (temporary — will be reassigned per-mouse in main())
        section_labels = adata_h5ad.obs["section"].values
        unique_sections = sorted(adata_h5ad.obs["section"].unique())
        section_to_z = {s: float(i) * SECTION_SPACING_UM for i, s in enumerate(unique_sections)}
        z_values = np.array([section_to_z[s] for s in section_labels])
        spatial_3d = np.column_stack([coords_2d, z_values])
        adata_h5ad.obsm["spatial"] = spatial_3d

        # Ensure cell_type
        if "cell_type" not in adata_h5ad.obs.columns:
            for alt in ["Cell_Type", "celltype", "cell_class", "cluster", "annotation"]:
                if alt in adata_h5ad.obs.columns:
                    adata_h5ad.obs["cell_type"] = adata_h5ad.obs[alt].astype(str)
                    break
            else:
                adata_h5ad.obs["cell_type"] = "unknown"
                print("  WARNING: No cell type annotation found in h5ad.")

        return adata_h5ad

    # The h5ad is the snRNA-seq reference; spatial data is in the ZIP
    print("  h5ad is snRNA-seq reference (no spatial coords). Extracting ZIP...")
    adata_h5ad.file.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        section_data = _load_visium_from_zip(zip_path, tmpdir)

        adatas = []
        for i, (name, adata_sec) in enumerate(section_data):
            print(f"Processing section: {name}")

            # Extract spatial coordinates and convert pixels → µm
            if _has_spatial_coords(adata_sec):
                coords_2d = _extract_spatial_coords(adata_sec)
                um_per_px = _get_visium_um_per_pixel(adata_sec)
                if um_per_px:
                    coords_2d = coords_2d * um_per_px
                    print(f"  Converted x,y to µm (scale={um_per_px:.4f} µm/px)")
            else:
                print(f"  WARNING: No spatial coords for {name}, using zeros.")
                coords_2d = np.zeros((adata_sec.n_obs, 2))

            adata_sec.obsm["spatial"] = build_spatial_3d(coords_2d, z=float(i) * SECTION_SPACING_UM)
            adata_sec.obs["section"] = name

            # Ensure cell_type
            if "cell_type" not in adata_sec.obs.columns:
                for alt in ["Cell_Type", "celltype", "cell_class", "cluster", "annotation"]:
                    if alt in adata_sec.obs.columns:
                        adata_sec.obs["cell_type"] = adata_sec.obs[alt].astype(str)
                        break
                else:
                    adata_sec.obs["cell_type"] = "unknown"

            adata_sec.var_names_make_unique()
            adatas.append(adata_sec)
            print(f"  {adata_sec.n_obs} spots, {adata_sec.n_vars} genes")

        print("Concatenating sections...")
        adata = ad.concat(adatas, join="outer", merge="first")
        adata.obs_names_make_unique()
        print(f"Combined: {adata.n_obs} spots × {adata.n_vars} genes")
        return adata


def main():
    check_raw_data()
    adata = process()
    adata.X = ensure_sparse_csr(adata.X)
    adata.var_names_make_unique()
    adata.uns["spatial_metadata"] = METADATA
    adata.uns["expression_type"] = METADATA["expression_type"]
    adata.uns["dataset_name"] = DATASET_NAME

    # Assign cell types from cell2location deconvolution results if available
    deconv_path = RAW_DIR / "cell2location_deconv.csv"
    if deconv_path.exists():
        deconv = pd.read_csv(deconv_path, index_col=0)
        # Clean column names: 'q05_spot_factorsAstro_CTX' -> 'Astro_CTX'
        deconv.columns = [c.replace("q05_spot_factors", "") for c in deconv.columns]
        # Match: our barcodes may have scanpy concat suffixes (-1-N)
        n_matched = 0
        for obs_idx in adata.obs.index:
            section = adata.obs.loc[obs_idx, "section"]
            bc = re.sub(r"(-\d+)-\d+$", r"\1", obs_idx)  # strip concat suffix
            deconv_id = f"{section}_{bc}"
            if deconv_id in deconv.index:
                adata.obs.loc[obs_idx, "cell_type"] = deconv.loc[deconv_id].idxmax()
                n_matched += 1
        print(f"  Cell2location deconv: {n_matched}/{adata.n_obs} spots matched")

    # Assign mouse ID from section→slide→mouse mapping
    if "section" in adata.obs.columns:
        adata.obs["mouse_id"] = adata.obs["section"].map(
            lambda s: SLIDE_TO_MOUSE.get(SECTION_TO_SLIDE.get(s, ""), "unknown")
        )
    else:
        adata.obs["mouse_id"] = "unknown"

    # Split by mouse and save per-mouse files with physical z
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old single-file output if it exists
    old_single = OUT_DIR.parent / f"{DATASET_NAME}.h5ad"
    if old_single.exists():
        old_single.unlink()
        print(f"Removed old single-file output: {old_single.name}")

    for mouse_id in sorted(adata.obs["mouse_id"].unique()):
        if mouse_id == "unknown":
            continue

        mask = adata.obs["mouse_id"] == mouse_id
        n_sections = len(adata.obs.loc[mask, "section"].unique())
        if n_sections < 3:
            print(f"\n  SKIP {mouse_id}: only {n_sections} sections (need ≥3 for 3D)")
            continue

        adata_mouse = adata[mask].copy()
        adata_mouse.X = ensure_sparse_csr(adata_mouse.X)

        # Re-assign z with physical spacing within this mouse
        mouse_sections = sorted(adata_mouse.obs["section"].unique())
        sec_to_z = {s: float(i) * SECTION_SPACING_UM for i, s in enumerate(mouse_sections)}
        z_new = adata_mouse.obs["section"].map(sec_to_z).values.astype(np.float64)
        adata_mouse.obsm["spatial"][:, 2] = z_new

        adata_mouse.uns["n_sections"] = len(mouse_sections)
        adata_mouse.uns["spatial_metadata"] = METADATA.copy()

        print(f"\n  {mouse_id}: {adata_mouse.n_obs} spots, {len(mouse_sections)} sections")
        verify(adata_mouse)

        out_path = OUT_DIR / f"{mouse_id}.h5ad"
        adata_mouse.write_h5ad(out_path)
        print(f"  Saved: {out_path}")

    print(f"\nDone. Output files in {OUT_DIR}")


if __name__ == "__main__":
    main()
