#!/usr/bin/env python
"""Process 3D MERFISH thick tissue dataset to standardized h5ad."""
import sys
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "merfish_thick_tissue"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME
METADATA = {
    "technology": "3D MERFISH",
    "species": "mouse",
    "tissue": None,  # set per region
    "n_sections": None,
    "section_spacing_um": None,
    "coordinate_units": "micrometers",
    "expression_type": "raw_counts",
    "source": "Dryad 10.5061/dryad.w0vt4b922",
}

EXTRACTED_DIR = RAW_DIR / "dryad-2024-08-14"
ZIP_FILE = RAW_DIR / "Fang_eLife_2023.zip"


def check_raw_data():
    # Either the extracted directory or the ZIP file must exist
    if EXTRACTED_DIR.exists():
        return
    if ZIP_FILE.exists():
        print(f"  Extracting {ZIP_FILE}...")
        with zipfile.ZipFile(ZIP_FILE, "r") as zf:
            zf.extractall(RAW_DIR)
        if EXTRACTED_DIR.exists():
            return
        # ZIP might extract to a different directory name
        subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
        if subdirs:
            print(f"  Found extracted directories: {[d.name for d in subdirs]}")
            return
    print(f"ERROR: Missing data in {RAW_DIR}:")
    print(f"  Need either {EXTRACTED_DIR.name}/ directory or {ZIP_FILE.name}")
    print(f"Run: python src/data/download/download_{DATASET_NAME}.py")
    sys.exit(1)


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


def load_region(region_dir, region_name):
    """Load transcripts_per_feature.csv and feature_metadata.csv for one region.

    Returns expression DataFrame (cells x genes) and metadata DataFrame with
    spatial coordinates.
    """
    expr_file = region_dir / "transcripts_per_feature.csv"
    meta_file = region_dir / "feature_metadata.csv"

    if not expr_file.exists():
        print(f"  ERROR: Missing {expr_file}")
        sys.exit(1)
    if not meta_file.exists():
        print(f"  ERROR: Missing {meta_file}")
        sys.exit(1)

    print(f"  Loading {region_name} expression: {expr_file}")
    expr = pd.read_csv(expr_file, index_col=0)
    print(f"    Raw shape: {expr.shape}")

    # Remove Blank barcode columns (negative controls)
    blank_cols = [c for c in expr.columns if c.startswith("Blank")]
    if blank_cols:
        expr = expr.drop(columns=blank_cols)
        print(f"    Dropped {len(blank_cols)} Blank columns -> {expr.shape}")

    print(f"  Loading {region_name} metadata: {meta_file}")
    meta = pd.read_csv(meta_file, index_col=0)
    print(f"    Metadata shape: {meta.shape}")

    # Align on common cell IDs
    common = expr.index.intersection(meta.index)
    if len(common) == 0:
        # Try string conversion
        expr.index = expr.index.astype(str)
        meta.index = meta.index.astype(str)
        common = expr.index.intersection(meta.index)
    print(f"    Common cells: {len(common)}")
    expr = expr.loc[common]
    meta = meta.loc[common]

    return expr, meta


def process_region(region_name, region_dir):
    """Process a single region and save as its own h5ad file."""
    print(f"\n  Processing region: {region_name}")
    expr, meta = load_region(region_dir, region_name)

    # Build sparse matrix
    X = ensure_sparse_csr(expr.values.astype(np.float32))

    # Build spatial coordinates from feature_metadata columns: center_x, center_y, center_z
    coords_x = meta["center_x"].values.astype(np.float64)
    coords_y = meta["center_y"].values.astype(np.float64)
    coords_z = meta["center_z"].values.astype(np.float64)
    spatial = np.column_stack([coords_x, coords_y, coords_z])

    # Assign section labels based on unique z-planes
    unique_z = np.unique(coords_z)
    z_to_section = {z: f"{region_name}_z{i}" for i, z in enumerate(sorted(unique_z))}
    section_labels = [z_to_section[z] for z in coords_z]

    # Load cell type annotations from Seurat object if available
    cell_type_map = {}
    seurat_path = region_dir / "seurat_obj.rds"
    if seurat_path.exists():
        try:
            import rpy2.robjects as ro
            # Hypothalamus: cell.type column; Cortex: subclass_label column
            ro.r(f'''
                obj <- readRDS("{seurat_path}")
                meta <- obj@meta.data
                # Try hypothalamus column names first, then cortex
                if ("cell.type" %in% colnames(meta)) {{
                    ct <- meta$cell.type
                }} else if ("subclass_label" %in% colnames(meta)) {{
                    ct <- meta$subclass_label
                }} else {{
                    ct <- rep("unknown", nrow(meta))
                }}
                write.csv(data.frame(cell_type=ct, row.names=rownames(meta)), "/tmp/merfish_thick_ct.csv")
            ''')
            ct_df = pd.read_csv("/tmp/merfish_thick_ct.csv", index_col=0)
            # Only use if cell IDs overlap with our expression data
            overlap = set(ct_df.index) & set(expr.index)
            if len(overlap) > 0:
                cell_type_map = ct_df["cell_type"].to_dict()
                # Filter to only cells that passed authors' QC (in Seurat object)
                n_before = len(expr)
                keep_ids = [cid for cid in expr.index if cid in overlap]
                expr = expr.loc[keep_ids]
                meta = meta.loc[keep_ids]
                coords_x = meta["center_x"].values.astype(np.float64)
                coords_y = meta["center_y"].values.astype(np.float64)
                coords_z = meta["center_z"].values.astype(np.float64)
                spatial = np.column_stack([coords_x, coords_y, coords_z])
                X = ensure_sparse_csr(expr.values.astype(np.float32))
                unique_z = np.unique(coords_z)
                z_to_section = {z: f"{region_name}_z{i}" for i, z in enumerate(sorted(unique_z))}
                section_labels = [z_to_section[z] for z in coords_z]
                print(f"    Seurat annotations: {len(overlap)}/{n_before} cells matched, filtered to {len(expr)} author-QC cells")
            else:
                # Seurat uses different cell segmentation — use Seurat as primary source
                print(f"    Seurat cell IDs don't match CSV (different segmentation)")
                print(f"    Switching to Seurat object as primary data source...")
                ro.r(f'''
                    write.csv(data.frame(
                        ct=ct,
                        gx=meta$global.x,
                        gy=meta$global.y,
                        gz=meta$global.z,
                        row.names=rownames(meta)
                    ), "/tmp/merfish_thick_spatial.csv")
                    # Also export expression matrix
                    counts <- as.matrix(obj@assays[["RNA"]]@counts)
                    write.csv(counts, "/tmp/merfish_thick_counts.csv")
                ''')
                seurat_meta = pd.read_csv("/tmp/merfish_thick_spatial.csv", index_col=0)
                seurat_expr = pd.read_csv("/tmp/merfish_thick_counts.csv", index_col=0).T
                # Remove Blank columns
                blank_cols = [c for c in seurat_expr.columns if c.startswith("Blank")]
                seurat_expr = seurat_expr.drop(columns=blank_cols, errors="ignore")
                # Replace expr, meta, coords, cell_type_map with Seurat data
                expr = seurat_expr
                coords_x = seurat_meta["gx"].values.astype(np.float64)
                coords_y = seurat_meta["gy"].values.astype(np.float64)
                coords_z = seurat_meta["gz"].values.astype(np.float64)
                spatial = np.column_stack([coords_x, coords_y, coords_z])
                X = ensure_sparse_csr(expr.values.astype(np.float32))
                unique_z = np.unique(coords_z)
                z_to_section = {z: f"{region_name}_z{i}" for i, z in enumerate(sorted(unique_z))}
                section_labels = [z_to_section[z] for z in coords_z]
                cell_type_map = seurat_meta["ct"].to_dict()
                # Reset meta to just spatial coords for obs copy
                meta = pd.DataFrame({
                    "center_x": coords_x, "center_y": coords_y, "center_z": coords_z
                }, index=expr.index)
                print(f"    Seurat primary: {len(expr)} cells × {expr.shape[1]} genes, {len(unique_z)} z-planes, 100% annotated")
        except Exception as e:
            print(f"    WARNING: Could not load Seurat annotations: {e}")

    # Build obs
    obs = pd.DataFrame(index=expr.index)
    obs["cell_type"] = [str(cell_type_map.get(cid, "unknown")) for cid in expr.index]
    obs["section"] = section_labels

    # Copy all remaining metadata columns (excluding spatial coords)
    skip_cols = {"center_x", "center_y", "center_z"}
    for col in meta.columns:
        if col not in skip_cols and col not in obs.columns:
            obs[col] = meta[col].values

    obs["cell_type"] = obs["cell_type"].astype(str)
    obs["section"] = obs["section"].astype(str)

    # Build var
    var = pd.DataFrame(index=expr.columns)
    var.index.name = "gene"

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    meta_dict = METADATA.copy()
    meta_dict["tissue"] = f"brain ({region_name})"
    meta_dict["n_sections"] = len(unique_z)
    adata.uns["dataset"] = meta_dict
    adata.uns["expression_type"] = METADATA["expression_type"]
    adata.uns["dataset_name"] = DATASET_NAME

    verify(adata)

    # Save
    out_path = OUT_DIR / f"{region_name}.h5ad"
    print(f"  Saving to {out_path}...")
    adata.write_h5ad(out_path)
    print(f"  Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")

    return adata


def process():
    print(f"Processing {DATASET_NAME}...")
    check_raw_data()

    data_base = EXTRACTED_DIR / "data"
    if not data_base.exists():
        print(f"ERROR: Expected data directory not found: {data_base}")
        sys.exit(1)

    # Define the two regions — each is an independent consecutive z-stack
    regions = {
        "cortex": data_base / "mouse.cortex.242genes.100um",
        "hypothalamus": data_base / "mouse.hypothalamus.156genes.200um",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old single-file output if it exists
    old_single = OUT_DIR.parent / f"{DATASET_NAME}.h5ad"
    if old_single.exists():
        old_single.unlink()
        print(f"  Removed old single-file output: {old_single.name}")

    for region_name, region_dir in regions.items():
        if not region_dir.exists():
            print(f"  WARNING: Region directory not found, skipping: {region_dir}")
            continue
        process_region(region_name, region_dir)

    print(f"\nDone. Output files in {OUT_DIR}")


if __name__ == "__main__":
    process()
