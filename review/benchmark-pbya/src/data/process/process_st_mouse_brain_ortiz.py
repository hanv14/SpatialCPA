#!/usr/bin/env python
"""Process ST mouse brain Ortiz dataset to standardized h5ad.

3 mice (A1, A2, A3) contribute interleaved sections to form one continuous
anterior-posterior atlas. All sections are registered to the Allen Brain Atlas
via WholeBrain, so they share a common stereotaxic coordinate frame. Kept as
one dataset (not split by animal) because per-animal coverage is non-consecutive.
"""
import gzip
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "st_mouse_brain_ortiz"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_PATH = PROJECT_ROOT / "data" / "processed" / f"{DATASET_NAME}.h5ad"
METADATA = {
    "technology": "ST",
    "species": "mouse",
    "tissue": "whole brain",
    "n_sections": 75,
    "section_spacing_um": None,
    "coordinate_units": "micrometers",
    "expression_type": "raw_counts",
    "source": "GEO GSE147747",
}


def parse_animal_ids() -> dict[str, str]:
    """Parse section -> animal_id mapping from GEO SOFT file."""
    soft_path = RAW_DIR / "GSE147747_family.soft.gz"
    if not soft_path.exists():
        print("  WARNING: SOFT file not found, animal_id will be missing")
        return {}
    animal_map = {}
    current_title = None
    current_animal = None
    with gzip.open(soft_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_title = "):
                current_title = line.split("= ", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1 = animal id:"):
                current_animal = line.split(":", 1)[1].strip()
            elif line.startswith("^SAMPLE = ") and current_title and current_animal:
                animal_map[current_title] = current_animal
                current_title = None
                current_animal = None
    if current_title and current_animal:
        animal_map[current_title] = current_animal
    return animal_map


def check_raw_data():
    files = [
        "GSE147747_expr_raw_counts_table.tsv.gz",
        "GSE147747_meta_table.tsv.gz",
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


def process():
    print(f"Processing {DATASET_NAME}...")
    check_raw_data()

    # Load expression matrix (gzipped TSV)
    print("  Loading expression matrix (163 MB gzipped)...")
    expr = pd.read_csv(
        RAW_DIR / "GSE147747_expr_raw_counts_table.tsv.gz",
        sep="\t",
        compression="gzip",
        index_col=0,
    )
    print(f"  Expression shape: {expr.shape}")

    # Load metadata (gzipped TSV)
    print("  Loading metadata...")
    meta = pd.read_csv(
        RAW_DIR / "GSE147747_meta_table.tsv.gz",
        sep="\t",
        compression="gzip",
        index_col=0,
    )
    print(f"  Metadata shape: {meta.shape}")
    print(f"  Metadata columns: {list(meta.columns)}")

    # Determine expression orientation (spots x genes or genes x spots)
    meta_ids = set(meta.index.astype(str))
    row_ids = set(expr.index.astype(str))
    col_ids = set(expr.columns.astype(str))

    row_overlap = len(meta_ids & row_ids)
    col_overlap = len(meta_ids & col_ids)

    if col_overlap > row_overlap:
        print("  Transposing expression matrix (genes x spots -> spots x genes)...")
        expr = expr.T

    print(f"  Expression (spots x genes): {expr.shape}")

    # Align spots
    common_spots = expr.index.intersection(meta.index)
    if len(common_spots) == 0:
        expr.index = expr.index.astype(str)
        meta.index = meta.index.astype(str)
        common_spots = expr.index.intersection(meta.index)
    print(f"  Common spots: {len(common_spots)}")

    expr = expr.loc[common_spots]
    meta = meta.loc[common_spots]

    # Build sparse matrix
    X = ensure_sparse_csr(expr.values.astype(np.float32))

    # Extract stereotactic spatial coordinates (mm → convert to µm)
    # stereo_ML = mediolateral (x), stereo_DV = dorsoventral (y), stereo_AP = anteroposterior (z)
    coords_x = meta["stereo_ML"].values.astype(np.float64) * 1000.0  # mm → µm
    coords_y = meta["stereo_DV"].values.astype(np.float64) * 1000.0
    coords_z = meta["stereo_AP"].values.astype(np.float64) * 1000.0
    print(f"  Using stereotactic coordinates: x=stereo_ML, y=stereo_DV, z=stereo_AP (converted mm → µm)")
    print(f"  Sections: {meta['section_index'].nunique()} unique")

    spatial = np.column_stack([coords_x, coords_y, coords_z])

    # Parse animal IDs from SOFT file
    animal_map = parse_animal_ids()
    if animal_map:
        print(f"  Animal IDs: {len(animal_map)} sections mapped to {len(set(animal_map.values()))} animals")

    # Build obs — keep all metadata columns
    obs = pd.DataFrame(index=common_spots)
    obs["cell_type"] = meta["cluster_name"].values.astype(str)
    obs["cell_type"] = obs["cell_type"].replace("nan", "unknown")
    obs["section"] = meta["section_index"].values.astype(str)
    obs["animal_id"] = obs["section"].map(animal_map).fillna("unknown")

    # Copy all remaining metadata columns (including HE_X, HE_Y, etc.)
    skip = {"cluster_name", "section_index"}
    for c in meta.columns:
        if c not in skip and c not in obs.columns:
            obs[c] = meta[c].values

    # Build var
    var = pd.DataFrame(index=expr.columns)
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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to {OUT_PATH}...")
    adata.write_h5ad(OUT_PATH)
    print(f"  Done. File size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")

    return adata


if __name__ == "__main__":
    process()
