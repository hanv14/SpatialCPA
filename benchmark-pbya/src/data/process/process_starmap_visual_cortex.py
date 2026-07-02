#!/usr/bin/env python
"""Process STARmap visual cortex dataset to standardized h5ad format.

Converts raw STARmap Wang2018 3D visual cortex data into the standardized format
with CSR sparse matrix, 3D spatial coordinates, and metadata.

Raw: data/raw/starmap_visual_cortex/STARmap_Wang2018three_data_3D_data.h5ad
Output: data/processed/starmap_visual_cortex.h5ad
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "starmap_visual_cortex"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "starmap_visual_cortex.h5ad"

RAW_FILE = RAW_DIR / "STARmap_Wang2018three_data_3D_data.h5ad"

# Voxel sizes for Leica TCS SP5 confocal, HC FLUOTAR L 25x/0.95 W objective
# at 1024x1024 scan format (Wang et al. 2018, Science).
# Lateral confirmed via He et al. 2021 ClusterMap (tissue extent 1545 µm matches
# 1800 voxels × 0.859 µm). Axial: 100 z-planes over ~100 µm depth (ibid.).
VOXEL_XY_UM = 0.859   # µm per voxel in x,y
VOXEL_Z_UM = 1.0      # µm per z-step


def ensure_sparse_csr(X):
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def build_spatial_3d(coords_2d, z=0.0):
    n = coords_2d.shape[0]
    z_col = np.full((n, 1), z, dtype=np.float64)
    return np.hstack([np.array(coords_2d, dtype=np.float64), z_col])


def verify(adata):
    assert sp.issparse(adata.X) and adata.X.format == 'csr', "X must be CSR sparse"
    assert 'spatial' in adata.obsm, "Missing obsm['spatial']"
    assert adata.obsm['spatial'].shape == (adata.n_obs, 3), f"spatial shape {adata.obsm['spatial'].shape} != ({adata.n_obs}, 3)"
    assert 'section' in adata.obs.columns, "Missing obs['section']"
    assert 'cell_type' in adata.obs.columns, "Missing obs['cell_type']"
    print(f"  Verified: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Spatial range: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
          f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
          f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]")
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")


def check_raw_data():
    if not RAW_FILE.exists():
        print(f"ERROR: Raw data not found: {RAW_FILE}")
        print("Please download by running:")
        print("  python src/data/download/download_starmap_visual_cortex.py")
        sys.exit(1)


def extract_spatial_coords(adata):
    """Extract spatial coordinates, returning an (n, 3) array.

    Priority: obs columns with z > obsm 3D > obsm 2D + obs z > obsm 2D alone.
    This ensures obs['z'] (89 consecutive planes) is used even when obsm['spatial']
    exists but is only 2D.
    """
    # Check obs columns first — this dataset has obs['z'] with 89 planes
    for xcol, ycol, zcol in [('x', 'y', 'z'), ('X', 'Y', 'Z')]:
        if xcol in adata.obs.columns and ycol in adata.obs.columns:
            x = adata.obs[xcol].values.astype(np.float64)
            y = adata.obs[ycol].values.astype(np.float64)
            if zcol in adata.obs.columns:
                z = adata.obs[zcol].values.astype(np.float64)
            else:
                z = np.zeros(len(x), dtype=np.float64)
            return np.column_stack([x, y, z])

    # Try obsm keys
    for key in ['spatial3d', 'spatial']:
        if key in adata.obsm:
            coords = np.array(adata.obsm[key], dtype=np.float64)
            if coords.shape[1] == 3:
                return coords
            elif coords.shape[1] == 2:
                # Check if obs has a z column we can combine
                for zcol in ['z', 'Z']:
                    if zcol in adata.obs.columns:
                        z = adata.obs[zcol].values.astype(np.float64)
                        return np.column_stack([coords, z.reshape(-1, 1)])
                return build_spatial_3d(coords, z=0.0)

    raise ValueError("Could not find spatial coordinates in obsm or obs columns")


def main():
    print("Processing STARmap visual cortex dataset...")
    check_raw_data()

    print(f"  Loading {RAW_FILE.name}...")
    adata = ad.read_h5ad(RAW_FILE)

    # Expression matrix
    adata.X = ensure_sparse_csr(adata.X)

    # Spatial coordinates (native 3D) — convert voxel indices to µm
    spatial = extract_spatial_coords(adata)
    spatial[:, 0] *= VOXEL_XY_UM  # x: voxels → µm
    spatial[:, 1] *= VOXEL_XY_UM  # y: voxels → µm
    spatial[:, 2] *= VOXEL_Z_UM   # z: voxels → µm
    adata.obsm['spatial'] = spatial

    # Gene symbols as var index (make unique)
    adata.var_names_make_unique()

    # Obs columns
    if 'cell_type' not in adata.obs.columns:
        if 'leiden' in adata.obs.columns:
            adata.obs['cell_type'] = adata.obs['leiden'].astype(str)
        else:
            adata.obs['cell_type'] = 'unannotated'
    # Use z-plane as section identifier (89 consecutive planes, z=6-94)
    if 'z' in adata.obs.columns:
        adata.obs['section'] = adata.obs['z'].astype(int).astype(str)
        n_sections = adata.obs['section'].nunique()
    elif 'section' not in adata.obs.columns:
        adata.obs['section'] = 'volume_1'
        n_sections = 1
    else:
        n_sections = adata.obs['section'].nunique()
    adata.obs['section'] = adata.obs['section'].astype(str)

    # Metadata
    adata.uns['spatial_metadata'] = {
        'technology': 'STARmap',
        'species': 'mouse',
        'tissue': 'visual cortex',
        'n_sections': int(n_sections),
        'section_spacing_um': VOXEL_Z_UM,
        'coordinate_units': 'micrometers',
        'expression_type': 'raw_counts',
        'source': 'starmapresources.org; Figshare 30418285',
    }
    adata.uns["expression_type"] = "raw_counts"
    adata.uns["dataset_name"] = "starmap_visual_cortex"

    verify(adata)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to {OUTPUT_PATH}...")
    adata.write_h5ad(OUTPUT_PATH)
    print("  Done.")


if __name__ == "__main__":
    main()
