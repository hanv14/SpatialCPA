#!/usr/bin/env python
"""Process 3D IMC breast cancer dataset to standardized h5ad format.

Converts 15 z-step slices of HER2+ breast cancer IMC data into a single
standardized h5ad with CSR sparse matrix, 3D spatial coordinates, and metadata.

Raw: data/raw/imc_breast_cancer/MainHer2BreastCancerModel_zstep10_{0..14}.h5ad
Output: data/processed/imc_breast_cancer.h5ad
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "imc_breast_cancer"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "imc_breast_cancer.h5ad"

N_SLICES = 15
Z_STEP_UM = 10.0


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
    missing = []
    for i in range(N_SLICES):
        fpath = RAW_DIR / f"MainHer2BreastCancerModel_zstep10_{i}.h5ad"
        if not fpath.exists():
            missing.append(fpath.name)
    if missing:
        print(f"ERROR: {len(missing)} raw file(s) not found in {RAW_DIR}/")
        for m in missing[:5]:
            print(f"  - {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
        print("Please download by running:")
        print("  python src/data/download/download_imc_breast_cancer.py")
        sys.exit(1)


def extract_spatial_2d(adata):
    """Extract 2D spatial coordinates from an AnnData object."""
    for key in ['spatial', 'spatial3d']:
        if key in adata.obsm:
            coords = np.array(adata.obsm[key], dtype=np.float64)
            return coords[:, :2]

    for xcol, ycol in [('x', 'y'), ('X', 'Y')]:
        if xcol in adata.obs.columns and ycol in adata.obs.columns:
            x = adata.obs[xcol].values.astype(np.float64)
            y = adata.obs[ycol].values.astype(np.float64)
            return np.column_stack([x, y])

    raise ValueError("Could not find spatial coordinates in obsm or obs columns")


def main():
    print("Processing 3D IMC breast cancer dataset...")
    check_raw_data()

    adatas = []
    for i in range(N_SLICES):
        fpath = RAW_DIR / f"MainHer2BreastCancerModel_zstep10_{i}.h5ad"
        print(f"  Loading slice {i}: {fpath.name}...")
        a = ad.read_h5ad(fpath)

        # Build 3D coordinates
        coords_2d = extract_spatial_2d(a)
        z_val = i * Z_STEP_UM
        a.obsm['spatial'] = build_spatial_3d(coords_2d, z=z_val)

        # Section label
        a.obs['section'] = f'z{i}'

        # Cell type
        if 'cell_type' not in a.obs.columns:
            if 'leiden' in a.obs.columns:
                a.obs['cell_type'] = a.obs['leiden'].astype(str)
            else:
                a.obs['cell_type'] = 'unannotated'

        # Ensure CSR
        a.X = ensure_sparse_csr(a.X)

        # Make var names unique per slice
        a.var_names_make_unique()

        # Unique obs names to avoid collision
        a.obs_names = [f"z{i}_{name}" for name in a.obs_names]

        adatas.append(a)

    print("  Concatenating slices...")
    adata = ad.concat(adatas, join='outer', merge='first')
    adata.obs_names_make_unique()

    # Ensure CSR after concat
    adata.X = ensure_sparse_csr(adata.X)

    # Gene symbols
    adata.var_names_make_unique()

    # Ensure string section
    adata.obs['section'] = adata.obs['section'].astype(str)

    # Metadata
    adata.uns['spatial_metadata'] = {
        'technology': '3D IMC',
        'species': 'human',
        'tissue': 'breast cancer (HER2+)',
        'n_sections': N_SLICES,
        'section_spacing_um': Z_STEP_UM,
        'coordinate_units': 'um',
        'expression_type': 'fluorescence_intensity',
        'source': 'Zenodo 10.5281/zenodo.4752030',
    }

    adata.uns["expression_type"] = "fluorescence_intensity"
    adata.uns["dataset_name"] = "imc_breast_cancer"

    verify(adata)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to {OUTPUT_PATH}...")
    adata.write_h5ad(OUTPUT_PATH)
    print("  Done.")


if __name__ == "__main__":
    main()
