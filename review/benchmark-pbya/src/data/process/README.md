# Processing Scripts

One standalone script per dataset. Each converts raw data from `data/raw/<dataset_name>/` to a standardized h5ad at `data/processed/<dataset_name>/data.h5ad`.

## Conventions

- File: `process_<dataset_name>.py`
- Each script is fully self-contained (no shared imports)
- Standard functions: `check_raw_data()`, `ensure_sparse_csr()`, `build_spatial_3d()`, `verify()`, `process()`, `main()`
- Code duplication across scripts is intentional (independence over DRY)

## Output format

Each processed h5ad contains:
- `X`: CSR sparse expression matrix
- `obsm['spatial']`: (n_cells, 3) array — x, y, z in micrometers
- `obs['section']`: section label per cell
- `obs['cell_type']`: cell type annotation (or "unknown")
- `uns['expression_type']`: one of `raw_counts`, `log1p_normalized`, `log2_normalized`, `normalized`, `fluorescence_intensity`, `mean_intensity`
- `uns['dataset_name']`: dataset identifier

## Coordinate standards

All coordinates in micrometers. Conversion factors:
- Visium: 55 um / spot_diameter_fullres
- Open-ST: 0.345 um/pixel
- CosMx SMI: 0.18 um/pixel
- Allen CCF (mm): multiply by 1000
- z = physical section depth (section_index * section_spacing)

## Usage

```bash
# Process one dataset
python src/data/process/process_cosmx_nsclc_3d.py

# Process all datasets (8 tiers, ordered by dependency)
python src/data/process/process_all.py

# Package datasets (generate READMEs + restructure)
python src/data/process/package_datasets.py

# QC report across all datasets
python src/data/process/report_all_datasets.py
```

## Scripts (24)

Includes `process_all.py` (orchestrator), `package_datasets.py` (README generation), and `report_all_datasets.py` (QC report), plus one `process_<name>.py` per dataset.

## Multi-specimen datasets

Some datasets produce multiple h5ad files (one per specimen):
- `merfish_hypothalamus/` — 11 animals (animal_1 through animal_11)
- `allen_zhuang_merfish/` — 4 regions (Zhuang-ABCA-1 through ABCA-4)
- `merfish_thick_tissue/` — 2 regions (cortex, hypothalamus)
- `easi_fish_hypothalamus/` — 3 samples (LHA1, LHA2, LHA3)
- `visium_mouse_brain_cell2location/` — 1 mouse (mouse_1; mouse_2 dropped, only 2 sections)
- `imc_breast_cancer_kuett/` — 4 tumors (MainHer2, SecondHer2, LVIBlood, LVILymph)
