# Download Scripts

One standalone script per dataset. Each downloads raw data from its source (GEO, Zenodo, Dryad, Allen Brain Atlas, etc.) to `data/raw/<dataset_name>/`.

## Conventions

- File: `download_<dataset_name>.py`
- Each script is fully self-contained (no shared imports)
- URL definitions at the top of the file
- Resumable downloads (skip existing files by size check)
- Post-download verification (file existence, expected sizes)
- `__main__` entry point

## Usage

```bash
# Download one dataset
python src/data/download/download_cosmx_nsclc_3d.py

# Download all datasets
python src/data/download/download_all.py
```

## Tracking

`data/raw/download_status.csv` tracks every dataset:
- download_status: COMPLETE, PARTIAL, WRONG_DATA, NO_DATA, REMOVED
- verified, disk_size, n_files, raw_format, processable, issues

## Scripts (22)

| Script | Source | Size |
|--------|--------|------|
| download_allen_merfish_brain.py | Allen Brain Atlas S3 | ~12 GB |
| download_allen_zhuang_merfish.py | Allen Brain Atlas S3 | ~8 GB |
| download_arrayseq_kidney.py | GEO GSE253355 | ~400 MB |
| download_cosmx_nsclc_3d.py | Zenodo 15240431 | ~2 GB |
| download_deep_starmap.py | Zenodo 8327576 | ~50 MB |
| download_easi_fish_hypothalamus.py | Zenodo 10932552 | ~3 GB |
| download_exseq_breast_cancer.py | Zenodo 4479018 | ~200 MB |
| download_exseq_visual_cortex.py | Zenodo 4560540 | ~1 GB |
| download_imc_breast_cancer.py | Zenodo 3518284 | ~100 MB |
| download_imc_breast_cancer_kuett.py | Zenodo 4752030 | ~200 MB |
| download_merfish_hypothalamus.py | Dryad | ~2 GB |
| download_merfish_thick_tissue.py | GitHub | ~500 MB |
| download_openst_lymph_node.py | GEO GSE233163 | ~15 GB |
| download_st_breast_cancer_stvgp.py | Zenodo/10x | ~100 MB |
| download_st_mouse_brain_ortiz.py | Zenodo/publication | ~200 MB |
| download_starmap_visual_cortex.py | Dropbox/publication | ~50 MB |
| download_visium_dlpfc_stvgp.py | spatialLIBD | ~300 MB |
| download_visium_mouse_brain_cell2location.py | GEO GSE199635 | ~500 MB |
| download_visium_spinal_cord_isost.py | GEO GSE234774 | ~2 GB |
| download_vizgen_merfish_brain.py | Vizgen (registration required) | N/A |
| download_hubmap_imc_spleen.py | HuBMAP (Globus) | N/A |
| download_all.py | Orchestrator | - |
