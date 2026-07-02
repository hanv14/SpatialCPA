# Benchmark Framework

Leave-one-out (LOO) evaluation of z-interpolation methods on 3D spatial transcriptomics datasets.

## Pipeline

Prediction and evaluation are **decoupled**:

1. **Predict**: `run_all.py` / `run_benchmark.py` spawn method wrappers that produce `prediction.h5`
2. **Evaluate**: `evaluate_all.py` / `evaluate.py` compare predictions against ground truth, write `metrics.json`
3. **Aggregate**: `aggregate_results.py` collects all `metrics.json` into summary CSVs

This separation means you can re-evaluate predictions without re-running methods, and method failures don't lose evaluation state.

## Architecture

```
config.py              # Dataset registry (17 datasets), method registry (5 methods), metric names
holdout.py             # Generate LOO / leave-k-out / alternating holdout configs from h5ad
run_benchmark.py       # Run one method x dataset x holdout: spawn conda env, produce prediction.h5
run_all.py             # Orchestrate full campaign: --tier, --methods, --datasets, --skip-existing
run_campaign.py        # Alternative campaign runner with job queuing
evaluate.py            # Compare one prediction.h5 against ground truth → metrics.json
evaluate_all.py        # Walk results tree, evaluate all prediction.h5 files (with --force to re-eval)
aggregate_results.py   # Collect all metrics.json into summary CSVs (all_metrics.csv, summary_by_method_dataset.csv)
resource_monitor.py    # Track wall time, peak RSS, GPU memory during method execution
plot_results.py        # Generate summary figures from aggregated results
```

## Methods

Each wrapper calls the **authors' code directly** — we do not reimplement models or training. See `methods/README.md` for details.

| Method | Paper | Code | Conda env | What our wrapper calls | What we implement ourselves |
|--------|-------|------|-----------|----------------------|---------------------------|
| **FEAST** | Chen et al. 2025 | `pip install FEAST-py` | bench_feast | `FEAST.interpolate_slices()`, `InterpolationConfig` | Alignment (PASTE2/PASTE1, as expected by FEAST), data prep |
| **SpatialZ** | Lin et al. 2025, Nat Methods | [Zenodo 17416727](https://doi.org/10.5281/zenodo.17416727) | bench_spatialz | `SpatialZ.Generate_spatialz()` | Data prep, Leiden fallback for missing cell types |
| **isoST** | Li et al. 2025 | [github.com/deng-ai-lab/isoST](https://github.com/deng-ai-lab/isoST) | bench_isost | `biaxial_train()`, `IsoST.fine_infer()` | Preprocessing (zscore→PCA→minmax→.pt; authors only show this in notebooks), inverse PCA for expression recovery |
| **stVGP** | Wang et al. 2026, Adv Sci | [github.com/wzdrgi/stVGP](https://github.com/wzdrgi/stVGP) | bench_stvgp | `gene_rigid_mapping_alignment()`, `adata_preprocess_adjnet()`, `train_stVGP()`, `get_3D_prediction()`, `gene_prediction()` | HVG selection (authors use Moran's I via R), GP subsampling for >5K cells |
| **SVGP (Spateo)** | Qiu et al. 2024, Cell | [github.com/aristoteleo/spateo-release](https://github.com/aristoteleo/spateo-release) | bench_spateo | GPyTorch SVGP directly (same algorithm as Spateo's `gp_interpolation`) | Full wrapper (Spateo's wrapper deadlocks; we call GPyTorch with identical model/params) |

## Paper datasets → our datasets

Methods were developed and evaluated on specific datasets. Our processed versions may have different names:

| Paper name | Used by | Our dataset name | Notes |
|------------|---------|------------------|-------|
| Zhuang-ABCA-1 (slices 005-009) | FEAST Fig 5c | `allen_zhuang_merfish/Zhuang-ABCA-1` | Subset of 5 slices for paper repro |
| Zhuang-ABCA-2 (54 sections) | isoST | `allen_zhuang_merfish/Zhuang-ABCA-2` | Subset of 54 specific section IDs |
| MERFISH hypothalamus (Animal 1) | SpatialZ | `merfish_hypothalamus/animal_1` | Paper ships pre-aligned h5ad with bregma minus signs stripped |
| STARmap visual cortex | SpatialZ Fig 2 | `starmap_visual_cortex` | Same data |
| 3D IMC breast cancer (Kuett 2022) | SpatialZ | `imc_breast_cancer_kuett/MainHer2` | Qualitative only in paper |
| DLPFC (151673-151676) | stVGP Supp Fig 16 | `visium_dlpfc_stvgp` | 4 Visium DLPFC sections |
| ADMB (Allen Mouse Brain) | stVGP | `st_mouse_brain_ortiz` | Ortiz et al. ISS data |
| BC (breast cancer, Layers 1-4) | stVGP | `st_breast_cancer_stvgp` | 4 ST breast cancer sections |
| Array-seq kidney | isoST | `arrayseq_kidney` | Same data |
| Drosophila embryo | Spateo demo | (not in collection) | Paper has no quantitative benchmark |

## Paper reproduction

Separate from the benchmark loop; these reproduce specific figures/tables from each method's paper:

| Script | Method | What it reproduces | Repro status |
|--------|--------|--------------------|--------------|
| `run_feast_paper_repro.py` | FEAST | Fig 5c (ABCA-1 slices 005-009) | mean_corr 0.955 vs paper 0.957 (Delta=-0.002) |
| `run_isost_paper_repro.py` | isoST | ABCA-2 alternating holdout | gene_mean_r median=0.955 (paper: boxplot only) |
| `run_stvgp_paper_repro.py` | stVGP | ADMB mouse brain (Tutorial params) | gene_mean_pearson 0.808 |
| `run_stvgp_dlpfc_repro.py` | stVGP | DLPFC MOBP PCC | 0.641 vs paper 0.682 (Delta=-0.041) |
| `run_stvgp_bc_repro_v2.py` | stVGP | Breast cancer cross-section | avg PCC 0.529 vs paper ~0.565 (Delta=-0.035, different task) |
| `run_spatialz_paper_audit.py` | SpatialZ | STARmap Fig 2e/g/h, MERFISH holdout | 10 Spearman R values: mean |Delta|=0.014 |
| `run_spatialz_stagate_eval.py` | SpatialZ | STAGATE ARI/NMI (Ext Data Fig 1j-k) | ARI 0.439 vs paper 0.492 (STAGATE params not published) |
| `run_spatialz_stagate_gridsearch.py` | SpatialZ | STAGATE hyperparameter grid search | Best: ARI=0.439, NMI=0.575 |
| `run_paper_validation.py` | All | Extract paper-specific data subsets | — |

Results: `results/<method>_paper_repro/` and `results/spatialz_paper_audit/`

## Usage

```bash
# Dry run to see campaign size
python -m src.benchmark.run_all --tier 1 --dry-run

# Run predictions only (no evaluation)
python -m src.benchmark.run_all --tier 1 --methods feast spatialz stvgp --no-eval

# Run predictions + evaluation in one step (default)
python -m src.benchmark.run_all --methods feast --datasets cosmx_nsclc_3d

# Evaluate all predictions (or re-evaluate with --force)
python -m src.benchmark.evaluate_all
python -m src.benchmark.evaluate_all --force --methods feast

# Evaluate a single prediction
python -m src.benchmark.evaluate \
    --prediction results/feast/cosmx_nsclc_3d/loo_section_10/prediction.h5 \
    --ground-truth data/processed/cosmx_nsclc_3d/data.h5ad \
    --output results/feast/cosmx_nsclc_3d/loo_section_10/metrics.json

# Aggregate results into CSV
python -m src.benchmark.aggregate_results
```

## Output structure

```
results/{method}/{dataset}/{holdout_id}/
    prediction.h5      # Predicted expression + coords
    metrics.json       # Evaluation metrics
    resources.json     # Wall time, peak RAM, GPU memory
    method_log.txt     # Stdout/stderr from method wrapper
results/summary/
    all_metrics.csv              # Per-holdout metrics (one row per LOO run)
    summary_by_method_dataset.csv # Aggregated means per method x dataset
```

## prediction.h5 format

```
prediction.h5
├── X/                          # CSR sparse expression matrix
│   ├── data      float32[]     # Non-zero values
│   ├── indices   int32[]       # Column indices
│   ├── indptr    int32[]       # Row pointers
│   └── shape     int64[2]      # (n_cells, n_genes)
├── obs/                        # Per-cell metadata
│   ├── cell_id   bytes[]       # "pred_0", "pred_1", ...
│   ├── x         float64[]     # Spatial x coordinate (µm)
│   ├── y         float64[]     # Spatial y coordinate (µm)
│   ├── z         float64[]     # Spatial z coordinate (µm)
│   ├── section   bytes[]       # Holdout section label
│   └── cell_type bytes[]       # Predicted cell type (or "NA")
├── var/                        # Per-gene metadata
│   └── gene_name bytes[]       # Gene symbols
└── uns/                        # Run metadata
    ├── method_name       str   # e.g. "feast", "spatialz"
    ├── holdout_sections  str   # JSON list of held-out section labels
    ├── method_params     str   # JSON dict of method parameters
    └── wall_time_seconds float # Training + prediction time
```
