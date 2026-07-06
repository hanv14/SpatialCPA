"""Dataset registry, paths, and metric names for the benchmarking framework."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURES_DIR = SUMMARY_DIR / "figures"
TOOLS_DIR = PROJECT_ROOT / "tools"

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry: dataset_name -> dict with path(s), metadata
# specimen=None means single h5ad at data.h5ad; otherwise list of specimen subdirs.

DATASETS = {
    # ── Tier 1 (small, fast iteration) ────────────────────────────────────────
    "cosmx_nsclc_3d": {
        "path": DATA_PROCESSED / "cosmx_nsclc_3d" / "data.h5ad",
        "specimens": None,
        "tier": 1,
        "technology": "CosMx",
        "species": "human",
    },
    "imc_breast_cancer": {
        "path": DATA_PROCESSED / "imc_breast_cancer" / "data.h5ad",
        "specimens": None,
        "tier": 1,
        "technology": "IMC",
        "species": "human",
    },
    "merfish_hypothalamus/animal_1": {
        "path": DATA_PROCESSED / "merfish_hypothalamus" / "animal_1" / "data.h5ad",
        "specimens": None,
        "tier": 1,
        "technology": "MERFISH",
        "species": "mouse",
    },

    # ── Tier 2 (larger / more specimens) ──────────────────────────────────────
    "allen_zhuang_merfish/Zhuang-ABCA-1": {
        "path": DATA_PROCESSED / "allen_zhuang_merfish" / "Zhuang-ABCA-1" / "data.h5ad",
        "specimens": None,
        "tier": 2,
        "technology": "MERFISH",
        "species": "mouse",
    },
    "allen_zhuang_merfish/Zhuang-ABCA-2": {
        "path": DATA_PROCESSED / "allen_zhuang_merfish" / "Zhuang-ABCA-2" / "data.h5ad",
        "specimens": None,
        "tier": 2,
        "technology": "MERFISH",
        "species": "mouse",
    },
    "allen_merfish_brain": {
        "path": DATA_PROCESSED / "allen_merfish_brain" / "data.h5ad",
        "specimens": None,
        "tier": 2,
        "technology": "MERFISH",
        "species": "mouse",
    },
    "openst_lymph_node": {
        "path": DATA_PROCESSED / "openst_lymph_node" / "data.h5ad",
        "specimens": None,
        "tier": 2,
        "technology": "Open-ST",
        "species": "human",
    },
    "st_mouse_brain_ortiz": {
        "path": DATA_PROCESSED / "st_mouse_brain_ortiz" / "data.h5ad",
        "specimens": None,
        "tier": 2,
        "technology": "ISS",
        "species": "mouse",
    },
    "visium_mouse_brain_cell2location": {
        "path": DATA_PROCESSED / "visium_mouse_brain_cell2location" / "mouse_1" / "data.h5ad",
        "specimens": None,
        "tier": 2,
        "technology": "Visium",
        "species": "mouse",
    },
    "deep_starmap": {
        "path": DATA_PROCESSED / "deep_starmap" / "data.h5ad",
        "specimens": None,
        "tier": 2,
        "technology": "Deep-STARmap",
        "species": "mouse",
    },
    "starmap_visual_cortex": {
        "path": DATA_PROCESSED / "starmap_visual_cortex" / "data.h5ad",
        "specimens": None,
        "tier": 2,
        "technology": "STARmap",
        "species": "mouse",
    },

    # ── Tier 3 (paper-specific datasets) ──────────────────────────────────────
    "arrayseq_kidney": {
        "path": DATA_PROCESSED / "arrayseq_kidney" / "data.h5ad",
        "specimens": None,
        "tier": 3,
        "technology": "Array-seq",
        "species": "mouse",
    },
    "visium_spinal_cord_isost": {
        "path": DATA_PROCESSED / "visium_spinal_cord_isost" / "data.h5ad",
        "specimens": None,
        "tier": 3,
        "technology": "Visium",
        "species": "mouse",
    },
    "visium_dlpfc_stvgp": {
        "path": DATA_PROCESSED / "visium_dlpfc_stvgp" / "data.h5ad",
        "specimens": None,
        "tier": 3,
        "technology": "Visium",
        "species": "human",
    },
    "st_breast_cancer_stvgp": {
        "path": DATA_PROCESSED / "st_breast_cancer_stvgp" / "data.h5ad",
        "specimens": None,
        "tier": 3,
        "technology": "ST",
        "species": "human",
    },
    "imc_breast_cancer_kuett/MainHer2": {
        "path": DATA_PROCESSED / "imc_breast_cancer_kuett" / "MainHer2" / "data.h5ad",
        "specimens": None,
        "tier": 3,
        "technology": "IMC",
        "species": "human",
    },
    "imc_breast_cancer_kuett/SecondHer2": {
        "path": DATA_PROCESSED / "imc_breast_cancer_kuett" / "SecondHer2" / "data.h5ad",
        "specimens": None,
        "tier": 3,
        "technology": "IMC",
        "species": "human",
    },
}

# ── Methods ───────────────────────────────────────────────────────────────────
METHODS = {
    "feast": {
        "wrapper": "src/benchmark/methods/run_feast.py",
        "conda_env": "bench_feast",
        "available": True,
    },
    "spatialz": {
        "wrapper": "src/benchmark/methods/run_spatialz.py",
        "conda_env": "bench_spatialz",
        "available": True,
    },
    "isost": {
        "wrapper": "src/benchmark/methods/run_isost.py",
        "conda_env": "bench_isost",
        "available": True,
    },
    "stvgp": {
        "wrapper": "src/benchmark/methods/run_stvgp.py",
        "conda_env": "bench_stvgp",
        "available": True,
    },
    "spateo_gp": {
        "wrapper": "src/benchmark/methods/run_spateo_gp.py",
        "conda_env": "bench_spateo",
        "available": True,
    },
    "spatialcpa": {
        "wrapper": "src/benchmark/methods/run_spatialcpa.py",
        "conda_env": "bench_spatialcpa",
        "available": True,
    },
    "spatialcpav4": {
        "wrapper": "src/benchmark/methods/run_spatialcpav4.py",
        "conda_env": "bench_spatialcpa",
        "available": True,
    },
    # De-novo generation variant: never sees the held-out (x, y); synthesizes
    # the slice via the occupancy head so the cell count is emergent (like
    # SpatialZ/FEAST/isoST). Same wrapper, --generate-mode forced.
    "spatialcpav4_gen": {
        "wrapper": "src/benchmark/methods/run_spatialcpav4_gen.py",
        "conda_env": "bench_spatialcpa",
        "available": True,
    },
    # STODE removed: it is a temporal interpolation method (developmental timepoints),
    # not a spatial z-interpolation method. See CLAUDE.md "STODE paper audit" for details.
}

# ── Metric names (canonical order for CSV columns) ────────────────────────────
METRIC_NAMES = [
    "pearson_median",
    "pearson_mean",
    "pearson_frac_gt05",
    "spearman_median",
    "spearman_mean",
    "rmse_median",
    "mae_median",
    "celltype_accuracy",
    "celltype_f1_macro",
    "ssim_median",
    "density_pearson",
    "matching_rate",
    # Additional metrics from paper audits
    "gene_mean_pearson",     # FEAST's primary: correlation of per-gene means
    "gene_var_pearson",      # FEAST: correlation of per-gene variances
    "morans_i_median",       # SpatialZ/FEAST: spatial autocorrelation preservation
    "dice_density",          # isoST: binarized density overlap
]

# ── Evaluation defaults ──────────────────────────────────────────────────────
NN_MATCH_THRESHOLD_UM = 50.0   # max distance (µm) for nearest-neighbor cell matching
SSIM_GRID_SIZE = 50            # bins per axis for SSIM grid
SSIM_TOP_GENES = 100           # number of HVGs for SSIM computation
RANDOM_SEED = 42
