"""Dataset registry, paths, and metric names for the benchmarking framework.

benchmark-pbya-v2 — leakage-hardened variant. See ``leakage_guard.py`` and
``README.md`` for the full leakage policy. Key differences from v1:
  * Processed datasets are SHARED with v1 (the processing pipeline is non-leaky
    and unchanged), so ``DATA_PROCESSED`` points at the v1 tree.
  * Every dataset carries a ``registration`` category that controls the
    training-only re-registration applied per holdout.
  * Methods run GENERATION-ONLY: they never receive the held-out (x, y); only a
    scalar target z and a training-only, re-registered input file.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]        # benchmark-pbya-v2/
V1_ROOT = PROJECT_ROOT.parent / "benchmark-pbya"          # shared data/tools

# Processed data and downloaded tools are shared with v1 (non-leaky, unchanged).
# Override with $BENCH_V2_DATA / $BENCH_V2_TOOLS if the layout differs.
import os as _os
DATA_PROCESSED = Path(_os.environ.get("BENCH_V2_DATA", V1_ROOT / "data" / "processed"))
TOOLS_DIR = Path(_os.environ.get("BENCH_V2_TOOLS", V1_ROOT / "tools"))
RESULTS_DIR = PROJECT_ROOT / "results"                    # v2 results kept separate
SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURES_DIR = SUMMARY_DIR / "figures"

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

# ── Re-registration policy (leakage-safe, training-only) ──────────────────────
# Applied PER HOLDOUT by run_benchmark before any method sees the data. See
# leakage_guard.reregister_training.
#   "rigid" — coordinate ICP re-alignment of the training slices (default).
#   "paste" — expression-aware PASTE if installed, else falls back to rigid.
#   "none"  — identity (single-volume data whose planes are already consistent).
#
# Volumetric datasets are one 3-D imaging block (z-planes inherently
# co-registered); re-registering them is unnecessary and could distort, so use
# identity. Everything else is re-registered training-only, which both removes
# any upstream registration that had used the held-out slice and makes
# not-aligned datasets interpolable.
_VOLUMETRIC = {
    "deep_starmap", "starmap_visual_cortex", "easi_fish_hypothalamus",
    "exseq_visual_cortex", "exseq_breast_cancer", "merfish_thick_tissue",
}
DEFAULT_REGISTRATION = "rigid"


def registration_for(dataset_name):
    """Return the training-only re-registration policy for a dataset."""
    base = str(dataset_name).split("/")[0]
    if base in _VOLUMETRIC:
        return "none"
    return DEFAULT_REGISTRATION


# ── Methods (GENERATION-ONLY) ─────────────────────────────────────────────────
# Every v2 method synthesizes the held-out slice from a training-only,
# re-registered input plus a scalar target z. None receives the held-out (x, y).
# `generation_native=True` methods already synthesize de novo; coordinate-query
# regressors (stVGP, Spateo SVGP) are NOT generation-native and are disabled in
# v2 (they would require the held-out query coordinates). See README.
METHODS = {
    "spatialcpav9_gen": {
        "wrapper": "src/benchmark/methods/run_spatialcpav9.py",
        "conda_env": "bench_spatialcpa",
        "available": True,
        "generation_native": True,
    },
    "spatialcpav8_gen": {
        "wrapper": "src/benchmark/methods/run_spatialcpav8.py",
        "conda_env": "bench_spatialcpa",
        "available": True,
        "generation_native": True,
    },
    "spatialcpav6_gen": {
        "wrapper": "src/benchmark/methods/run_spatialcpav6.py",
        "conda_env": "bench_spatialcpa",
        "available": True,
        "generation_native": True,
    },
    "spatialcpav5_gen": {
        "wrapper": "src/benchmark/methods/run_spatialcpav5.py",
        "conda_env": "bench_spatialcpa",
        "available": True,
        "generation_native": True,
    },
    "spatialcpav4_gen": {
        "wrapper": "src/benchmark/methods/run_spatialcpav4.py",
        "conda_env": "bench_spatialcpa",
        "available": True,
        "generation_native": True,
    },
    "spatialz": {
        "wrapper": "src/benchmark/methods/run_spatialz.py",
        "conda_env": "bench_spatialz",
        "available": True,
        "generation_native": True,
    },
    "feast": {
        "wrapper": "src/benchmark/methods/run_feast.py",
        "conda_env": "bench_feast",
        "available": True,
        "generation_native": True,
    },
    "isost": {
        "wrapper": "src/benchmark/methods/run_isost.py",
        "conda_env": "bench_isost",
        "available": True,
        "generation_native": True,
    },
    # Coordinate-query regressors — incompatible with generation-only evaluation
    # because they predict AT supplied coordinates. Disabled in v2.
    "stvgp": {
        "wrapper": "src/benchmark/methods/run_stvgp.py",
        "conda_env": "bench_stvgp",
        "available": False,
        "generation_native": False,
        "disabled_reason": "coordinate-query method; needs held-out (x, y)",
    },
    "spateo_gp": {
        "wrapper": "src/benchmark/methods/run_spateo_gp.py",
        "conda_env": "bench_spateo",
        "available": False,
        "generation_native": False,
        "disabled_reason": "coordinate-query method; needs held-out (x, y)",
    },
}

# ── Metric names (canonical order for CSV columns) ────────────────────────────
# PRIMARY generation metrics (gen_*) come first: they are correspondence-free
# and (mostly) alignment-free, so they are the meaningful measurement for de-novo
# slice generation. The cell-matched block below is kept for reference / coverage
# but is NOT the primary score for generation (it needs a cell correspondence
# that generation does not produce — see evaluate_generation.py).
METRIC_NAMES = [
    # ── Primary: correspondence-free generation metrics ──
    "gen_coexpression_agreement",  # gene-gene structure agreement (scale-fair, alignment-free)
    "gen_morans_agreement",        # per-gene Moran's I agreement (scale-fair, alignment-free)
    "gen_sinkhorn",                # OT distance between expression distributions (lower=better)
    "gen_celltype_composition",    # cell-type proportion (mix) agreement (correspondence-free)
    "gen_celltype_nhood_agreement",  # cell-type SPATIAL organization agreement (correspondence-free)
    "gen_gene_mean_pearson",       # per-gene mean agreement (scale-sensitive, secondary)
    "gen_gene_var_pearson",        # per-gene variance agreement (scale-sensitive, secondary)
    "gen_field_pearson",           # binned spatial-field agreement (needs alignment)
    "gen_field_ssim",              # binned structural similarity (needs alignment)
    "gen_density_pearson",         # bin-wise cell-density agreement (needs alignment)
    "gen_morans_i_pred_median",    # spatial structure of the prediction alone
    # ── Reference: cell-matched (correspondence-dependent) metrics ──
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
    "gene_mean_pearson",
    "gene_var_pearson",
    "morans_i_median",
    "dice_density",
]

# ── Evaluation defaults ──────────────────────────────────────────────────────
NN_MATCH_THRESHOLD_UM = 50.0   # max distance (µm) for nearest-neighbor cell matching
SSIM_GRID_SIZE = 50            # bins per axis for SSIM grid
SSIM_TOP_GENES = 100           # number of HVGs for SSIM computation
RANDOM_SEED = 42
