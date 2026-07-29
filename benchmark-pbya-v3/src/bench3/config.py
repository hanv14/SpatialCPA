"""Registry, paths and protocol constants for benchmark-pbya-v3.

benchmark-pbya-v3 is a **paper-faithful reproduction benchmark**: it reproduces
the STARmap visual-cortex evaluation described in the SpatialZ paper (Lin et al.,
2025, Nat Methods) and runs *every* method through that exact protocol, so a
published method's numbers are reproducible and the new SpatialCPA variants are
measured against them on identical terms.

Differences from ``benchmark-pbya-v2``
--------------------------------------
* **One dataset, one protocol.** v2 sweeps 17 datasets with leave-one-out. v3
  runs the single design the SpatialZ paper used: partition the STARmap volume
  into 7 consecutive 2-D sections, hold out sections 2/4/6 *simultaneously*, and
  reconstruct them from sections 1/3/5/7.
* **Paper validation strategy.** In addition to v2's correspondence-free
  generation metrics, v3 computes the quantities the paper actually validated on:
  UMAP continuity between real and reconstructed cells, marker-gene spatial
  patterns (Flt1 / Pcp4 / Cux2), Moran's I *and* Geary's C, and preservation of
  per-cell-type spatial localization. See ``evaluate_paper.py``.
* **Shared method code.** The method wrappers are v2's, invoked unchanged by
  path (see ``METHODS``). Reusing them rather than forking keeps a single source
  of truth: whatever v2 measures, v3 measures with the same synthesis code.

The leakage policy is inherited from v2 wholesale (``_v2bridge``): the held-out
sections are physically absent from the method input, methods receive only a
scalar target z per held-out section, and label vocabularies are training-only.
"""

from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]        # benchmark-pbya-v3/
REPO_ROOT = PROJECT_ROOT.parent                           # SpatialCPA/
V1_ROOT = REPO_ROOT / "benchmark-pbya"                    # shared tools/ + envs/
V2_ROOT = REPO_ROOT / "benchmark-pbya-v2"                 # shared method wrappers
V2_SRC = V2_ROOT / "src"                                  # importable `benchmark` pkg

# Raw STARmap volume (Wang et al. 2018). Different checkouts keep it in
# different places, so resolve against the known locations rather than hard-coding
# one; $BENCH_V3_RAW_STARMAP wins outright, and --raw overrides per invocation.
#
# The v1-processed file is an accepted input too: it is the same cells and the
# same 89 z-planes, just with coordinates already converted to micrometres —
# ``prepare_starmap`` detects that and skips the voxel conversion.
_RAW_STARMAP_NAME = "STARmap_Wang2018three_data_3D_data.h5ad"
RAW_STARMAP_CANDIDATES = (
    REPO_ROOT / "data" / "starmap" / _RAW_STARMAP_NAME,
    V1_ROOT / "data" / "raw" / "starmap_visual_cortex" / _RAW_STARMAP_NAME,
    V1_ROOT / "data" / "processed" / "starmap_visual_cortex" / "data.h5ad",
    V1_ROOT / "data" / "processed" / "starmap_visual_cortex.h5ad",
)


def resolve_raw_starmap():
    """First existing candidate, else the preferred path (for the error message)."""
    env = os.environ.get("BENCH_V3_RAW_STARMAP")
    if env:
        return Path(env)
    for cand in RAW_STARMAP_CANDIDATES:
        if cand.exists():
            return cand
    return RAW_STARMAP_CANDIDATES[0]


RAW_STARMAP = resolve_raw_starmap()

# The v3-built, paper-protocol dataset (produced by ``prepare_starmap.py``).
# Laid out as ``data/processed/<dataset>/data.h5ad``, the same convention v1 and
# v2 use, so the tree is navigable the same way — but under benchmark-pbya-v3,
# because this is a *differently partitioned* view of the STARmap volume (seven
# paper sections, not the raw 89 z-planes) and must not be confused with, or
# written over, v1's processed copy. Which protocol produced a given result stays
# visible in the holdout id (``paper_2_4_6``) and in ``uns['paper_protocol']``.
DATA_DIR = Path(os.environ.get("BENCH_V3_DATA", PROJECT_ROOT / "data" / "processed"))
DATASET_NAME = "starmap_visual_cortex"
DATASET_PATH = DATA_DIR / DATASET_NAME / "data.h5ad"

RESULTS_DIR = Path(os.environ.get("BENCH_V3_RESULTS", PROJECT_ROOT / "results"))
SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURES_DIR = SUMMARY_DIR / "figures"
INPUTS_CACHE = RESULTS_DIR / "_inputs"   # training-only inputs, one per holdout


# ── The SpatialZ-paper STARmap protocol ───────────────────────────────────────
# "The original 3D volume was partitioned into consecutive 2D sections. Uppermost
#  layers (z = 6-13) and lowermost layers (z = 91-94) were removed to reduce
#  technical noise. The remaining tissue was divided into seven consecutive
#  sections. Sections 2, 4 and 6 were held out to simulate missing slices, while
#  sections 1, 3, 5 and 7 were used as input."
#
# The raw volume carries 89 z-planes (z = 6..94). Dropping 6-13 and 91-94 leaves
# 77 planes (z = 14..90), which divides into 7 sections of exactly 11 planes.
DROP_Z_LOW = (6, 13)        # inclusive
DROP_Z_HIGH = (91, 94)      # inclusive
N_SECTIONS = 7
HELD_OUT_SECTION_INDICES = (2, 4, 6)
INPUT_SECTION_INDICES = (1, 3, 5, 7)

def section_label(index):
    """Canonical section label for a 1-based paper section index."""
    return f"section_{int(index)}"

HELD_OUT_SECTIONS = tuple(section_label(i) for i in HELD_OUT_SECTION_INDICES)
INPUT_SECTIONS = tuple(section_label(i) for i in INPUT_SECTION_INDICES)

# Voxel calibration — identical to the v1 processor so v3 coordinates are
# directly comparable with the v1/v2 ``starmap_visual_cortex`` dataset.
# Leica TCS SP5, HC FLUOTAR L 25x/0.95 W at 1024x1024 (Wang et al. 2018);
# lateral confirmed via He et al. 2021 (tissue extent 1545 um = 1800 x 0.859).
VOXEL_XY_UM = 0.859
VOXEL_Z_UM = 1.0

# Marker genes the paper compares spatial expression patterns for (Fig. 2).
MARKER_GENES = ("Flt1", "Pcp4", "Cux2")

# Layer-marker composite used to derive the cortical laminar (depth) axis from
# the ground-truth slice; see ``evaluate_paper.laminar_axis``. Only the genes
# actually present in the panel are used.
LAYER_SUPERFICIAL = ("Cux2", "Rasgrf2", "Nov")
LAYER_DEEP = ("Pcp4", "Ctgf", "Sema3e")

# STARmap is a single 3-D imaging block: the z-planes are inherently
# co-registered, so the training-only re-registration is the identity. (v2 makes
# the same call for every volumetric dataset — re-registering would only
# introduce distortion.)
REGISTRATION = "none"

RANDOM_SEED = 42


# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = {
    DATASET_NAME: {
        "path": DATASET_PATH,
        "technology": "STARmap",
        "species": "mouse",
        "tissue": "visual cortex",
        "registration": REGISTRATION,
        "protocol": "SpatialZ paper (7 consecutive sections, hold out 2/4/6)",
    },
}


# ── Methods ───────────────────────────────────────────────────────────────────
# Generation-only, exactly as in v2: each method receives a training-only,
# re-registered input plus a scalar target z per held-out section, and
# synthesizes the slice de novo.
#
# ``wrapper`` points into benchmark-pbya-v2 on purpose. The wrappers are plain
# CLI scripts over the stable ``methods/_v2_io.py`` contract, so reusing them
# (rather than forking) guarantees v3 and v2 exercise the *same* method code.
def _v2_wrapper(name):
    return V2_ROOT / "src" / "benchmark" / "methods" / name


METHODS = {
    "spatialcpav8_gen": {
        "wrapper": _v2_wrapper("run_spatialcpav8.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "training-free symmetric entropic-OT / McCann bridge",
    },
    "spatialcpav11_gen": {
        "wrapper": _v2_wrapper("run_spatialcpav11.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "teacher-distilled conditional generator",
    },
    "spatialcpav14_gen": {
        "wrapper": _v2_wrapper("run_spatialcpav14.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "joint latent flow matching",
    },
    "spatialcpav15_gen": {
        "wrapper": _v2_wrapper("run_spatialcpav15.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "structure-field diffusion + NB expression VAE + marked point process",
    },
    "spatialz": {
        "wrapper": _v2_wrapper("run_spatialz.py"),
        "conda_env": "bench_spatialz",
        "available": True,
        "family": "published",
        "notes": "Lin et al. 2025, Nat Methods — the method this protocol comes from",
    },
    "feast": {
        "wrapper": _v2_wrapper("run_feast.py"),
        "conda_env": "bench_feast",
        "available": True,
        "family": "published",
        "notes": "Chen et al. 2025 — parameter-cloud interpolation over a PASTE2 alignment",
    },
    "isost": {
        "wrapper": _v2_wrapper("run_isost.py"),
        "conda_env": "bench_isost",
        "available": True,
        "family": "published",
        "notes": "Li et al. 2025 — biaxial SDE generation",
    },
}

# Order used in tables and figures: published baselines first, then SpatialCPA.
METHOD_ORDER = [
    "spatialz", "feast", "isost",
    "spatialcpav8_gen", "spatialcpav11_gen", "spatialcpav14_gen", "spatialcpav15_gen",
]


# ── Metric names (canonical column order) ─────────────────────────────────────
# Tier A — the paper's own validation strategy (v3-specific, ``paper_*``).
PAPER_METRIC_NAMES = [
    # UMAP continuity between real and reconstructed cells
    "paper_umap_mixing",
    "paper_umap_centroid_dist",
    "paper_embedding_mixing_pca",
    # spatial autocorrelation: Moran's I and Geary's C
    "paper_morans_pearson",
    "paper_morans_spearman",
    "paper_morans_mae",
    "paper_morans_median_pred",
    "paper_morans_median_gt",
    "paper_gearys_pearson",
    "paper_gearys_spearman",
    "paper_gearys_mae",
    "paper_gearys_median_pred",
    "paper_gearys_median_gt",
    # marker-gene spatial patterns (Flt1 / Pcp4 / Cux2)
    "paper_marker_field_r",
    "paper_marker_depth_r",
    "paper_marker_morans_mae",
    # gene expression similarity
    "paper_gene_mean_spearman",
    "paper_gene_var_spearman",
    # preservation of cell spatial localization
    "paper_celltype_localization",
    "paper_celltype_ot",
    # bookkeeping
    "paper_cell_count_ratio",
]

# Per-marker breakdowns (populated per gene in MARKER_GENES).
PAPER_MARKER_METRIC_NAMES = [
    f"paper_marker_{g}_{suffix}"
    for g in MARKER_GENES
    for suffix in ("field_r", "depth_r", "morans_delta")
]

# Tier B — v2's correspondence-free generation metrics, computed identically so
# v3 numbers can be read alongside the v2 sweep.
GEN_METRIC_NAMES = [
    "gen_coexpression_agreement",
    "gen_morans_agreement",
    "gen_sinkhorn",
    "gen_celltype_composition",
    "gen_celltype_nhood_agreement",
    "gen_gene_mean_pearson",
    "gen_gene_var_pearson",
    "gen_field_pearson",
    "gen_field_ssim",
    "gen_density_pearson",
    "gen_morans_i_pred_median",
]

# Tier C — v2's cell-matched metrics. Kept for reference only: de-novo generation
# produces no cell-to-cell correspondence, so these are NOT a valid score here
# (see benchmark-pbya-v2/README.md).
MATCHED_METRIC_NAMES = [
    "pearson_median", "spearman_median", "celltype_accuracy",
    "celltype_f1_macro", "ssim_median", "density_pearson", "matching_rate",
]

METRIC_NAMES = (PAPER_METRIC_NAMES + PAPER_MARKER_METRIC_NAMES
                + GEN_METRIC_NAMES + MATCHED_METRIC_NAMES)


# ── Evaluation defaults ───────────────────────────────────────────────────────
SPATIAL_K = 10          # kNN degree for Moran's I / Geary's C spatial weights
FIELD_GRID = 20         # bins per axis for binned spatial fields
DEPTH_BINS = 20         # bins along the cortical laminar axis
EMBED_NEIGHBORS = 15    # kNN degree for the UMAP/PCA mixing score
