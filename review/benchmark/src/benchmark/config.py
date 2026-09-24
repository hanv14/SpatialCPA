"""Constants for the shared evaluators (``evaluate.py``).

``evaluate.py`` imports ``METRIC_NAMES``, ``NN_MATCH_THRESHOLD_UM``,
``SSIM_GRID_SIZE`` and ``SSIM_TOP_GENES`` from here; the values are unchanged
from the evaluators' published configuration (``tests/test_pins.py`` asserts
them). The benchmark's own registry, paths and protocol constants are in
``src/bench3/config.py``.
"""

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
