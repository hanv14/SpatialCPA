# T09 on real data — shipped config, full calibration, tier-1 STARmap

Dataset `starmap_visual_cortex`, holdout `paper_2_4_6` (**tier 1**, `specs/10` §1). Config `751b73d4b97338f6` from defaults, seed 2.

| gate | value |
|---|---|
| `layout_mode` | `resample` |
| `prior_mode` | `correlated` |
| `expr_mode` | `zinb-flow` |
| `text_emb_mode` | `medcpt` |
| `train_steps` | 1200 |
| weights (autocorr / profile / distribution) | 0 / 0 / 0 |

**The text channel is live for the first time on this dataset.** Every prior STARmap number was produced with embeddings built from zeros or from a bare symbol, so `text_emb_mode=medcpt` was `lookup` in all but name (ablation A3). Descriptors here come from `resources/gene_meta.parquet` through `model.build_entity_embeddings`.

## The six target metrics, medians over the three held-out sections

At ground-truth-matched density: each section subsampled to its own true cell count, because a denser point set puts kNN neighbours closer and inflates every graph-based metric. Medians, never means (`specs/10` §4.6) — `section_2` carries a fixed one-sided-evidence penalty that a mean over n = 3 would launder into the headline.

| metric | `uncalibrated` | `detection-calibrated` |
|---|---|---|
| `paper_morans_pearson` | +0.4179 | +0.5112 |
| `paper_gearys_pearson` | +0.4205 | +0.5133 |
| `paper_umap_mixing` | +0.8493 | +0.8431 |
| `paper_marker_field_r` | +0.5316 | +0.5403 |
| `paper_marker_depth_r` | +0.7100 | +0.7306 |
| `paper_celltype_localization` | +0.7499 | +0.7499 |
| `paper_gene_mean_spearman` | +0.9792 | +0.9557 |
| `paper_cell_count_ratio` (raw pass) | 0.988 | 0.988 |

Per section, matched density:

| arm | metric | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|---|
| `uncalibrated` | `morans_pearson` | +0.4075 | +0.4179 | +0.4467 | +0.4179 |
| `uncalibrated` | `gearys_pearson` | +0.4040 | +0.4205 | +0.4440 | +0.4205 |
| `uncalibrated` | `umap_mixing` | +0.8713 | +0.8167 | +0.8493 | +0.8493 |
| `uncalibrated` | `marker_field_r` | +0.5129 | +0.5436 | +0.5316 | +0.5316 |
| `uncalibrated` | `marker_depth_r` | +0.6266 | +0.7432 | +0.7100 | +0.7100 |
| `uncalibrated` | `celltype_localization` | +0.6925 | +0.7499 | +0.7761 | +0.7499 |
| `detection-calibrated` | `morans_pearson` | +0.5705 | +0.5112 | +0.4866 | +0.5112 |
| `detection-calibrated` | `gearys_pearson` | +0.5712 | +0.5133 | +0.4876 | +0.5133 |
| `detection-calibrated` | `umap_mixing` | +0.8757 | +0.8070 | +0.8431 | +0.8431 |
| `detection-calibrated` | `marker_field_r` | +0.5593 | +0.5380 | +0.5403 | +0.5403 |
| `detection-calibrated` | `marker_depth_r` | +0.6962 | +0.7582 | +0.7306 | +0.7306 |
| `detection-calibrated` | `celltype_localization` | +0.6771 | +0.7499 | +0.7761 | +0.7499 |

Emitted cell counts (generated / ground truth):

* `uncalibrated`: section_2=4073/4187, section_4=4169/4102, section_6=4110/4162
* `detection-calibrated`: section_2=4073/4187, section_4=4169/4102, section_6=4110/4162

## Calibration statuses

```
{
  "retrieval_window": {
    "window": 3.0,
    "max_gap_um": 22.0,
    "configured": 3.0
  },
  "lengthscale": {
    "ell_xy": 41.35089121180491,
    "ell_z": 26.944387170614956,
    "status": "converged",
    "ell_z_status": "converged",
    "ell_fitted_xy": 125.62451351795204,
    "ell_fitted_z": 132.0,
    "i_gen": 0.24634964764118195,
    "i_target": 0.2577052041888237,
    "z_achieved": 0.9464790820342298,
    "z_target": 0.9570073884273124,
    "iterations": 2,
    "applied_ell_xy": 41.35089121180491,
    "applied_ell_z": 26.944387170614956
  },
  "detection": {
    "measured": true,
    "shipped": false,
    "fold_sections": [
      "section_1",
      "section_3",
      "section_5",
      "section_7"
    ],
    "detection_mae_gen_vs_real": 0.0029964004900481034,
    "seconds": 7.4
  },
  "anchor": {
    "fitted": false,
    "why": "expr_mode=zinb-flow: w(v) is only read under auto-blend"
  }
}
```

## Calibration (leakage-free, flanking training sections only)

| quantity | value | status |
|---|---|---|
| `ell_xy` | 41.4 um | converged |
| `ell_z` | 26.9 um | converged |
| fitted `ell` | 125.6 / 132.0 um | variogram |
| Moran's I | gen 0.2463 vs flanking 0.2577 | 2 iterations |
| between-section r | gen 0.9465 vs observed 0.9570 | R1 remedy 2 |

Derived `retrieval_z_window` = **3** spacings (largest section gap 22 um). `Config.retrieval_z_window` remains the fallback and the ablation handle.

## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)

One **global** `ell` is calibrated; this table says whether it serves every gene module equally. It is not a target, and a poor table is evidence for the per-channel-group escalation, which is a design change to be decided explicitly.

⚠️ **`mean |I_gen - I_real|` is a per-gene mean of absolute differences, and does NOT equal the difference of the two columns beside it.** `I_gen` and `I_real` are means over the module's genes; a module whose genes miss in both directions has a small column difference and a large mean absolute one. Measured on tier-1 at module 1: the columns differ by 0.0114 and the per-gene mean by **0.1639**, 14x apart. The old header read `|diff|`, which invited exactly the wrong subtraction.

| module | genes | mean I_gen | mean I_real | mean |I_gen - I_real| |
|---|---|---|---|---|
| 0 | 9 | 0.2488 | 0.3871 | 0.1582 |
| 1 | 8 | 0.3098 | 0.3357 | 0.1420 |
| 2 | 6 | 0.1983 | 0.1614 | 0.0416 |
| 3 | 5 | 0.3705 | 0.4899 | 0.1351 |


## Collapse alarms

⚠️ **Not measured on this run** — the fit was reused from `model.pt`, which carries weights and config but never the `TrainHistory`. That is not the same as *did not fire*; nothing here can distinguish the two. Re-fit to get an alarm record.
## Provenance

```
{
  "source": "defaults",
  "selection_path": null
}
```

Training volume: 16527 cells x 28 genes over 4 sections, flattened=True. `use_umap=True`.

**One seed.** `specs/09` §3's repeated-seed rule requires `claim_min_seeds` = 3 for any measurement that reaches a paper claim. This table is a single-seed measurement — admissible as a diagnostic, not as a headline.

⚠️ **No envelope is quoted here, deliberately.** This footer used to cite the fixture's 0.0335 (`reports/envelope_synthetic.md`) and call smaller differences ties. `specs/10` §4.2a says an envelope is **per-metric and per-arm**, and A7 measured what carrying a fixture figure to real data costs: on this dataset the per-arm across-seed spreads run **0.0033 to 0.3687**, a **112x** range, with `morans_pearson`'s alone at **0.2684 — 8x** the fixture number. A tie declared against 0.0335 here would be the cross-dataset comparison §4.2a exists to forbid. Measure the envelope on **these** arms, on this metric, before calling anything a tie.
