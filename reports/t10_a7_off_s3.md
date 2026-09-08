# T09 on real data — shipped config, full calibration, tier-1 STARmap

Dataset `starmap_visual_cortex`, holdout `paper_2_4_6` (**tier 1**, `specs/10` §1). Config `8e0fcee60de7f6a0` from defaults, seed 3.

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
| `paper_morans_pearson` | +0.5024 | +0.5504 |
| `paper_gearys_pearson` | +0.5041 | +0.5516 |
| `paper_umap_mixing` | +0.8504 | +0.8281 |
| `paper_marker_field_r` | +0.5512 | +0.5488 |
| `paper_marker_depth_r` | +0.6973 | +0.7229 |
| `paper_celltype_localization` | +0.7410 | +0.7410 |
| `paper_gene_mean_spearman` | +0.9797 | +0.9551 |
| `paper_cell_count_ratio` (raw pass) | 0.988 | 0.988 |

Per section, matched density:

| arm | metric | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|---|
| `uncalibrated` | `morans_pearson` | +0.5261 | +0.5024 | +0.4765 | +0.5024 |
| `uncalibrated` | `gearys_pearson` | +0.5223 | +0.5041 | +0.4805 | +0.5041 |
| `uncalibrated` | `umap_mixing` | +0.8756 | +0.8330 | +0.8504 | +0.8504 |
| `uncalibrated` | `marker_field_r` | +0.5208 | +0.5610 | +0.5512 | +0.5512 |
| `uncalibrated` | `marker_depth_r` | +0.5380 | +0.7186 | +0.6973 | +0.6973 |
| `uncalibrated` | `celltype_localization` | +0.6964 | +0.7410 | +0.7672 | +0.7410 |
| `detection-calibrated` | `morans_pearson` | +0.6638 | +0.5504 | +0.5193 | +0.5504 |
| `detection-calibrated` | `gearys_pearson` | +0.6609 | +0.5516 | +0.5175 | +0.5516 |
| `detection-calibrated` | `umap_mixing` | +0.8662 | +0.8281 | +0.8040 | +0.8281 |
| `detection-calibrated` | `marker_field_r` | +0.5335 | +0.5553 | +0.5488 | +0.5488 |
| `detection-calibrated` | `marker_depth_r` | +0.7752 | +0.7229 | +0.7004 | +0.7229 |
| `detection-calibrated` | `celltype_localization` | +0.7008 | +0.7410 | +0.7858 | +0.7410 |

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
    "ell_xy": 56.47779858297338,
    "ell_z": 23.32015516343746,
    "status": "converged",
    "ell_z_status": "converged",
    "ell_fitted_xy": 116.29317396449403,
    "ell_fitted_z": 132.0,
    "i_gen": 0.23837240040302277,
    "i_target": 0.2532505840063095,
    "z_achieved": 0.9427450849137977,
    "z_target": 0.9570073884273124,
    "iterations": 3,
    "applied_ell_xy": 56.47779858297338,
    "applied_ell_z": 23.32015516343746
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
    "detection_mae_gen_vs_real": 0.0035723782432295465,
    "seconds": 6.9
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
| `ell_xy` | 56.5 um | converged |
| `ell_z` | 23.3 um | converged |
| fitted `ell` | 116.3 / 132.0 um | variogram |
| Moran's I | gen 0.2384 vs flanking 0.2533 | 3 iterations |
| between-section r | gen 0.9427 vs observed 0.9570 | R1 remedy 2 |

Derived `retrieval_z_window` = **3** spacings (largest section gap 22 um). `Config.retrieval_z_window` remains the fallback and the ablation handle.

## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)

One **global** `ell` is calibrated; this table says whether it serves every gene module equally. It is not a target, and a poor table is evidence for the per-channel-group escalation, which is a design change to be decided explicitly.

| module | genes | I_gen | I_real | |diff| |
|---|---|---|---|---|
| 0 | 9 | 0.2673 | 0.3871 | 0.1519 |
| 1 | 8 | 0.2472 | 0.3357 | 0.1465 |
| 2 | 6 | 0.1948 | 0.1614 | 0.0502 |
| 3 | 5 | 0.3672 | 0.4899 | 0.1227 |

## Provenance

```
{
  "source": "defaults",
  "selection_path": null
}
```

Training volume: 16527 cells x 28 genes over 4 sections, flattened=True. `use_umap=True`.

**One seed.** `specs/09` §3's repeated-seed rule requires `claim_min_seeds` = 3 for any measurement that reaches a paper claim, and the across-seed envelope measured on the fixture is 0.0335 (`reports/envelope_synthetic.md`). Any difference here smaller than that is a tie, and this table is a single-seed measurement — admissible as a diagnostic, not as a headline.
