# T09 on real data — shipped config, full calibration, tier-1 STARmap

Dataset `starmap_visual_cortex`, holdout `paper_2_4_6` (**tier 1**, `specs/10` §1). Config `ef5b44cd4bd5e260` from defaults, seed 2.

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
| `paper_morans_pearson` | +0.1375 | +0.2738 |
| `paper_gearys_pearson` | +0.1135 | +0.2593 |
| `paper_umap_mixing` | +0.5537 | +0.5444 |
| `paper_marker_field_r` | +0.1412 | +0.1325 |
| `paper_marker_depth_r` | +0.3366 | +0.0885 |
| `paper_celltype_localization` | +0.5359 | +0.1441 |
| `paper_gene_mean_spearman` | +0.7586 | +0.7444 |
| `paper_cell_count_ratio` (raw pass) | 0.988 | 0.988 |

Per section, matched density:

| arm | metric | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|---|
| `uncalibrated` | `morans_pearson` | +0.1375 | +0.2191 | -0.1521 | +0.1375 |
| `uncalibrated` | `gearys_pearson` | +0.1135 | +0.2113 | -0.1319 | +0.1135 |
| `uncalibrated` | `umap_mixing` | +0.5537 | +0.5596 | +0.5414 | +0.5537 |
| `uncalibrated` | `marker_field_r` | +0.1505 | +0.1412 | +0.0960 | +0.1412 |
| `uncalibrated` | `marker_depth_r` | +0.3582 | +0.3366 | +0.3017 | +0.3366 |
| `uncalibrated` | `celltype_localization` | +0.5359 | +0.5367 | +0.5168 | +0.5359 |
| `detection-calibrated` | `morans_pearson` | +0.4016 | +0.2738 | +0.1500 | +0.2738 |
| `detection-calibrated` | `gearys_pearson` | +0.4125 | +0.2593 | +0.1678 | +0.2593 |
| `detection-calibrated` | `umap_mixing` | +0.5444 | +0.5660 | +0.5357 | +0.5444 |
| `detection-calibrated` | `marker_field_r` | +0.1252 | +0.1325 | +0.1794 | +0.1325 |
| `detection-calibrated` | `marker_depth_r` | +0.0513 | +0.0885 | +0.3997 | +0.0885 |
| `detection-calibrated` | `celltype_localization` | +0.0265 | +0.4721 | +0.1441 | +0.1441 |

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
    "ell_xy": 20.09264301728191,
    "ell_z": 41.56032491930344,
    "status": "target_unreachable",
    "ell_z_status": "target_unreachable",
    "ell_fitted_xy": 125.62451351795204,
    "ell_fitted_z": 132.0,
    "i_gen": 0.006126999389380217,
    "i_target": 0.2577052041888237,
    "z_achieved": 0.8789381422389275,
    "z_target": 0.9570073884273124,
    "iterations": 0,
    "applied_ell_xy": 100.0,
    "applied_ell_z": 100.0
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
    "detection_mae_gen_vs_real": 0.006072830738211168,
    "seconds": 6.7
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
| `ell_xy` | 20.1 um | target_unreachable |
| `ell_z` | 41.6 um | target_unreachable (bound, not a fit — R1) |
| fitted `ell` | 125.6 / 132.0 um | variogram |
| Moran's I | gen 0.0061 vs flanking 0.2577 | 0 iterations |
| between-section r | gen 0.8789 vs observed 0.9570 | R1 remedy 2 |

Derived `retrieval_z_window` = **3** spacings (largest section gap 22 um). `Config.retrieval_z_window` remains the fallback and the ablation handle.

## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)

One **global** `ell` is calibrated; this table says whether it serves every gene module equally. It is not a target, and a poor table is evidence for the per-channel-group escalation, which is a design change to be decided explicitly.

| module | genes | I_gen | I_real | |diff| |
|---|---|---|---|---|
| 0 | 9 | 0.0059 | 0.3871 | 0.3812 |
| 1 | 8 | 0.0037 | 0.3357 | 0.3320 |
| 2 | 6 | -0.0007 | 0.1614 | 0.1621 |
| 3 | 5 | 0.0012 | 0.4899 | 0.4887 |

## Provenance

```
{
  "source": "defaults",
  "selection_path": null
}
```

Training volume: 16527 cells x 28 genes over 4 sections, flattened=True. `use_umap=True`.

**One seed.** `specs/09` §3's repeated-seed rule requires `claim_min_seeds` = 3 for any measurement that reaches a paper claim, and the across-seed envelope measured on the fixture is 0.0335 (`reports/envelope_synthetic.md`). Any difference here smaller than that is a tie, and this table is a single-seed measurement — admissible as a diagnostic, not as a headline.
