# T09 on real data — shipped config, full calibration, tier-1 STARmap

Dataset `starmap_visual_cortex`, holdout `paper_2_4_6` (**tier 1**, `specs/10` §1). Config `9f245ba6d4dcb155` from defaults, seed 3.

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
| `paper_morans_pearson` | -0.0748 | +0.1029 |
| `paper_gearys_pearson` | -0.0659 | +0.0309 |
| `paper_umap_mixing` | +0.4905 | +0.4864 |
| `paper_marker_field_r` | +0.0962 | +0.0412 |
| `paper_marker_depth_r` | +0.0198 | +0.0194 |
| `paper_celltype_localization` | +0.3149 | +0.0000 |
| `paper_gene_mean_spearman` | +0.6393 | +0.7406 |
| `paper_cell_count_ratio` (raw pass) | 0.988 | 0.988 |

Per section, matched density:

| arm | metric | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|---|
| `uncalibrated` | `morans_pearson` | -0.0748 | -0.0184 | -0.1304 | -0.0748 |
| `uncalibrated` | `gearys_pearson` | -0.0877 | +0.0229 | -0.0659 | -0.0659 |
| `uncalibrated` | `umap_mixing` | +0.4603 | +0.5085 | +0.4905 | +0.4905 |
| `uncalibrated` | `marker_field_r` | +0.0698 | +0.0962 | +0.1139 | +0.0962 |
| `uncalibrated` | `marker_depth_r` | +0.0528 | -0.0399 | +0.0198 | +0.0198 |
| `uncalibrated` | `celltype_localization` | +0.0000 | +0.3149 | +0.6252 | +0.3149 |
| `detection-calibrated` | `morans_pearson` | +0.1029 | +0.2637 | -0.2009 | +0.1029 |
| `detection-calibrated` | `gearys_pearson` | +0.0309 | +0.2291 | -0.2613 | +0.0309 |
| `detection-calibrated` | `umap_mixing` | +0.4463 | +0.4864 | +0.4993 | +0.4864 |
| `detection-calibrated` | `marker_field_r` | +0.0877 | +0.0412 | +0.0358 | +0.0412 |
| `detection-calibrated` | `marker_depth_r` | +0.0194 | +0.1998 | -0.0658 | +0.0194 |
| `detection-calibrated` | `celltype_localization` | +0.0471 | +0.0000 | +0.0000 | +0.0000 |

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
    "ell_xy": 163.27045560586424,
    "ell_z": 98.87811579032706,
    "status": "target_unreachable",
    "ell_z_status": "target_unreachable",
    "ell_fitted_xy": 116.29317396449403,
    "ell_fitted_z": 132.0,
    "i_gen": 0.003422762732952833,
    "i_target": 0.2532505840063095,
    "z_achieved": 0.85223466339149,
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
    "detection_mae_gen_vs_real": 0.00591159694602831,
    "seconds": 7.5
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
| `ell_xy` | 163.3 um | target_unreachable |
| `ell_z` | 98.9 um | target_unreachable (bound, not a fit — R1) |
| fitted `ell` | 116.3 / 132.0 um | variogram |
| Moran's I | gen 0.0034 vs flanking 0.2533 | 0 iterations |
| between-section r | gen 0.8522 vs observed 0.9570 | R1 remedy 2 |

Derived `retrieval_z_window` = **3** spacings (largest section gap 22 um). `Config.retrieval_z_window` remains the fallback and the ablation handle.

## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)

One **global** `ell` is calibrated; this table says whether it serves every gene module equally. It is not a target, and a poor table is evidence for the per-channel-group escalation, which is a design change to be decided explicitly.

| module | genes | I_gen | I_real | |diff| |
|---|---|---|---|---|
| 0 | 9 | 0.0014 | 0.3871 | 0.3857 |
| 1 | 8 | -0.0028 | 0.3357 | 0.3385 |
| 2 | 6 | 0.0008 | 0.1614 | 0.1606 |
| 3 | 5 | -0.0013 | 0.4899 | 0.4912 |

## Provenance

```
{
  "source": "defaults",
  "selection_path": null
}
```

Training volume: 16527 cells x 28 genes over 4 sections, flattened=True. `use_umap=True`.

**One seed.** `specs/09` §3's repeated-seed rule requires `claim_min_seeds` = 3 for any measurement that reaches a paper claim, and the across-seed envelope measured on the fixture is 0.0335 (`reports/envelope_synthetic.md`). Any difference here smaller than that is a tie, and this table is a single-seed measurement — admissible as a diagnostic, not as a headline.
