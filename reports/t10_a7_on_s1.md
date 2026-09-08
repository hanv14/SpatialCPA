# T09 on real data — shipped config, full calibration, tier-1 STARmap

Dataset `starmap_visual_cortex`, holdout `paper_2_4_6` (**tier 1**, `specs/10` §1). Config `9209091d5d27d668` from defaults, seed 1.

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
| `paper_morans_pearson` | +0.2858 | +0.0704 |
| `paper_gearys_pearson` | +0.3028 | -0.0208 |
| `paper_umap_mixing` | +0.5090 | +0.4922 |
| `paper_marker_field_r` | +0.1171 | +0.0948 |
| `paper_marker_depth_r` | +0.2154 | +0.1114 |
| `paper_celltype_localization` | +0.4167 | +0.6083 |
| `paper_gene_mean_spearman` | +0.6546 | +0.7033 |
| `paper_cell_count_ratio` (raw pass) | 0.988 | 0.988 |

Per section, matched density:

| arm | metric | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|---|
| `uncalibrated` | `morans_pearson` | +0.3408 | +0.2858 | +0.1889 | +0.2858 |
| `uncalibrated` | `gearys_pearson` | +0.3228 | +0.3028 | +0.1797 | +0.3028 |
| `uncalibrated` | `umap_mixing` | +0.4665 | +0.5090 | +0.5188 | +0.5090 |
| `uncalibrated` | `marker_field_r` | +0.1805 | +0.0811 | +0.1171 | +0.1171 |
| `uncalibrated` | `marker_depth_r` | +0.4901 | +0.1240 | +0.2154 | +0.2154 |
| `uncalibrated` | `celltype_localization` | +0.4167 | +0.6700 | +0.0711 | +0.4167 |
| `detection-calibrated` | `morans_pearson` | +0.0192 | +0.2366 | +0.0704 | +0.0704 |
| `detection-calibrated` | `gearys_pearson` | -0.0208 | +0.2089 | -0.0485 | -0.0208 |
| `detection-calibrated` | `umap_mixing` | +0.4596 | +0.4922 | +0.5127 | +0.4922 |
| `detection-calibrated` | `marker_field_r` | +0.1154 | +0.0544 | +0.0948 | +0.0948 |
| `detection-calibrated` | `marker_depth_r` | +0.1114 | +0.0529 | +0.1678 | +0.1114 |
| `detection-calibrated` | `celltype_localization` | +0.6083 | +0.4324 | +0.7255 | +0.6083 |

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
    "ell_xy": 56.48369711544424,
    "ell_z": 31.131868330715914,
    "status": "target_unreachable",
    "ell_z_status": "target_unreachable",
    "ell_fitted_xy": 116.31226054120532,
    "ell_fitted_z": 132.0,
    "i_gen": 0.004210726357996464,
    "i_target": 0.2611113265156746,
    "z_achieved": 0.8379970548992737,
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
    "detection_mae_gen_vs_real": 0.00602085110380112,
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
| `ell_xy` | 56.5 um | target_unreachable |
| `ell_z` | 31.1 um | target_unreachable (bound, not a fit — R1) |
| fitted `ell` | 116.3 / 132.0 um | variogram |
| Moran's I | gen 0.0042 vs flanking 0.2611 | 0 iterations |
| between-section r | gen 0.8380 vs observed 0.9570 | R1 remedy 2 |

Derived `retrieval_z_window` = **3** spacings (largest section gap 22 um). `Config.retrieval_z_window` remains the fallback and the ablation handle.

## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)

One **global** `ell` is calibrated; this table says whether it serves every gene module equally. It is not a target, and a poor table is evidence for the per-channel-group escalation, which is a design change to be decided explicitly.

| module | genes | I_gen | I_real | |diff| |
|---|---|---|---|---|
| 0 | 9 | -0.0012 | 0.3871 | 0.3883 |
| 1 | 8 | 0.0013 | 0.3357 | 0.3345 |
| 2 | 6 | 0.0005 | 0.1614 | 0.1609 |
| 3 | 5 | 0.0057 | 0.4899 | 0.4842 |

## Provenance

```
{
  "source": "defaults",
  "selection_path": null
}
```

Training volume: 16527 cells x 28 genes over 4 sections, flattened=True. `use_umap=True`.

**One seed.** `specs/09` §3's repeated-seed rule requires `claim_min_seeds` = 3 for any measurement that reaches a paper claim, and the across-seed envelope measured on the fixture is 0.0335 (`reports/envelope_synthetic.md`). Any difference here smaller than that is a tie, and this table is a single-seed measurement — admissible as a diagnostic, not as a headline.
