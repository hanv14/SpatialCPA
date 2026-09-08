# T09 on real data — shipped config, full calibration, tier-1 STARmap

Dataset `starmap_visual_cortex`, holdout `paper_2_4_6` (**tier 1**, `specs/10` §1). Config `19cf1544f0cc5fcc` from defaults, seed 1.

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
| `paper_morans_pearson` | +0.2340 | +0.3316 |
| `paper_gearys_pearson` | +0.2357 | +0.3365 |
| `paper_umap_mixing` | +0.7976 | +0.7765 |
| `paper_marker_field_r` | +0.4472 | +0.4677 |
| `paper_marker_depth_r` | +0.6289 | +0.6431 |
| `paper_celltype_localization` | +0.7591 | +0.7601 |
| `paper_gene_mean_spearman` | +0.9825 | +0.9611 |
| `paper_cell_count_ratio` (raw pass) | 0.988 | 0.988 |

Per section, matched density:

| arm | metric | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|---|
| `uncalibrated` | `morans_pearson` | +0.2337 | +0.2340 | +0.2831 | +0.2340 |
| `uncalibrated` | `gearys_pearson` | +0.2357 | +0.2352 | +0.2801 | +0.2357 |
| `uncalibrated` | `umap_mixing` | +0.8311 | +0.7837 | +0.7976 | +0.7976 |
| `uncalibrated` | `marker_field_r` | +0.4472 | +0.4563 | +0.4446 | +0.4472 |
| `uncalibrated` | `marker_depth_r` | +0.5224 | +0.6559 | +0.6289 | +0.6289 |
| `uncalibrated` | `celltype_localization` | +0.6771 | +0.7591 | +0.7812 | +0.7591 |
| `detection-calibrated` | `morans_pearson` | +0.3670 | +0.2927 | +0.3316 | +0.3316 |
| `detection-calibrated` | `gearys_pearson` | +0.3699 | +0.2903 | +0.3365 | +0.3365 |
| `detection-calibrated` | `umap_mixing` | +0.8247 | +0.7765 | +0.7765 | +0.7765 |
| `detection-calibrated` | `marker_field_r` | +0.4519 | +0.4677 | +0.4763 | +0.4677 |
| `detection-calibrated` | `marker_depth_r` | +0.6563 | +0.6431 | +0.5816 | +0.6431 |
| `detection-calibrated` | `celltype_localization` | +0.6771 | +0.7601 | +0.7864 | +0.7601 |

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
    "ell_xy": 76.9831757532092,
    "ell_z": 48.019298229969614,
    "status": "converged",
    "ell_z_status": "converged",
    "ell_fitted_xy": 116.31226054120532,
    "ell_fitted_z": 132.0,
    "i_gen": 0.25465428829193115,
    "i_target": 0.2611113265156746,
    "z_achieved": 0.9500558174634642,
    "z_target": 0.9570073884273124,
    "iterations": 3,
    "applied_ell_xy": 76.9831757532092,
    "applied_ell_z": 48.019298229969614
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
    "detection_mae_gen_vs_real": 0.0026703540209965593,
    "seconds": 8.2
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
| `ell_xy` | 77.0 um | converged |
| `ell_z` | 48.0 um | converged |
| fitted `ell` | 116.3 / 132.0 um | variogram |
| Moran's I | gen 0.2547 vs flanking 0.2611 | 3 iterations |
| between-section r | gen 0.9501 vs observed 0.9570 | R1 remedy 2 |

Derived `retrieval_z_window` = **3** spacings (largest section gap 22 um). `Config.retrieval_z_window` remains the fallback and the ablation handle.

## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)

One **global** `ell` is calibrated; this table says whether it serves every gene module equally. It is not a target, and a poor table is evidence for the per-channel-group escalation, which is a design change to be decided explicitly.

| module | genes | I_gen | I_real | |diff| |
|---|---|---|---|---|
| 0 | 9 | 0.2556 | 0.3871 | 0.1694 |
| 1 | 8 | 0.3243 | 0.3357 | 0.1639 |
| 2 | 6 | 0.1968 | 0.1614 | 0.0364 |
| 3 | 5 | 0.3646 | 0.4899 | 0.1369 |

## Provenance

```
{
  "source": "defaults",
  "selection_path": null
}
```

Training volume: 16527 cells x 28 genes over 4 sections, flattened=True. `use_umap=True`.

**One seed.** `specs/09` §3's repeated-seed rule requires `claim_min_seeds` = 3 for any measurement that reaches a paper claim, and the across-seed envelope measured on the fixture is 0.0335 (`reports/envelope_synthetic.md`). Any difference here smaller than that is a tie, and this table is a single-seed measurement — admissible as a diagnostic, not as a headline.
