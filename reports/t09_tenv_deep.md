# Repeated-seed verdict — `medcpt` vs `lookup` on `deep_starmap`

Holdout **`paper_2_4_6`**, measured under **{'expr_mode': 'zinb-flow'}**, 2400 steps, LOSO folds `section_3`, `section_5` (2), seeds [2, 3, 4] (3 of `Config.claim_min_seeds`=3). Margins are `medcpt` minus `lookup`; positive favours `medcpt`.

| metric | seed 2 | seed 3 | seed 4 | mean margin | **per-metric envelope** | vs it | margin's own seed spread | fold spread | verdict | was, vs pooled 0.0335 |
|---|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | -0.1108 | -0.1167 | -0.1358 | -0.1211 | **0.0246** | 4.9x | 0.0250 | 0.0169 | **STANDS** | STANDS |
| `gearys_pearson` | -0.1126 | -0.1022 | -0.1239 | -0.1129 | **0.0501** | 2.3x | 0.0217 | 0.0746 | **STANDS** | STANDS |
| `umap_mixing` | +0.0021 | +0.0029 | -0.0105 | -0.0018 | **0.0201** | 0.1x | 0.0134 | 0.0642 | not established | not established |
| `marker_field_r` | -0.0412 | -0.0291 | -0.0485 | -0.0396 | **0.0197** | 2.0x | 0.0194 | 0.1140 | not established | not established |
| `marker_depth_r` | -0.0205 | -0.0106 | +0.0139 | -0.0058 | **0.0427** | 0.1x | 0.0345 | 0.1274 | not established | not established |
| `celltype_localization` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | **0.0000** | infx | 0.0000 | 0.1430 | not established | not established |

### The envelope, per metric and per arm

R10's **0.0335** was one pooled figure, measured on the synthetic fixture, applied to all six metrics and both arms. Measured here on real data it is neither constant across metrics nor across arms:

| metric | `medcpt` across-seed spread | `lookup` across-seed spread | envelope used |
|---|---|---|---|
| `morans_pearson` | 0.0246 | 0.0073 | **0.0246** |
| `gearys_pearson` | 0.0398 | 0.0501 | **0.0501** |
| `umap_mixing` | 0.0067 | 0.0201 | **0.0201** |
| `marker_field_r` | 0.0152 | 0.0197 | **0.0197** |
| `marker_depth_r` | 0.0182 | 0.0427 | **0.0427** |
| `celltype_localization` | 0.0000 | 0.0000 | **0.0000** |

A **copying** arm barely uses the fitted weights, so its score is nearly seed-invariant; a **generative** arm's is not. The margin therefore inherits almost all of its seed variance from one side, which a pooled envelope cannot express.

**The four conditions.** A margin STANDS only if every seed agrees on the sign, the mean margin exceeds the spread of the margin itself across seeds, exceeds **that metric's own envelope** — the worse arm's across-seed spread from the table above, not R10's pooled 0.0335 — and exceeds the largest within-arm fold spread. Which condition failed, per metric:

| metric | signs agree | > seed spread | > envelope | > fold spread |
|---|---|---|---|---|
| `morans_pearson` | yes | yes | yes | yes |
| `gearys_pearson` | yes | yes | yes | yes |
| `umap_mixing` | **no** | **no** | **no** | **no** |
| `marker_field_r` | yes | yes | yes | **no** |
| `marker_depth_r` | **no** | **no** | **no** | **no** |
| `celltype_localization` | **no** | **no** | **no** | **no** |

`specs/09` §3 requires only that the spread be reported rather than a point estimate; the four conditions are this file's and are stricter, so a reader who disagrees with any one of them can recompute from the per-seed columns above.

**"Not established" is not "refuted."** A margin that fails a condition has not been shown; it may still be real and under-powered at this number of seeds and folds.
