# T09's `layout_mode` gate, re-run on the grid sampler

Synthetic fixture, `alternating` holdout, internal LOSO over training sections, 2400 steps, seeds [1, 2, 3], `layout_sampler=grid`.

**One fit per seed.** `layout_mode` is read only at generation time, and fitting the
fixture at all three modes with one seed gives bitwise identical weights over all 96
parameter and buffer tensors — so the three arms below share one model and the contrast
carries no fit-to-fit noise. `text_emb_mode=lookup` (ablation A3): the encoder needs a
network. It is constant across arms and cannot affect the contrast.

## Median over seeds, per metric (higher is better)

| arm | `morans_pearson` | `gearys_pearson` | `umap_mixing` | `marker_field_r` | `marker_depth_r` | `celltype_localization` | median rank |
|---|---|---|---|---|---|---|---|
| `field` | +0.7292 | +0.6786 | +0.3032 | +0.0124 | +0.0505 | -0.0853 | **2.0** |
| `hybrid` | +0.7400 | +0.6794 | +0.3071 | +0.0052 | +0.0448 | -0.0864 | **2.0** |
| `resample` | +0.7394 | +0.6844 | +0.3025 | +0.0053 | +0.0433 | -0.0660 | **2.0** |

## Across-seed spread (max minus min over the seeds)

| arm | `morans_pearson` | `gearys_pearson` | `umap_mixing` | `marker_field_r` | `marker_depth_r` | `celltype_localization` | max |
|---|---|---|---|---|---|---|---|
| `field` | 0.0203 | 0.0082 | 0.0041 | 0.0251 | 0.0213 | 0.0241 | **0.0251** |
| `hybrid` | 0.0189 | 0.0408 | 0.0060 | 0.0456 | 0.0304 | 0.0267 | **0.0456** |
| `resample` | 0.0254 | 0.0279 | 0.0172 | 0.0394 | 0.0290 | 0.0000 | **0.0394** |

R10's measured across-seed envelope is **0.0335**. A gap below it is not a gap.

## Median rank per seed

| seed | `field` | `hybrid` | `resample` |
|---|---|---|---|
| 1 | 1.5 | 2.0 | 2.0 |
| 2 | 2.5 | 1.5 | 2.5 |
| 3 | 2.5 | 2.5 | 1.5 |

## Verdict, computed from the rows above

* **Each seed picks a different winner**: seed 1 -> `field`, seed 2 -> `hybrid`, seed 3 -> `resample`.
* Pooled median rank is a **3-way tie** at 2.0 (`field`, `hybrid`, `resample`).
* On **5 of 6** metrics the spread between the three arms is smaller than the arms' own spread across seeds — the gate is reading seed variation, not layout.
* `resample`'s `celltype_localization` has an across-seed spread of **exactly zero**, and that is a correctness check rather than a coincidence: `_resample_layout` copies the flanking section's coordinates and types unchanged, so the layout carries no seed, and this metric is a per-type field correlation over positions and type labels only. Its expression-driven metrics do vary across seeds, which is how you can tell the three fits really are different.

## Pairwise gaps against `resample`, per metric (median over seeds)

| metric | `field` less `resample` | `hybrid` less `resample` | vs envelope |
|---|---|---|---|
| `morans_pearson` | -0.0103 | +0.0005 | 0.3x |
| `gearys_pearson` | -0.0058 | -0.0050 | 0.2x |
| `umap_mixing` | +0.0007 | +0.0046 | 0.1x |
| `marker_field_r` | +0.0071 | -0.0001 | 0.2x |
| `marker_depth_r` | +0.0072 | +0.0015 | 0.2x |
| `celltype_localization` | -0.0193 | -0.0204 | 0.6x |
