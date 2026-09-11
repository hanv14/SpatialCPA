# The angle budget across every built specimen

**Free: no fit, no model, no generation.** For each dataset, real cells only, asking what
a plane tilted by each angle actually cuts through — and whether what it cuts is enough
for `celltype_localization` to score.

Gates are the metric's **own** constants: **G1** scorable types (≥ `min_gt_cells` = 20) ≥ 60% of the coronal plane's; **G2** largest type ≥ `max_n` = 250, its subsample cap. The 60% is the one number that is mine.

| dataset | cells | sections | extent x/y/z µm | **in-plane : depth** | **budget** | cells there | first failure |
|---|---|---|---|---|---|---|---|
| `starmap_visual_cortex` | 16527 | 4 | 1545 × 1301 × 66 | **21.6 : 1** | **5°** | 3211 | 10° — the largest type has 187 cells, under the metric's own max_n = 250 subsample cap (G2) |
| `deep_starmap` | 115830 | 4 | 4385 × 4155 × 125 | **34.1 : 1** | **5°** | 14209 | 10° — 59 scorable types against the coronal plane's 124 — below 60% (G1) |
| `merfish_thick_cortex` | 17467 | 4 | 2160 × 1950 × 83 | **24.8 : 1** | **5°** | 2791 | 10° — the largest type has 207 cells, under the metric's own max_n = 250 subsample cap (G2) |
| `merfish_thick_hypothalamus` | 47189 | 4 | 1613 × 1884 × 170 | **10.3 : 1** | **90°** | 1988 | clears every angle measured |
| `cosmx_nsclc_3d` | — | — | — | — | *not read* | — | the leakage-guarded training input not found at /data/han/projects/Spatial3D/benchmark-pbya-v3/results/_inputs/cosmx_nsclc_3d/paper_2_4_6/train_registered.h5ad. Set it with --input or BENCH_V3_RESULTS (build it with `python -m bench3.selftest`). Nothing is guessed: a scored number has to say which tree it came from. |
| `exseq_breast_cancer` | — | — | — | — | *not read* | — | the leakage-guarded training input not found at /data/han/projects/Spatial3D/benchmark-pbya-v3/results/_inputs/exseq_breast_cancer/paper_2_4_6/train_registered.h5ad. Set it with --input or BENCH_V3_RESULTS (build it with `python -m bench3.selftest`). Nothing is guessed: a scored number has to say which tree it came from. |
| `exseq_visual_cortex` | — | — | — | — | *not read* | — | the leakage-guarded training input not found at /data/han/projects/Spatial3D/benchmark-pbya-v3/results/_inputs/exseq_visual_cortex/paper_2_4_6/train_registered.h5ad. Set it with --input or BENCH_V3_RESULTS (build it with `python -m bench3.selftest`). Nothing is guessed: a scored number has to say which tree it came from. |
| `allen_merfish_brain` | — | — | — | — | *not read* | — | the leakage-guarded training input not found at /data/han/projects/Spatial3D/benchmark-pbya-v3/results/_inputs/allen_merfish_brain/paper_2_4_6/train_registered.h5ad. Set it with --input or BENCH_V3_RESULTS (build it with `python -m bench3.selftest`). Nothing is guessed: a scored number has to say which tree it came from. |

## What this decides

**`merfish_thick_hypothalamus` clears 90°** with 1988 cells, at an in-plane : depth ratio of 10.3 : 1. The oblique demonstration is **scored** rather
than shown, and the paper's claim is a measured one.

⚠️ **A budget is not a result.** It says what a specimen permits, not what the method
achieves at that angle. `reports/oblique_layout_cost.md` §3c: there is no real oblique
section to score against, so the evaluation set is the real cells near the plane — which
are also the donors, and must be excluded from them.

---

## `starmap_visual_cortex` — the angle budget

16527 cells, 4 sections, 19 types. Extent **1545 × 1301 × 66 µm**, section spacing **22.0 µm**, slab thickness **22.0 µm**.

The reference plane is centred on the **real section at z = 41.0 µm**, not on the volume's z-midpoint — see `reports/retractions.md` R1 for why that distinction cost a published reference row.

### IN-PLANE : DEPTH = **21.6 : 1**

That ratio is the whole constraint. GATE 2's synthetic fixture was 3000 µm across and
400 µm deep — **7.5 : 1** — and it is the only geometry oblique parity has ever been
measured on. A plane tilted by θ exits the thin dimension after `(D + t) / sin θ`.

| θ | cells in slab | donors | sections | scorable types | largest type | strip µm | aspect | clears |
|---|---|---|---|---|---|---|---|---|
| 0° | 4169 | 4169 | 1 | 19 | 456 | 1301 | 0.859 | **yes** |
| 5° | 3211 | 3211 | 4 | 19 | 337 | 1008 | 0.715 | **yes** |
| 10° | 1669 | 1669 | 4 | 17 | 187 | 504 | 0.326 | no |
| 15° | 1134 | 1134 | 4 | 17 | 117 | 337 | 0.218 | no |
| 20° | 889 | 889 | 4 | 13 | 99 | 252 | 0.163 | no |
| 30° | 567 | 567 | 4 | 10 | 65 | 169 | 0.127 | no |
| 45° | 441 | 441 | 4 | 9 | 50 | 115 | 0.086 | no |
| 60° | 355 | 355 | 4 | 8 | 38 | 88 | 0.066 | no |
| 90° | 304 | 304 | 4 | 7 | 39 | 66 | 0.050 | no |

- **G1** — scorable types (≥ `min_gt_cells` = 20 cells) must be at least **60%** of the coronal plane's 19. *The fraction is mine; the cell count is the metric's.*
- **G2** — the largest type must have ≥ `max_n` = 250 cells, the metric's own subsample cap. Below it the strip sits under the design point of the statistic.
- **cells in slab** is what is available to *evaluate*; **donors** is what the layout would *reuse*. They differ only when the slab is empty — the generation setting, and the case the first version of the selection rule got wrong.

**Budget: 5°.**

First angle that fails: **10°** — the largest type has 187 cells, under the metric's own max_n = 250 subsample cap (G2)

## `deep_starmap` — the angle budget

115830 cells, 4 sections, 137 types. Extent **4385 × 4155 × 125 µm**, section spacing **42.0 µm**, slab thickness **42.0 µm**.

The reference plane is centred on the **real section at z = 52.5 µm**, not on the volume's z-midpoint — see `reports/retractions.md` R1 for why that distinction cost a published reference row.

### IN-PLANE : DEPTH = **34.1 : 1**

That ratio is the whole constraint. GATE 2's synthetic fixture was 3000 µm across and
400 µm deep — **7.5 : 1** — and it is the only geometry oblique parity has ever been
measured on. A plane tilted by θ exits the thin dimension after `(D + t) / sin θ`.

| θ | cells in slab | donors | sections | scorable types | largest type | strip µm | aspect | clears |
|---|---|---|---|---|---|---|---|---|
| 0° | 29842 | 29842 | 1 | 124 | 3101 | 4154 | 0.954 | **yes** |
| 5° | 14209 | 14209 | 4 | 99 | 2014 | 1917 | 0.467 | **yes** |
| 10° | 6873 | 6873 | 4 | 59 | 1248 | 960 | 0.239 | no |
| 15° | 4654 | 4654 | 4 | 45 | 870 | 640 | 0.160 | no |
| 20° | 3602 | 3602 | 4 | 37 | 686 | 481 | 0.121 | no |
| 30° | 2502 | 2502 | 4 | 26 | 514 | 322 | 0.082 | no |
| 45° | 1774 | 1774 | 4 | 20 | 379 | 217 | 0.056 | no |
| 60° | 1504 | 1504 | 4 | 18 | 316 | 169 | 0.043 | no |
| 90° | 1309 | 1309 | 4 | 14 | 267 | 125 | 0.032 | no |

- **G1** — scorable types (≥ `min_gt_cells` = 20 cells) must be at least **60%** of the coronal plane's 124. *The fraction is mine; the cell count is the metric's.*
- **G2** — the largest type must have ≥ `max_n` = 250 cells, the metric's own subsample cap. Below it the strip sits under the design point of the statistic.
- **cells in slab** is what is available to *evaluate*; **donors** is what the layout would *reuse*. They differ only when the slab is empty — the generation setting, and the case the first version of the selection rule got wrong.

**Budget: 5°.**

First angle that fails: **10°** — 59 scorable types against the coronal plane's 124 — below 60% (G1)

## `merfish_thick_cortex` — the angle budget

17467 cells, 4 sections, 24 types. Extent **2160 × 1950 × 83 µm**, section spacing **27.5 µm**, slab thickness **27.5 µm**.

The reference plane is centred on the **real section at z = 37.0 µm**, not on the volume's z-midpoint — see `reports/retractions.md` R1 for why that distinction cost a published reference row.

### IN-PLANE : DEPTH = **24.8 : 1**

That ratio is the whole constraint. GATE 2's synthetic fixture was 3000 µm across and
400 µm deep — **7.5 : 1** — and it is the only geometry oblique parity has ever been
measured on. A plane tilted by θ exits the thin dimension after `(D + t) / sin θ`.

| θ | cells in slab | donors | sections | scorable types | largest type | strip µm | aspect | clears |
|---|---|---|---|---|---|---|---|---|
| 0° | 3875 | 3875 | 1 | 19 | 672 | 1949 | 0.902 | **yes** |
| 5° | 2791 | 2791 | 4 | 19 | 533 | 1266 | 0.600 | **yes** |
| 10° | 1386 | 1386 | 4 | 16 | 207 | 633 | 0.391 | no |
| 15° | 926 | 926 | 4 | 14 | 124 | 423 | 0.271 | no |
| 20° | 701 | 701 | 4 | 12 | 105 | 317 | 0.210 | no |
| 30° | 486 | 486 | 4 | 8 | 79 | 213 | 0.145 | no |
| 45° | 340 | 340 | 4 | 8 | 50 | 144 | 0.100 | no |
| 60° | 265 | 265 | 4 | 4 | 47 | 111 | 0.078 | no |
| 90° | 232 | 232 | 4 | 3 | 40 | 83 | 0.059 | no |

- **G1** — scorable types (≥ `min_gt_cells` = 20 cells) must be at least **60%** of the coronal plane's 19. *The fraction is mine; the cell count is the metric's.*
- **G2** — the largest type must have ≥ `max_n` = 250 cells, the metric's own subsample cap. Below it the strip sits under the design point of the statistic.
- **cells in slab** is what is available to *evaluate*; **donors** is what the layout would *reuse*. They differ only when the slab is empty — the generation setting, and the case the first version of the selection rule got wrong.

**Budget: 5°.**

First angle that fails: **10°** — the largest type has 207 cells, under the metric's own max_n = 250 subsample cap (G2)

## `merfish_thick_hypothalamus` — the angle budget

47189 cells, 4 sections, 9 types. Extent **1613 × 1884 × 170 µm**, section spacing **57.5 µm**, slab thickness **57.5 µm**.

The reference plane is centred on the **real section at z = 73.0 µm**, not on the volume's z-midpoint — see `reports/retractions.md` R1 for why that distinction cost a published reference row.

### IN-PLANE : DEPTH = **10.3 : 1**

That ratio is the whole constraint. GATE 2's synthetic fixture was 3000 µm across and
400 µm deep — **7.5 : 1** — and it is the only geometry oblique parity has ever been
measured on. A plane tilted by θ exits the thin dimension after `(D + t) / sin θ`.

| θ | cells in slab | donors | sections | scorable types | largest type | strip µm | aspect | clears |
|---|---|---|---|---|---|---|---|---|
| 0° | 11104 | 11104 | 1 | 9 | 4473 | 1878 | 0.858 | **yes** |
| 5° | 12234 | 12234 | 3 | 9 | 5128 | 1885 | 0.855 | **yes** |
| 10° | 9392 | 9392 | 4 | 9 | 4462 | 1308 | 0.812 | **yes** |
| 15° | 6921 | 6921 | 4 | 9 | 3420 | 873 | 0.541 | **yes** |
| 20° | 5455 | 5455 | 4 | 9 | 2556 | 656 | 0.407 | **yes** |
| 30° | 3874 | 3874 | 4 | 9 | 1441 | 440 | 0.273 | **yes** |
| 45° | 2583 | 2583 | 4 | 9 | 815 | 298 | 0.185 | **yes** |
| 60° | 2140 | 2140 | 4 | 9 | 735 | 230 | 0.143 | **yes** |
| 90° | 1988 | 1988 | 4 | 8 | 931 | 170 | 0.106 | **yes** |

- **G1** — scorable types (≥ `min_gt_cells` = 20 cells) must be at least **60%** of the coronal plane's 9. *The fraction is mine; the cell count is the metric's.*
- **G2** — the largest type must have ≥ `max_n` = 250 cells, the metric's own subsample cap. Below it the strip sits under the design point of the statistic.
- **cells in slab** is what is available to *evaluate*; **donors** is what the layout would *reuse*. They differ only when the slab is empty — the generation setting, and the case the first version of the selection rule got wrong.

**Budget: 90°.**
