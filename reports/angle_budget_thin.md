## `merfish_thick_hypothalamus` — the angle budget

47189 cells, 4 sections, 9 types. Extent **1613 × 1884 × 170 µm**, section spacing **57.5 µm**, slab thickness **13.5 µm**.

The reference plane is centred on the **real section at z = 73.0 µm**, not on the volume's z-midpoint — see `reports/retractions.md` R1 for why that distinction cost a published reference row.

### IN-PLANE : DEPTH = **10.3 : 1**

That ratio is the whole constraint. GATE 2's synthetic fixture was 3000 µm across and
400 µm deep — **7.5 : 1** — and it is the only geometry oblique parity has ever been
measured on. A plane tilted by θ exits the thin dimension after `(D + t) / sin θ`.

| θ | cells in slab | donors | sections | scorable types | largest type | strip µm | aspect | clears |
|---|---|---|---|---|---|---|---|---|
| 0° | 11104 | 11104 | 1 | 9 | 4473 | 1878 | 0.858 | **yes** |
| 5° | 2997 | 2997 | 3 | 9 | 959 | 1450 | 0.902 | **yes** |
| 10° | 2242 | 2242 | 4 | 9 | 1001 | 1058 | 0.657 | **yes** |
| 15° | 1736 | 1736 | 4 | 8 | 797 | 709 | 0.441 | **yes** |
| 20° | 1363 | 1363 | 4 | 8 | 638 | 535 | 0.333 | **yes** |
| 30° | 940 | 940 | 4 | 7 | 351 | 364 | 0.227 | **yes** |
| 45° | 662 | 662 | 4 | 5 | 250 | 255 | 0.159 | no |
| 60° | 480 | 480 | 4 | 4 | 165 | 205 | 0.128 | no |
| 90° | 486 | 486 | 4 | 4 | 241 | 170 | 0.107 | no |

- **G1** — scorable types (≥ `min_gt_cells` = 20 cells) must be at least **60%** of the coronal plane's 9. *The fraction is mine; the cell count is the metric's.*
- **G2** — the largest type must have ≥ `max_n` = 250 cells, the metric's own subsample cap. Below it the strip sits under the design point of the statistic.
- **cells in slab** is what is available to *evaluate*; **donors** is what the layout would *reuse*. They differ only when the slab is empty — the generation setting, and the case the first version of the selection rule got wrong.

**Budget: 30°.**

First angle that fails: **45°** — 5 scorable types against the coronal plane's 9 — below 60% (G1)
