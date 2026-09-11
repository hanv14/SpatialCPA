## `merfish_thick_hypothalamus` — the angle budget

47189 cells, 4 sections, 9 types. Extent **1613 × 1884 × 170 µm**, section spacing **57.5 µm**, slab thickness **57.5 µm**.

The reference plane is centred on the **real section at z = 73.0 µm**, not on the volume's z-midpoint — see `reports/retractions.md` R1 for why that distinction cost a published reference row.

### IN-PLANE : DEPTH = **10.3 : 1**

That ratio is the whole constraint. GATE 2's synthetic fixture was 3000 µm across and
400 µm deep — **7.5 : 1** — and it is the only geometry oblique parity has ever been
measured on. A plane tilted by θ exits the thin dimension after `(D + t) / sin θ`.

| θ | cells in slab | donors | strata | **fill** | scorable types | largest type | strip µm | clears |
|---|---|---|---|---|---|---|---|---|
| 0° | 11104 | 11104 | 1 | **1.00** | 9 | 4473 | 1878 | **yes** |
| 5° | 12234 | 12234 | 3 | **1.00** | 9 | 5128 | 1885 | **yes** |
| 10° | 9392 | 9392 | 4 | **0.98** | 9 | 4462 | 1308 | **yes** |
| 15° | 6921 | 6921 | 4 | **0.97** | 9 | 3420 | 873 | **yes** |
| 20° | 5455 | 5455 | 4 | **0.94** | 9 | 2556 | 656 | **yes** |
| 30° | 3874 | 3874 | 4 | **0.87** | 9 | 1441 | 440 | **yes** |
| 45° | 2583 | 2583 | 4 | **0.71** | 9 | 815 | 298 | **yes** |
| 60° | 2140 | 2140 | 4 | **0.50** | 9 | 735 | 230 | **yes** |
| 90° | 1988 | 1988 | 4 | **0.00** | 8 | 931 | 170 | **yes** |

- **G1** — scorable types (≥ `min_gt_cells` = 20 cells) must be at least **60%** of the coronal plane's 9. *The fraction is mine; the cell count is the metric's.*
- **G2** — the largest type must have ≥ `max_n` = 250 cells, the metric's own subsample cap. Below it the strip sits under the design point of the statistic.
- **cells in slab** is what is available to *evaluate*; **donors** is what the layout would *reuse*. They differ only when the slab is empty — the generation setting, and the case the first version of the selection rule got wrong.
- **fill** = `t·cos θ / s` — the fraction of the oblique plane the real cells can cover. **Not a gate, and a limit on the field rather than on this method**: an oblique ground truth drawn from `N` serial sections has only `N` samples along depth, so at 90° it is `N` parallel lines whatever generated it. See `reports/the_comb_limit.md`. Arms that reproduce real cells are combs too and compare like with like; an arm that generates a continuous fill does not, and is not comparable where this is small.
- **thickness** came from the volume's median section spacing -- Section.thickness is assumed on every section, so the file carries no measured slab thickness. On a leakage-guarded input this OVERSTATES the slab: held-out sections are removed, so the spacing between the ones that remain is a multiple of the real pitch.

**Budget: 90°.**
