# Can `marker_depth_r` measure anything on the held-out genes?

Dataset **`deep_starmap`**, holdout **`paper_2_4_6`** — 115830 cells x 1017 genes. Split: **813 kept / 204 held out**, stratified 5x5 on mean expression x Moran's I (ranked on `section_1`), seed 7. 20 split-half repeats, 20 shuffles. **No model, no fit.**

| target | side | genes | markers | split-half R | **ceiling √R** | constant field | shuffled | best copy (context only) |
|---|---|---|---|---|---|---|---|---|
| `section_3` | `held_out` | 204 | 32 | +0.9620 | **0.9808** | -0.018802 | +0.0120 | +0.9459 |
| `section_3` | `kept` | 813 | 32 | +0.9897 | **0.9948** | -0.039651 | +0.0078 | +0.9816 |
| `section_5` | `held_out` | 204 | 32 | +0.9678 | **0.9838** | +0.012384 | +0.0012 | +0.9499 |
| `section_5` | `kept` | 813 | 32 | +0.9874 | **0.9937** | -0.015606 | +0.0356 | +0.9719 |

`self` is 1.000000 on every row or the run aborts — a check on this file, not a result.

### The two questions this decides

**1. Is the ceiling clear of the floor on the held-out genes?** Median √R **0.9823**; the largest **constant-field** referent is 0.0188 and the largest shuffled floor 0.0120, so the room available above the **usable** floor is **0.9703**. A zero-shot arm cannot copy — `cross-mix` reads the full count matrix and would emit the held-out genes verbatim, so it is excluded from the experiment rather than handicapped.

⚠️ For this metric the **constant field is degenerate and is not the floor.** It has exactly zero per-gene variance after normalisation, so Moran's I is `0/0` and what comes back is float32 round-off that scales with the gene's own magnitude — which is why it correlates with the real statistic at all. The floor here is the **shuffled** referent, and the room above is quoted against that.

**2. Is the split representative?** Held-out ceiling is **99%** of the kept genes' on the same sections. Far from 100% would mean the stratified draw still took systematically easier or harder genes, and a result on the held-out set would not generalise to the panel.

**`best copy` is context only.** It is what copying a whole real section scores on these genes, and it is reported so the numbers can be placed beside the reconstruction ceilings — **no arm in the zero-shot experiment may use it.**
