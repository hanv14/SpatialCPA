# Can `morans_pearson` measure anything on the held-out genes?

Dataset **`cosmx_nsclc_3d`**, holdout **`paper_2_4`** — 225981 cells x 960 genes. Split: **769 kept / 191 held out**, stratified 5x5 on mean expression x Moran's I (ranked on `section_5`), seed 7. 20 split-half repeats, 20 shuffles. **No model, no fit.**

| target | side | genes | markers | split-half R | **ceiling √R** | constant field | shuffled | best copy (context only) |
|---|---|---|---|---|---|---|---|---|
| `section_3` | `held_out` | 191 | 191 | +0.9910 | **0.9955** | +0.239959 | +0.0137 | +0.9806 |
| `section_3` | `kept` | 769 | 769 | +0.9938 | **0.9969** | +0.297287 | +0.0228 | +0.9698 |
| `section_5` | `held_out` | 191 | 191 | +0.9930 | **0.9965** | +0.232081 | +0.0397 | +0.9806 |
| `section_5` | `kept` | 769 | 769 | +0.9944 | **0.9972** | +0.292933 | -0.0165 | +0.9750 |

`self` is 1.000000 on every row or the run aborts — a check on this file, not a result.

### The two questions this decides

**1. Is the ceiling clear of the floor on the held-out genes?** Median √R **0.9960**; the largest **constant-field** referent is 0.2400 and the largest shuffled floor 0.0397, so the room available above the **usable** floor is **0.9563**. A zero-shot arm cannot copy — `cross-mix` reads the full count matrix and would emit the held-out genes verbatim, so it is excluded from the experiment rather than handicapped.

⚠️ For this metric the **constant field is degenerate and is not the floor.** It has exactly zero per-gene variance after normalisation, so Moran's I is `0/0` and what comes back is float32 round-off that scales with the gene's own magnitude — which is why it correlates with the real statistic at all. The floor here is the **shuffled** referent, and the room above is quoted against that.

**2. Is the split representative?** Held-out ceiling is **100%** of the kept genes' on the same sections. Far from 100% would mean the stratified draw still took systematically easier or harder genes, and a result on the held-out set would not generalise to the panel.

**`best copy` is context only.** It is what copying a whole real section scores on these genes, and it is reported so the numbers can be placed beside the reconstruction ceilings — **no arm in the zero-shot experiment may use it.**
