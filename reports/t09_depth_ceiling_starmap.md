# `marker_depth_r` — the metric's ceiling and floor, measured from data alone

Dataset **`starmap_visual_cortex`**, holdout **`paper_2_4_6`** — 16527 cells x 28 genes over 4 training sections. 32 marker genes per target, 24 depth bins, 20 split-half repeats, 20 shuffles, seed 1. **No model, no fit, no generation** — the metric's own kernels applied to the built input.

| target | cells | genes | self | split-half R | **noiseless ceiling sqrt(R)** | nearest other section | best other section | mean other | shuffled (floor) |
|---|---|---|---|---|---|---|---|---|---|
| `section_1` | 4073 | 28 | 1.000000 | +0.8704 | **0.9330** | +0.7408 | +0.7408 | +0.6699 | +0.0144 |
| `section_3` | 4169 | 28 | 1.000000 | +0.8784 | **0.9372** | +0.7434 | +0.7791 | +0.7442 | +0.0001 |
| `section_5` | 4110 | 28 | 1.000000 | +0.8815 | **0.9389** | +0.7839 | +0.7839 | +0.7286 | -0.0127 |
| `section_7` | 4175 | 28 | 1.000000 | +0.8594 | **0.9270** | +0.6989 | +0.7134 | +0.6702 | -0.0367 |

`self` must be exactly 1.0; it is a correctness check on the profile code in `scripts/t09_depth_ceiling.py`, not a result. The run aborts if it is not.

### Every donor section, by distance

| target | donor | \|dz\| (um) | `marker_depth_r` |
|---|---|---|---|
| `section_1` | `section_3` | 22 | +0.7408 |
| `section_1` | `section_5` | 44 | +0.6666 |
| `section_1` | `section_7` | 66 | +0.6024 |
| `section_3` | `section_1` | 22 | +0.7434 |
| `section_3` | `section_5` | 22 | +0.7791 |
| `section_3` | `section_7` | 44 | +0.7100 |
| `section_5` | `section_3` | 22 | +0.7839 |
| `section_5` | `section_7` | 22 | +0.7227 |
| `section_5` | `section_1` | 44 | +0.6793 |
| `section_7` | `section_5` | 22 | +0.6989 |
| `section_7` | `section_3` | 44 | +0.7134 |
| `section_7` | `section_1` | 66 | +0.5984 |

### How to read this against the audits

`starmap_visual_cortex`'s `expr_mode` audit (`reports/t09_audit_expr_mode.json`) put `cross-mix` at **+0.7256**  `zinb-flow` at **+0.6292** on this metric. Place them against the columns above:

* **noiseless ceiling `sqrt(R)`** is the hard bound: even a method with no noise of its own cannot correlate with the observed target above this, because the target is itself a finite-sample measurement. **split-half R** is the softer bound, for a method as noisy as the data.
* **best other section** is the copying ceiling: the most a donor-copying method could score if it copied a whole real section. `cross-mix` copies donor by donor, so it is competing against this number, not against 1.0.
* **shuffled** is the floor. An arm at the floor is not modelling depth at all.

If the copying ceiling is near zero, `cross-mix`'s score is not a model failure — no donor-copying method could have done better, and the `expr_mode` margin on this metric says nothing about the flow head. If the copying ceiling is high and `cross-mix` is at the floor, the deficit is `cross-mix`'s donor retrieval, still not the flow head's modelling. Only a high ceiling *with* `zinb-flow` well above the copying ceiling makes the margin a statement about the generative path.

**One caveat on the split-half.** Each half carries half the cells, and Spearman-Brown corrects for that only under assumptions this profile does not exactly meet. On a section with few cells the correction under-estimates R, which is why the per-target cell count is printed beside it.
