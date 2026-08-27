# `marker_depth_r` — the metric's ceiling and floor, measured from data alone

Dataset **`deep_starmap`**, holdout **`paper_2_4_6`** — 115830 cells x 1017 genes over 4 training sections. 32 marker genes per target, 24 depth bins, 20 split-half repeats, 20 shuffles, seed 1. **No model, no fit, no generation** — the metric's own kernels applied to the built input.

| target | cells | genes | self | split-half R | **noiseless ceiling sqrt(R)** | nearest other section | best other section | mean other | shuffled (floor) |
|---|---|---|---|---|---|---|---|---|---|
| `section_1` | 39327 | 32 | 1.000000 | +0.9914 | **0.9957** | +0.9859 | +0.9859 | +0.9028 | -0.0217 |
| `section_3` | 29842 | 32 | 1.000000 | +0.9924 | **0.9962** | +0.9631 | +0.9861 | +0.9167 | +0.0036 |
| `section_5` | 28654 | 32 | 1.000000 | +0.9900 | **0.9950** | +0.8578 | +0.9739 | +0.9298 | +0.0304 |
| `section_7` | 18007 | 32 | 1.000000 | +0.9871 | **0.9935** | +0.9293 | +0.9293 | +0.9010 | +0.0029 |

`self` must be exactly 1.0; it is a correctness check on the profile code in `scripts/t09_depth_ceiling.py`, not a result. The run aborts if it is not.

### Every donor section, by distance

| target | donor | \|dz\| (um) | `marker_depth_r` |
|---|---|---|---|
| `section_1` | `section_3` | 43 | +0.9859 |
| `section_1` | `section_5` | 85 | +0.9426 |
| `section_1` | `section_7` | 125 | +0.7799 |
| `section_3` | `section_5` | 42 | +0.9631 |
| `section_3` | `section_1` | 43 | +0.9861 |
| `section_3` | `section_7` | 83 | +0.8009 |
| `section_5` | `section_7` | 41 | +0.8578 |
| `section_5` | `section_3` | 42 | +0.9739 |
| `section_5` | `section_1` | 85 | +0.9577 |
| `section_7` | `section_5` | 41 | +0.9293 |
| `section_7` | `section_3` | 83 | +0.8953 |
| `section_7` | `section_1` | 125 | +0.8784 |

### How to read this against the audits

`deep_starmap`'s `expr_mode` audit (`reports/t09_audit_deep_expr_mode.json`) put `cross-mix` at **+0.0147**  `zinb-flow` at **+0.2745** on this metric. Place them against the columns above:

* **noiseless ceiling `sqrt(R)`** is the hard bound: even a method with no noise of its own cannot correlate with the observed target above this, because the target is itself a finite-sample measurement. **split-half R** is the softer bound, for a method as noisy as the data.
* **best other section** is the copying ceiling: the most a donor-copying method could score if it copied a whole real section. `cross-mix` copies donor by donor, so it is competing against this number, not against 1.0.
* **shuffled** is the floor. An arm at the floor is not modelling depth at all.

If the copying ceiling is near zero, `cross-mix`'s score is not a model failure — no donor-copying method could have done better, and the `expr_mode` margin on this metric says nothing about the flow head. If the copying ceiling is high and `cross-mix` is at the floor, the deficit is `cross-mix`'s donor retrieval, still not the flow head's modelling. Only a high ceiling *with* `zinb-flow` well above the copying ceiling makes the margin a statement about the generative path.

**One caveat on the split-half.** Each half carries half the cells, and Spearman-Brown corrects for that only under assumptions this profile does not exactly meet. On a section with few cells the correction under-estimates R, which is why the per-target cell count is printed beside it.
