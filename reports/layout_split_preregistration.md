# Pre-registration — splitting the layout deficit into placement and typing

**Committed before the runner is built or run.** Nobody has separated the two, the fix differs by
which it is, and this is step 1 of the method paper's layout section.

## 1. What is being split

`paper_celltype_localization` on `layout_mode=field` with a shippable cell count is **0.5371**
against the model-free copy floor of **0.7765** (`reports/test1_field_count.md`). That deficit of
**0.2394** is currently attributed to "the layout", which is two mechanisms:

- **placement** — the point process: are cells where the tissue puts them?
- **typing** — the mark model: given a position, is it the right cell type?

They have different fixes. Placement is the intensity field and the sampler (T05); typing is the
per-position categorical drawn from `lam / lam.sum(axis=1)`.

## 2. How the metric works, and what that forces on the construction

Read from `bench3/evaluate_paper.py::celltype_localization` before the arms were designed:

- It compares, **per cell type**, the *point cloud* of predicted cells of that type against the
  ground truth's, by debiased Sinkhorn divergence, calibrated against a within-tissue null
  (`score_c = 1 - D(pred_c, gt_c) / D(random, gt_c)`). **No cell-to-cell correspondence is used.**
- A type with fewer than `min_pred_cells = 5` predicted cells scores **0** — composition matters.
- Scores are averaged weighted by ground-truth type frequency.

**And the pose is aligned by EXPRESSION**: `pred_xy_al, ainfo = align_by_expression(pred_xy,
pR[:, cols], gt_xy, gR[:, cols], ...)`. So an arm's expression field decides its alignment, and
therefore its localization score. **A chimeric arm that fabricates or drops expression would be
scored at a different pose from the arm it is compared with** — the cross-construction error this
project has been caught by repeatedly.

**Therefore every arm carries a real, coherent expression field matched to its own positions**, and
arms that share positions share an alignment exactly. Each arm is a complete prediction scored
through the **full pinned `evaluate_paper`**, the same path that produced 0.7765 and 0.7546.

## 3. The arms

Type transfer between point sets is by **nearest neighbour** — a type field is a piecewise-constant
function of position, and NN is its transfer.

| arm | positions | expression | types | alignment shared with |
|---|---|---|---|---|
| **base** | model | model | model | — |
| **fix_types** | model | model | **ground truth, by NN** | base |
| **fix_positions** | the copy's | the copy's | **model, by NN** | `flanking_copy` |
| **both_oracle** | the copy's | the copy's | ground truth, by NN | `flanking_copy` |
| **null_types** | model | model | model, **permuted among its own cells** | base |

`fix_types` keeps the model's expression on purpose: the question *"if the types were right, would
it score?"* does not require the expression to change, and holding it fixed holds the pose fixed.

## 4. The statistic

On medians over held-out sections 2/4/6, with `floor` = `flanking_copy` = **0.7765**:

```
recovered_types     = (fix_types     - base) / (floor - base)
recovered_positions = (fix_positions - base) / (floor - base)
```

## 5. Outcomes, fixed now

| | condition | reading, and the fix it implies |
|---|---|---|
| **TYPING** | `recovered_types >= 0.60` and `recovered_positions <= 0.30` | the cells are in the right places and the wrong types are on them. The fix is the **mark model** — the per-position categorical — not the point process |
| **PLACEMENT** | `recovered_positions >= 0.60` and `recovered_types <= 0.30` | the typing is sound and the cells are in the wrong places. The fix is the **intensity field and sampler** |
| **EITHER SUFFICES** | both `>= 0.60` | each fix alone nearly closes the deficit; it lives in the joint assignment and either route works. **A good outcome, not an ambiguous one** |
| **NEITHER SUFFICES** | both `<= 0.30` | the deficit is genuinely joint and not separable by this decomposition. The layout section reports the deficit undivided |
| **MIXED** | anything else | both shares reported, no single-mechanism claim |

`recovered_types + recovered_positions` is reported beside them: **its distance from 1.0 is the
interaction**, and it is the number that distinguishes EITHER SUFFICES from a clean split.

## 6. Preconditions — any failing and the split is NOT READABLE

1. **`both_oracle >= floor - 0.05`.** If painting the ground truth's types onto the copy's positions
   does not reach the copy floor, the NN transfer is lossy and every arm is attenuated by an
   unknown amount.
2. **`null_types <= 0.10`.** Permuting types among the model's own cells destroys the
   type-position association while preserving composition and pose. The metric's own null is 0 by
   construction, so anything above this means it is not responding to the association the split
   assumes.
3. **`floor - base >= 0.10`**, or the denominator is unstable.
4. **Pose.** Arms sharing positions must report **identical** `align_rotation_deg`; arms with
   different positions must be within **5°**. Otherwise the comparison is across poses.

## 6-bis. AMENDMENT — precondition 4 forbids the comparison this test exists to make

*Added after the three-seed run returned **NOT READABLE** on precondition 4. Recorded as a design
error in the pre-registration, not as a run failure, and written before the split is re-read.*

**What happened.** Model-position arms aligned at 6.000°, copy-position arms at 0.000°; the span is
6.00° against a 5° bound. Precondition 4 was written to catch an *accidental* pose divergence. But
the split's entire content is model positions **versus** copy positions, and `align_by_expression`
gives those two groups different poses by construction. **The precondition cannot ever pass.** It
rules out the primary comparison structurally and permanently.

That the gate fired is the system working. The repair is to the design, not to the gate.

**The repair: the split is a WITHIN-pose-group comparison.**

| readable | contrast | what it isolates |
|---|---|---|
| ✅ | `base` → `fix_types`, both at 6° | oracle typing **on model positions** |
| ✅ | `fix_positions` → `both_oracle`, both at 0° | oracle typing **on copy positions** |
| ❌ | anything crossing the two groups | across poses — and now across cell counts too |

**Three consequences, all accepted rather than argued around:**

1. **`recovered_positions` is retired as a statistic.** There is no within-pose contrast that
   changes positions while holding types, because changing positions *is* what changes the pose. The
   single headline percentage does not survive this amendment.
2. **What survives is a comparison of two typing gains**, which is a real and readable statement:
   oracle typing is worth **+0.101** on model positions and **+0.230** on copy positions — typing
   helps *more* when placement is already right. Read as medians and subject to §4.2o's spread
   check before it is quoted.
3. **The cross-group contrast is confounded by more than pose.** The arms do not even hold cell
   count fixed: model-position arms carry 4165/4276/4289 cells, copy-position arms 4073/4169/4110.
   Precondition 4 was right to refuse it; it was merely right for a narrower reason than the full
   one.

**One further precondition defect, from `reports/test1b_estimator_discrepancy.md`.** `pose_deg` is a
median over three sections, then a median over three seeds — one number standing for nine — and
precondition 4 is applied to *that*. `fix_types`'s reported pose moved 1.5° → 6.0° between two runs
whose three scores are bitwise identical, which is the compression moving, not the poses that were
used. **Preconditions must be evaluated on the (section, seed) pose that entered each score.** Until
they are, precondition 4's verdict is correct for the structural reason above but was not reached by
the evidence it claims to read.

## 7. Predictions, recorded now

1. **TYPING**, moderate confidence. The intensity's error enters placement through a *normalised*
   route — `celltype_localization` divides positions by the ground truth's centre and radius, so
   coverage matters more than density — but it enters typing *directly*, because the marks are a
   categorical drawn from `lam / lam.sum(axis=1)` at each position. Same broken field, more direct
   path.
2. `both_oracle` **exceeds** the copy floor, probably near oracle: the copy's positions carrying the
   truth's own type field is close to the truth's type clouds by construction.
   *Confirmed at 0.8375 against the floor's 0.7765 — and see `reports/retractions.md` **R3**: this
   is what a partial oracle on the scored quantity buys, and may never be quoted as the method
   beating the copy. The prediction was right; the favourable reading of it is withdrawn.*
3. `null_types` lands **below 0.05**.
4. `recovered_types + recovered_positions` is **above 1.0** — the two fixes overlap rather than
   partition, because each repairs part of the same joint assignment.

## 8. What this cannot decide

It splits the deficit on **one metric, `paper_celltype_localization`, at axis-aligned planes**. It
says nothing about oblique planes, where no layout mode currently works and where the ground truth
for scoring is a different construction entirely
(`reports/oblique_layout_cost.md`). It is the diagnosis that tells the method paper's layout section
which mechanism to fix, not a demonstration that fixing it works.
