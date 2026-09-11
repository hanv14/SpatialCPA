# Settled: 25.2% vs 29.4% is two estimators, not one bug

**Step 1 of the six. Nothing quotes either number until this is read.** A records read of the two
JSONs; no run, no fit.

## The diff

Fifteen section scores, five arms, two runs of the same checkpoint.

| arm | positions | types | changed? |
|---|---|---|---|
| `base` | model | **model** | `section_4` +0.0144 |
| `fix_positions` | copy | **model** | `section_2` +0.0333 |
| `fix_types` | model | oracle | — bitwise identical, all three |
| `both_oracle` | copy | oracle | — bitwise identical, all three |
| `null_types` | model | permuted | — bitwise identical, all three |

Plus `pose_deg` on the three model-position arms: **1.5° → 6.0°**.

## What it is not

**It is not a median-selection bug.** Two underlying section scores moved; the medians followed.

**It is not a Convention 3 violation.** I said last round that the typing path looked
non-reproducible, because the two arms that moved are exactly the two that use model types. That
reading was wrong, and `both_oracle` is the disproof: its across-seed spread is **0.0 on all three
sections**, and it is bitwise identical across the two runs to sixteen digits. Determinism holds.

## What it is

`_finish` collapses seeds with a **per-section median across seeds**:

```python
vals = {n: float(np.median([per_seed[s][arm]["per_section"][n] for s in seeds])) ...}
```

The earlier run passed one seed, so the median of one value *is* that value: the table was **seed 1**.
The later run passed three, so the table is the **per-section median over seeds 1–3**. Arms whose
across-seed spread is zero are unaffected; arms with spread move wherever seed 1 was not the middle
value. Both numbers are correct for their own estimator. **25.2% is the seed-1 split; 29.4% is the
3-seed median split.** Neither is quotable, for a different reason: the verdict is NOT READABLE.

## Three defects this exposes, all worth fixing before the split is re-run

**1. The report's own caption names the wrong estimator.** `test1b_layout_split_seeds.md` says
"grid sampler, shippable cell count, **seed 1**" above a table that is a three-seed median. That is
§4.2n — a statistic whose estimator was inherited from a branch rather than chosen — with a caption
that actively misdescribes it.

**2. The median is applied twice, and at n = 3 each application IS a selection.** Median over seeds,
then median over sections. The headline is therefore one seed's score on one section, chosen twice
over, and the per-section table beneath it is a **mosaic**: each row may come from a different seed.

**3. The pose precondition is evaluated on a triple median.** `pose_deg` is the median over three
sections, then the median over three seeds — one number standing for nine — and the precondition
that produced NOT READABLE is applied to *that*, not to the per-section poses that actually entered
the scoring. This is why `fix_types`'s reported pose moves 1.5° → 6.0° while all three of its scores
stay bitwise identical: the compressed number moved, the poses that were used did not.

## What to do

Report per seed and per section, and evaluate preconditions on the **(section, seed)** pose that
entered each score rather than on a compressed aggregate. Until then the pose precondition is
measuring an artefact of two medians, and its NOT READABLE verdict — though correct for a separate
and structural reason, see `layout_split_preregistration.md` §6-bis — was not reached by the
evidence it claims to be reading.
