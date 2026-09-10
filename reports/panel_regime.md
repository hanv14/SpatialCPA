# The two datasets are different measurement regimes, and F3 measures it model-free

## 1. The measurement

`F3_copy` — the score `flanking_copy` retains when gene identity is destroyed *within abundance
strata*, i.e. what its score would be from abundance matching alone:

| `F3_copy` | section_2 | section_4 | section_6 |
|---|---|---|---|
| **deep**, 1017 genes | +0.4762 | +0.5049 | +0.5579 |
| **tier-1**, 28 genes | +0.1231 | −0.0325 | +0.0179 |

Individually the tier-1 figures are too noisy to quote — at 28 genes the per-seed intervals span
~0.4. Three independent estimates all within 0.13 of zero, against three all within 0.05 of 0.50,
is not. Corroborated by `R1(I_real, detection)` — how much of the **tissue's own** ordering is a
sparsity ordering: **0.28 / 0.25 / 0.28** on tier-1 against **0.62 / 0.68 / 0.70** on deep.

## 2. What it says

**On tier-1's 28 curated markers, `paper_morans_pearson` is almost entirely gene-specific spatial
fidelity. On deep's 1017 genes, about half of it is per-gene abundance.**

The two datasets' scores are therefore **not the same quantity**, and a method's figure on one is
not comparable to its figure on the other. This campaign suspected it for four rounds — from
`R1` = 0.68 against 0.29 — and F3 measures it without a control specification anyone chose.

## 3. What follows

- Any statement of the form *"v25 scores X on `paper_morans_pearson`"* must name the dataset, and
  the paper may not average or pool the two.
- The abundance-floor rescaling (`abundance_floor_preregistration.md`) is **deep-only**: tier-1's
  floor is indistinguishable from the permutation null on its own interval, so §4 refuses it there,
  and that refusal is the correct answer rather than a limitation to work around.
- It is a property of the **gene panel**, not of the tissue or the method: a curated marker set is
  chosen for spatial informativeness, so its `I` ordering carries little abundance signal; a
  1017-gene set is mostly sparse genes whose `I` is bounded by detection.

## 4. What it is not

It is **not** a benchmark critique. `flanking_copy` retains 95.1–100.8 % under all three controls on
all six sections, and `spatial_scramble` is at the null everywhere: the copy floor is genuine
spatial fidelity on both datasets. That deep's metric additionally rewards abundance matching does
not make the floor reachable by matching abundance — the copy's own abundance-only score is 0.48,
against its 0.99.
