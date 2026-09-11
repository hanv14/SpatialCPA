# Retractions

Claims this campaign made and has since withdrawn, each with what disproved it. Recorded as
retractions rather than quiet corrections: a claim that reached a report, a review or the advisor
has to be visibly withdrawn in the same place it was made.

---

## R1 — the 0° reference row was a double-thickness plane

**Claimed** (`reports/angle_budget.md`, first version): tier-1's coronal reference plane contains
**8279 cells** and **19 scorable types**, and G1's threshold is 60% of that 19.

**Withdrawn.** The plane's origin was `0.5 * (lo + hi)`, the volume's z-**midpoint** — 52.0 µm on
tier-1, exactly midway between the sections at z = 41 and z = 63. With a half-thickness of 11.0 µm
the band is precisely [41.0, 63.0], and `<=` admitted **both** sections on an exact floating-point
tie. The reference row was two sections where every other row is a slab through one.

**What disproved it**, independently and arithmetically:

- `test1b`'s own `n_cells` records ~**4165** cells per real tier-1 section. 8279 ≈ 2 × 4140.
- Strip area scales correctly only after halving the row: area(5°)/area(0°) = (1008×1412)/(1300×1514)
  = **0.72**, while cells = 3248/8279 = **0.39** — off by exactly 2×. Halved: 3248/4140 = **0.78**.
- Every tilted row draws from 4 sections; only the 0° row reports 2. Once tilted, the tie is gone.

**Class.** `specs/10` §4.2n — a statistic whose reference was inherited from where the code happened
to put the plane rather than chosen. Aggravated: the number sat on an exact float tie, one
recompilation away from being 0 or 8279.

**What survives.** The **budget of 5° is unchanged.** G2 is absolute (largest type ≥ 250) and a
single real section gives ~546, still clearing; re-centred, G1's denominator falls to ~18 and 10°
still fails on G2 at 159 < 250, 45° still fails G1 at 9 types. The defect is in the reported
reference, not the verdict.

**Fixed at source.** The reference plane is now centred on the volume's median **real section**, and
`reports/angle_budget.md` says so in the table's own caption.

---

## R2 — "the layout's positions are unstable across sections"

**Claimed** (review of `test1b_layout_split.md`, last round): the ~25× ratio between model-position
and copy-position arms' across-section spread shows the model's *placement* is unstable.

**Withdrawn.** It was a post-hoc reading of a table already seen, it was flagged as such, and the
three-seed run disproves it.

**What disproved it.** `both_oracle`'s across-seed spread is **0.0 on all three sections** — exactly
zero, not small. `celltype_localization` touches generated expression only through
`align_by_expression`; with positions fixed, types fixed and pose 0°, nothing stochastic remains.
So the metric is nearly **blind to the expression head**, and the across-section spread it reports
cannot be attributed to placement instability by that argument. `null_types` (model positions,
permuted types) has an across-section spread of 0.054 — but it sits at 0.03–0.08 and a metric
bounded below cannot spread at its floor, so it is uninterpretable in either direction.

**What replaces it**, on a clean single-variable contrast both of whose arms are far from any floor:

| | positions | types | across-section spread |
|---|---|---|---|
| `fix_types` | model | oracle | **0.4306** |
| `both_oracle` | copy | oracle | **0.0140** |

Types held identical, one variable changed, **31×**. Worst across-seed spreads are 0.095 and 0.000,
so the variation is across **sections**, not across seeds. The statement the paper can make is
**"the model's layout is wildly section-dependent"**, not "its placement is unstable" — and nothing
in `test1b` may be read as a statement about expression at all.

---

## R3 — `both_oracle` beating the copy floor

**Claimed** (implicit in reading `test1b`'s 125.5% recovered-both, and in the temptation to quote
0.8375 > 0.7765): giving the model the copy's positions and true types produces a section that
**beats `flanking_copy`**.

**Withdrawn.** `both_oracle` is donor positions carrying **ground-truth** cell types NN-transferred
onto them. `celltype_localization` scores, per type, the Sinkhorn divergence between predicted and
GT point clouds — so GT types on donor positions is a **partial oracle on precisely the quantity
being scored**, while the copy carries the *donor's* types. 0.8375 > 0.7765 is what that oracle buys
and is not evidence about the method.

**And a coincidence not to be seduced by.** 0.8375 sits next to the chain diagnostic's A1b ceiling
of 0.8369. **These are different metrics.** The ladder is `paper_morans_pearson` (its `oracle` is
1.0000); this table is `celltype_localization` (its `oracle` is 0.9808). Reading the two as
corroborating each other is the cross-scope comparison this campaign has been caught by four times
and will not make a fifth.

---

## R4 — "an exact tie is not a case the old rule handled meaningfully"

**Claimed** (`cells_near_plane`'s docstring, and `progress/t09_inference_and_calibration.md`, one
commit ago): the `plane-distance` rule's one divergence from `nearest-z` is that a plane sitting
*exactly midway* between two sections returns both, where `nearest-z` broke the tie on `section_id`
and took one — and this was "documented rather than hidden" as a curiosity about exact ties.

**Withdrawn.** It is not a curiosity. **With evenly spaced sections it is every target plane there
is.** `tests/test_layout.py::target_plane` says so in its own docstring — *"A plane halfway between
two training sections: what generation actually asks for"* — and on tier-1 the two flanking sections
sit at exactly ±22 µm from every held-out plane. I read the central case as an edge case and wrote a
paragraph excusing it.

**What disproved it.** The load-bearing test, on the second try:

```
test_plane_distance_is_BITWISE_identical_to_nearest_z_at_a_coronal_plane
  AssertionError: positions must be bitwise identical
test_the_empty_slab_means_two_different_things_and_the_flag_says_which
  expanding reached {'synthetic_s00', 'synthetic_s01'}, expected {'synthetic_s00'}
```

Two failures, one cause. Until they passed, `plane-distance` was not a strict generalisation and
every tier-1 number measured under `nearest-z` would have become cross-construction.

**Fixed at source.** The empty-slab fallback now selects a **section**, not a distance stratum, and
breaks its tie exactly as `nearest-z` does — on `(perpendicular distance, section_id)`. Verified
against `nearest-z`'s key transcribed verbatim, on all **8** midway planes of the fixture's stack,
plus the on-section and post-exclusion cases.

**And one assertion of mine was simply false**, on the same wrong premise: the exclusion test
asserted the band widens to cells *strictly further* from the plane (`min > max`). At a midway plane
the section the exclusion falls through to is at **the same** distance — 50.0 against 50.0 on the
fixture. Corrected to `>=`, with the equality asserted positively instead: the result must be
exactly the section `nearest-z` would pick under the same exclusion.

**Class.** Not §4.2n or §4.2o. This one is: *a divergence I documented instead of measuring.* Writing
the exception down is not the same as checking how often it fires, and the paragraph excusing it was
doing the work a test should have done.
