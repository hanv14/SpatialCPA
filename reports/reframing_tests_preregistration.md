# Pre-registration — the two tests that gate the copy-based-field framing

**Committed before either test is built or run.** The framing they gate is: *a continuous field
enabling copy-based generation at arbitrary orientations, with coherent volumes.* Both tests are
**zero fits** — `layout_mode` and `expr_mode` are in `FIT_INVARIANT_GATES`, and the r11 checkpoint
recovery proved all five r11 arms are one fit differing only in generation-time gates
(`test_layout_mode_does_not_enter_the_fit`, bitwise identical across all 96 tensors).

## 0. What each test is actually for

The framing's novelty claim is **arbitrary orientations**. Before either test:

- **`layout_mode=resample` cannot produce an oblique section.** `_resample_layout` picks the
  nearest flanking section by `abs(f.z - plane.origin[2])` — a *z*-distance, meaningless for a plane
  that spans the whole stack — and pastes that section's in-plane `(u, v)` coordinates onto the new
  plane verbatim. At an oblique angle it will not crash; it emits a coronal point pattern relabelled
  as oblique, carrying no information about the tissue the plane actually cuts. **The shipped
  configuration cannot do the thing the framing is named for.**
- So **Test 1 is not an ablation rescue — it is the existence test for the framing's central
  claim.** If the field layout cannot be made to work, no configuration generates an oblique
  section and the claim cannot be made in any form.

## 1. Test 1 — the field layout with the cell count supplied externally

**What.** `layout_mode=field` on the r11 checkpoint (`runs/pilot/model_exp_2400.pt`,
`decoder_mu_link=exp`, 2400 steps, seed 1), with the cell count taken from outside the intensity
integral. Scored on `paper_celltype_localization`, medians over sections 2/4/6, on the pinned
`bench3.evaluate_paper`.

**Two count sources, and they are not equals:**

| arm | count from | status |
|---|---|---|
| **1a** | the ground-truth section's own cell count | an **oracle input**. An upper bound, never a shippable configuration |
| **1b** | the two flanking sections' measured density × the plane's area | **the shippable one. 1b GOVERNS every verdict below.** |

1a is reported to separate "the count was the defect" from "the pattern is the defect". It may not
be quoted as the method's score, and no band below is read from it.

**Reference points, already on record** (`r11_probes_recheck.md`, one fit, one seed):

| | `paper_celltype_localization` |
|---|---|
| `field`, count from the broken integral | 0.6607 |
| `resample` (ships) | 0.7546 |
| `flanking_copy` — the model-free floor | **0.7765** |
| `oracle` | 0.9808 |

**Outcomes, fixed now, read on 1b:**

| | band | reading |
|---|---|---|
| **CONFIRMED** | ≥ **0.7765** | the field layout clears the model-free copy floor. The diagnosis holds, the framing has a mechanism for arbitrary orientations, and claim 2 becomes makeable |
| **PARTIAL** | ≥ **0.7546** and < 0.7765 | matches the shipped layout but does not clear the floor. Claim 2 may be made only in the weak form: *"oblique generation is possible at the quality of axis-aligned generation, which is itself below a copy"* |
| **REFUTED** | < **0.7546** | the count was not the defect. **No configuration generates a meaningful oblique section, and claim 2 cannot be made at all** |
| **UNINFORMATIVE** | 1a and 1b differ by more than the across-section spread of 1b | the count source is doing the work rather than the layout; no reading |

**The default on anything other than CONFIRMED or PARTIAL is that the framing loses its novelty
claim**, and the paper returns to the negative result of `reports/diagnostic_programme_closed.md`.

**Cost.** Zero fits. Three sections × two count arms of generation + scoring on tier-1's ~4 100
cells. Comparable to an r11 arm, which the record puts in minutes. Plus one code change: the count
must enter `_field_layout` from a caller rather than from the integral — small, and it needs a test
that the integral is genuinely bypassed rather than overridden downstream.

**One seed.** The r11 arms are one fit and one seed, so this inherits that: **no across-seed spread
exists for `paper_celltype_localization` on the field arms** (`envelope_correction.md` §3). Bands
are read on raw deficits and no multiple-of-envelope may be quoted, exactly as the A4 row was
corrected to do.

## 2. Test 2 — does `cross-mix` survive an oblique plane, and does it inherit intersection consistency?

### 2a. Does it run, and is the output sane?

`cross-mix` draws donors from retrieval, which is defined at any physical point, so it *should*
run. Confirm rather than assume, at **30°** and **45°**, with the layout mode Test 1 endorses.

**If Test 1 returns REFUTED, 2a is reported NOT APPLICABLE** — there is no layout that produces an
oblique section to run it on, and running it on a `resample` layout would be scoring a coronal point
pattern wearing an oblique label (§0).

Pass conditions, all three: it completes without `EmptyCandidatePoolError`; every emitted cell has
≥ 1 admissible donor; and the emitted counts are non-degenerate (more than one distinct donor used
per gene across the section).

### 2b. Intersection consistency — the prediction is recorded now, before the test

`architecture_ceiling.md` and the record's claim 2.3 — *two crossing sections emit bitwise identical
expression along their intersection, exactly and untrained* — is proved for the **`zinb-flow`**
path, where every conditioning pathway is queried at physical points.

**Read from source (`infer/generate.py`), before running anything: `cross-mix` does NOT inherit
it.** `_cross_mix` ends in `cross_mix_counts(donors, w, gen, cfg)` and `anchor_blend` draws
`gen.random(a.shape)`, where `gen` is a **per-generation `np.random.Generator`**. The per-gene
Bernoulli selection is keyed to a call-ordered RNG stream, not to the cell's physical position, so
two crossing planes will select different donors for the same physical cell.

**So claim 3 is false of the configuration the framing proposes to ship**, and this test is expected
to confirm that rather than to discover it. It is registered because a prediction read from source
and never executed is not a result.

**The repair, and its acceptance condition.** Key the per-gene selection to a query of the 3D noise
field at the cell's physical point instead of to `gen`. If that change is made, the acceptance
condition is the **same one `zinb-flow` already meets** — bitwise identical expression along the
intersection, on an **untrained** model, with no consistency loss applied, asserted in
`tests/test_sefl.py` beside the existing test. Anything less than bitwise is a failure, because the
existing claim is bitwise and a weaker one would be a different claim wearing its name
(`specs/10` §4.2l).

**Cost.** 2a: two generations, zero fits, minutes. 2b: a unit test on an untrained model, no data,
seconds. The repair, if made: small, and gated on the bitwise test.

## 3. Predictions, recorded now

1. **Test 1 returns PARTIAL, not CONFIRMED.** The record already says the field layout's problem is
   not only the count: `pilot.md` §6.3 calls the layout "a separate, real defect" *after* noting
   `celltype_localization` improves to 0.5822 and "stays well below the floor". Moderate confidence.
2. **2a passes** — cross-mix runs at both angles. High confidence; retrieval is position-defined.
3. **2b fails as written**, for the reason in §2b. High confidence — it is read from source.
4. The repair in §2b makes 2b pass bitwise. Moderate confidence: the GRF is already queried at
   physical points, so the mechanism exists; whether the selection can be keyed to it without
   changing the marginal distribution of the mix is not established.

## 4. What these two tests do NOT establish

Neither touches the two claims that are in the worst shape (`reports/framing_honesty_review.md`):

- **that oblique generation works on real data.** GATE 2 ran on a **synthetic fixture** and measured
  a **linear probe on 32 expression PCs**, not the generation pipeline. E3, the real-data oblique
  validation, has never run.
- **that reconstruction is competitive with published methods.** SpatialZ, FEAST and isoST numbers
  are **not in this repository** (`specs/10` line 2435, which calls the comparator run required),
  and the v20-vs-SpatialZ comparison the record does hold was corrected to **9–9 per dataset**.
