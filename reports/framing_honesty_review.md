# Is the copy-based-field framing honest? — claim by claim, against the record

**Asked before the negative-result paper is drafted.** Short answer in §6.

## 1. The structure of the argument is legitimate

*Build X, measure that a component of X loses, replace it with the simpler thing that wins, and
report the negative as the justification* is how a method paper should work. It is more honest than
most published methods, which assert the design choice. **The framing is not inherently dressing a
negative result as a method.**

The question is only whether the four claims are supported. Three are not.

## 2. Claim 4 — "a design choice proved rather than asserted" ✅ SUPPORTED

The ceiling analysis is anchored to the published scale on tier-1 (exact per-section reproduction of
`flanking_copy`, `reports/deep_anchoring_check.md` §2), pre-registered, and replicated 6/6 sections
across both datasets. **This is the paper's best contribution and it stands as stated.**

## 3. Claim 3 — "exact agreement between crossing sections" ⚠️ TRUE OF A CONFIGURATION THE FRAMING DOES NOT SHIP

Bitwise, untrained, with a test — for **`zinb-flow`**, where every conditioning pathway is queried
at physical points.

`cross-mix` does **not** inherit it. `_cross_mix` ends in `cross_mix_counts(donors, w, gen, cfg)`
and `anchor_blend` draws `gen.random(...)`, where `gen` is a per-generation `np.random.Generator`:
the per-gene Bernoulli is keyed to a call-ordered stream, not to physical position. Two crossing
planes select different donors for the same physical cell.

**Under the new framing, expression comes from cross-mix. So claim 3 is currently false of the
proposed method.**

⚠️ **AMENDED 2026-09-11, and it is worse than written above.** Building the fix exposed that **the
bitwise test is weaker than the claim it is cited for.**
`test_generation_is_intersection_consistent_by_construction` computes `points = segment.points(64)`
**once** and hands the *same array* to both branches. It establishes that `evaluate_branch` is a
pure function of `(points, labels, neighbours)` and ignores the plane — true, and the mechanism —
but not that two independently generated crossing sections agree, because each derives its own
coordinates and GATE 1 G1.2a measures those agreeing to **1.14e-13 um, "to rounding, not exactly"**.

So the record's *"bitwise identical, exactly and without training"* holds under identical supplied
coordinates and is **unmeasured under independent derivation — for `zinb-flow` as well as for
cross-mix**. `tests/test_sefl.py::test_intersection_survives_independent_coordinate_derivation` now
measures that gap. Same family as everything else this campaign has found: a test that verifies the
code path it exercises rather than the claim it is quoted for.

**The fix is built** — `Config.cross_mix_position_keyed`, default **off** — and my cost estimate of
"seconds" was wrong in a way worth recording (`reports/position_key_cost_correction.md`).

## 4. Claim 2 — "arbitrary planes, which no published method can do" ⛔ THE WEAKEST, AND THE NOVELTY

Three separate problems, any one of which would block it.

**4a. The shipped layout cannot produce an oblique section.** `_resample_layout` selects the nearest
flanking section by `abs(f.z - plane.origin[2])` — a *z*-distance, meaningless for a plane spanning
the stack — and pastes its in-plane `(u, v)` coordinates onto the new plane. At an oblique angle it
does not crash; it emits a coronal point pattern relabelled as oblique, carrying no information
about the tissue the plane cuts. **The configuration the framing proposes to ship cannot do the
thing the framing is named for.** This is what makes Test 1 an existence test rather than an
ablation rescue.

**4b. GATE 2 ran on a synthetic fixture, and the record says "real data" in two places.**
`gate2.md` states its fixture as `make_synthetic_volume(seed=0, extent_xy=3000)` — 9 synthetic
sections, "the same fixture GATE 1 was measured on" — and its §309 says *"Measured on the 3000 um
synthetic fixture; T10's E3 is where it"* goes to real data.

But `reports/t09_closeout.md` §2.2 reads *"A clean, unqualified pass on **real data**, and the
strongest single result in the project"*, and `reports/advisor_report.md` line 71 tags the same row
*"`reports/gate2.md` — real data"*. **Both are false, and both cite the report that contradicts
them.** This is `specs/10` §4.2's own family — a claim about provenance that the cited source
refutes — sitting on the result the new framing would make its headline.

**4c. Even on synthetic data it is not the generation pipeline.** GATE 2's probe is a **linear head
predicting the top 32 expression PCs**, and `gate2.md` says why: *"The full generative heads do not
exist yet."* It shows the **representation** is not orientation-biased. It does not show the method
produces good sections at oblique angles. **E3, the real-data oblique validation, has never run** —
no report exists.

## 5. Claim 1 — "competitive reconstruction, 5 of 6 against SpatialZ" ⛔ NOT SUPPORTED AS STATED

- **The comparator numbers are not in this repository.** `specs/10` line 2435 lists SpatialZ, FEAST
  and isoST as **required** runs precisely because "the existing numbers are not in this repo".
- **The comparison the record does hold was already corrected against us.** Per dataset it is
  **9–9**; the version favouring us was a cross-dataset average that `specs/10` §4.2a forbids by
  name (`advisor_report.md` §"A third and fourth appearance ... are withdrawn").
- **It is v20's number, not v25's.** v25 at `expr_mode=cross-mix` shares v20's *expression* step
  and differs in layout, retrieval and conditioning. "Cross-mix is v20's path" is true of one step.

I could not locate a "5 of 6 against SpatialZ" result in the record at all.

## 6. The verdict

**The framing is honest in structure and would be dishonest as written today — and the specific
danger is that its own virtue is what it would violate.**

Claim 4 says *we proved the design choice instead of asserting it*. On today's evidence the paper
would assert claims 1, 2 and 3 — with claim 2, the novelty, resting on a synthetic probe result that
two documents mislabel as real data, for a configuration that cannot produce an oblique section at
all. A paper whose selling point is proof-not-assertion cannot carry three asserted claims.

**That is a statement about the evidence, not about the idea.** The framing is worth pursuing. It is
not yet a paper.

**What it needs, in order of how much it costs:**

| | claim | what would support it | cost |
|---|---|---|---|
| 1 | claim 3 | key cross-mix's Bernoulli to the 3D field; the existing bitwise test | small code change, seconds to test |
| 2 | claim 2, mechanism | **Test 1** — the field layout with the count supplied externally | zero fits, minutes |
| 3 | claim 2, real data | **E3** — oblique validation on the re-sectioned STARmap, through the generation pipeline rather than a linear probe | a re-sectioning build plus generation; the largest item, and unavoidable |
| 4 | claim 1 | the comparator run: SpatialZ, FEAST, isoST on tier-1, same instrument and holdout | `specs/10` step 5, already scoped as required |

**And two corrections owed regardless of what is decided:** `t09_closeout.md` §2.2 and
`advisor_report.md` line 71 both say GATE 2 was on real data. It was not.

## 7. The fallback, so the choice is between two real options

If Test 1 returns REFUTED, or E3 is not affordable, the negative-result paper of
`diagnostic_programme_closed.md` stands and is publishable as it is: anchored, pre-registered,
replicated 6/6, with a ceiling that says the design cannot win. **That paper is finished and this
one is not.** The reframing is worth the four items above; it is not worth writing on their absence.
