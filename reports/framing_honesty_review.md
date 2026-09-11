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

## 3. Claim 3 — "exact agreement between crossing sections" ✅ SUPPORTED, once the fix lands

*(This section originally read ⚠️ TRUE OF A CONFIGURATION THE FRAMING DOES NOT SHIP. Both the
diagnosis and my subsequent amendment to it were partly wrong; the sequence is kept below because
the second error is the more instructive one.)*

Bitwise, untrained, with a test — for **`zinb-flow`**, where every conditioning pathway is queried
at physical points.

`cross-mix` does **not** inherit it. `_cross_mix` ends in `cross_mix_counts(donors, w, gen, cfg)`
and `anchor_blend` draws `gen.random(...)`, where `gen` is a per-generation `np.random.Generator`:
the per-gene Bernoulli is keyed to a call-ordered stream, not to physical position. Two crossing
planes select different donors for the same physical cell.

**Under the new framing, expression comes from cross-mix. So claim 3 is currently false of the
proposed method.**

⚠️ **MY AMENDMENT OF 2026-09-11 IS WITHDRAWN — I overstated the gap.** It said the bitwise claim
holds only "to rounding, not exactly", for `zinb-flow` as well as cross-mix. That is wrong, and the
mechanism I missed was written in the record I was reading: `progress/t03_noise_field.md` G1.2
records *"coords agree to 2.8e-14 um and **round to the same float32**"*.

**`CTFFlow.prior_latent` casts to float32 before querying the GRF.** The float32 step at these
coordinates is ~1e-5 to ~1e-4 um; GATE 1 G1.2a's pathway disagreement is 1.14e-13 um. So the two
pathways round to the same float32 with about **nine orders of margin** — measured here at **0 of
6,000,000 coordinates** changing under that drift. And G1.2 *does* test independently derived
coordinates (256 points, two plane pathways, **max diff exactly 0.0**), which is the test I said did
not exist.

**So "bitwise" stands.** What it needs is its mechanism stated, because "exact by construction" is
half the story: it is physical-point conditioning **plus** float32 quantisation of a disagreement
nine orders below the quantisation step. Structural *and* numerical, with an enormous margin — not
merely structural.

**What genuinely remains untested**, and it is much narrower than what I claimed: the generation
test supplies `points`, `labels` *and* `neighbours` to both branches, so **retrieval's neighbour
selection under independently derived coordinates is not exercised**. That channel is *discrete* —
a tie between two equidistant donors could in principle break differently — and float32
quantisation covers it for the same reason it covers the rest, but it has not been measured. One
test, and worth having.

**Cross-mix now inherits the property at the same standard**: `position_keyed_uniforms` quantises to
float32 first, and is bitwise under G1.2a's drift. So claim 3 is **true for both expression paths**
with the flag on — which is the one place in this review where the answer improved on checking.

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
would assert claims 1 and 2 — with claim 2, the novelty, resting on a synthetic probe result that
two documents mislabel as real data, for a configuration that cannot produce an oblique section at
all. A paper whose selling point is proof-not-assertion cannot carry asserted claims.

**Updated 2026-09-11: claim 3 has moved to supported** (§3), and claim 1 has been **struck and
restated** (§5). The verdict now rests on claim 2 alone — which is the right place for it, because
claim 2 is the novelty.

**That is a statement about the evidence, not about the idea.** The framing is worth pursuing. It is
not yet a paper.

**What it needs, in order of how much it costs:**

| | claim | what would support it | cost |
|---|---|---|---|
| 1 | ~~claim 3~~ | ✅ **DONE.** `position_keyed_uniforms`, float32-quantised, bitwise under G1.2a's drift; `Config.cross_mix_position_keyed` (default off). Cost estimate was wrong — `reports/position_key_cost_correction.md` | done |
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
