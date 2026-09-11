# SpatialCPA-v25-Gen — paper draft

**Status.** Sections 1–4 and 6–7 are written from settled numbers. Section 5's oblique table awaits
`scripts/oblique_demo.py` scoring at 30/45/60°; every cell in it is marked `[PENDING]` and no number
is placeholdered with a guess. Nothing here may be revised on the strength of a score: θ\*, the
angles, the gates and the outcomes are pre-registered in
`reports/oblique_demonstration_preregistration.md`.

## The claim, in one sentence

A continuous field makes section generation well-defined at an arbitrary position and orientation,
which lets sections be **generated and scored off-axis** for the first time — and the two bounds
that govern how far off-axis anyone can go are stated, with their arithmetic, because neither is in
the literature.

## Structure

| § | title | rests on | status |
|---|---|---|---|
| 1 | Introduction | — | drafted |
| 2 | Method: the continuous field, and what it makes well-defined | T03–T07 | drafted |
| 3 | **Two bounds on oblique evaluation** | `the_comb_limit.md`, `metric_resolution.md` | drafted |
| 4 | The angle budget across specimens | `angle_budget.md` | drafted |
| 5 | **The oblique demonstration** | `oblique_demo.md` | drafted; scores `[PENDING]` |
| 6 | What the method does not do | `architecture_ceiling.md`, `t09_closeout.md` | drafted |
| 7 | Limitations, and the record of what we withdrew | `retractions.md` | drafted |

§3 leads §5. The bounds are stated before our result is shown, so the result is read inside them
rather than defended against them afterwards.

## What §5 may say, fixed now

Per `oblique_demonstration_preregistration.md` §6 with 90° replaced by θ\*:

- **DEMONSTRATED** — `resample-pd(θ*) ≥ copy(θ*) − 0.05`, spread reported beside it.
- **DEMONSTRATED WITH A COST** — below that band; the price is a number in the abstract, not in a
  limitations paragraph.
- **PARTIAL** — θ\* fails a precondition, a smaller qualifying angle passes.
- **NOT DEMONSTRATED** — no qualifying angle passes; §5 becomes a negative section and §3 and §4
  carry the paper.

## The headline angle is an author's decision, not a derived one

F2's **second form** (§2-quater) tests the stratum in micrometres against the volume's median
nearest-neighbour distance. It excludes 75°, 85° and 90°, and admits 30/45/60/70 — so it excludes a
*regime*, not a value, which is the test that it is not reverse-engineered to a preferred answer.
It selects **θ\* = 60°, fill 0.25**.

Whether the headline is 60°, or 45° at fill 0.35, or 30° at fill 0.43, is a judgement about how much
coverage a claim needs. **It is recorded as a judgement, not derived.** All three are scored and
tabulated either way.
