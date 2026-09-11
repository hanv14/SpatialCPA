# SpatialCPA-v25-Gen — full draft

**Status: complete end to end.** Every number is measured and sourced; nothing is placeholdered.
Read `DRAFT.md` for the assembled document, or the sections individually.

## The claim, in one sentence

A continuous field makes section generation well-defined at an arbitrary position and orientation;
the evaluation of that capability is bounded in three independent ways we quantify for the first
time; and within those bounds the standard metric cannot separate our layout from the previous one.

## Structure

| § | title | rests on | status |
|---|---|---|---|
| 1 | Introduction | — | drafted |
| 2 | Method | T03–T09 | drafted |
| 3 | **Two bounds on oblique evaluation** | `the_comb_limit.md`, `metric_resolution.md` | drafted |
| 4 | **What geometry an oblique evaluation requires** | `angle_budget.md` | drafted |
| 5 | **Off-axis evaluation, and why it could not be resolved** | `oblique_demo.md` | drafted, scored, final |
| 6 | What the method does not do | `architecture_ceiling.md`, `t09_closeout.md` | drafted |
| 7 | Limitations, and what we withdrew | `retractions.md` | drafted |

**§3 and §4 are the spine.** They are stated before §5 so the result is read inside the bounds rather
than defended against them afterwards. §5 is the explained negative. §6 and §7 carry the
reconstruction deficit and the nineteen retractions.

## The result

**NOT DISTINGUISHABLE at every readable angle** — 0.39σ at 30°, 0.52σ at 45°; 60° is not readable.
The capability was not demonstrated, and four measurements say why it could not be.

## The three numbers a reader should leave with

| | bound | value |
|---|---|---|
| the comb limit | what an oblique ground truth can contain | `fill = t·cos θ / s`; 0.25 at 60°, **0** at 90° |
| the metric's resolution | what the statistic can distinguish | `blur/radius = √(eps·scale)` ≈ **0.26–0.30**, any dataset |
| the scrambled-section floor | what a section with no type information scores | **0.03–0.24**, and flat in `n` |

## Open editorial decisions

1. **The headline angle.** F2 selects θ\* = 60°; 60° is not readable on P2, and 30° and 45° are.
   Which angle leads §5's abstract sentence is an author's judgement about how much coverage a claim
   needs, and it is recorded as a judgement rather than derived.
2. **Whether §4.7's two-specimen scope is enough**, or the full eight-dataset sweep table from
   `reports/angle_budget.md` should be inlined.
