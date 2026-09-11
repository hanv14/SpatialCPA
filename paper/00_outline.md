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
| — | Abstract (280 words) | §3-§8 | drafted; long form in `00_abstract_extended.md` |
| 1 | Introduction | — | drafted |
| 2 | Method | T03–T09 | drafted |
| 3 | **Two bounds on oblique evaluation** | `the_comb_limit.md`, `metric_resolution.md` | drafted |
| 4 | **What geometry an oblique evaluation requires** | `angle_budget.md` | drafted |
| 5 | **Off-axis evaluation, and why it could not be resolved** | `oblique_demo.md` | drafted, scored, final |
| 6 | What the method does not do | `architecture_ceiling.md`, `t09_closeout.md` | drafted |
| 7 | Limitations, and what we withdrew | `retractions.md` | drafted |
| 8 | **Related work** | the tier-1 comparator re-score, `reference/SpatialZ.py`, `benchmark.md` | drafted |

§8 names the published competitor and carries the six-method tier-1 table, re-scored on the pinned
evaluator, against the copy floor and the oracle. It states that SpatialZ beats this method on five
of six and why §6 predicts that — and it reports the table's larger finding: **on four of the five
metrics that have a floor, no method in it reaches a model-free copy of the flanking sections.**

**§3 and §4 are the spine.** They are stated before §5 so the result is read inside the bounds rather
than defended against them afterwards. §5 is the explained negative. §6 and §7 carry the
reconstruction deficit, the eighteen retractions and the two self-caught fabrications.

## Figures

Five are rendered into `paper/figures/` by `scripts/make_figures.py` and placed in `DRAFT.md` by
`scripts/assemble_draft.py`. Every number in them, captions included, is read from
`reports/oblique_demo.json` at draw time rather than typed. `FIGURES.md` is the list, including the
one deliberate omission (the qualitative side-by-side, refused with its reason) and the standing
condition on F3.

| | what | state |
|---|---|---|
| F1 | the comb, per scored angle | rendered (extent form) |
| F2 | fill and resolution against angle | rendered |
| F4 | the footprint | rendered (extent form) — **point-cloud version required before submission** |
| F5 | the scrambled-section floor, and the arms against it | rendered |
| F6 | the result, as a forest | rendered |
| F3 | the angle budget across specimens | supplementary, not drawn — §4.4's table carries it |
| F7 | the architecture ceiling | not drawn |

## The result

**NOT DISTINGUISHABLE at every readable angle** — 0.39σ at 30°, 0.52σ at 45°; 60° is not readable.
The capability was not demonstrated, and four measurements say why it could not be.

## The three numbers a reader should leave with

| | bound | value |
|---|---|---|
| the comb limit | what an oblique ground truth can contain | `fill = t·cos θ / s`; 0.25 at 60°, **0** at 90° |
| the metric's resolution | what the statistic can distinguish | `blur/radius = √(eps·scale)` ≈ **0.26–0.30**, any dataset |
| the scrambled-section floor | what a section with no type information scores | **0.03–0.24**, and flat in `n` |

## Editorial decisions, settled

1. **The headline angle: 45°.** θ\* = 60° is the largest *scorable* angle but fails P2, and an angle
   whose preconditions fail has no readable score. Leading with θ\* would report a number the
   protocol had already disqualified. Recorded as a judgement, in §5.2.
2. **The full eight-dataset sweep is inlined** in §4.4, with the four unbuilt datasets as named rows
   carrying their reasons. The result is **bimodal** (three specimens at 5°, one at 60°) and does not
   track aspect ratio — which is §4's actual claim, and the reason F3 stays a table.
3. **`merfish_thick_hypothalamus`'s budget is 60°, not 90°.** The 90° in the sweep is what a slab
   thickness defaulted to the section spacing gives; it is `retractions.md` R5, reappearing through a
   sweep that predates R5's own fix, and is withdrawn again as R18. §4.7 carries the note.
4. **F3 is supplementary and F4's extent form is internal-read-only** — see `FIGURES.md`.
