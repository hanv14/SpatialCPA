# 7. Limitations, and what we withdrew

## 7.1 Limitations

1. **One specimen carries §5, and the replication candidate has been checked and fails.**
   `merfish_thick_hypothalamus` is the only built volume whose geometry admits a scorable oblique
   angle. **`merfish_thick_cortex` — the closest analogue, same holdout design, same thick-slab
   preparation at half the thickness — was measured and stops at 5°**, failing G2 at 10° with a
   largest type of 207 against the metric's 250 (§4.4). `deep_starmap`, with seven times the cells,
   also stops at 5°. So the result rests on one specimen and the most likely replication has been
   attempted and did not succeed. A pre-build screen rules two further candidates out on arithmetic
   before they are built: an oblique strip retains 2–4% of a volume, so a specimen needs ~8 300
   cells in its largest type, and those two hold fewer than 2 000 in total.
2. **Four sections.** The training volume has four, so the comb has four teeth. §3.1's bound is
   correspondingly tight and every fill figure is specific to this geometry.
3. **`celltype_localization` resolves only ~0.26–0.30 of the tissue radius** — 186 µm on a coronal
   section (§3.2). Every score in this paper inherits it, and so does every score in the literature
   it is compared against.
4. **The oblique ground truth is assembled, not observed.** No obliquely-cut section exists in any
   dataset we know of; §3.1 is the statement of what the assembly can and cannot be.
5. **The expression head is not evidenced by §5.** `celltype_localization` touches generated
   expression only through pose estimation; an arm with fixed positions and types scores identically
   across generation seeds. §5 is a claim about layout.

## 7.2 What we withdrew, and why it is here

**Eighteen** claims of our own were retracted during this work, each with the evidence that
disproved it (`reports/retractions.md`). The eight in the first table below are those that would
have changed a number a reader of this paper would otherwise have seen; the rest are listed in that
file. **Two further withdrawals follow in a second table, and they are the ones to read first** — not
measurements that turned out wrong, but a number that was never measured and a caption that claimed
more than its panel showed. We give all of them because a reader who cannot see what an analysis
rejected cannot calibrate what it accepted.

| | claim withdrawn | what disproved it |
|---|---|---|
| R1 | a coronal reference row's cell count and type denominator | the plane sat exactly between two sections and admitted both on a floating-point tie; three independent arithmetic checks |
| R2 | "the layout's positions are unstable across sections" | an oracle arm's across-seed spread is exactly zero — the metric is nearly blind to the expression head |
| R3 | an oracle arm "beating the copy floor" | it carries ground-truth types on donor positions: a partial oracle on the scored quantity |
| R4 | that an exact tie between equidistant sections was an edge case | it is *every* target plane in an evenly spaced stack; two tests caught it before a run did |
| R5 | "this specimen clears 90°" | measured on a slab 2.1× too thick, because a leakage-guarded input's section spacing is a multiple of the real slab pitch |
| R6 | a screening ratio | mixed a full-dataset cell count with a training-volume count |
| R7 | a residual-modulation table | an FFT on a discrete grid, whose value moved two orders of magnitude with the grid size against a true value of 3 × 10⁻¹² |
| R18 | "this specimen clears 90°", **a second time** | the cross-dataset sweep that reports it predates R5's own fix: its records carry no `thickness_source`, a field only the fixed runner writes |

**R5, R7 and R18 are the three that would have reached print.** R5 would have put a headline claim
at 90° on a doubled slab; R7 would have published floating-point noise as a measurement; R18 is R5
returning through a report generated before its fix, and it was still in §4's table while this paper
was being assembled. R5 and R7 were caught by a self-check that asserted a *margin* rather than a
value — a practice we would recommend to anyone reporting a derived quantity. **R18 was not caught by
any check at all**: it was caught by drawing a figure, which forced the thickness to be read out of
the record instead of off the table.

Two further withdrawals are of a different kind and are listed separately, because they are not
measurements that turned out wrong. **Neither was caught by a check; both were caught by preparing a
figure**, which is the only reason they are here rather than in the submitted paper.

| | what was withdrawn | why it is a different failure |
|---|---|---|
| **an invented number** | §4.7 stated that a gate failed *"at 159 cells against 250"*. **That figure is in no committed report.** The measured value is 187, it is printed in §4.4's own table, and 159 appears nowhere in `reports/` except as an unrelated `fill` value in a different specimen's table | not a measurement error — **a number that was never measured, written in the voice of one that was**, in a sentence whose argument did not need it. The surrounding claim (the verdict is unaffected) was true; the evidence offered for it was fabricated |
| **a caption claiming more than its measurement** | the specification for Figure 5 said its visual point was that the arm scores *"sit **inside**"* the scrambled-section band. They do not: ours is inside at 30° and 45°, **the baseline's is above it at every angle**. The defensible claim — that the arm-to-arm differences, 0.046–0.217, are the size of the band's own width, 0.203 — was available the whole time and is what the figure now says | not wrong about the data — **wrong about what the data licensed**, in the one place a reader cannot check it against a table. A caption is read as a summary of the panel and is rarely audited against the source; a figure is where an overclaim is least likely to be caught and most likely to be believed |

**These are the two failures a reader should weigh this paper's other numbers against**, because they
are the two that no procedure in this work caught. The retraction table above is evidence that the
checks work; these two are evidence of where they do not reach. A self-check asserts a *relation*
between quantities and cannot tell that a quantity was never measured, nor that a sentence about a
panel says more than the panel shows. **Both were found by drawing the figure**, which forced the
numbers to be read out of the record instead of off the prose — the closest thing to a control we
have for this class of error, and it is not one, because it only works where a figure happens to be
drawn.

## 7.3 Four methodological rules this work paid for

- **Vary the implementation knob before reporting the number.** A convergence check on R7's FFT cost
  four lines and would have caught it immediately.
- **A default is a claim about the data.** R1 and R5 are the same defect: a rule that is right in
  general ("a section's thickness is the volume's section spacing") and wrong on the particular input
  (a leakage-guarded volume with alternate sections removed). Both were in a *reference* quantity
  rather than in a gate, where they are hardest to see.
- **A retraction is closed by a re-run, not by a patch.** R5 recorded its runner as fixed and the
  fix was real, but §4's sweep had already been run and was never redone, so the withdrawn number
  reappeared in a later report as R18. Nothing in `reports/` distinguished a pre-fix run from a
  post-fix one until we looked for a field only the fixed runner writes. **Make the fix change the
  artifact's shape, not only its numbers** — then a stale run is visible without recomputing it.
- **A warning is not a control. It is only a control where it lands in front of the number it
  qualifies.** This is the defect that cost the most, and it happened twice, in two different shapes.

  The post-fix sweep record (R18) carries, in its own `thickness_source` field, the sentence *"on a
  leakage-guarded input this OVERSTATES the slab"*. The budget it qualifies sits in the **same
  record**, and was read off and carried into a paper section. **The warning was in the
  artifact, and the number it qualifies was read off anyway.**

  Earlier, the `deep_starmap` fit's spatial-collapse alarm fired at **122 training steps including
  the last**, and its spatial *inversion* check fired at 79 — into `stderr`, while the report that
  run produced read as a clean measurement and said nothing about it. The alarm was built, it was
  correct, and it worked. It fired where nobody was looking.

  These are the same failure: a correct warning in a channel the reader of the number does not read.
  The remedy in both cases is the same and it is not "look harder" — **a qualifier must travel with
  the quantity it qualifies, in the same field a consumer reads, or the run must refuse to report
  the quantity at all.** A run whose alarm fired should not emit a clean-looking headline; a record
  whose thickness is assumed should not emit a budget as a bare number. We have applied this to the
  second case (`angle_budget.py --audit`) and not yet to the first.
