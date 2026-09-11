# Figure list

**Producible now** = every number is in a committed artifact under `reports/`; the figure is a
rendering job, no new measurement. **Needs a run** = the data does not exist yet, and what it would
change is stated.

---

## F1 — The comb (§3.1) · **PRODUCED** · `paper/figures/F1_comb.svg`

**Shows** why an oblique ground truth is not a section: one panel per scored angle, drawn to scale
from the measured `comb_period_um` and `comb_gap_um`, so the strata and the gaps between them are
the measurement rather than an illustration of it.

**Drawn in extent form, not point-cloud form.** The originally-specified version — the cells
themselves, highlighted in the plane's `(u, v)` frame — needs per-cell coordinates, which
`--emit-coords` now serialises but which no committed run has yet produced. The extent form makes
the same point from numbers that are already published, and the point-cloud form is a drop-in
replacement once a run carries `coords`.

**Why it earns its place.** This is the paper's first bound and it is far easier to see than to read.
A reader who sees four stripes understands `fill = t·cos θ / s` immediately.

**Source** `reports/oblique_demo.json` — `comb_gap_um`, `comb_period_um`, `n_strata`, `fill_ratio`,
`section_spacing_um`.

---

## F2 — Fill and resolution against angle (§3) · **PRODUCED** · `paper/figures/F2_bounds.svg`

**Shows** both bounds on one axis. Panel A: `fill(θ) = t·cos θ / s`, measured at 0/30/45/60/90°,
falling to 0 at 90°, with the angles that qualify marked solid and the ones that do not marked
hollow. Panel B: `blur/radius` against θ, flat at ≈ 0.26–0.30 —
deliberately flat, because the point is that it is a constant of the metric and not of the tissue.

**Why.** The contrast between a bound that collapses with angle and one that does not is the
structure of §3, and the flat line is the surprising half.

**Source** `reports/oblique_demo.json` — `fill_ratio`, `metric_blur_um`, `metric_radius_um`.

---

## F3 — The angle budget across specimens (§4) · **producible now** · *dot plot*

**Shows** each readable dataset as a dot at (in-plane : depth ratio, largest scorable angle), with the
four unbuilt datasets on their own row labelled as not read.

**⚠️ The prediction in the previous version of this list was wrong, and the correction is the
figure's point.** It said to expect *"a single monotone band — depth buys angle"*. The sweep arrived
and the result is **bimodal**: three specimens at exactly 5°, one at 60°, nothing between, and
**nothing at 10°**. It also does **not** track the aspect ratio — the three that stop at 5° span
21.6 : 1, 24.8 : 1 and 34.1 : 1 in no order, and `deep_starmap` has seven times the cells of
`starmap_visual_cortex` with the same budget. A band would have suggested a dial a preparation can
turn; four points in two clusters say the opposite, which is §4's actual claim.

With four points this is honestly a table. **Decided: F3 stays supplementary and is not drawn now.**
The standing condition for ever drawing it is that the axis be **`depth`, not aspect ratio** — depth
is what separates the clusters (170 µm against 66, 83, 125) and is the variable the section argues
for; aspect ratio is the variable that visibly does *not* explain them. §4.4's table carries the
argument in the main text meanwhile.

**Source** `reports/angle_budget.md` / `.json`. Now committed. ⚠️ **The sweep's records are stale**
(`retractions.md` R18): they predate the runner's own slab-thickness fix and default `t = s`, so the
90° in the JSON is not the budget the paper quotes. **A figure drawn from this file must use 60° for
`merfish_thick_hypothalamus`**, per §4.7, or it will reprint a withdrawn number. `angle_budget.py
--audit` flags the file.

---

## F4 — The footprint (§5.3) · **PRODUCED** · `paper/figures/F4_footprint.svg`

**Shows**, at all three scored angles, the in-plane extent along the comb axis of each arm against
the plane's own footprint (shaded): the ground truth a 270 µm ribbon at 45°, `resample-pd` the same
ribbon at 1.00×, `copy-nearest-z` a 1328 µm face at 4.92×.

**⚠️ Extent form is an internal read only. The point-cloud version must be produced before
submission** — author's decision, recorded here so it cannot be forgotten at the deadline. Seeing the
cells is more persuasive than seeing a bar, and this is the figure carrying the paper's strongest
positive claim, so it is the one that most deserves the stronger form. `--emit-coords` exists; the
run is a re-serialisation, not a re-measurement, and `--verify-unchanged` asserts that.

The extent form has one advantage worth keeping either way: it shows all three angles at once, where
the point cloud shows one. **The submitted figure should keep both** — the point cloud at 45° as the
main panel, the extents across angles as an inset or companion.

**Why it is the most important.** It is the paper's strongest positive claim and the one a reviewer
will press on. The numbers — 4.92× and 75% outside — are convincing; the picture is unarguable. It
should also make our own arm's 22% honestly visible, so the figure does not overstate the contrast
its own caption warns against.

**Caption must carry** the §5.3 caveat verbatim: not a section *for the purpose of scoring a section*,
not as a general claim about the previous method's output.

**Source** `reports/oblique_demo.json` — `footprint` per arm per angle (`u_extent_um`,
`u_extent_ratio`, `frac_outside`, `truth_u_extent_um`).

---

## F5 — The scrambled-section floor (§5.6) · **PRODUCED** · `paper/figures/F5_null_floor.svg`

**Shows**, panel A, the self-null against `n` for the three scored angles — flat, not falling — and,
panel B, the two arms at each angle on the *same* axis, with the difference each comparison rests on
printed between them.

**⚠️ Corrected from the previous version of this list**, which said the visual point was that "the
arm scores sit **inside** the band". They do not, and the figure must not say so. Measured: our arm
is inside the band at 30° (0.130) and 45° (0.117); **the baseline's scores are above it at every
angle** (0.249, 0.334, 0.452). What is true, and what the figure now says, is that the **arm-to-arm
differences** — 0.046, 0.118, 0.217 — are of the same size as the band's own width, 0.203. That is
the §5.5 result and it survives the correction; the stronger claim did not, and would have been a
figure caption asserting more than the measurement.

60° is greyed and labelled, because its null control fails P2.

**Source** `reports/oblique_demo.json` — `self_null` (four `n` per angle, three seeds each), `arms`,
`null_ceiling`.

---

## F6 — The result (§5.5) · **PRODUCED** · `paper/figures/F6_result.svg`

**Shows** the difference `resample-pd − copy-nearest-z` at each scored angle with its combined
precision bound, against a zero line. Every interval spans zero. 60° is drawn greyed and labelled
*not readable — P2 fails* rather than omitted.

**Why.** One panel showing every interval crossing zero is the honest form of this result, and
showing the disqualified angle greyed rather than dropped is the protocol made visible.

**The figure reproduces §5.5's table from the JSON**, which is the point of drawing it this way:
−0.1183 ± 0.2999 (0.39σ), −0.2173 ± 0.4171 (0.52σ), −0.0462 ± 0.6492 (0.07σ). The bounds are
recomputed at draw time as `√(se_ours² + se_base²)` from the per-arm `jk_se`, not copied from the
text, so the figure and §5.5 cannot disagree.

**Caption carries** the §5.4 statement verbatim: these are an upper bound on precision, not
confidence intervals, and they are not narrowed.

**Source** `reports/oblique_demo.json` — `arms` (`celltype_localization`, `jk_se`), `null_ceiling`.

---

## F7 — The architecture ceiling (§6.1) · **producible now** · *ladder / bar*

**Shows** the diagnostic ladder on tier-1: the method at 0.5574, the architecture ceiling at 0.8369,
an optimal copier at 0.9836, oracle at 1.0.

**Why.** §6.1 is the volunteered deficit and it is more credible shown than asserted — the gap
between 0.8369 and 0.9836 is the part no tuning closes.

**Source** the chain-diagnostic reports under `reports/`.

---

## Recommended set

**Main text, five:** F1, F2, F4, F5, F6. F1 and F2 carry §3; F4 carries §5.3; F5 and F6 carry §5.5.

**Supplementary:** F3 and F7. F3 stays supplementary — with four points in two clusters it is a
table, and §4.4 already prints that table (see the entry above for the form that would earn a panel).

**Produced this round**, from committed JSON, by `python scripts/make_figures.py --out paper/figures`:

| | file | form |
|---|---|---|
| F1 | `paper/figures/F1_comb.svg` | extent (point-cloud form pending a run with `--emit-coords`) |
| F2 | `paper/figures/F2_bounds.svg` | as specified |
| F4 | `paper/figures/F4_footprint.svg` | extent (as above) |
| F5 | `paper/figures/F5_null_floor.svg` | as specified, **after the correction in its entry** |
| F6 | `paper/figures/F6_result.svg` | as specified |

**Before submission:** F4 must be redrawn in point-cloud form (see its entry). Nothing else in this
list is pending.

SVG, rendered without a plotting library — this container has neither matplotlib nor a rasteriser.
Every number in them is read from `reports/oblique_demo.json` at draw time rather than typed in,
including the numbers inside the captions, so a figure cannot drift from the report it is drawn
from.

F6 was listed as blocked in the previous draft of this file, on the belief that `jk_se` had been
retained at only one angle. That was wrong — it is present for every arm at every scored angle — so
F6 is produced, and it reproduces §5.5's three rows to four decimal places.

---

## Deliberate omissions

Figures a reader may expect and will not find, with the reason — so that an absence is not read as
an inability.

### A qualitative side-by-side of a generated oblique section against a real one

**Not produced, and it should not be.** The panel a reader expects here is our generated section
beside the real one at the same angle, so the eye can judge the resemblance. **There is no real one.**
That is the premise of §3.1: an oblique plane through a stack of serial sections passes through
tissue only inside the slabs, so the "ground truth" at 45° is four strata covering 0.35 of the plane,
not a section. There is nothing to put in the right-hand panel that is not itself a comb.

**And the version that could be drawn would mislead.** Rendering the comb as a filled sheet — by
interpolating between strata, or by drawing markers large enough to close the gaps — produces exactly
the picture the paper spends §3.1 arguing does not exist, and it would do so in the one format
readers trust most. A figure that quietly contradicts the paper's first bound is worse than no
figure.

**What stands in its place.** F1 shows the comb at the same three angles, to scale and from the
measured periods, which is the honest form of "here is what an oblique ground truth looks like". F4
shows the spatial claim the side-by-side would have been used to make — that the previous method's
off-axis output is not a section of the plane — as measured extents, where it is checkable rather
than impressionistic.

**This is a refusal, not a limitation.** The coordinates needed to draw the misleading version are
available; `--emit-coords` would serialise them. We are choosing not to draw it.

---

## What is not producible without a new run

- **F1 and F4 in point-cloud form.** `scripts/oblique_demo.py --emit-coords` now retains each arm's
  `(u, v)` coordinates, and `--verify-unchanged` asserts that every previously reported number is
  bitwise identical to the committed record, so the run that produces them is a re-serialisation and
  not a re-measurement. Until then both figures are drawn in extent form and say so.
- **F3 as a panel** rather than a table, which needs no run at all. Decided against for now; the
  condition under which it would be drawn is recorded in its entry.

Everything else in this list is a rendering of committed JSON, and five of the seven are rendered.
