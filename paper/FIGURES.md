# Figure list

**Producible now** = every number is in a committed artifact under `reports/`; the figure is a
rendering job, no new measurement. **Needs a run** = the data does not exist yet, and what it would
change is stated.

---

## F1 — The comb (§3.1) · **producible now** · *schematic + measured overlay*

**Shows** why an oblique ground truth is not a section. Left: a stack of four sections and a tilted
plane, with the cells the plane passes near highlighted — visibly four strata, not a filled sheet.
Right: the same cells in the plane's own `(u, v)` frame, showing the comb directly.

**Why it earns its place.** This is the paper's first bound and it is far easier to see than to read.
A reader who sees four stripes understands `fill = t·cos θ / s` immediately.

**Source** `reports/oblique_demo.json` — `comb_gap_um`, `comb_period_um`, `n_strata`, `fill_ratio` per
angle. The cell positions need one geometry pass (`--score` not required).

---

## F2 — Fill and resolution against angle (§3) · **producible now** · *two-panel line plot*

**Shows** both bounds on one axis. Panel A: `fill(θ) = t·cos θ / s` for the two specimens, falling to
0 at 90°, with the scored angles marked. Panel B: `blur/radius` against θ, flat at ≈ 0.26–0.30 —
deliberately flat, because the point is that it is a constant of the metric and not of the tissue.

**Why.** The contrast between a bound that collapses with angle and one that does not is the
structure of §3, and the flat line is the surprising half.

**Source** `reports/oblique_demo.json` — `fill_ratio`, `metric_blur_um`, `metric_radius_um`.

---

## F3 — The angle budget across specimens (§4) · **needs `reports/angle_budget.md`** · *dot plot*

**Shows** each built dataset as a dot at (in-plane : depth ratio, largest scorable angle), with the
unreadable datasets on their own row labelled as not built. The expected shape is a single monotone
band — depth buys angle — with `merfish_thick_hypothalamus` alone in the top-right.

**Why.** It is §4's whole argument in one panel: this is a property of preparation, not of method.

**Blocked on.** The eight-dataset sweep report, which has not been read into the draft. *No new
measurement is required — the run has already happened.* Until it arrives the figure would have two
points, which is a table, not a figure.

---

## F4 — The footprint (§5.3) · **producible now** · *the paper's most important figure*

**Shows**, at 45°, three point clouds in the plane's `(u, v)` frame, on one pair of axes:
the ground truth (a 270 µm ribbon), `resample-pd` (the same ribbon, offset), and `copy-nearest-z`
(a 1328 µm face spilling far outside both). A shaded band marks the plane's own footprint.

**Why it is the most important.** It is the paper's strongest positive claim and the one a reviewer
will press on. The numbers — 4.92× and 75% outside — are convincing; the picture is unarguable. It
should also make our own arm's 22% honestly visible, so the figure does not overstate the contrast
its own caption warns against.

**Caption must carry** the §5.3 caveat verbatim: not a section *for the purpose of scoring a section*,
not as a general claim about the previous method's output.

**Source** `reports/oblique_demo.json` — `footprint` per arm per angle. Cell coordinates need one
scoring pass with the arms retained.

---

## F5 — The scrambled-section floor (§5.6) · **producible now** · *dot plot with the null band*

**Shows** self-null against `n` for the three angles — flat, not falling — with each angle's arm
scores overlaid on the same axis. The visual point is that the arm scores sit **inside** the band a
section with randomised labels reaches.

**Why.** It carries the §5.5 result on its own. A reader who sees the arm scores inside the scrambled
band needs no further argument for "not distinguishable", and it is the figure that makes the
negative result a finding rather than an absence.

**Source** `reports/oblique_demo.json` — `self_null` (four `n` per angle, three seeds each) and
`arms`.

---

## F6 — The result (§5.5) · **producible now** · *forest plot*

**Shows** the difference `resample-pd − copy-nearest-z` at each readable angle, with its combined
precision bound, against a zero line. 45° leads. 60° appears struck through and labelled *not
readable — P2 fails*.

**Why.** One panel showing every interval crossing zero is the honest form of this result, and
showing the disqualified angle struck through rather than omitted is the protocol made visible.

**Caption must say** that the bounds are an upper bound on precision, not confidence intervals
(§5.4).

**Source** `reports/oblique_demo.json` — `arms`, `jk_se`, `precondition_checks`.

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

**Supplementary:** F3 (when the sweep report arrives — it would move to the main text if the band is
as clean as expected, since it is §4's argument in one panel) and F7.

**Not proposed.** A qualitative side-by-side of a generated oblique section against a real one. There
is no real oblique section to put beside it — that is the premise of §3.1 — and a figure implying
otherwise would undercut the paper's own first bound.

---

## What is not producible without a new run

Only **F3**, and it needs a *report already generated*, not a new measurement. Everything else is a
rendering of committed JSON, except that **F1 and F4 need cell coordinates**, which the current
artifacts summarise but do not store. Emitting them is a serialisation change to
`scripts/oblique_demo.py` — no re-measurement, no change to any number — and I would rather flag it
now than have it surface as "one more run" during figure preparation.
