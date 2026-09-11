# 6. What the method does not do

*Volunteered, ahead of the demonstration section rather than after it. Everything here was measured
by us, pre-registered before it was measured, and several items withdraw earlier claims of our own.*

## 6.1 It does not reach the copy floor on axis-aligned reconstruction

On tier-1 STARmap, `paper_morans_pearson`: an optimal copier of the flanking sections scores
**0.9836**; the method scores **0.5574**. A diagnostic ladder that hands the model progressively more
of the truth puts the **architecture ceiling at 0.8369** — that is, even given oracle means, the
decoder's own dispersion and dropout cannot reproduce the copy. The deficit is not a tuning failure
and no route inside this architecture closes it. The diagnostic programme establishing this is closed
and reported (`reports/architecture_ceiling.md`, `reports/diagnostic_programme_closed.md`).

**We say this first because §5's claim is not a reconstruction claim.** The contribution is that
sections become well-defined and scorable off-axis, which no previous method offers at all. It is
not that they are better on-axis. They are not.

## 6.2 The intensity-field layout is refuted, and we report the arm built to fail

`layout_mode=field` scores **0.6607** against `resample`'s 0.7546 and a copy floor of 0.7765 — below
the model-free floor on the metric the layout head exists to win. A pre-registered test supplying the
cell count externally (removing the suspected cause) did **not** rescue it: R11's "pattern good,
scale wrong" diagnosis is refuted, and the flanking-density count estimator is accurate to 4%.
`resample` ships, and `field` is reported as ablation A4.

## 6.3 The layout is wildly section-dependent, and that is the useful finding

Holding cell types at oracle and changing only positions, across-section spread is **0.4306** with
model positions and **0.0140** with copied positions — a 31× difference, with worst across-seed
spreads of 0.095 and 0.000. The variation is across *sections*, not across seeds. The model's layout
is not uniformly worse than copying; it is inconsistent, and inconsistency is a different problem
with different remedies.

**Two readings we withdrew.** We first called this positional *instability*; the disproof is that the
oracle-position arm has an across-seed spread of exactly zero, which shows the metric is nearly blind
to the expression head and that the argument we had made did not support the word. And we do not
claim that oracle types on copied positions "beat the copy" at 0.8375 against 0.7765: that arm
carries ground-truth types transferred onto donor positions, a partial oracle on precisely the
quantity being scored.

## 6.4 Everything in §6 is bounded by §3.2

All of the above is `paper_celltype_localization` or `paper_morans_pearson`. The first is blind below
~110 µm. **None of these numbers is evidence about placement finer than that**, including the ones
that favour us.
