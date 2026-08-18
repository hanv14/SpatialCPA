# Open risks carried forward

Part of [PROGRESS.md](../PROGRESS.md).

## Open risks carried forward

| # | Risk | Raised | Owed to | Decision due |
|---|---|---|---|---|
| R1 | ~~**`ell_z` cannot be resolved by a 9-section stack** (fit 561 µm against a 200 µm ground truth)~~ — **DECIDED at T07, 2026-08-17.** Remedy **2** (calibrate `ell_z` at inference against observed between-section correlation) is adopted, **owed to T09**, with remedy **3** (treat the fit as a bracket endpoint and fail loudly) shipped alongside it as the guard; remedy 1 is rejected as the primary. The decision is evidence-based: T07 measured whether `L_cross` can serve as the training-time instrument remedy 2 might have used, and it **cannot** — see the R1 section below | T03 | T07 → T09 | **decided; implementation owed at T09** |
| R7 | **SEFL as a whole is unverified and, on every statistic that could be measured, harmful.** With `w_thick = w_prog = 0.2` a model trained at **T06's own budget and config** fails three T06 acceptance tests: detection MAD **0.0551** (< 0.05, T06 recorded 0.0191), mean-variance slope error **0.2838** (< 0.15, recorded 0.0084), Frobenius covariance **20.301** (recorded 9.316, baseline 7.563). All three SEFL weights therefore ship at **0** and A7 becomes an *addition* experiment | T07 | T10 (A7) | **at T10** — SEFL is used only if A7 shows a gain on the six target metrics |
| R6 | **`L_cross` is vacuous in v25 and its residual minimiser destroys the anatomical field.** Two crossing planes already emit bitwise identical expression (by construction, untrained); the only plane-dependent channel left is the augmentation pose, which T04 made pose-dependent on purpose. At `specs/07`'s `w_cross = 0.3` the generated section's per-gene variance falls to **0.065** of the real one's against **0.711** with SEFL off. `w_cross` ships at **0**; `loss_cross` is kept and pinned by a strict xfail | T07 | T10 (A7, E5) | **at T10**, or sooner if the spec's owner takes SPEC_QUESTIONS C19's option 1 |
| R2 | ~~GATE 2's attention is near-uniform (0.987 × log K)~~ — **CLOSED at T06.** With the flow-matching head trained the entropy falls to **0.8563 × log K** (a fall of 0.132, required 0.05) and stays above the 0.5 collapse line. The query is what changed: T04's probe queried with the field feature alone, T06's with `[F(p), fourier(p), type_emb, region_emb, z_embed]`, and a query that knows its own cell type can prefer a donor that shares it. | T04 | T06 | **closed 2026-08-16** |
| R5 | **C14 blocks the paper's headline novelty.** *(Sharpened: the table also has to be *built correctly* — see B19, four defects in `build_gene_meta` that produced four species' genes for a one-species request, all fixed and replayed, but the live mouse-only summary coverage is still unmeasured here.)* Open-vocabulary decoding has no positive evidence: zero-shot is r = −0.368 on the fixture (B18) because arbitrary gene names carry no text signal, and `resources/gene_meta.parquet` cannot be built here — every gene-annotation host is 403'd by the network policy. Needs one command on a networked machine (see C14). | T02, T06 | T10 (E1) | **before the first real run** |
| R4 | **Likelihood/fidelity divergence — one pathology, two heads.** *(T08 ran: the cure is **built and measured**, it is **not** enabled by default, and R4 stays **OPEN**. The covariance criterion is missed on both regimes — 11.02 vs 7.73 and 13.39 vs 11.38 — so the covariance claim is downgraded; but at 2400 steps the arm **with** the terms improves its covariance while the arm without it degrades, which is the first direct evidence that anything in this project can arrest the divergence. Decision owed to T09 §3's joint selection gate and reported by T10's A2, at two budgets.)* Fitting by likelihood alone makes the likelihood better and the *generated section* worse. On the **layout intensity** head (SPEC_QUESTIONS B10, T05): recovered intensity correlation 0.97 → 0.28 over 300 → 1200 steps while the Poisson NLL keeps falling. On the **expression** head (T06): 1200 → 2400 steps lowers the ZINB NLL 1.589 → 1.578 nats/pair while Frobenius covariance error goes 17.7 → 21.3 and detection MAD 0.056 → 0.069. Neither head has a term that could stop it. **`TRAIN_STEPS = 1200` is early stopping fitted to this fixture and will not transfer to real data.** | T05 (B10), T06 | ~~T08~~, T09, T10 (A2) | **T08 ran and did not close it**; the criterion is a strict xfail carrying 11.02 / 13.39, and the enable/disable decision moves to **T09 §3's joint selection gate** (`train_steps` × the metric weights, scored per dataset on internal LOSO), reported by **T10's A2** at both budgets. The 0 is a starting point, not a shipped decision |
| R3 | **The stack's ends reconstruct far worse than its interior.** Per-section R² was **0.2912** at the first section and **0.3642** at the last, against an interior mean of **0.4474** — a 20–35 % deficit. One-sided evidence at the volume boundary. | T04 | T09, T10 | **at T09** |

### R4 — likelihood/fidelity divergence: one pathology, two heads, and 1200 steps is not a fix

B10 and what T06 first recorded as a separate risk are **the same failure**, and they are merged here
so that nobody fixes one and thinks the class is closed. In both cases a head fitted by its own
likelihood gets monotonically better at that likelihood while the *generated section* gets worse:

| head | fitted by | likelihood over the budget | fidelity over the same budget |
|---|---|---|---|
| layout intensity (T05, **B10**) | inhomogeneous Poisson NLL | keeps falling | recovered intensity r **0.97 → 0.28** (300 → 1200 steps, default basis) |
| expression (T06, **R4**) | ZINB NLL | **1.589 → 1.578** nats/pair | Frobenius covariance **17.7 → 21.3**, detection MAD **0.056 → 0.069** (1200 → 2400 steps) |

Two partial mitigations exist and neither is a cure:

* **Capacity control on the intensity head** — `fourier_bands_for_lengthscale`, T06's answer to B10.
  Real and trainer-level (every `CTFFlow` gets it), but partial: r still decays 0.979 → 0.861, just
  2.6× less than the default basis's 0.835 → 0.527.
* **Early stopping on the expression head** — `TRAIN_STEPS = 1200` in `tests/test_expression.py`.
  **This is the part that must not be mistaken for a fix.** 1200 was chosen by measuring the fixture's
  own degradation curve, which is a form of fitting to the test set, and on a real dataset there is no
  such curve to read: the sections are fewer, the panel is wider, the degradation sets in at a
  different step count, and nobody will know where. A stopping signal has to come from *inside the
  training run* — internal LOSO over training sections, which is what `specs/08` §4 builds.

**Why T06 could not fix it.** `forward_train` returns `recon`, `cfm`, `size`, `layout`, `distill` and
`tv_z`. Not one of them is a statement about the distribution of a generated section; every one is a
statement about the training data. The terms that would be are T08's — `w_autocorr`, `w_profile`,
`w_distribution`, all sitting in `Config` at 0.5 with nothing yet to weight — and the mean–variance
and detection calibrators are T09 §2's.

**The success criterion is now written into `specs/08`** as
`test_metric_losses_close_the_covariance_loss`: model Frobenius covariance error below the
independent-donor baseline's **7.783** on the default holdout **and** holding at `consecutive-3`
(where the model currently scores 17.7 against 11.3). If T08 cannot deliver both, the covariance claim
is downgraded to a mechanism claim and `specs/10` §2 frames it that way — a decision to record, not to
drift into.

### R3 — the boundary is a different regime, and it is not a fixture artefact

Every serial-section dataset has two ends, and a cell at either of them has training sections and
retrieval donors on **one side only**. Measured on the gate fixture (coronal arms at the common
`n` = 1011): 0.2912 / 0.4234 / 0.4364 / 0.4280 / 0.4567 / 0.4532 / 0.4625 / 0.4715 / 0.3642 — the
interior is homogeneous to within 0.0481, which is inside the criterion's own resolution, and the
whole spread is the two boundary sections.

This was found while diagnosing GATE 2 and it is the reason the criterion needed amending, but it
does not stop being true once the gate passes. Two places it lands:

* **T09** generates at arbitrary planes, routinely including planes at or beyond the outermost
  sections, where the model extrapolates rather than interpolates. The uncertainty gate is the
  natural place to surface it; if the latent variance it already estimates is *not* elevated there,
  that is itself a finding. Written into `specs/09` §1.
* **T10** must stratify the six headline metrics by distance to the boundary rather than pooling. A
  method strong in the interior and weak at the ends is a different claim from one that is uniformly
  mediocre, and `alternating` never holds out an end section while `consecutive-5` on a short stack
  pushes the held-out run close to one. Written into `specs/10` §4.

**R1 update at T04.** GATE 2 could not test it and did not: the probe is deterministic and never
queries the GRF, so a wrong `ell_z` has no path into the oblique-parity number. The risk is unchanged
and the decision is still owed before T07 — T04's oblique numbers, which were supposed to inform the
choice, turn out not to bear on it. Remedy 2 (calibrate `ell_z` against observed between-section
correlation) is therefore still the inclination, and it will have to be decided on T07's `L_cross`
evidence rather than on this gate's.

### R1 — **DECIDED at T07**: calibrate `ell_z` at inference (remedy 2), gate on it (remedy 3), and stop waiting for SEFL to fix it

**The decision.** Remedy **2** — calibrate `ell_z` at inference against the *observed*
between-section correlation, exactly as T09 already calibrates `ell_xy` against Moran's I — with
remedy **3** shipped with it: the fitted value enters T09 as a **bracket endpoint**, not a value, and
a volume that cannot constrain it fails loudly rather than generating over-smooth z structure.
Remedy 1 (a joint fit under an anisotropy prior) is **rejected as the primary**: it regularises the
under-determined direction with a prior on tissue anisotropy, i.e. it decides by assumption the very
quantity the design says the anisotropy exists to measure. It is still fine as remedy 2's
*initialiser*, which is where it lands.

**The evidence T07 owed, and it is negative.** T04 could not test R1 (its probe never queries the
GRF), and the note carried into T07 said the choice "will have to be decided on T07's `L_cross`
evidence". That evidence now exists: `L_cross` **cannot** discriminate `ell_z`. Measured on the
fixture with everything else fixed, `L_cross` across `ell_z` = 100 / 200 / 561 / 1000 µm varies by
less than **25 %** relative — the criterion `test_cross_loss_sensitivity_to_ell_z` now asserts, so a
future change that made it sensitive re-opens this decision instead of passing silently. The reason
is structural and was predictable once T03 landed: **the GRF is continuous in 3-D, so both branches
of `L_cross` receive the identical noise realisation whatever `ell` is**, and the loss measures
conditioning disagreement, not the prior. The same argument applies to `L_thick`.

So no SEFL term can calibrate `ell_z`, and the hypothesis that T07 would supply the instrument is
closed. What remains is T09's, and it is now written down as such.

**Why it still matters** (unchanged from T03, restated because the decision rests on it):

A 9-section stack at 50 µm spacing spans **400 µm**, so the largest along-z lag the variogram can
form is 400 µm. A 200 µm correlation length has decayed to `matern(2) ≈ 0.14` there — the empirical
variogram reaches only **35 %** of its fitted sill at the largest lag (60 % on the narrow fixture), so
the fit is extrapolating past its data and reads high. `fit_lengthscale_from_sections` warns
(`LengthscaleFitWarning`, `Config.variogram_min_saturation = 0.75`) rather than returning the number
quietly, and GATE 1 records it.

**This is real-volume geometry, not a fixture artefact.** Serial-section datasets are tens of
sections at 10–50 µm; the z extent is small by construction and always will be. It matters because
SEFL's claim to oblique correctness rests on `ell` being *anisotropic*: an oblique plane mixes the
in-plane and along-z correlation structure, so a `ell_z` that is 2.8× too long makes a 45° section's
in-plane correlation wrong by a factor that depends on the angle — precisely the error the design
says the anisotropy exists to remove (`design/v23_sectioning_equivariance.md` §2, point 3). T04's
GATE 2 (oblique parity ≥ 0.90 × axis-aligned) is the first place a wrong `ell_z` can show up, and
T07's `L_thick` and `L_cross` are the first places it can be trained on.

The three candidate remedies, and what became of each:

1. **Joint fit under a shared anisotropy prior.** Fit `(ell_xy, ell_z)` together against the in-plane
   *and* along-z variograms with a prior on the ratio `ell_z / ell_xy` (tissue anisotropy is bounded
   in practice), so the under-determined direction is regularised by the well-determined one rather
   than left to extrapolation. Cheapest; the fit is already a grid scan, so it becomes a 2-D scan.
2. **Calibrate `ell_z` at inference against between-section correlation**, the way T09 already
   calibrates `ell_xy` against Moran's I: hold out a flanking section, generate it, and match the
   *observed* section-to-section correlation decay rather than the fitted one. Leakage-free by the
   same construction as T09 §2, and it measures the quantity that actually matters instead of a
   parameter of a model of it.
3. **Treat the fitted value as an upper bound and gate on it.** Keep the warning, pass `ell_z` to
   T09 as a bracket endpoint rather than a value, and add an `ell_z` criterion to T09's gates so a
   volume that cannot constrain it fails loudly instead of silently generating over-smooth z
   structure.

**Adopted: 2, with 1 as its initialiser and 3 as the guard.** The criterion T09 inherits: a run whose
`ell_z` bracket cannot be closed against observed between-section correlation reports
`target_unreachable` and the selected `ell_z` is recorded as an upper bound, not a fitted value.
