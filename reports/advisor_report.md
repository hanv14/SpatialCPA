# SpatialCPA-v25-Gen — advisor report

**2026-09-07.** Regenerated from `reports/t09_closeout.md` and the record, not from the earlier
draft, which predates A9, the `marker_field_r` correction and the collapse-alarm finding, and which
carried a framing since shown to be wrong (§7).

**Every number below names the file it comes from.** Where no current measurement exists, this
report says so rather than substituting an older one. §8 splits the provenance into three tiers:
**8a** results whose artifacts exist but are **not yet committed**, **8b** results that were measured
but whose files are not held — attested by a progress entry alone — and **8c** why the split matters. A ⚠️
against a number in the body means it is 8b.

⚠️ **The six-metric table changed materially since the earlier draft** — `marker_field_r` 0.638 →
**0.683**, `marker_depth_r` 0.748 → **0.855** — because that draft used the superseded `hybrid`
pilot arm rather than the shipped `resample` one. The older figures have already been quoted; §5
states the change and what it affects.

---

## 1. What the method is, for a reader who has not seen the design

Spatial transcriptomics gives you a **stack of tissue sections**: a few parallel 2D slices through
a 3D volume, each with cells at known positions and a count for every gene in a fixed panel. The
sections are far apart relative to a cell, so most of the volume is missing.

**The task.** Given the stack, produce a plausible section at a plane you never measured — a gap
between two slices, or a plane cutting at an angle. That means inventing (a) **where the cells are**
and (b) **what each one expresses**.

**The v25 idea, in three pieces.**

1. **A continuous 3D field.** Rather than model each section separately, learn one function of
   3D position. Any plane is then just a set of query points, so an oblique cut is no harder than
   an axis-aligned one and two planes that cross must agree where they meet.
2. **A correlated noise prior.** Real expression is spatially smooth: neighbouring cells resemble
   each other. Draw the model's randomness from a **3D Gaussian random field** with a fitted
   length-scale, so the smoothness is built in rather than hoped for.
3. **Open-vocabulary genes.** Each gene is represented by an embedding built from its **text**
   description — name and NCBI summary, encoded by MedCPT — rather than by a free per-gene lookup
   row. If that works, the model can place a gene it was never trained on.

Plus **SEFL** (Sectioning-Equivariant Field Learning), three auxiliary losses meant to make the
model's answer independent of how the tissue happened to be sliced. The method is named for it.

**How anything is judged.** Every generated section is scored against the real one it replaced, on
six metrics (§5). Two model-free referents bracket what is achievable:

* **`flanking_copy`** (the **floor**) — just copy the nearest real section. Any method that cannot
  beat this is not adding anything.
* **`oracle`** (the **ceiling**) — score the real cells against themselves, through the same
  pipeline. This is what a perfect method would get, and it is not 1.0, because the metric has its
  own noise.

A score means nothing until it is placed between those two.

**The evaluation vocabulary used throughout.** An **envelope** is how much a number moves between
identical runs at different random seeds. A margin smaller than the envelope is not a result. A
**claim** here needs three seeds (`Config.claim_min_seeds = 3`), signs agreeing on every seed and
every fold, and a margin exceeding the envelope. **Pre-registration** means the criteria and the
"this run cannot answer" conditions are written down before the fits run.

---

## 2. What works

Three claims, all about **encoding** — representing the volume. None is about generation.

| # | claim | measurement | source |
|---|---|---|---|
| 2.1 | The correlated prior controls per-gene spatial autocorrelation | error ratio **0.130** against an i.i.d. prior; per-gene `I_gen` vs `I_real` at **r = 0.917**; median `I_gen` monotone as `ell` sweeps 0.25x–4x | `reports/gate1.md` — **synthetic fixture**, stated as such |
| 2.2 | Oblique planes reconstruct as well as axis-aligned | depth-matched parity **0.955**, edge-excluded **0.979**, against a pre-registered ≥ 0.90 | `reports/gate2.md` — real data |
| 2.3 | Two crossing sections agree exactly where they meet | **bitwise identical** expression along the intersection, on an **untrained** model, no consistency loss applied | `tests/test_sefl.py::test_generation_is_intersection_consistent_by_construction` |

**2.2 is the strongest single result in the project** and it is unqualified on real data.

**2.3 needs three sentences of explanation**, because it is easy to read as an implementation note:

* It is a **stronger** claim than the loss written to earn it. `L_cross` exists in `specs/07` to
  *train* approximate agreement; the architecture already has exact agreement, at every checkpoint,
  for every model of this shape, because the noise field is continuous and every conditioning
  pathway is queried at **physical** points.
* It **survived that loss being shown both unnecessary and harmful.** At `w_cross = 0.3` the
  generated per-gene variance falls to **0.065** of real, against **0.711** with SEFL off
  (`progress/risks.md`, R6). `w_cross` ships at 0 — and the property was unaffected, because it
  never depended on the loss.
* **No competing method has it.** Independent per-plane generation gives crossing sections no
  reason to agree anywhere.

**2.4 One capability claim survives replication.** Text embeddings place genes the model never saw
above the `shuffled` floor: **2.52x** the shared envelope on `deep_starmap`
(`reports/t09_zeroshot_deep.md`) and **2.08x** on `cosmx_nsclc_3d` ⚠️ (held locally, not committed, §8a). *Which* path
in the text channel does it is **not** established — see §6.

---

## 3. What does not work

**Every generative component loses to copying a real section.** Same trivial baseline each time.

| component | verdict | measurement | source |
|---|---|---|---|
| intensity-field layout | **REFUTED** | `field` **0.6607**, `hybrid` **0.6692** against `resample` **0.7546**, a copy floor of **0.7765** and an oracle ceiling of **0.9808** — both field modes score *below the floor* on the metric the layout head exists to win. `resample` ships. | `reports/r11_starmap_layout_modes.json` |
| flow-matching expression head | **REFUTED, both datasets** | on `deep_starmap` copying wins **every live metric**; on tier-1 by 4.6–5.3x the envelope on three | `reports/t09_audit_deep_expr_mode.json`, `reports/t09_audit_expr_mode.json` |
| SEFL — the mechanism the method is named for | **REFUTED, 3 seeds** | the SEFL arm **collapses** the anatomical field: `i_gen` at **1.6–2.4 %** of target against 95.6–97.5 % off; `check_collapse` fired **218 times**. All three weights ship at 0. | ⚠️ **held locally, not committed** (§8a) |
| the *mechanism* half of the zero-shot claim | **PARTIAL** | A2 − A3 = **+0.0450**, **0.22x** the shared envelope, where `deep_starmap` had +0.2514 at 2.7x. `specs/10` §7's mechanism sentence is **withdrawn**. | ⚠️ held locally, not committed (§8a) |
| the seen/unseen sign flip | **DOES NOT REPLICATE** | signs reversed on **12 of 12** seed x fold cells; **both** magnitudes inside their pool's envelope (0.22x, 0.51x). Direction replicated, effect size did not. | ⚠️ held locally, not committed (§8a) |
| the metric-aware losses | **UNINFORMATIVE** — see §6 | condition (a) fired: worst primary envelope **0.4323** against the **0.067** bound, **6.5x over** | `reports/t10_a9.md` |

**And the sentence a reader will derive for themselves, so the paper should say it first.** v25
ships `resample` + `zinb-flow` — real positions, generated expression. v20's fallback is `resample`
+ `cross-mix` — the same real positions with the donor's counts **copied**. The difference between
them is exactly v25's contribution to that pairing, and **it is negative on every live metric**
(`reports/t09_audit_deep_expr_mode.json`).

---

## 4. The defects table — read as the contribution

Nine rules (`specs/10` §4.2a–j). They are the transferable half of this project, and the reason is
not that any one is clever:

> Every claim in this literature has the form **"the margin exceeds the noise."** That sentence
> hides seven independent choices, and in this project **each one silently decided a verdict before
> anyone noticed it was a choice.**

| § | the choice | the verdict it decided |
|---|---|---|
| 4.2a | **which arm's** variance is the noise | a pooled envelope was too lenient on three metrics and too strict on two; the worse arm alternates by metric, so it cannot be reasoned about in advance |
| 4.2b | **which comparison's** envelope a clearance takes | two arms 0.004 apart landed on opposite sides of the line, because one varied less — the steadier arm was being credited with a capability |
| 4.2c | **which referent** is a floor at all | the pre-registered "constant-field band" had bitwise-identical input; three instruments were needed to establish it, two of them thresholds that failed |
| 4.2d | **how the spread** is aggregated | an effect read 1.12x under fold-averaged noise and 0.75x under per-fold; reported as standing, withdrawn |
| 4.2g | **which arms may contribute** a spread | on the replication the envelope was set on *both* gene pools by a **degenerate** member, 6–18x every real arm's variance, so an effect consistent on 12 of 12 cells could not clear it |
| 4.2h | **which side** of the threshold refutes | a two-sided "within 1.5x" band on a one-sided hypothesis returned *false* on the result that refuted it most strongly |
| 4.2j | **whether the check ran at all** | both collapse alarms were armed only while SEFL was on, so on the **shipped** configuration neither ever ran — and two reports described that as a check performed (§4a) |

Plus two of a different kind, and one standing requirement:

* **4.2e** — a metric that is a *correlation on a dead field*. One seed inverted a headline's sign
  on two of seven metrics (−0.0519 / +0.2804 / +0.5772), because a correlation across genes has no
  amplitude and correlates noise when the field is flat. This is why `claim_min_seeds = 3` is not
  conservatism.
* **4.2f** — *an alarm that fires where nobody looks.* A7's alarm fired 218 times across three
  fits, was persisted, and no report read it. A diagnostic that is computed, recorded and never
  surfaced is **worse** than one never built: a blind spot is honestly missing, a silent one
  creates the impression of a watched run.
* **4.2i** — **quote the null's sampling distribution beside every measured spread.** Every envelope
  in this project is a three-seed estimate and none was ever reported as one. For a correlation
  over `n` genes the null has `sd ≈ 1/sqrt(n−1)`, and the **range of three draws** has expectation
  `1.693 sd` with its own `sd = 0.888 sd` — a **CV above 50 %**. So a 2.2x gap between two datasets'
  envelopes, which reads like a fact about the datasets, is noise in the noise estimate.

⚠️ **The honest discount, and it belongs in the paper.** Every one of these was found *because a
criterion in this project nearly returned the wrong answer*, and in four cases it had already
returned one that was recorded and later withdrawn. **We found these by getting them wrong first.**

### 4a. 🚨 The collapse alarm had never been armed on the shipped configuration

The newest defect and the one with the widest blast radius.

`train_ctfflow` armed both collapse alarms only when `sefl_teacher is not None`, and the teacher is
built only when a SEFL weight exceeds zero. **The shipped configuration has all four at zero.**
Therefore, on every run that shipped, **`check_collapse` and `check_spatial_collapse` never ran.**

**Every empty alarm record in every SEFL-off campaign means *never armed*, not *did not fire*** —
and it was read as health at least twice, including by this project's own report generator
(`t09_ship_starmap.py` printed *"Both were **checked** — a measured silence rather than an absent
measurement"*) and by the A9 aggregator, **written after §4.2f to enforce §4.2f**.

Neither statistic is about SEFL: one reads per-gene variance of generated expression against real,
the other their Moran's I ratio. Both are properties of the emission, which every configuration
has. Gating them on a loss that ships at zero armed the diagnostic **only in the arm that does not
ship**.

**Fixed 2026-09-07** — `_log_sefl` now arms on the step floor alone, with
`test_both_collapse_alarms_are_armed_with_sefl_off` pinning the gate **from source** (a fit that
happens not to collapse cannot distinguish "armed and silent" from "never armed", which is the
defect itself). Every citation of an empty record as a performed check has been corrected.

---

## 5. The six-metric table — the shipped configuration, with floor and ceiling

`layout_mode=resample` (shipped), `decoder_mu_link=exp`, 2400 steps, **one seed**, tier-1 STARmap,
holdout `paper_2_4_6`, medians over the three held-out sections, ground-truth-matched density,
pinned `bench3.evaluate_paper`. Source: `reports/r11_starmap_layout_modes.json` (arm
`resample-grid`); referents from the same file.

🚨 **These numbers changed materially between drafts, and the older ones have been quoted.** The
earlier advisor draft built this table from the **superseded `hybrid` pilot row** (rejection
sampler, `reports/t10_rescore_exp.json`) rather than the shipped `resample` arm. Two cells moved a
long way:

| metric | earlier draft (`hybrid`, superseded) | this report (`resample`, shipped) | change |
|---|---|---|---|
| `marker_field_r` | 0.6384 | **0.6830** | **+0.045** |
| `marker_depth_r` | 0.7478 | **0.8554** | **+0.108** |
| `morans_pearson` | 0.5749 | **0.6465** | +0.072 |
| `celltype_localization` | 0.6572 | **0.7546** | +0.097 |

**The shipped configuration is better than the earlier draft said**, on every cell that moved. The
cause is R11: `hybrid` was replaced by `resample` as the default, and the grid sampler replaced the
rejection sampler. Anywhere `marker_field_r = 0.6384` or its "0.247 below the floor" is quoted —
including in earlier progress entries — it is describing a **layout mode that no longer ships**.
The shipped deficit is **0.203**.

| metric | v25 shipped | `flanking_copy` floor | `oracle` ceiling | v25 − floor |
|---|---|---|---|---|
| `morans_pearson` | 0.6541 | **0.9836** | 1.0000 | **−0.329** |
| `gearys_pearson` | 0.6535 | **0.9840** | 1.0000 | **−0.331** |
| `umap_mixing` | 0.9262 | — | — | ⚠️ no probe measured |
| `marker_field_r` | 0.6824 | **0.8857** | 0.9997 | **−0.203** |
| `marker_depth_r` | 0.8331 | **0.9794** | 1.0000 | **−0.146** |
| `celltype_localization` | 0.7546 | **0.7765** | 0.9808 | −0.022 |
| `gene_mean_spearman` | 0.9901 | **0.9863** | 1.0000 | **+0.004** |

**Every `v25 shipped` cell is the 2026-09-08 re-measurement**
(`reports/r11_resample_grid_umap.json`, confirmed bitwise by
`reports/r11_determinism_{a,b}.json`), **not the r11 campaign numbers this table carried until
2026-09-08.** Those were stale; §Provenance below gives the evidence and the deltas.

✅ **The floor and ceiling columns were re-measured on 2026-09-08 and reproduce exactly.** Both
probes were regenerated from `bench3.selftest` and re-scored under today's code
(`reports/r11_probes_recheck.json`); all twelve values match the r11-era ones this table carried, to
four decimals, on both probes and all six metrics. An earlier draft flagged them as an inference —
that `flanking_copy` and `oracle` are probes that copy rather than run the model, so a decoder
change cannot move them. **It is now a measurement, and the inference it replaces was correct.** The
`v25 − floor` column is sound.

  ⚠️ **One cell is still empty and the reason is not what an earlier draft said.** `umap_mixing` has
  no floor and no ceiling because `scripts/t10_layout_modes_table.py` does not compute it for the
  probes — see the two-`SIX` defect below. It is not a matter of passing or omitting `--no-umap`; an
  earlier instruction in this campaign said it was, and that was wrong.

**What this table is not, and every caveat attached:**

* **It is one seed.** `claim_min_seeds = 3`. Nothing here is a claim; it is a description.
* ⚠️ **`celltype_localization` is NOT inert — an earlier draft of this report said it was, and
  A7's re-scores refute it.** The claim was that under `resample` the cell types come from the
  copied layout, so the column says nothing about the expression path. A7's six re-scores share
  `layout_mode=resample`, emit **identical cell counts** in all three sections, and differ only in
  the fitted weights — yet localization ranges **0.3149 to 0.7591** across them
  (`reports/t10_a7_{off,on}_s{1,2,3}.json`). A quantity that moves by 0.44 when only the weights
  change is reading generated expression. It is still the metric closest to its floor, but that is
  a result, not a construction.
* **`gene_mean_spearman` being *above* the floor is the one genuinely solved thing**: per-gene
  average magnitude is right. It is also the easiest of the six.
* **The two autocorrelation metrics carry the largest deficit** — 0.34 below a floor that a
  literal copy reaches. This is R12, and §7 gives the mechanism.
* 🚨 **Provenance: the r11 numbers this table carried were stale, and the check that found it is
  now the strongest reproducibility statement the project has.** Filling `umap_mixing` required
  re-scoring the shipped arm, which re-measured the other five metrics from the same checkpoint,
  seed, mode and sampler — and five of six disagreed with r11. The pair that separates a code change
  from broken determinism was then run, and it is decisive:

  | metric | r11 (2026-08-25) | re-score | det_a | det_b |
  |---|---|---|---|---|
  | `morans_pearson` | 0.6465 | 0.6541 | 0.6541 | 0.6541 |
  | `gearys_pearson` | 0.6469 | 0.6535 | 0.6535 | 0.6535 |
  | `marker_field_r` | 0.6830 | 0.6824 | 0.6824 | 0.6824 |
  | `marker_depth_r` | 0.8554 | 0.8331 | 0.8331 | 0.8331 |
  | `gene_mean_spearman` | 0.9880 | 0.9901 | 0.9901 | 0.9901 |
  | `celltype_localization` | 0.7546 | 0.7546 | 0.7546 | 0.7546 |
  | `umap_mixing` | (not measured) | 0.9262 | 0.9262 | 0.9262 |

  **All three of today's runs agree bitwise, on every metric, to full printed precision.**
  Sources: `reports/r11_resample_grid.json`, `reports/r11_resample_grid_umap.json`,
  `reports/r11_determinism_a.json`, `reports/r11_determinism_b.json`.

  ✅ **Convention 3 holds.** Generation and scoring are deterministic under a fixed seed. The
  alternative — that single-seed numbers carry hidden run-to-run noise — is **ruled out**, and with
  it the worry that every single-seed figure in this report needed an error bar. This is a positive
  result and it was obtained for the price of two re-scores.

  🚨 **The r11 numbers were stale.** `reports/r11_resample_grid.json` was written on 2026-08-25 and
  never regenerated. Four commits touching the model and scoring path landed after it:

  | commit | date | what it changed |
  |---|---|---|
  | `eedd003` | 2026-08-27 | stopped the `resample` layout copying the section it is scored against |
  | `ba3c474` | 2026-08-28 | fixed a per-process seeding bug |
  | `14306d8` | 2026-08-30 | closed the `expr_pca` leak at the size factor |
  | `b6bf123` | 2026-08-30 | pool-restricted metrics; a fifth leak of the same shape |

  ⚠️ **Which one moved the numbers is not established and is not claimed here.** Three of the four
  are leak or seeding fixes, so the r11 values were measured on a pipeline since found defective;
  the direction is not uniform, though (`morans` rose, `marker_depth_r` fell), so "the leak was
  inflating it" is *not* the story. The point that matters needs no attribution: **a results table
  is only valid for the code state that produced it, and this one had drifted two weeks.**

  🚨 **A second defect this run surfaced: the project has two different six-metric sets under the
  same name.** `scripts/t09_ship_starmap.py:106` defines `SIX` as morans, gearys, **`umap_mixing`**,
  marker_field_r, marker_depth_r, celltype_localization. `scripts/t10_layout_modes_table.py:36`
  defines `SIX` as the same list with **`gene_mean_spearman` substituted for `umap_mixing`**. Neither
  says it differs from the other.

  The consequence is not cosmetic. **R11's layout-mode table — the evidence that made `resample` the
  shipped default — was scored on the second set**, so the ranking that decided the default has
  never included `umap_mixing`, while the headline table above calls its own set "the six metrics"
  and does include it. A reader comparing the two tables is comparing different panels under one
  name. This is `specs/10` §4.2j's shape again — an instrument reporting under a label that means
  something else elsewhere — and it belongs in the rules table as its own instance.

  ⚠️ **Not yet fixed, and the fix is not free.** Adding `umap_mixing` to the layout-mode table means
  re-scoring both probes and every layout arm with UMAP on. It changes no verdict I can foresee —
  `resample` beat the field modes by 0.09 on localization, far outside any envelope — but "changes
  no verdict I can foresee" is exactly the phrase this campaign has now been wrong about twice.

  **Scope of the damage, stated honestly.** Within-campaign comparisons are unaffected: r11's
  layout-mode ranking (`field` / `hybrid` / `resample`) was measured in one code state, so the
  comparison stands even though the absolute values moved. What is affected is any **absolute**
  number quoted from a campaign and compared against a number from another campaign — which is what
  the six-metric table does. ⚠️ **The zero-shot, envelope and ceiling campaigns have not been
  re-checked this way**, and the same question applies to each.

* **The superseded pilot row** (`hybrid`, `reports/t10_rescore_exp.json`: morans 0.5749, gearys
  0.5716, marker_field 0.6384, marker_depth 0.7478, localization 0.6572) is the source of several
  numbers in the older write-ups. It is **not** the shipped configuration: R11 replaced `hybrid`
  with `resample` and the grid sampler replaced the rejection sampler.
* **The instrument is `bench3.evaluate_paper`.** Numbers elsewhere in the record scored on *internal
  LOSO with T08 kernels* are a different quantity and must not be placed beside these — a mistake
  this project made once, in an uninformative condition that could only ever fire (§6).

---

## 6. A9 — the metric-aware weights, and the standing recommendation

**The experiment.** `w_autocorr = w_profile = w_distribution` at the shipped **0.5** against **0**,
three seeds, six fits, shipped configuration, pinned evaluator. Source: `reports/t10_a9.md`.

**The verdict: UNINFORMATIVE.** Pre-registered condition (a) fired — the worst primary envelope is
**0.4323** against the **0.067** the condition names, **6.5x over** — and both autocorrelation
primaries had signs disagreeing across seeds. Margins for the record, none readable: `morans`
+0.0057, `gearys` +0.0032, `marker_depth` −0.0709.

**So the weights remain unestablished, and a three-seed real-data design could not resolve them.**
"Run more seeds" is not a cheap path: this design was already too noisy at three.

### 🚨 Standing recommendation: set the three weights to **zero** in the next campaign

Stated as a recommendation rather than a note, so whoever runs next decides with the evidence in
front of them instead of inheriting a coin-flip rank. Four reasons:

1. **They were selected on the synthetic fixture** — an aggregate rank over six metrics, per-metric
   margins of 0.0052 / 0.0101 / 0.0018 inside a 0.0335 envelope, **one seed** — on a fixture
   documented to over-reward exactly this kind of addition (R11: its flanking baseline sits at 58 %
   of its ceiling against real tissue's 79 %; it was *"underpowered, not wrong"*, and real data
   reversed its verdict).
2. **The one real-data test could not resolve them** (above).
3. **They cost 1.63x the compute** — 93 minutes a fit against 57 (`reports/t10_a9.md`).
4. **On the diagnostic the shipped model did not watch, they sit in the collapse regime** — median
   `variance_ratio` **0.140 / 0.083 / 0.171** on against **0.856 / 0.771 / 0.813** off, threshold
   **0.25**, where A7's *collapsed* arm settled at 0.105–0.193. Candidate, not established (§7a).

**Nothing was turned off post-hoc, deliberately.** Disabling them now would re-open every fitted
number in the project for a change that is, by the only real-data test available, within noise — a
threshold moved after seeing where the data fell. The recommendation is for the **next** campaign,
which refits anyway and can adopt it at no cost.

**What the paper must say either way**: the weights ship at 0.5 on a **fixture-selected** aggregate
rank with margins inside the reproducibility envelope, a three-seed real-data test could not resolve
their contribution, and **every absolute number in this project was produced with them active**.

⚠️ **A pre-registered condition was withdrawn as malformed**, on fit 1 and before the other five
landed: it compared these real-data, held-out, `bench3.evaluate_paper` numbers against a
**synthetic-fixture**, T08-kernel, `alternating`-holdout row. Different dataset, holdout and
instrument — it could only ever fire. Withdrawing it established the §6.1 point above: **these
weights had never been measured on real data at all.**

---

## 7. The mechanism under the generative failure

**Where the structure is lost.** Median Moran's I along the chain: **0.9714** at the GRF prior,
**0.9015** after the flow, **0.8607** at the decoded mean — then **0.1297** at the sampled counts
(`reports/chain_2400.md`). Stages 1–3 lose 0.11 in total; **the count draw loses 0.73 in one
operation.** Real tissue retains **62.2 %** across the same latent→counts step
(`reports/chain_2400_calibrated.md`); the model retains ~14 %.

**Why.** The decoder reproduces the *pattern* of between-cell variation almost perfectly and a
fraction of its *amplitude*, and the ZINB objective closes the gap with **dispersion** rather than
by sharpening the mean. **The three supporting figures landed on 2026-09-08 and were checked against the
artifact one by one. They do not all survive:**

| figure as stated | in the artifact | status |
|---|---|---|
| Spearman **0.068** between `theta` and the data's own dispersion, over **1017 genes** | `log_ratio_spearman = 0.06801`, `n_genes = 1017` (`reports/t09_theta_learned_s2.json`) | ✅ **sourced exactly** |
| `theta` carrying **57–63 %** of conditional variance | `f_overdispersion` = 0.6110, 0.6304 — the file has two folds and both are 61–63 % | ⚠️ **upper bound sourced, lower bound is not.** The honest range from this artifact is **61–63 %**; where 57 % came from is not in the file |
| `mu`'s Moran's I **0.861** against the tissue's own latent **0.745** | ❌ **neither number appears.** A scan of all thirteen recovered files finds no value within 0.004 of either. The nearest is `cond_mean_morans_top` = 0.8240 / 0.8153 | 🚨 **unsourced — stays in §8b** |

**So the attribution to `theta` is now two-thirds sourced, not sourced.** The chain figures in the
paragraph above **are** sourced, so the *localisation* to the count draw survives regardless. ⚠️
**Until the 0.861 / 0.745 pair is either found or re-measured, this paragraph should quote 61–63 %
and the Spearman, and drop the Moran's comparison** — it is the one number here with no file behind
it.

That is a property of the **objective**, not a tuning error: a likelihood that can be reduced by
moving explanatory power out of the structured component into the unstructured one will be, and
nothing in the objective opposes it. No data-derived value of `theta` exists to match to, which is
why the moment-matching experiment was stopped after one fit.

### 7a. A fifth instance — candidate, not established

A9's metric-aware arm drove median `variance_ratio` to **0.140 / 0.083 / 0.171** (off: 0.856 /
0.771 / 0.813) while `spatial_ratio` moved the **other way**: **2.43 / 2.54 / 2.17** on against
**1.64 / 1.39 / 1.39** off (`reports/t10_a9.md`).

Both at once is not a contradiction. **Moran's I is variance-normalised** — it measures how much of
a field's variance is spatially structured, not how much variance there is. So a field can score
*better* on it while the amplitude it measures drains away, and that is what the autocorrelation
term appears to do: **it hits its target statistic by removing the structure the statistic was meant
to certify.** Same shape as the four instances above.

⚠️ **Candidate.** Two diagnostic trajectories, one budget, one dataset. It decides no A9 branch and
the verdict stays UNINFORMATIVE. It is reported because a number that would have raised an alarm
had the alarm been armed must not reach a reader as silence.

### 7b. `marker_field_r` — stated at its real strength

**Two clean v25 appearances**: its single losing metric on the T09 fixture, and 0.1611 in the smoke
run. On the shipped arm it sits **0.203 below its copy floor** (0.6830 against 0.8857,
`reports/r11_starmap_layout_modes.json`) — the largest deficit of the six after the two
autocorrelation metrics.

⚠️ **A third and fourth appearance were claimed and are withdrawn.** "The pooled loss for v20 and
v21 against SpatialZ" is a **cross-dataset average, which `specs/10` §4.2a forbids by name.** Read
per dataset the comparison is **9–9**; on tier-1 v20 (0.8804) and v21 (0.8881) both **beat**
SpatialZ (0.8522); in the wide regime v20 wins **7 of 7**. Only the pooled figure favours SpatialZ.
**The honest count is two clean v25 appearances plus a comparison this project's own methodology
rejects** — a real pattern in v25, not the three-generation weakness it was written as.

**Where it lives.** A boundary stratification (zero fits — both sides were on disk) returned
**BOUNDARY ELIMINATED** on the shipped arm: deficits **0.1729 / 0.1877 / 0.2043** across the three
held-out sections, boundary-vs-interior gap **0.69x** the envelope
(`reports/t10_marker_field_boundary.json`). The weakness is **uniform along the stack**, so R3's
one-sided-evidence regime is out.

**And the redirect is worth more than the elimination**: `resample` does not use the intensity head
to place cells **at all**, and still carries a 0.19 deficit. The weakness survives removing the
layout head from the picture entirely — so it is in the **expression path**, not T05's.

---

## 8. ⚠️ Provenance — what has an artifact, what is recoverable, and what is not

Listed because "flag anything you cannot source" is the instruction, and because the split matters:
a gap that can be closed by committing a file is bookkeeping, and a gap that cannot is a limit on
what the paper may assert.

### 8a. HELD LOCALLY — measured, the files exist, NOT COMMITTED (as of 2026-09-08)

⚠️ **This heading said "being committed" and has been corrected twice.** The first correction: it
was a claim about the future, which is the thing §8c says not to do. The second, on 2026-09-08, is
sharper and is the reason this section is worth its length.

**Twelve of these landed.** A9's six and A7's six are in the branch, verified by `git ls-files`, and
A9's fit 1 is already named `t10_a9_0_s1.*` — the rename this table asked for is discharged.

✅ **And the remaining thirteen landed too, after a false alarm worth recording.** They were
reported committed; four searches — branch, full history on all branches, checkout, whole machine —
found none of them, so §8b took the row back. The cause was then found in the **reflog**: commit
`571e769` held all thirteen, and a later `git reset` to `origin/…` discarded that commit *and* the
working-tree copies with it. **Both the report and the searches were correct about different
moments.** The files were recovered from the reflog — the original commit's content, not a copy —
and are now in the branch.

🚨 **A regeneration had been accepted in place of an original, and this patch exposed it.** The A9
and A7 reports committed earlier carried `fit_seconds: null` and `alarms: null`: they were
**re-scores**, not the fitting runs' own output, and this table's own instruction says *"a copy of
the file that produced the number, not a regeneration."* The originals are now in place. They carry
identical metrics — verified field by field on all twelve — plus A9's `fit_seconds` and its full
alarm record (`variance_ratio`, `spatial_ratio`, the three alarm lists). **The rule was written here
and then broken here**, by me, and nothing in the report would have shown it: the metrics matched,
so only the provenance fields differed.

⚠️ **A7's alarm record is still not in these files.** Neither the originals nor the re-scores carry
`collapse_alarms` — the A7 reports predate the schema. So §3's *"`check_collapse` fired 218 times
across the three ON fits"* remains attested by `progress/`, not by these artifacts.

✅ **Landed 2026-09-08** — every path below is in the branch, `git ls-files` verified, checked
against the branch of record rather than against a report:

| result | path(s) |
|---|---|
| **A9** — six input reports, **originals** | `reports/t10_a9_{0,05}_s{1,2,3}.{json,md}` |
| **A7** — six reports, **originals** | `reports/t10_a7_{off,on}_s{1,2,3}.{json,md}` |
| **R12/R4** — the `theta` attribution | `reports/t09_theta_learned_s2.json`; six `t09_structured_share_deep{,_lookup,_lookup_s3,_lookup_s4,_medcpt_s3,_medcpt_s4}.json`; six `t09_retention_{medcpt,lookup}_s{2,3,4}.json`. ⚠️ **Named wrongly in earlier revisions of this table** — the structured-share files carry a `_deep` segment this table omitted, so the paths it listed matched nothing even once the files existed |
| **A7** — the `L_thick` binding check | `reports/t10_a7_thick_binding_deep.json`, `t10_a7_thick_binding_tier1.json`. ⚠️ Earlier revisions named **one** file, `t10_a7_thick_binding.json`; there are two, per dataset |
| **cosmx replication** — three scored seeds | `reports/t09_zeroshot_cosmx_seed{2,3,4}.json` |
| **cosmx** — the gene split | `reports/t09_gene_split_cosmx.json` |
| **cosmx** — the model-free ceiling | `reports/t09_zeroshot_ceiling_morans_cosmx.{json,md}`. Only the Moran's variant was ever run on cosmx; `deep_starmap` has both |
| **cosmx** — descriptor coverage | `reports/t09_text_coverage_cosmx_human.json`, and `t09_text_coverage_cosmx.json` — **two** files, where this table named one |
| **cosmx** — the degeneracy check | `reports/t09_degeneracy_cosmx.json` |
| **pool sparsity**, both datasets | `reports/t09_pool_sparsity_{cosmx,deep}.json` |
| **the human gene-metadata table** | `resources/gene_meta.cosmx_human.parquet` — the table the fits actually read, not a rebuild |

**This list is empty of outstanding items.** Everything §8a has ever asked for is in the branch.

⚠️ **Landing a file is necessary and not sufficient**, and §7 is the proof: its three supporting
figures now have an artifact, and checking them against it found one sourced exactly, one whose
stated range is wider than the file supports, and one that appears nowhere in any of the thirteen.
**The next revision of this section should verify each cited number against its file**, not merely
that the file exists — that is the same distinction §4.2f draws, one level up.

**Those verdicts are now reproducible from a clone** by re-running the aggregators already in
`scripts/`. §8b is the whole of the remaining gap.

**Two of them carry a caveat worth keeping with the file.** The A7 reports were produced **before**
the collapse alarm was armed independent of SEFL — but A7 ran SEFL **on**, so its alarm record is a
real measurement and the 218 firings stand. The A9 reports were produced with SEFL **off**, so their
empty alarm records mean *never armed* (§4a); the aggregator says so, and the JSONs should not be
read without it.

### 8a-bis. LANDED 2026-09-08 — 24 artifacts that were untracked, not missing

A `git status` on the campaign machine found **45 untracked files** that no one had committed. 24
are now in the repository, and two of them change what this report can claim:

| landed | what it sources |
|---|---|
| `reports/t09_envelope_starmap_seed{2,3,4}.{json,md}` (+ a `_dup`) | the **three-seed real-data envelope** |
| `reports/t09_ceiling_bootstrap_{deep,starmap}.{json,md}` | the ceiling instrument's uncertainty |
| `reports/t09_zeroshot_ceiling{,_morans}_deep.md`, `t09_gene_split_deep.json` | the `deep_starmap` zero-shot ceilings and split |
| `reports/t09_tenv_deep*.{json,md}` (3 seeds) | the `text_emb_mode` envelope on `deep_starmap` |
| `resources/cosmx_panel_symbols.txt` (959 symbols) | **the input needed to rebuild the cosmx gene-metadata table** |

**✅ `specs/10` §4.2a's table is now sourced, and verified rather than asserted.** Recomputing the
across-seed spread from the three committed seed files reproduces its every cell:

| metric | `cross-mix` | `zinb-flow` | shared | vs the 0.0335 used throughout |
|---|---|---|---|---|
| `morans_pearson` | 0.0054 | 0.0574 | 0.0574 | 1.71x |
| `gearys_pearson` | 0.0027 | 0.0595 | 0.0595 | 1.78x |
| `umap_mixing` | 0.0068 | 0.0190 | 0.0190 | 0.57x |
| `marker_field_r` | 0.0049 | 0.0148 | 0.0148 | 0.44x |
| `marker_depth_r` | 0.0084 | 0.0472 | 0.0472 | 1.41x |

⚠️ **And that exposes something about the 0.0335 every ratio in this report is divided by.** It is
**sourced** — `reports/envelope_synthetic.md`, R10, nine fits at three seeds — but it is a
**pooled figure measured on the synthetic fixture**, while the real-data per-metric envelopes
above span **0.0148 to 0.0595**, a 4.0x range straddling it. §4.2a states the rule that a pooled
envelope errs in both directions; the practice throughout this project, this report included, has
been to divide by the pooled fixture number anyway. **Every "Nx the envelope" here should be read
as "N times a synthetic pooled figure", not as a real-data noise scale.** The material to redo them
per-metric is now committed; doing so is a scoring-free re-derivation and is the cheapest
outstanding correction in this document.

**Still not committed**: the 21 `logs_*.txt` console logs from the same listing.

### 8b. MEASURED BUT UNRECOVERABLE — no artifact, and no file to commit

Distinct from the above: these numbers were measured and are in the record, and **the file that
produced them is not held**. They are attested by a progress entry alone. A referee cannot check
them and neither can we.

| result | what is missing | consequence |
|---|---|---|
| **§7's Moran's comparison** — `mu`'s Moran's I **0.861** against the tissue's own latent **0.745** | no file. The thirteen recovered artifacts were scanned numerically and **no value falls within 0.004 of either number**; the nearest is `cond_mean_morans_top` = 0.8240 / 0.8153. | ⚠️ **This row is what is left of a bigger one.** Its two companions — the Spearman and the conditional-variance share — landed and check out (§7), so the attribution to `theta` is now mostly sourced. This pair is not, and §7 should drop it until it is found or re-measured. ✅ **Regenerable**: `scripts/t09_theta_mode.py` and `t09_structured_share.py` are in the repo, and the latter states it measures on an existing fit with no refit — so this costs a scoring pass **if that fit is still held**. A regeneration is a new measurement and must be labelled so. |
| **The T09 selection table** that put the metric-aware weights at 0.5 — the four `(budget x weights)` cells, ranks 3.0 / 3.5 / 2.0 / 1.0 | the `scripts/t09_report.py` run's output. `reports/config_selection_synthetic.md` covers the synthetic selection but not this table. | §6's recommendation rests on *how* the weights were chosen. The claim "selected on the fixture by an aggregate rank, one seed, margins inside the envelope" is currently attested by `progress/` only. It is **cheap to regenerate** — the fixture run is ~8 minutes a fit — but a regeneration is a new measurement, not the one that made the decision, and should be labelled as such. |

✅ **This table did shrink on 2026-09-08 — on the third attempt, and only the third one counted.**
The `theta` row was moved out on a report that its files existed, moved back when four searches
found nothing, and finally reduced to a fragment when the files were recovered from the reflog and
**checked figure by figure against their contents**. Only the last of those three was evidence.

**What each round cost and bought.** The first cost a wrong edit. The second cost a search and
bought the correct classification at the time. The third cost a reflog recovery and bought the thing
neither of the others could: the knowledge that two of §7's three figures are sourced and one is
not — which no amount of confirming that *a file exists* would ever have revealed.

### 8c. Why this section exists at all

**8b is two rows and did not shrink.** The length of §8a **plus** §8b, not §8b alone, is the
finding. **A headline result attested only by a progress entry is
the same class of problem as §4.2f**: the difference between a measurement and a claim that one was
made. §4.2j sharpens it — an artifact may state a check was performed only if the record shows it
ran. A progress entry saying a number was measured is not that record.

**The rule this implies, for the next campaign:** the artifact is committed in the same change as
the claim, or the claim is labelled as attested-by-log. There is no third state, and "we still have
it locally" is the first state only until someone's disk is reimaged.

⚠️ **2026-09-08 adds a fourth failure mode, and it is worse than the three above: an artifact
believed to exist that does not.** Thirteen files were reported measured and committed. They are in
no branch, in no commit on any branch, in no checkout and nowhere on the machine that ran the fits.
Nobody was being careless — the numbers **were** measured, and the belief that a file must therefore
be somewhere is the ordinary one.

**That belief is the failure mode.** It moved a row out of this section on nothing but recollection,
and only a search put it back.

🚨 **And a fifth, which is what actually happened here: a local commit that is then reset away.**
Commit `571e769` really did hold all thirteen files. A later `git reset` to `origin/…` discarded the
commit and the working-tree copies together, leaving **no trace in `git ls-files`, in `git log
--all`, or in the checkout** — while the person who made the commit correctly remembers committing
it. Both parties then asserted the files existed, on the strength of an `ls` from five days earlier,
and both were describing a state that had ceased to exist. Only the **reflog** still held it.

**The consequences for how §8 is maintained:**

1. **`git ls-files` on the branch of record is the only test this section may use.** Every revision
   re-runs it rather than carrying the previous split forward.
2. **A report that a file exists — from anyone, including whoever ran the campaign, and including
   me — is a hypothesis to check, not a provenance.** Recollection and an old `ls` are the same
   evidence.
3. **When a search comes back empty and someone is confident, check `git reflog` before concluding
   the file was never written.** That step was missing here and it cost a round trip.
4. **A landed file is not a sourced figure.** Verify the cited number against the artifact's
   contents — §7 had three figures, one file, and only two of the figures survived contact with it.
5. ⚠️ **Pushes from the campaign machine return 403, so everything travels by patch.** Until that is
   fixed, *"committed on the campaign machine"* and *"in the branch of record"* are different states
   and this section must keep treating them as different. Every landing recorded above went through
   a patch for that reason.

## 9. Claim-by-claim status

| claim | status | source |
|---|---|---|
| Oblique planes reconstruct as well as axis-aligned | **PASSES** — 0.955 / 0.979 against ≥ 0.90 | `reports/gate2.md` |
| A 3D GRF prior controls per-gene spatial autocorrelation | **PASSES, on the fixture** — 0.130, r 0.917 | `reports/gate1.md` |
| Crossing sections agree along their intersection | **PASSES, exactly and untrained** | `tests/test_sefl.py` |
| The learned intensity field places cells better than copying | **REFUTED on real data** — below the copy floor | `reports/r11_starmap_layout_modes.json` |
| Generated expression beats copying a real section | **REFUTED, both datasets** | `reports/t09_audit_deep_expr_mode.json` |
| Text embeddings help genes the model was fitted on | **REFUTED, three three-seed negatives** | `reports/t09_zeroshot_deep.md` |
| Text embeddings place genes the model never saw | ⚠️ **PARTIAL, replicated on two datasets** — clears the floor at 2.52x and 2.08x; *which path* does it is not established | `reports/t09_zeroshot_deep.md`; cosmx half held locally, not committed (§8a) |
| SEFL improves anything | **REFUTED — it makes things worse, 3 seeds** | ⚠️ held locally, not committed (§8a) |
| Metric-aware losses improve the metrics they are made of | ⚠️ **UNINFORMATIVE at 3 seeds; ships ON, established by nothing** | `reports/t10_a9.md` |
| The model reproduces gene–gene covariance | **DOWNGRADED to a mechanism claim** — criterion unsatisfiable as stated | `progress/` |
| Both collapse alarms watch the shipped model | 🚨 **WAS FALSE until 2026-09-07** — never armed with SEFL off; fixed, with a test | `spatialcpav25_gen/model/spatialcpav25_gen.py`, `tests/test_sefl.py` |

---

## 10. The assessment

**A continuous 3D field is a good *representation* of a tissue volume and a bad *generator* from
it.** Off-axis reconstruction reaches 95 % of on-axis quality, per-gene spatial autocorrelation is
controllable, crossing sections agree exactly and without training, and text embeddings place unseen
genes above a measured floor on two datasets. Against that, every generative component built on the
representation loses to copying a real section — including the sectioning-equivariant losses the
method is named for, which make it actively worse.

The failure is **localised and mechanical**, not diffuse: the decoder reproduces the pattern of
between-cell variation and a fraction of its amplitude, and the ZINB objective closes the gap with
dispersion instead of with the structured mean.

**This is a negative-results paper with a methods contribution, and the methods contribution is the
stronger half** — because it transfers. The negative is about one method on two datasets; §4 is
about how anyone should read a repeated-seed benchmark, and none of it is reported in this
literature.

**What would change the assessment:** a change to the emission model or its objective that closes
the amplitude/dispersion trade, followed by a re-run of the six-metric comparison against copying.
This project cannot run it. **What would not:** more seeds — §4.2i is the reason.
