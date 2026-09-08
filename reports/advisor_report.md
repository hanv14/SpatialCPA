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
| `morans_pearson` | 0.6465 | **0.9836** | 1.0000 | **−0.337** |
| `gearys_pearson` | 0.6469 | **0.9840** | 1.0000 | **−0.337** |
| `marker_field_r` | 0.6830 | **0.8857** | 0.9997 | **−0.203** |
| `marker_depth_r` | 0.8554 | **0.9794** | 1.0000 | **−0.124** |
| `celltype_localization` | 0.7546 | **0.7765** | 0.9808 | −0.022 |
| `gene_mean_spearman` | 0.9880 | **0.9863** | 1.0000 | **+0.002** |
| `umap_mixing` | ⚠️ **NaN in this run** — no current measurement on the shipped arm | — | — | — |

**What this table is not, and every caveat attached:**

* **It is one seed.** `claim_min_seeds = 3`. Nothing here is a claim; it is a description.
* **`celltype_localization` is inert.** Under `resample` the cell types come from the copied
  layout, so this column says nothing about the model's expression path. It is on the floor by
  construction, not by achievement.
* **`gene_mean_spearman` being *above* the floor is the one genuinely solved thing**: per-gene
  average magnitude is right. It is also the easiest of the six.
* **The two autocorrelation metrics carry the largest deficit** — 0.34 below a floor that a
  literal copy reaches. This is R12, and §7 gives the mechanism.
* 🚨 **`umap_mixing` is a hole in the shipped configuration's own characterisation, not a missing
  cell.** It is NaN on **all five** arms of the r11 run, which means `--no-umap` was passed to that
  whole campaign — a deliberate skip, not a failure. So **one of the six headline metrics has never
  been measured on the shipped layout mode under the current sampler.** The superseded pilot arm
  measured 0.9152 (`reports/t10_rescore_exp.json`) and is **not substituted here**: different layout
  mode, different sampler.

  **Cost to fill it: one generation-and-scoring pass, no fit.** The checkpoint the r11 run used is
  named in that file — `runs/pilot/model_exp_2400.pt` — and `layout_mode` is a generation-time gate,
  so nothing needs refitting:

  ```
  python scripts/t10_rescore_saved.py --model runs/pilot/model_exp_2400.pt \
      --modes resample --out reports/r11_resample_grid_umap.md
  ```

  One arm, three sections, UMAP on (omit `--no-umap`). It is the same operation the r11 campaign
  performed five times, so it is minutes-to-tens-of-minutes rather than the ~56 minutes a fit takes
  — ⚠️ **I cannot give a measured duration**: `fit_seconds` in these reports times the fit alone and
  the generation-plus-scoring portion is not separately recorded anywhere. UMAP is the expensive
  part, which is presumably why the flag exists. **If the checkpoint is gone, this becomes a refit**
  and the cost is ~1 hour, at which point it should be folded into whatever the next campaign runs
  rather than done alone.
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
by sharpening the mean. ⚠️ The supporting figures — `mu`'s Moran's I 0.861 against the tissue's own
latent 0.745, `theta` carrying 57–63 % of conditional variance, and Spearman **0.068** between
`theta` and the data's own dispersion over 1017 genes — are **measured but unrecoverable** (§8b):
no artifact is held. The chain figures in the paragraph above **are** sourced, so the *localisation*
to the count draw survives; what lacks a file is the *attribution to* `theta`.

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

⚠️ **This heading said "being committed" and has been corrected.** That was a claim about the
future, which is the thing §8c says not to do. Two attempts to land these files have failed: the
first because a push returned 403 on the campaign machine, the second because the artifacts are not
under the repository working tree at all — a `find` over the checkout matched **none** of them. They
demonstrably exist (they were transferred out of that machine), but their location is not currently
known, and until they are in a commit the correct present-tense statement is **held locally, not
committed**.

These results were measured and the artifacts are held on the campaign machine. The paths below are
the **destinations**, not current locations. Each should be a **copy of the file that produced the
number**, not a regeneration.

| result | destination path(s) |
|---|---|
| **A9** — six input reports | `reports/t10_a9_0_s1.json`, `_0_s2`, `_0_s3`, `t10_a9_05_s1.json`, `_05_s2`, `_05_s3` (+ the `.md` sibling of each). ⚠️ Fit 1 was written as `t10_a9_off_s1.*`; **rename to `t10_a9_0_s1.*`** so the arm is in the filename in one form only — the aggregator reads the arm from the config, but a reader should not have to. |
| **A7** — six reports | `reports/t10_a7_off_s1.json`, `_off_s2`, `_off_s3`, `t10_a7_on_s1.json`, `_on_s2`, `_on_s3` (+ `.md` siblings) |
| **A7** — the `L_thick` binding check | `reports/t10_a7_thick_binding.json` |
| **cosmx replication** — three scored seeds | `reports/t09_zeroshot_cosmx_seed2.json`, `_seed3`, `_seed4` (matching the existing `t09_zeroshot_deep_seed*.json` convention) |
| **cosmx** — the gene split | `reports/t09_gene_split_cosmx.json` |
| **cosmx** — the model-free ceilings | `reports/t09_zeroshot_ceiling_cosmx.json`, `reports/t09_zeroshot_ceiling_morans_cosmx.json` |
| **cosmx** — descriptor coverage | `reports/t09_text_coverage_cosmx_human.json` |
| **cosmx** — the degeneracy check | `reports/t09_degeneracy_cosmx.json` |
| **pool sparsity**, both datasets | `reports/t09_pool_sparsity_cosmx.json`, `reports/t09_pool_sparsity_deep.json` |
| **the human gene-metadata table** | `resources/gene_meta.cosmx_human.parquet` — needed to reproduce anything on the cosmx side at all |

**Until these land, every verdict in §3, §6 and §9 that cites them is attested by a progress entry
rather than by an artifact** — the same status as §8b, differing only in that it is expected to be
fixable. Once they land, those verdicts become reproducible from a clone by re-running the
aggregators already in `scripts/`, and §8b is the whole of the remaining gap.

**Two of them carry a caveat worth keeping with the file.** The A7 reports were produced **before**
the collapse alarm was armed independent of SEFL — but A7 ran SEFL **on**, so its alarm record is a
real measurement and the 218 firings stand. The A9 reports were produced with SEFL **off**, so their
empty alarm records mean *never armed* (§4a); the aggregator says so, and the JSONs should not be
read without it.

### 8b. MEASURED BUT UNRECOVERABLE — no artifact, and no file to commit

Distinct from the above: these numbers were measured and are in the record, and **the file that
produced them is not held**. They are attested by a progress entry alone. A referee cannot check
them and neither can we.

| result | what is missing | consequence |
|---|---|---|
| **R12/R4's supporting figures** — `mu`'s Moran's I **0.861** against the tissue's own latent 0.745; `theta` carrying **57–63 %** of conditional variance; Spearman **0.068** between `theta` and the data's own dispersion over 1017 genes | the structured-share audit's output and `t09_theta_learned_s2.json` | These are the quantitative core of §7, the paper's strongest mechanism claim. The **chain** figures beside them (0.9714 → 0.9015 → 0.8607 → 0.1297, and real tissue's 62.2 %) **are** sourced, to `reports/chain_2400.md` and `chain_2400_calibrated.md`, so the *localisation* survives; what is unattested is the *attribution to `theta`*. |
| **The T09 selection table** that put the metric-aware weights at 0.5 — the four `(budget x weights)` cells, ranks 3.0 / 3.5 / 2.0 / 1.0 | the `scripts/t09_report.py` run's output. `reports/config_selection_synthetic.md` covers the synthetic selection but not this table. | §6's recommendation rests on *how* the weights were chosen. The claim "selected on the fixture by an aggregate rank, one seed, margins inside the envelope" is currently attested by `progress/` only. It is **cheap to regenerate** — the fixture run is ~8 minutes a fit — but a regeneration is a new measurement, not the one that made the decision, and should be labelled as such. |

⚠️ **If either of these files does turn out to be held, they belong in 8a and this table shrinks.**
The split is a claim about what is available, and it is the kind of claim that should be checked
rather than inherited.

### 8c. Why this section exists at all

The length of 8b, not 8a, is the finding. **A headline result attested only by a progress entry is
the same class of problem as §4.2f**: the difference between a measurement and a claim that one was
made. §4.2j sharpens it — an artifact may state a check was performed only if the record shows it
ran. A progress entry saying a number was measured is not that record.

**The rule this implies, for the next campaign:** the artifact is committed in the same change as
the claim, or the claim is labelled as attested-by-log. There is no third state, and "we still have
it locally" is the first state only until someone's disk is reimaged.

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
