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
| 2.2 | Oblique planes reconstruct as well as axis-aligned | depth-matched parity **0.955**, edge-excluded **0.979**, against a pre-registered ≥ 0.90 | `reports/gate2.md` — ⚠️ **synthetic fixture, corrected 2026-09-11**; linear probe on 32 expression PCs, not the generation pipeline. Real-data oblique validation (E3) has never run (`reports/framing_honesty_review.md` §4) |
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
| intensity-field layout | **REFUTED** | `field` **0.6607**, `hybrid` **0.6692** against `resample` **0.7546**, a copy floor of **0.7765** and an oracle ceiling of **0.9808** — both field modes score *below the floor* on the metric the layout head exists to win, by **0.1158 and 0.1073 raw**. 🚩 No envelope multiple is quotable: all five r11 arms are **one seed**, so `field` and `hybrid` have no across-seed spread at all (`reports/envelope_correction.md` §3). `resample` ships. | `reports/r11_starmap_layout_modes.json` |
| flow-matching expression head | **REFUTED, both datasets** | on `deep_starmap` copying wins **every live metric**; on tier-1 by **2.2x, 2.3x and 7.4x** their own per-metric per-arm envelopes on three (⚠️ corrected 2026-09-08 from "4.6–5.3x", which was one seed, pre-frame-fix, ÷ the pooled fixture 0.0335) | `reports/t09_envelope_starmap_seed{2,3,4}.json`; `reports/envelope_correction.md` §2.1 |
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

Eleven rules (`specs/10` §4.2a–j, plus §4.2a-i and §4.2a-ii from the envelope correction). They are
the transferable half of this project, and the reason is not that any one is clever:

> Every claim in this literature has the form **"the margin exceeds the noise."** That sentence
> hides seven independent choices, and in this project **each one silently decided a verdict before
> anyone noticed it was a choice.**

| § | the choice | the verdict it decided |
|---|---|---|
| 4.2a | **which arm's** variance is the noise | a pooled envelope was too lenient on three metrics and too strict on two; the worse arm alternates by metric, so it cannot be reasoned about in advance. 🚨 **The strongest instance in this table, because the rule was written and still did not bite**: the per-metric envelope existed from the day R10 measured it, in a report that said so, and this project divided by the pooled scalar for three weeks afterwards across all three documents |
| 4.2b | **which comparison's** envelope a clearance takes | two arms 0.004 apart landed on opposite sides of the line, because one varied less — the steadier arm was being credited with a capability |
| 4.2c | **which referent** is a floor at all | the pre-registered "constant-field band" had bitwise-identical input; three instruments were needed to establish it, two of them thresholds that failed |
| 4.2d | **how the spread** is aggregated | an effect read 1.12x under fold-averaged noise and 0.75x under per-fold; reported as standing, withdrawn |
| 4.2g | **which arms may contribute** a spread | on the replication the envelope was set on *both* gene pools by a **degenerate** member, 6–18x every real arm's variance, so an effect consistent on 12 of 12 cells could not clear it |
| 4.2h | **which side** of the threshold refutes | a two-sided "within 1.5x" band on a one-sided hypothesis returned *false* on the result that refuted it most strongly |
| 4.2j | **whether the check ran at all** | both collapse alarms were armed only while SEFL was on, so on the **shipped** configuration neither ever ran — and two reports described that as a check performed (§4a) |
| **4.2a-i** | 🚨 **which instrument the envelope was measured on** | the corpus holds two scorers over the same six metric names, and **every three-seed envelope ever quoted came from the one the headline numbers were not scored on**. On tier-1 their per-metric envelopes differ by 2.6x–6.7x. `specs/10` §5 forbade the mixture in prose while every "Nx the envelope" in the project performed it (`reports/envelope_correction.md`, 2026-09-08) |
| **4.2a-ii** | 🚨 **whether the artifact records the arm it describes** | the committed, verified, bitwise-reproducible files behind the six-metric table carry no `config_hash`, no `text_emb_mode` and no metric-aware weights, so a correctly measured envelope **cannot be matched to them**. Six clearance figures are flagged rather than numbered for this reason alone |

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

### 4b. 🚨 Every envelope in this project was measured on a different instrument from the numbers it judged

The newest defect and, by blast radius, the largest of the set — it touches **every** "Nx the
envelope" this project has ever written about real data, which is every clearance in §2, §3 and §5.
Found 2026-09-08 by tracing each figure to the script that produced it
(`reports/envelope_correction.md`).

**There are two scorers in this repository, over the same six metric names:**

| | instrument A | instrument B |
|---|---|---|
| scorer | `train/select.py::section_scores` (T08 kernels) | `bench3.evaluate_paper`, SHA-pinned |
| design | internal LOSO, **interior training** sections (`section_3`, `section_5`) | **`paper_2_4_6`** held-out (`section_2/4/6`) |
| metric names | `morans_pearson`, … | `paper_morans_pearson`, … |
| what it produced | the envelope files, the gate audits, the depth ceilings | the six-metric table, R11, A7, A9, the marker deficits, the boundary work |

**Every three-seed envelope this project has ever quoted is instrument A's. Every headline number is
instrument B's.** `specs/10` §5 forbids the mixture in as many words — *"Numbers … scored on
internal LOSO with T08 kernels are a different quantity and must not be placed beside these"* — and
the practice violated it in every division. Measured on tier-1, the per-metric envelopes differ by
**2.6x to 6.7x** between the two (`morans` 0.0574 → 0.2894, `gearys` 0.0595 → 0.2861, `umap_mixing`
0.0190 → 0.1281, `marker_field_r` 0.0148 → 0.0596, `marker_depth_r` 0.0472 → 0.1225).

**So the 0.0335 was wrong on three axes, not two.** Pooled across metrics (§4.2a's known defect),
measured on the synthetic fixture rather than the dataset — and measured by a scorer that never
touched the numbers it was dividing. The third is the one nobody had named, and it is the one that
runs in the direction that **flatters** every clearance: on the two autocorrelation metrics the real
instrument-B envelope is **~9x** the pooled figure that was used.

**What it does and does not change.** It moves no verdict in §3: every negative is a
within-configuration contrast, so a wrong divisor rescales a margin without touching its sign. It
does mean that **no "Nx the envelope" attached to a `paper_*` number in this report was ever a
statement about that number's noise**, and six of them are now flagged rather than numbered (§8).

### 4c. ⚠️ A9's envelope existed for a month and nobody looked, because the run was filed by its verdict

A smaller finding and a different kind, and it is the one most likely to recur.

A9 was run to answer one question — do the metric-aware weights help — and returned
**UNINFORMATIVE** (§6). It was filed under that verdict, in the reports, in `PROGRESS.md` and in
this document. What nobody recorded is that the same six fits are **three seeds of two arms of the
shipped-shape configuration, at 2400 steps, on `paper_2_4_6`, scored by the pinned evaluator** —
i.e. exactly the instrument-B envelope §4b says the project does not have, including
`paper_gene_mean_spearman`, which appears in no instrument-A envelope at all.

For a month, "we have no real-data envelope on the pinned instrument" and "A9 measured one" were
both true and only the first was written down. **The failure is retrieval, not measurement**: a run
was indexed by the question it was designed to answer, so the quantities it happened to measure
became unfindable to anyone asking a different question.

**The rule this earns**: an experiment's record states **what it measured**, not only **what it
concluded** — the arms, seeds, dataset, holdout, instrument and every metric it emitted — because a
null result's *measurements* stay valid after its *verdict* stops being interesting. This is
§4.2f's shape one level up: not a diagnostic that fires where nobody looks, but a **measurement
filed where nobody will think to look.**

---

## 5. The six-metric table

### 5.0 🚨 THE STATED ABSENCE: this project has never measured its own shipped configuration on the headline table

Not a caveat on a column. **There is no draft of this table, at any point in the project, that
measured the configuration v25 ships.** Both drafts are the same ablation:

| draft | arm | `text_emb_mode` | `expr_pca_dim` | layout | seeds | what it is |
|---|---|---|---|---|---|---|
| earlier (withdrawn) | `hybrid`, rejection sampler | `lookup` | 16 | `hybrid` | 1 | **ablation A3** |
| corrected 2026-09-08 | `resample-grid` | `lookup` | 16 | `field` fitted, `resample` generated | 1 | **ablation A3** |
| **the shipped configuration** | — | `medcpt` | 28 | `resample` | — | **never measured here** |

The 2026-09-08 correction moved the layout gate from `hybrid` to `resample` and relabelled the
result "shipped". It fixed the gate that was in dispute and inherited the label on the two that were
not. **Both drafts are `text_emb_mode=lookup` at `expr_pca_dim=16`** — and `specs/10` §3 already
names that arm: *"any local run is forced to `text_emb_mode="lookup"` — which is ablation A3, not
the shipped method."* The pilot ran in the development container, where the MedCPT encoder is
unreachable, so it could not have been anything else.

**So every absolute number this report has ever published about v25's own performance is A3's.**
Not the layout claim — that survives, see below — but every level: the deficits below the floor, the
fractions of the achievable range, the "one genuinely solved thing". The paper cannot report v25's
performance from this table, in any draft.

### 5.1 ✅ And the measurement exists — it is A9, filed as an ablation

The remedy named in `reports/envelope_correction.md` was *three seeds of the shipped arm on
instrument B*. **That is what A9's fits are**, and this is the A9 retrieval failure (§4c) in its most
consequential form: A9 was indexed by the question it answered, so nobody noticed it had also
measured the headline table.

`reports/t10_a9_{0,05}_s{1,2,3}.json`: tier-1 STARmap, `paper_2_4_6`, `bench3.evaluate_paper`,
`text_emb_mode=medcpt`, `expr_pca_dim=28`, `layout_mode=resample`, `layout_sampler=grid`,
`prior_mode=correlated`, `expr_mode=zinb-flow`, `decoder_mu_link=exp`, 2400 steps, SEFL at zero —
**the shipped configuration on every gate except the one still in dispute** (§6a), which is present
at both of its candidate values. Medians over the three held-out sections, three seeds, min–max
beside them. Referents are `flanking_copy` and `oracle` from `r11_starmap_layout_modes.json`: they
are **model-free probes** that copy real cells rather than running the model, re-measured on
2026-09-08 and reproducing to four decimals, so they are arm-independent and transfer.

| metric | **`w = 0` (what `Config` produces)** | min–max | `w = 0.5` (what the selection chose) | min–max | floor | oracle | `w=0` − floor |
|---|---|---|---|---|---|---|---|
| `paper_morans_pearson` | **0.5574** | 0.5438–0.5640 | 0.6422 | 0.3753–0.6647 | 0.9836 | 1.0000 | **−0.4262** |
| `paper_gearys_pearson` | **0.5543** | 0.5419–0.5659 | 0.6343 | 0.3757–0.6618 | 0.9840 | 1.0000 | **−0.4297** |
| `paper_umap_mixing` | **0.8318** | 0.8201–0.8665 | 0.6197 | 0.6067–0.7348 | — | — | ⚠️ no probe |
| `paper_marker_field_r` | **0.5655** | 0.5426–0.5732 | 0.4139 | 0.4122–0.4718 | 0.8857 | 0.9997 | **−0.3201** |
| `paper_marker_depth_r` | **0.7228** | 0.6508–0.7733 | 0.6391 | 0.5974–0.6977 | 0.9794 | 1.0000 | **−0.2566** |
| `paper_celltype_localization` | **0.7591** | 0.7582–0.7591 | 0.7591 | 0.7540–0.7601 | 0.7765 | 0.9808 | −0.0174 |
| `paper_gene_mean_spearman` | **0.9721** | 0.9475–0.9874 | 0.9611 | 0.9376–0.9677 | 0.9863 | 1.0000 | **−0.0142** |

**Cost: zero fits.** A9's six fits are already paid for and already committed — 2.85 core-hours at
`w = 0` (56 / 59 / 56 min) and 4.65 at `w = 0.5` (91 / 92 / 97 min), read out of `fit_seconds`. This
is a re-read, not a campaign.

### 5.1a ✅ And these are the first deficits in the project readable against an admissible envelope

A9's `w = 0` arm **is** the shipped configuration (§6c), so its own across-seed spreads are the
envelope for its own clearances — right instrument, right design, right arm, three seeds. Per §4.2b
the referent is a fixed model-free probe, so its envelope is exactly zero and the arm's decides.
**No deficit in this project has ever been divisible by an admissible figure before.**

| metric | deficit below floor | env (`w=0` arm) | **vs its own envelope** | vs the retired 0.0335 |
|---|---|---|---|---|
| `paper_morans_pearson` | −0.4262 | 0.0202 | **21.1x** | 12.7x |
| `paper_gearys_pearson` | −0.4297 | 0.0241 | **17.8x** | 12.8x |
| `paper_marker_field_r` | −0.3201 | 0.0307 | **10.4x** | 9.6x |
| `paper_marker_depth_r` | −0.2566 | 0.1225 | **2.1x** | 7.7x |
| `paper_celltype_localization` | −0.0174 | 0.0009 | ⚠️ **not readable — see below** | 0.5x |
| `paper_gene_mean_spearman` | −0.0142 | 0.0400 | **0.4x — inside** | 0.4x |
| `paper_umap_mixing` | — | 0.0464 | no probe exists | — |

**The four large deficits are established and the pooled figure was mis-scaling them in both
directions**: the autocorrelation pair reads *higher* than 0.0335 suggested (12.7x → **21.1x**,
12.8x → **17.8x**) while `marker_depth_r` reads far *lower* (7.7x → **2.1x**), because its own arm
moves by 0.12 between seeds. §4.2a's claim, once more, in the project's own headline numbers.

⚠️ **Two cells are not what the point estimates suggest, and §5.2 is corrected by them.**

* 🚩 **`celltype_localization` is NOT READABLE, not a 19x deficit.** Its `w = 0` spread is **0.0009**
  — and `n_pred` is **bitwise identical across all three seeds** (4073 / 4169 / 4110), because
  `resample` copies the donor's layout and cell types. Two of three seeds return exactly 0.7591.
  That is an arm **near-degenerate by construction on this metric**, and §4.2g's principle applies:
  a degenerate member contributes its **level**, not its spread. Dividing by 0.0009 manufactures
  significance from an arm that barely moves. ⚠️ §4.2g was written about a *referent* being
  degenerate; here it is the **arm under test**, which the rule does not currently cover. The
  honest statement is that the deficit is small (**−0.017**) and its noise scale is not measurable
  from this arm.
* ⚠️ **`gene_mean_spearman`'s deficit is INSIDE its envelope at 0.4x**, so §5.2's "flips sign
  against its floor" overstates it. The point estimate does move from +0.0038 above the floor to
  −0.0142 below. But −0.0142 against a 0.0400 spread is a **tie**. The withdrawal of *"the one
  genuinely solved thing"* stands — it is not established as solved — but it is **not established
  as worse either**, and this report should not claim it is.

### 5.2 🚨 The shipped configuration is WORSE than the arm labelled as it, on six of seven metrics

| metric | A3 column (1 seed) | shipped, `w=0` (3 seeds) | change | A3 − floor | **shipped − floor** |
|---|---|---|---|---|---|
| `morans_pearson` | 0.6541 | 0.5574 | **−0.097** | −0.3294 | **−0.4262** |
| `gearys_pearson` | 0.6535 | 0.5543 | **−0.099** | −0.3306 | **−0.4297** |
| `umap_mixing` | 0.9262 | 0.8318 | **−0.094** | — | — |
| `marker_field_r` | 0.6824 | 0.5655 | **−0.117** | −0.2032 | **−0.3201** |
| `marker_depth_r` | 0.8331 | 0.7228 | **−0.110** | −0.1464 | **−0.2566** |
| `celltype_localization` | 0.7546 | 0.7591 | +0.005 | −0.0219 | −0.0174 |
| `gene_mean_spearman` | 0.9901 | 0.9721 | **−0.018** | **+0.0038** | **−0.0142** |

**Three consequences, and they run in the same direction as everything else in this report.**

1. **Every deficit below the copy floor gets larger.** `marker_field_r` goes from 0.203 below to
   **0.320** below; the two autocorrelation metrics from ~0.33 to **~0.43**. The negative column is
   stronger on the shipped arm than on the arm that was standing in for it.
2. 🚨 **`gene_mean_spearman` no longer clears its floor.** This report calls it *"the one genuinely
   solved thing"* at **+0.0038** above the copy floor; on the shipped configuration the point
   estimate is **−0.0142 below** it. **That claim is withdrawn** — per-gene average magnitude is not
   established as solved, and what solved it was A3. ⚠️ But the deficit is **0.4x its own envelope**
   (§5.1a), so it is a **tie**, not an established loss; do not quote it as one.
3. **`celltype_localization` is the only cell that improves**, by 0.005 — inside the `w=0.5` arm's
   0.0061 spread, and on an arm whose `w=0` spread is 0.0009 with bitwise-identical cell counts
   across seeds. A tie, and see §5.1a: this metric's noise scale is not measurable under `resample`.

⚠️ **Two things the new table does not carry.** `paper_umap_mixing` still has no probe — the
probes are not scored for it (the two-`SIX` defect below), so it has no floor or ceiling in any
draft. And A9 records `paper_gene_mean_spearman` pooled only, with no per-section breakdown, where
`specs/10` §4.6 requires per-section values beside every tier-1 median.

⚠️ **What is still genuinely missing, and what it costs.** The shipped arm's own numbers are now
in hand at three seeds. What no draft of this table has ever had is the **comparator set** —
SpatialZ, FEAST, isoST and v20 on the same instrument and holdout (`specs/10` §3: the prior
campaign's numbers are not in this repository and its two sides were scored by different
`evaluate_paper` revisions). Until those run there is a v25 row and nothing to read it against
except the two model-free probes. That is the campaign §12 already budgets, and it is the remaining
cost of a headline table — not more v25 fits.

---

### 5.3 The superseded A3 column, kept for the record

🚨 **MISLABELLED, and corrected 2026-09-09 when the arm was finally read out of its own checkpoint.**
This column has been headed "v25 shipped" through every draft. It is not. Every cell comes from
`runs/pilot/model_exp_2400.pt` (via `reports/r11_resample_grid_umap.json`, confirmed bitwise by
`r11_determinism_{a,b}.json`, all five r11 arms sharing that one checkpoint), and the recovery run
of `scripts/t09_recover_checkpoint_config.py` reads its gates directly:

| gate | this column | shipped | when it is applied | verdict |
|---|---|---|---|---|
| `layout_mode` | fitted `field`, **generated `resample`** | `resample` | **generation-time, and provably fit-invariant** | ✅ **sound** |
| `layout_sampler` | `grid` | `grid` | generation-time | ✅ |
| `expr_mode` / `prior_mode` / `decoder_mu_link` / `train_steps` | `zinb-flow` / `correlated` / `exp` / 2400 | same | fit | ✅ |
| **`text_emb_mode`** | **`lookup`** | **`medcpt`** | **fit — not overridable** | ❌ **this is ablation A3** |
| **`expr_pca_dim`** | **16** | **28** (`clamp_config_to_volume`) | **fit — not overridable** | ❌ the pilot's stand-in |
| metric-aware weights | 0 / 0 / 0 | see §6a — the record says 0.5 | fit | ⚠️ **inverts a claim, §6a** |

**The layout half is sound and that matters**, because it is what R11 turns on:
`FIT_INVARIANT_GATES = ("layout_mode",)`, and `tests/test_select.py::test_layout_mode_does_not_enter_the_fit`
asserts **bitwise identical weights across all 96 parameter and buffer tensors** when only that gate
moves. So the arm really does generate under `resample`, and R11's *ordering* across the five arms is
if anything **firmer** than before: they are now provably one checkpoint differing only in
generation-time gates.

**The expression half is not.** `text_emb_mode=lookup` is **ablation A3**, and this report already
says so in §3's own words, quoting `specs/10` §3: *"any local run is forced to
`text_emb_mode="lookup"` — which is ablation A3, not the shipped method."* The rule was written,
and then applied to everything except this table. `expr_pca_dim=16` is the pilot stand-in that
`PROGRESS.md` already records as superseded by the clamp rule's 28.

⚠️ **This is the same failure the table was corrected FOR.** The 2026-09-08 revision fixed it from
the superseded `hybrid` pilot row to the `resample` arm and called the result "shipped". The layout
label became right and the expression label stayed wrong, because only the layout gate was checked.
**A column is not the shipped configuration until every fit-time gate has been read out of the
artifact** — §4.2a-ii, arriving one level up from where it was written.

**What the table is, stated exactly**: tier-1 STARmap, `paper_2_4_6`, 2400 steps, **one seed**,
medians over the three held-out sections, ground-truth-matched density, pinned
`bench3.evaluate_paper`, `layout_mode=resample` at generation — on a model fitted with the
**lookup** gene embedding at **`expr_pca_dim=16`** and the metric-aware terms off. It is a valid
measurement of that arm and every within-r11 comparison built on it stands. It is not a
characterisation of the shipped method, and no absolute number in it should be quoted as one.

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

**The `resample` A3 arm is better than the `hybrid` A3 arm**, on every cell that moved — ⚠️ this
paragraph said "the shipped configuration is better than the earlier draft said", and neither arm is
the shipped configuration (§5.0). The cause is R11: `hybrid` was replaced by `resample` as the
default, and the grid sampler replaced the rejection sampler. Anywhere `marker_field_r = 0.6384` or
its "0.247 below the floor" is quoted it describes a layout mode that no longer ships; the
`resample` **A3** deficit is 0.203, and the **shipped** deficit is **0.320** (§5.2).

| metric | **A3 arm** (not v25 shipped — §5.0) | `flanking_copy` floor | `oracle` ceiling | A3 − floor |
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
* 🚨 **`gene_mean_spearman` being *above* the floor was called "the one genuinely solved thing".
  WITHDRAWN 2026-09-09.** It is +0.0038 above the floor **on the A3 arm** and **−0.0142 below** it
  on the shipped configuration (§5.2). Per-gene average magnitude is not solved; it was solved on an
  arm that is not the method.
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
  `resample` beat the field modes by 0.09 on localization, ⚠️ which the earlier draft called "far
  outside any envelope" and which is **not a statement this project can currently make**: no
  across-seed envelope exists for `paper_celltype_localization` on the `field`/`hybrid` arms
  (one seed each). The comparison to A9's `resample`-arm figure of 0.0061 is suggestive and is not
  admissible — different arm (`reports/envelope_correction.md` §3). "Changes no verdict I can
  foresee" is exactly the phrase this campaign has now been wrong about twice.

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

### 🚩 Standing recommendation — **WITHDRAWN 2026-09-09 as vacuous**, see §6c: set the three weights to **zero** in the next campaign

Stated as a recommendation rather than a note, so whoever runs next decides with the evidence in
front of them instead of inheriting a coin-flip rank. Four reasons:

1. **They were selected on the synthetic fixture** — an aggregate rank over six metrics, per-metric
   margins of 0.0052 / 0.0101 / 0.0018 inside their **own** fixture envelopes (0.0160 and 0.0335 on
   the autocorrelation metrics, so inside by **1.6x to 19x**; ⚠️ corrected 2026-09-08 from "inside a
   0.0335 envelope … 3 to 19", `reports/envelope_correction.md` §2.3), **one seed** — on a fixture
   documented to over-reward exactly this kind of addition (R11: its flanking baseline sits at 58 %
   of its ceiling against real tissue's 79 %, and real data reversed its verdict). ⚠️ The
   *"underpowered, not wrong"* gloss is withdrawn: read per metric the fixture `layout_mode` gate
   separated on **two** metrics that pointed **opposite ways** (`umap_mixing` 3.49x to `resample`,
   `celltype_localization` 2.84x to `hybrid`). Over-rewarding a generative addition survives as the
   objection; bluntness does not.
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
their contribution, and — ⚠️ **corrected 2026-09-09, see §6a** — the claim that **every absolute
number in this project was produced with them active is backwards**: `Config` ships them at `0.0`,
and A9's `05` arm is the only run in the corpus that had them on.

⚠️ **A pre-registered condition was withdrawn as malformed**, on fit 1 and before the other five
landed: it compared these real-data, held-out, `bench3.evaluate_paper` numbers against a
**synthetic-fixture**, T08-kernel, `alternating`-holdout row. Different dataset, holdout and
instrument — it could only ever fire. Withdrawing it established the §6.1 point above: **these
weights had never been measured on real data at all.**

### 6a. 🚨 "Every absolute number was produced with them active" is backwards

Found 2026-09-09 by the same checkpoint recovery, and it inverts a sentence this report and the
close-out both carry.

**`Config` ships the three weights at `0.0`, not 0.5.** `config.py` declares
`w_autocorr = w_profile = w_distribution = 0.0`; `scripts/_starmap_run.py::base_config` overrides
only the four data keys in `BENCH3_KEYS` and never the weights; and
`t09_ship_starmap.py --w-metric-aware` is an **override flag defaulting to unset**. Read off the
artifacts, every real-data fit in this repository:

| run | weights | text_emb_mode | expr_pca_dim | steps |
|---|---|---|---|---|
| the six-metric table's checkpoint (r11, all five arms) | **0 / 0 / 0** | lookup | 16 | 2400 |
| A7, both arms (SEFL) | **0 / 0 / 0** | medcpt | 28 | 1200 |
| A9 `0` arm | **0 / 0 / 0** | medcpt | 28 | 2400 |
| **A9 `05` arm** | **0.5 / 0.5 / 0.5** | medcpt | 28 | 2400 |

**So the exposure runs the other way.** *"Every absolute number in this project was produced with
them active"* is false: **A9's `05` arm is the only thing in the corpus produced with them active,
and it is the experiment that tested them.** Everything else — the six-metric table, A7, A9's own
control — was produced with them at zero.

⚠️ **And the fixture envelope was the opposite way round again.** `scripts/t09_envelope.py` sets
`w_autocorr = w_profile = w_distribution = 0.5` explicitly for all nine of its fits. So R10's
**0.0335**, the figure every real-data clearance was divided by until 2026-09-08, was measured on
the **weights-on** arm while every number it judged was **weights-off** — the wrong arm on the very
gate §6 is about, on top of being pooled, off-dataset and off-instrument (§4b). A fourth axis, in
the one place it is most embarrassing.

**What survives and what changes.** §6's *experiment* is untouched: A9 compared 0.5 against 0 at
three seeds and returned UNINFORMATIVE. What changes is the framing around it:

* **The absolute numbers are cleaner than claimed, not dirtier.** They were not produced with an
  unestablished component active. The paragraph warning that they were is withdrawn.
* **But they were not produced with the configuration the record calls shipped, either** — which is
  §5's finding arriving from the other side. The gap between "the value T09's selection chose" and
  "the value any code path produces" is the real defect, and it is a labelling failure rather than a
  contamination one.
* **The standing recommendation is vacuous, not merely a no-op** (§6c): `Config` already does what
  it asks, and after §6b there is no artifact anywhere that ever held 0.5. What the next campaign
  needs is not a decision to turn them off; it is for the record to stop saying they are on.

### 6b. 🚨 RESOLVED 2026-09-09 — the 0.5 has no machine-readable source anywhere in the project

**A persisted selection does exist**, and it is neither branch this report pre-registered.
`runs/select/starmap_visual_cortex/selected.yaml` was added at `3d57725` ("T10 pilot: halted at
step 6") and deleted at `5cd1fd6`. It is still in history, and it records:

| field | value | what it says |
|---|---|---|
| `w_autocorr` / `w_profile` / `w_distribution` | **0.0 / 0.0 / 0.0** | **not 0.5** |
| `train_steps` | **20** | a smoke run; the selected budget is 2400 |
| `decoder_mu_link` | `softplus` | predates the `exp` default (2026-08-21) |
| `expr_pca_dim` | 32 | predates `clamp_config_to_volume`'s 28 |
| `layout_mode` / `text_emb_mode` | `field` / `lookup` | predates R11 |

So it is a **20-step smoke artifact from the halted pilot**, not a selection anyone would ship —
and it still carries the weights at zero.

**The record therefore reads, exactly:**

> **`w_autocorr = w_profile = w_distribution = 0.5` appears in no machine-readable artifact this
> project produced.** Not in `Config`. Not in any fit's recorded config — the six-metric table's
> checkpoint, A7's two arms and A9's control all read back 0.0. Not in the one persisted selection
> that exists. It appears **only in prose**: `Config`'s docstrings, `specs/10`, this report, the
> close-out, `PROGRESS.md`, and `t09_ship_starmap.py --w-metric-aware`'s own help text — each
> citing the others.

🚨 **That is the seventh provenance failure mode, and it is stronger than "never persisted".** The
six in §8c are all about a *file*: missing, unrecoverable, reset away, believed-present, regenerated,
or searched in the wrong corpus. This one has no file at its centre. **A value was carried through
the entire written record, cited as shipped, reasoned from in two standing arguments — and there is
no artifact anywhere that ever held it.** Nothing is missing, because nothing was ever written.
Every artifact is internally consistent and every document is wrong, and no provenance check that
compares artifacts to each other can detect it: they all agree, at 0.0.

**The rule it earns.** A value the record calls *shipped* must be traceable to an artifact that a
run produced or consumed — a persisted config, a recorded fit config, or a declared default in
code. A number attested only by documents is a **claim about the code**, and it must be checked
against the code before it is repeated, not after it has been reasoned from twice.

⚠️ **The instrument that answered this question got it wrong first, and that is §4.2j again.**
`scripts/t09_find_selection.py` reported **NONE FOUND**: it walked filesystem roots, and the file is
tracked in the very repository it was run from — present in history, absent from the tree. Its scope
caveat warned about *roots*, i.e. space, while the file was missing in *time*, and it quoted §8c's
reflog lesson in its own output while committing it. Fixed with `scan_history` (`--all --reflog`, so
a reset-away commit is covered), and the self-check now carries a **regression on the false negative
itself** — it asserts the history scan finds that exact deleted file. Second time in this thread an
instrument I wrote committed the failure it was built to detect.

### 6c. What this does to the two arguments that were reasoned from the 0.5

Both §6's standing recommendation and R10's envelope took the 0.5 as real. Neither survives intact.

**1. The standing recommendation is withdrawn — it is vacuous, not merely a no-op.**
*"Set the three metric-aware weights to zero in the next campaign"* recommends changing a state that
does not exist. They are zero in `Config`, zero in every recorded fit config, and zero in the only
persisted selection. There is nothing to turn off.

⚠️ **And its first reason no longer has a source either.** Reason 1 was *"they were selected on the
synthetic fixture by an aggregate rank"* — but the selection table that chose 0.5 is §8b
(**measured but unrecoverable**), and the one selection file that *is* recoverable carries 0.0. So
**the claim that a selection ever chose 0.5 is itself prose-only.** Reasons 3 (1.63x compute) and 4
(the collapse regime) are measured from A9 and stand as measurements — but they describe an arm the
project never ran anywhere else, not the shipped one.

**What replaces it**: a documentary correction, not a configuration change. The record must stop
asserting 0.5 — in `Config`'s docstrings, `specs/10`, both reports and the `--w-metric-aware` help
text — and say that the weights are zero and always were.

🚨 **And A9 was framed backwards.** Its pre-registration reads *"Unlike A7 this is a REMOVAL
experiment: the three ship at 0.5, so the arms are `--w-metric-aware 0.5` (shipped) and 0."* With
the 0.5 unsourced, **A9 is an ADDITION experiment** — the same inversion A2, A4 and A7 each went
through — and **its `0` arm is the shipped configuration while its `05` arm is the addition.** The
UNINFORMATIVE verdict is untouched, since a two-arm contrast does not care which arm is called the
baseline; what inverts is which arm §5.1's table should be read as v25, and it is the `w = 0`
column.

**2. R10's envelope was measured on an arm with no other instance in the project.**
`scripts/t09_envelope.py` sets `w_autocorr = w_profile = w_distribution = **0.5**` explicitly for
all nine of its fits — presumably because 0.5 was believed shipped. So the 0.0335, and the
per-metric decomposition this report now quotes from it, were measured on a configuration that
appears **nowhere else**: not in `Config`, not in any real-data fit, not in the persisted selection.

Last revision called this "the wrong arm on the very gate §6 is about". It is stronger than that:
**there is nothing it is the envelope *of*.** Every other mismatch in §4b is a comparison between
two arms that both exist. This one has a single instance, created by a script, on the strength of a
value with no source.

✅ **What is unaffected, and it is what the corrected numbers now rest on.** A9's per-metric spreads
are measured on A9's own arms, and **A9's `0` arm is the shipped configuration** — so
`paper_morans_pearson` 0.0202, `paper_gearys_pearson` 0.0241, `paper_umap_mixing` 0.0464,
`paper_marker_field_r` 0.0307, `paper_marker_depth_r` 0.1225, `paper_celltype_localization` 0.0009,
`paper_gene_mean_spearman` 0.0400 are **the only envelope this project has ever measured on the
configuration it actually ships**, on the pinned instrument, at the right design, at three seeds.
That is the figure the next campaign should quote, and it did not exist as a usable number until
this week.

⚠️ **The fixture-only corrections in `reports/envelope_correction.md` §2.3 and §2.4 inherit the
orphan arm.** Both divide fixture margins by fixture per-metric envelopes, and the envelope run is
`w = 0.5` throughout. §2.4's layout tie-break is internally consistent — both arms are the same
`w = 0.5` base — so its finding (two metrics separating and disagreeing) stands **as a statement
about that arm**. §2.3's margins come from the unrecoverable selection table, so which arm they were
measured on is not established at all. Neither conclusion moves; both need the arm named beside
them.
---

## 7. The mechanism under the generative failure

**Where the structure is lost.** Median Moran's I along the chain: **0.9714** at the GRF prior,
**0.9015** after the flow, **0.8607** at the decoded mean — then **0.1297** at the sampled counts
(`reports/chain_2400.md`). Stages 1–3 lose 0.11 in total; **the count draw loses 0.73 in one
operation.** Real tissue retains **62.2 %** across the same latent→counts step
(`reports/chain_2400_calibrated.md`); the model retains ~14 %.

**Why.** The decoder reproduces the *pattern* of between-cell variation almost perfectly and a
fraction of its *amplitude*, and the ZINB objective closes the gap with **dispersion** rather than
by sharpening the mean.

🚨 **The `mu` = 0.8607 vs latent 0.7449 comparison licenses less than it has been quoted for, and
the limitation belongs here rather than in a footnote.** The 0.8607 was measured on **the encoder's
latent for a real section** — never on the flow's latent at generated positions — so the comparison
is **between two latents**, not between the generator's output and the tissue. What it supports is
narrow and worth stating exactly:

> At the stage where a latent is decoded to a mean, spatial structure is present and is not the
> thing being lost.

It does **not** support "the conditional mean is healthy in the generative path", which is the
stronger reading this project has used it for in review. The generative path's own conditional mean
at generated positions is a different measurement and was not made here.
(`progress/t09_inference_and_calibration.md`.)

**The localisation to the count draw does not depend on it.** That rests on the chain — 0.9714 →
0.9015 → 0.8607 → 0.1297, and real tissue's 62.2 % retention — all four sourced to
`reports/chain_2400.md` and `chain_2400_calibrated.md`. The narrower reading of the pair leaves §7's
conclusion standing and removes a claim that was travelling on it.

**The three supporting figures were checked against artifacts on 2026-09-08, one by one:**

| figure | artifact | status |
|---|---|---|
| Spearman **0.068** between `theta` and the data's own dispersion, over **1017 genes** | `log_ratio_spearman = 0.06801`, `n_genes = 1017` — `reports/t09_theta_learned_s2.json` | ✅ **sourced exactly** |
| `theta` carrying **61–63 %** of conditional variance | `f_overdispersion` = 0.6110 and 0.6304, the two folds in that file | ✅ **sourced.** ⚠️ Stated as **57–63 %** until 2026-09-08; the landed artifact covers **two folds of one arm at one seed**, and the 57 % lower bound is in none of it. The record calls the quantity arm- and split-independent, so 57 % plausibly comes from an arm whose file did not land — but an unlanded arm is not a source, so the range is narrowed to what the artifact carries |
| `mu`'s Moran's I **0.8607** against the tissue's own latent **0.7449** | `reports/chain_2400.md`, rows 3 and REF — the **chain** artifact, sourced all along | ✅ **sourced.** ⚠️ Quoted as 0.861 / 0.745; the artifact's values are 0.8607 and 0.7449 with IQRs 0.8508–0.8722 and 0.6752–0.7912 |

🚨 **A correction to this report's own §8b, and to the instruction that followed from it.** An
earlier revision reported this last pair as appearing **nowhere**, on a numeric scan of the thirteen
files recovered from the reflog. The scan was correct and its **scope was wrong**: the pair was
never in those files because it belongs to the chain measurement, whose artifacts this report has
cited as sourced throughout. A withdrawal of the figure was authorised on the strength of that
finding and is **not being carried out**, because its premise does not hold.

⚠️ **Provenance is not the only test the pair has to pass, and the other one is stated above where
the number appears** — not here and not in `progress/` alone, because a reader meets the figure
before they reach a provenance table. The pair is sourced *and* its claim is narrower than the way
it has been quoted.

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
rejects** 🚩 **STRUCK 2026-09-11 — "v20 beats SpatialZ 5 of 6" is not in the record at all, and even the real figures are unreadable: no floor column, cross-instrument, and v20 is mechanically a copier (`reports/spatialz_claim_struck.md`).** — a real pattern in v25, not the three-generation weakness it was written as.

**Where it lives.** A boundary stratification (zero fits — both sides were on disk) measured
deficits of **0.1729 / 0.1877 / 0.2043** across the three held-out sections on the shipped arm, a
boundary-vs-interior gap of **−0.0231** (`reports/t10_marker_field_boundary.json`).

🚩 **The BOUNDARY ELIMINATED verdict is downgraded to NOT READABLE, 2026-09-08.** It rested on that
gap being *"0.69x the envelope"*, and the instrument hard-codes `"envelope": 0.0335` — the pooled
**synthetic-fixture** figure, applied to a `bench3.evaluate_paper` number. Against the only
real-data envelope on that instrument (A9's `paper_marker_field_r`, **0.0596**) the same gap is
**0.39x** — still inside — while against instrument A's `marker_field_r` envelope (0.0148) the same
gap is **1.56x**, outside. ⚠️ **Corrected: an earlier revision attributed the 1.56x to A9's
envelope; it belongs to instrument A's.** **The two candidate divisors give opposite answers**, and
neither is admissible — A is the wrong instrument, B is the wrong arm and is fold-aggregated where
the effect is per section. A verdict that flips with the divisor is not a verdict. `reports/envelope_correction.md` §3.

**What survives without an envelope, and it is what the redirect rests on**: the boundary section
carries the **smallest** of the three deficits (0.1729 against 0.1877 and 0.2043), so nothing in
the data points toward a boundary mechanism even before a threshold is applied. R3's
one-sided-evidence regime stays out on that reading; the *quantitative* claim that the three agree
within one envelope is not currently supportable.

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

⚠️ **And that exposes something about the 0.0335.** It is **sourced** — `reports/envelope_synthetic.md`,
R10, nine fits at three seeds — but it is a **pooled figure measured on the synthetic fixture**,
while the real-data per-metric envelopes above span **0.0148 to 0.0595**, a 4.0x range straddling
it. §4.2a states the rule that a pooled envelope errs in both directions; the practice throughout
this project, this report included, was to divide by the pooled fixture number anyway.

✅ **DONE 2026-09-08 — `reports/envelope_correction.md`.** Every clearance figure in this report,
the close-out and `specs/10` was re-read against its own metric's envelope on the dataset and
instrument it was measured on. Zero fits. Four things came out of it, and only the first was
expected:

1. **Nothing in the negative column moved.** Every negative is a within-configuration contrast, so
   the divisor changes the scale and not the sign. The multiples moved a lot: `expr_mode` on tier-1
   goes 4.6–5.3x → **2.2x / 2.3x / 7.4x**, with the error running in **opposite directions** on
   different metrics, which is §4.2a's own point arriving in this report's own numbers.
2. 🚨 **A third axis the rule does not name: an envelope is per *instrument*.** The corpus holds two
   scorers over the same six metric names — `train/select.py::section_scores` on internal LOSO, and
   `bench3.evaluate_paper` on `paper_2_4_6` — and **every three-seed envelope ever quoted is on the
   first while every headline number is on the second.** `specs/10` §5 forbids mixing them by name.
   On tier-1 their per-metric envelopes differ by 2.6x–6.7x.
3. ✅ **The missing envelope existed inside A9.** `t10_a9_{0,05}_s{1,2,3}` are three seeds of the
   shipped-shape configuration on the pinned instrument at `paper_2_4_6` — the real-data envelope
   this project has been saying it lacks. Its per-metric spreads run **0.0009 to 0.2894**, against
   a pooled fixture figure of 0.0335.
4. 🚩 **Six figures could not be recomputed and are now flagged rather than numbered**, because the
   only admissible envelope would have to come from an arm this project can no longer identify —
   see the correction's §3 and the new failure mode below.

⚠️ **And a correction to this paragraph's own earlier wording**, which said *"the 0.0335 every ratio
in this report is divided by"*. That was an overstatement: the zero-shot ratios — most of §2 and §6
— divide by their own measured envelopes, as do A7's and A9's. The defect is real, and it is
narrower than this report claimed.

🚨 **A seventh failure mode, and it is new: a landed artifact that does not record its own
configuration.** `r11_starmap_layout_modes.json`, `r11_resample_grid{,_umap}.json` and
`r11_determinism_{a,b}.json` — the files behind §5's six-metric table, all committed, all verified,
all reproducible bitwise — carry `model`, `decoder_mu_link`, `train_steps` and `seed`, and carry
**no `config_hash`, no `text_emb_mode` and no metric-aware weights**. So A9's envelopes cannot be
matched to that arm, and §4.2a forbids assuming they can: on the `deep_starmap` `text_emb_mode`
gate the two arms' envelopes differ by up to 2.6x and the worse arm alternates by metric. §8's
tiers assume the problem is a **missing** file. This is a present, verified file that is missing its
own identity — and it is the single blocker on five of the six flagged rows.

**Still not committed**: the 21 `logs_*.txt` console logs from the same listing.

### 8b. MEASURED BUT UNRECOVERABLE — no artifact, and no file to commit

Distinct from the above: these numbers were measured and are in the record, and **the file that
produced them is not held**. They are attested by a progress entry alone. A referee cannot check
them and neither can we.

| result | what is missing | consequence |
|---|---|---|
| **The T09 selection table** that put the metric-aware weights at 0.5 — the four `(budget x weights)` cells, ranks 3.0 / 3.5 / 2.0 / 1.0 | the `scripts/t09_report.py` run's output. `reports/config_selection_synthetic.md` covers the synthetic selection but not this table. | §6's recommendation rests on *how* the weights were chosen. The claim "selected on the fixture by an aggregate rank, one seed, margins inside the envelope" is currently attested by `progress/` only. It is **cheap to regenerate** — the fixture run is ~8 minutes a fit — but a regeneration is a new measurement, not the one that made the decision, and should be labelled as such. |

✅ **This table is one row, and it took four passes to get there.** The `theta` row was moved out on
a report that its files existed; moved back when four searches found nothing; reduced to a fragment
when the files were recovered from the reflog and checked figure by figure; and removed entirely
when that fragment turned out to be sourced in `reports/chain_2400.md` — an artifact this report had
been citing as sourced on the very same page.

**Only the last pass was right, and the error in the third was mine.** A numeric scan of the
thirteen recovered files came back empty and I reported the figure as unsourced. The scan was
correct; the **scope** was wrong, because the figure was never going to be in those files. **A
negative result is only as broad as what it searched**, and a search of the wrong corpus reads
exactly like an absence.

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
   contents — §7 had three figures, one file, and checking them against it moved every one of them.
5. 🚨 **A regeneration substituted for an original passes every check except a field-by-field
   comparison.** The A9 and A7 reports committed on 2026-09-08 were **re-scores**, not the fitting
   runs' output. Their metric values were **identical** to the originals — that is why nothing
   caught it: no table, no verdict, no aggregate would have differed by a digit. Only
   `fit_seconds` and `alarms` differed, and those are exactly the fields a re-score cannot carry.
   **So a provenance check that compares numbers will always pass here**; the check has to be on
   the file. This is the sixth failure mode and the most invisible of them: §8a states the rule
   ("a copy of the file that produced the number, not a regeneration"), and §8a is where it was
   broken.
6. 🚨 **A negative result is only as broad as the corpus it searched.** §7's Moran's pair was
   reported as appearing *nowhere*, on a scan of the thirteen files just recovered. The scan was
   correct and the corpus was the wrong one: the pair lives in `reports/chain_2400.md`, cited as
   sourced elsewhere on the same page. A withdrawal was authorised on that finding and had to be
   stopped. **State what a search covered whenever reporting that something is absent** — "not in
   these thirteen files" and "unsourced" are different claims, and only the first was measured.
7. ⚠️ **Pushes from the campaign machine return 403, so everything travels by patch.** Until that is
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
stronger half** — because it transfers. The strongest single item in it is §4b: **every envelope in
this project was measured by a different scorer from the numbers it was used to judge**, a mistake
that survived a written rule forbidding it, and one that any repeated-seed benchmark with two
scoring paths can make. The negative is about one method on two datasets; §4 is
about how anyone should read a repeated-seed benchmark, and none of it is reported in this
literature.

**What would change the assessment:** a change to the emission model or its objective that closes
the amplitude/dispersion trade, followed by a re-run of the six-metric comparison against copying.
This project cannot run it. **What would not:** more seeds — §4.2i is the reason.
