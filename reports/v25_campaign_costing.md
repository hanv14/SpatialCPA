# Costing the v25 benchmark campaign — CPU, and what GPU would cost to qualify first

**2026-09-08. No fits were run to produce this.** Every number is either measured and sourced, or
derived from a measured anchor and labelled as derived. Where nothing has been measured, this
report says so rather than substituting a model — `specs/10` §12's own model was **2.3x optimistic**
against the first real fit, and the record notes fit cost was under-estimated on **five consecutive
occasions**.

---

## 0. The finding that reorders everything below

**T10 is `TODO`.** `PROGRESS.md`'s T10 row, confirmed against the tree:

| `specs/10` deliverable | state |
|---|---|
| `spatialcpav25_gen/eval/baselines.py` | exists (152 lines, T06's independent-donor) |
| `eval/metrics.py`, `eval/bench3_driver.py`, `eval/stats.py`, `eval/ceiling.py`, `eval/resection.py`, `eval/experiments.py` | **absent** |
| `spatialcpav25_gen/cli.py` | **absent** |
| `tests/test_metrics.py`, `test_baselines.py`, `test_bench3_driver.py`, `test_resection.py`, `test_stats.py` | **absent** |
| bench3 wrapper `run_spatialcpav25_gen.py` + the one `METHODS` entry | **exist** — the additive footprint is done |

The T09/T10 campaign that closed ran through **one-off `scripts/`**, not through a driver. Everything
this costing prices — per-seed results roots (§4.1), `CLAIM_BEARING` enforcement (§4.2),
`assert_tier_purity`, the Wilcoxon/BH/bootstrap/Cliff's-delta layer bench3 does not have (§4.5), the
boundary holdout (§4.3) — is in the absent column.

**Consequence for the bill: the compute is the cheap half.** ~256 core-hours of fits sit behind
roughly **2–3 engineering weeks** of driver work. Nothing in §3–§5 below can start until that lands.

---

## 1. Which of the 18 are informative — answer the screen before costing fits

You asked for this first, and it is the right order: `specs/10` §0a records that measuring it
**inverted how the campaign had been prioritised**.

### 1a. Eleven of the eighteen are excluded before any measurement

| # | dataset | out because |
|---|---|---|
| 1 | `imc_breast_cancer` | `fluorescence_intensity` → `zigamma`, never trained/generated/calibrated; `calibrate_detection` is ZINB-only (`infer/calibrate.py:1086`). **Retained as one labelled non-claim-bearing diagnostic row** — it is the only thing discharging the non-transcriptomic half of `design/v23_design.md` §7 |
| 2 | `merfish_hypothalamus` | `zigamma` |
| 3–5 | `easi_fish_lha1/2/3` | `zigamma`, tiny panel |
| 6–7 | `allen_zhuang_abca1/2` | `zigamma`; ABCA-2 additionally runs `paper_alt7of15`, so its rows cannot sit beside the headline |
| 8 | `openst_lymph_node` | `n_cell_types = 0` → Potts degenerate, the `localization` metric group empty, composite over 4 groups not 5. Plus bench3's own documented OOM (uncapped whole-transcriptome, ~1.55 M cells) |
| 9 | `visium_mouse_brain_c2l` | `n_cell_types = 0`; **spot** resolution; 3 sections, 1 held out |
| 10 | `st_mouse_brain_ortiz` | **spot** resolution |
| 11 | `allen_merfish_brain` | 1.17 M cells, 59 sections — ~45 FEF/fit, cost-prohibitive |

⚠️ **Spec defect.** `specs/10` §5.1 says *"Six datasets fall on the wrong side"* and then lists
**seven** (`imc_breast_cancer`, `merfish_hypothalamus`, `easi_fish_lha1/2/3`,
`allen_zhuang_abca1/2`). The list is right and the count is wrong; the exclusion is unaffected.
Recorded here rather than silently corrected.

### 1b. Seven survive. Exactly one is measured informative.

The saturation criterion is `specs/10` §0a's: **headroom of the noiseless ceiling `√R` over the best
available copy**, model-free, no fits.

| dataset | training cells | genes | design | headroom over best copy | copying as % of ceiling | verdict |
|---|---|---|---|---|---|---|
| `starmap_visual_cortex` | 16 527 | 28 | `paper_2_4_6` | **+0.1551 [+0.1075, +0.2186]** | 81 % | ✅ **INFORMATIVE — measured** |
| `deep_starmap` | ~113 000 | 1 017 | `paper_2_4_6` | **+0.0160 [+0.0104, +0.0243]** | **98 %** | ⛔ **SATURATED for reconstruction — measured** |
| `cosmx_nsclc_3d` | ~227 000 | 960 | `paper_2_4` | **unmeasured** | — | ❓ a priori best candidate (6 cryosections, **30 µm** apart — bench3's widest uniform gaps) |
| `merfish_thick_hypothalamus` | ~45 000 | 156 | `paper_2_4_6` | **unmeasured** | — | ❓ a priori strong (~29 µm slabs from a 200 µm block — thickest in the benchmark) |
| `merfish_thick_cortex` | ~16 500 | 254 | `paper_2_4_6` | **unmeasured** | — | ❓ a priori weak (~14 µm slabs at STARmap's scale) |
| `exseq_breast_cancer` | ~1 300 | 297 | `z_width`, 5 sections | **unmeasured** | — | ❓ small-n (1 979 cells, thinnest section 57 — just over bench3's 50-cell floor) |
| `exseq_visual_cortex` | ~1 130 | 28 | `paper_2_4_6` | **unmeasured** | — | ⚠️ marginal: 5 sections, 28 % unannotated, **1 of 3 markers resolved** |

Two intervals are **disjoint**, P(deep > tier-1) = 0.000, difference −0.1389 [−0.2017, −0.0884],
400-replicate bootstrap. Sources: `reports/t09_depth_ceiling_{starmap,deep}.md`,
`reports/t09_ceiling_bootstrap_{starmap,deep}.md`.

**`deep_starmap` being saturated does not remove it from the campaign.** It removes it from the
*reconstruction* claim. Two things are unaffected:

* **E1 zero-shot** — copying has no output at all for an unmeasured gene, so saturation is not
  defined there. `deep_starmap` is the only `raw_counts` dataset with a panel wide enough to run E1,
  and E1's is the one capability claim that survived replication (A2 clears the `shuffled` floor at
  2.52x / 2.08x on two datasets; A2 − A4 = +0.3949, 1.96x the shared envelope, 3/3 seeds, 6/6 folds).
* **The *operational* copy.** Against the copy the shipped configuration actually performs,
  `deep_starmap` has **+0.0855 [+0.0489, +0.1351]**, 2.6x the envelope. That is **R14** — a defect
  in the copier, not a property of the tissue — and it is deliberately unfixed because fixing it
  makes the negatives stronger.

### 1c. The screen for the remaining five: zero fits, and it already exists

Run these **before** committing a core-hour of fits.

| step | tool | scope | cost |
|---|---|---|---|
| 1 | `python -m src.bench3.survey_datasets --csv survey.csv` | all built datasets at once — `planes`, `cells/section`, `spacing`, `markers`, **`flank_r`** | one pass, no fits. `flank_r` is bench3's own discriminability probe, on the same `_morans_i` definition `paper_morans_pearson` uses. **STARmap measures 0.98**; the README's rule is "read that column before anything else" |
| 2 | `scripts/t09_depth_ceiling.py --dataset X` | split-half reliability R, ceiling `√R`, best copy, operational copy, shuffled floor | no model, no fit, no generation — reads the built input |
| 3 | `scripts/t09_ceiling_bootstrap.py --dataset X --compare reports/t09_ceiling_bootstrap_starmap.json` | 400-replicate CI on the headroom, and P(X > tier-1) | hours on the large volumes, still zero fits |

⚠️ **The screen covers 2 of the 6 headline metrics.** `t09_depth_ceiling.py` is `marker_depth_r`-only
(no `--metric` flag); `t09_zeroshot_ceiling.py` offers `marker_depth_r` and `morans_pearson`.
Extending it to `gearys_pearson`, `umap_mixing`, `marker_field_r` and `celltype_localization` reuses
the kernels already in `losses/metric_aware.py` — **1–2 days**, and it is `eval/ceiling.py`, which
`specs/10` requires anyway. Do it as part of the screen, not after the fits.

**Prerequisite, unknown from here and checkable in minutes:** which of the seven are *built*
(`prepare_dataset`) on your box. An unbuilt dataset costs a build before it costs a screen.

### 1d. What the screen cannot tell you, and which matters more

On tier-1 — the *most* informative of the eighteen — the shipped configuration sits **below the
`flanking_copy` floor on four of six metrics**, by 0.33 on both autocorrelation metrics
(`reports/advisor_report.md` §5, re-measured 2026-09-08, bitwise-confirmed across three runs):

| metric | v25 shipped | `flanking_copy` floor | `oracle` ceiling | v25 − floor |
|---|---|---|---|---|
| `morans_pearson` | 0.6541 | 0.9836 | 1.0000 | **−0.329** |
| `gearys_pearson` | 0.6535 | 0.9840 | 1.0000 | **−0.331** |
| `marker_field_r` | 0.6824 | 0.8857 | 0.9997 | **−0.203** |
| `marker_depth_r` | 0.8331 | 0.9794 | 1.0000 | **−0.146** |
| `celltype_localization` | 0.7546 | 0.7765 | 0.9808 | −0.022 |
| `gene_mean_spearman` | 0.9901 | 0.9863 | 1.0000 | **+0.004** |

A dataset's informativeness is a **necessary** condition for separating methods. It is not
sufficient for v25 to win, and on the one dataset where the room is measured, v25 does not currently
reach the floor a model-free copy reaches. **Cost the campaign as producing a negative-results table
with floors and ceilings beside it** — which is what `reports/advisor_report.md` §10 says the paper
is — not as a search for a win.

---

## 2. The unit, and the measured anchors

**Every timing in this project is CPU, single-threaded** (`scripts/_bench3_paths.py::set_torch_threads`
pins `torch.set_num_threads($OMP_NUM_THREADS)`; the campaign box is shared and wide, and the scripts
are written to run several to a box at one thread each). That is exactly your 192-core EPYC, so the
anchors transfer directly.

**Unit: one cold 2400-step fit, one thread, uncontended.**

| dataset | training cells | genes | per cold fit @2400 | provenance |
|---|---|---|---|---|
| `starmap_visual_cortex` | 16 527 | 28 | **62 min** | **measured** (`progress/t09_…` cost table; pilot measured 1712.7 s at 1200 steps = 28.5 min, 1709 MB peak RSS) |
| `deep_starmap` | ~113 000 | 1 017 | **3.82–4.09 h, mean 3.92** | **measured**, 6 fits, 84 615 s total. *"has never come in under 3.8 h on this box"* |
| `cosmx_nsclc_3d` | ~227 000 | 960 | **~8 h** | extrapolated by the record, stated there as **a lower bound** |
| `merfish_thick_hypothalamus` | ~45 000 | 156 | ~2.1 h | derived here |
| `merfish_thick_cortex` | ~16 500 | 254 | ~1.3 h | derived here |
| `exseq_breast_cancer` | ~1 300 | 297 | ~0.4 h | derived here, overhead-floored |
| `starmap` @ `consecutive-5` | ~8 200 | 28 | ~0.65 h | derived here (the design clamps to n−2 = 5, leaving only sections 1 and 7 as input) |

*Derivation, stated so it can be rejected:* a power law in training cells fitted to the two measured
points, `t = 0.0740 · N^0.693` minutes. It reproduces both anchors by construction and absorbs the
gene-count effect into the cell exponent, which the record supports — *"cell count has driven the
scaling on both measured points more than gene count"*, `genes_per_step = 128` caps the per-step gene
work. It is still an extrapolation.

🚨 **Carry a contingency, and the reason is in the record, not in caution.** *"I have under-estimated
fit cost on five consecutive occasions in this project."* The most recent miss was **1.4–1.7x**
(2.3 h predicted, 3.82–4.09 h measured). Every derived row above should be read as a lower bound.

⚠️ **One measurement is missing and it decides your wall clock: peak RSS on `deep_starmap`.** Only
tier-1's 1709 MB was ever recorded. Concurrency on a shared 192-core box is bounded by RAM, not
cores. **One fit with `/usr/bin/time -v`** converts a 10-hour campaign into a known quantity, or
reveals a 3-day one.

---

## 3. Per-dataset selection at full budget — confirmed, it cannot be shortened

You are right, and there are three independent reasons, all measured:

1. **The training-free-option rule** (`specs/09` §3). `layout_mode`, `prior_mode` and `expr_mode`
   each have an option that reaches final behaviour **without training** — `resample`, `iid`,
   `cross-mix`. A reduced-budget comparison therefore measures the budget, not the gate. They merge
   into **one 18-cell joint gate, every cell at the selected budget**, and are scored jointly because
   their errors compound through coordinate descent's ordering.
2. **R8, the measurement behind it.** At 25 % of the budget `cross-mix` won under both priors and at
   full budget came **last** under both; `iid` won at 25 % on exactly the two expression paths where
   the prior can act, and lost both at full budget. **The shipped configuration ranked fifth of six
   at the reduced budget.** From 600 → 2400 steps `morans_pearson` gains **+0.3432** for `zinb-flow`
   and **−0.0180** for `cross-mix`.
3. **The budget gate is invalid at a reduced budget by construction.** A reduced-epoch fit of the
   `2×` candidate *is* the `1×` candidate. Pinned by `test_budget_gate_is_not_scored_at_a_reduced_budget`.

And a fourth that fires at run time: **condition (2)**, unconverged incumbent. On the fixture the
incumbent fell short by **+0.3609 / +0.4260 / +0.8095** on three metrics against
`selection_convergence_tol = 0.05` (minimum two), so `text_emb_mode` was escalated to full budget
too. Expect it to fire on real data; budget for it.

**Measured composition of one selection run** (`reports/config_selection_synthetic.md`, "Fits
issued: 23"): **2 × 1200 + 19 × 2400 + 2 × 600**.

In full-budget-equivalents (FBE): `2(0.5) + 19(1.0) + 2(0.25)` = **20.5 FBE**, plus **+2 FBE** for
the condition-(2) escalation → **≈ 22.5 FBE per dataset**.

> **This is the lever, and it is dataset count — not the selector.** Each additional headline dataset
> costs ~22.5 full-budget fits *before it produces a single headline number.*

---

## 4. The CPU bill

### 4a. Comparators — costed separately, as asked

**Do this check first; it costs minutes and it forks the line item by an order of magnitude.**

`specs/10` §3 verified 2026-08-20 that `benchmark-pbya-v3/results/`, `benchmark-pbya-v2/results/`
and `benchmark-pbya/results/` **do not exist in this repository** — the tree is gitignored and was
never committed. Only a `per_section_metrics.csv` was recovered. On *your* box the predictions may
still exist.

| fork | what runs | cost |
|---|---|---|
| **A1 — predictions exist** | `evaluate_all --force` re-scores them on the pinned evaluator. **No method is re-run.** | 4 methods × 3 regimes × 3 seeds × 4 datasets ≈ 144 scorings. Scoring measured at **408–414 s/arm on `deep_starmap`**, less on the smaller ones → **≈ 12–24 core-h** |
| **A2 — predictions do not exist** | SpatialZ, FEAST, isoST and v20 re-run from scratch, per dataset, per regime, per seed | ⚠️ **Unmeasured, and I will not quote a number.** `specs/10` §12 called them *"individually far cheaper than a fit"* — that was a model, and the same model was 2.3x optimistic on v25. SpatialZ's published defaults are `nb_iter_max=3000`, `num_projections=80`, on up to 227 k cells |

**If A2: put 16 timing runs in the bill** — 4 methods × 4 datasets, one each, single seed — and
re-derive the line item from them before committing the rest. That is the same discipline
`progress/` applies to v25 fits (*"One fit, timed, before the other five"*), and it is what stops
the sixth consecutive under-estimate.

**Why re-scoring is not optional even if numbers exist elsewhere.** §13.1: the recovered file is
**evaluator-heterogeneous**. `paper_marker_field_ssim`, `paper_gene_detection_spearman` and
`paper_rare_celltype_localization` are populated on **132/132** rows for v18/v20/v21 and **0/3** for
`spatialz` on STARmap. The two sides of the tier-1 head-to-head were scored by **different revisions
of `evaluate_paper.py`**, and whether the *shared* metrics moved between them is unknown and
undeterminable from the CSV — `celltype_localization` gained its rare-type split in the same change.

**Three seeds on the comparator side too.** §4.2b: a clearance against a referent takes the **worst
envelope in the comparison**, not the arm's own. SpatialZ is stochastic, so it carries an envelope
and needs seeding. The two probes (`oracle`, `flanking_copy`) are deterministic copies — **1 seed,
and free**; both are already measured on tier-1 (`reports/pilot.md` §3) and reproduced exactly on
2026-09-08 (`reports/r11_probes_recheck.json`, all twelve values to four decimals).

**Also free:** the `evaluate_paper.py` SHA-256 assertion before and after every run —
`7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992`, 764 lines.

### 4b. Per-metric per-arm envelopes — 0 extra v25 fits, but they set the shape

§4.2a is a per-metric **and per-arm** envelope, measured on each dataset, never borrowed. The
measurement that forces it, on tier-1's `expr_mode` gate at three post-fix seeds:

| metric | `cross-mix` | `zinb-flow` | envelope | vs a pooled 0.0335 |
|---|---|---|---|---|
| `morans_pearson` | 0.0054 | **0.0574** | 0.0574 | pooled too **small**, 1.7x |
| `gearys_pearson` | 0.0027 | **0.0595** | 0.0595 | pooled too **small**, 1.8x |
| `umap_mixing` | 0.0068 | 0.0190 | 0.0190 | pooled too large, 1.8x |
| `marker_field_r` | 0.0049 | 0.0148 | 0.0148 | pooled too large, 2.3x |
| `marker_depth_r` | 0.0084 | **0.0472** | 0.0472 | pooled too **small**, 1.4x |

A 4.0x range across metrics, up to **22x across arms**, and **the worse arm is not predictable** —
on `deep_starmap`'s `text_emb_mode` gate both arms are `zinb-flow` and the worse arm *alternates by
metric*. So it must be measured, which means **seeding both arms and reporting them separately**.

**Cost: zero extra fits, computed from the 3-seed runs by `scripts/t09_seed_claim.py`** — provided
both arms of every claim-bearing contrast run at 3 seeds. That proviso is the whole cost, and it is
what lands in §4a.

Two things ride along free:
* **§4.2i** — quote the null's sampling distribution beside every spread. For a correlation over `n`
  genes, `sd ≈ 1/√(n−1)`; the range of three draws has expectation `1.693 sd` with its own
  `sd = 0.888 sd`, a **CV above 50 %**. Analytic, no compute. It also means the 2.2x gap between two
  datasets' measured envelopes is noise in the noise estimate, not a fact about the datasets.
* **R10 is retired and the replacement is stronger.** The 0.0120 same-config drift was a defect
  (a salted builtin `hash()` in two RNG seeds), not run-to-run variation. Post-fix, refitting one
  configuration in a separate process agrees **bitwise — 36 of 36 values, largest difference exactly
  zero**. Envelopes are now genuinely seed-to-seed. *Note this for §5: it is the guarantee a GPU
  backend puts at risk.*

### 4c. The v25 fits — recommended four-dataset core

| line | fits | composition | core-h |
|---|---|---|---|
| **Selection @ full budget** (§3, 22.5 FBE each) | ~92 | starmap 23.2 · `deep_starmap` 88.2 · `exseq_breast_cancer` 9.0 · `merfish_thick_hypothalamus` 47.3 | **167.7** |
| **Headline six-metric table, 3 seeds** | 24 | starmap 3 regimes (8.1) · deep `paper` only (11.8) · exseq 2 regimes (2.4) · thick-hypo 2 regimes (12.6) | **34.9** |
| **Boundary rows, R3** (starmap, both ends) | 6 | 6 × 1.03 | **6.2** |
| **E1 zero-shot** (`deep_starmap`) | 12 | 2 gene splits × 2 arms × 3 seeds × 3.92 h. *The replication measured 6 such fits at 23.5 core-h* | **47.0** |
| **Per-arm envelopes** | 0 | computed from the rows above | **0** |
| **Ceilings, floors, probes, E5, E4** | 0 | model-free or training-free | **0** |
| **v25 subtotal** | **~134** | | **≈ 256 core-h** |

**Honest range: 256–400 core-h**, applying the record's documented 1.4–1.7x under-estimate history
to the derived rows.

**Wall clock is a scheduling choice, and RAM sets it.** The runs are independent per
(dataset, arm, seed) at one thread each:

| concurrency | wall clock | gated by |
|---|---|---|
| 24-way | **~11 h** | needs ~24 × peak RSS. Tier-1 is 1.7 GB; **`deep_starmap`'s is unmeasured** |
| 8-way | ~32 h | the conservative assumption if deep fits are large |

### 4d. What a fifth dataset costs, and the two the spec would add

| candidate | selection | headline (2 regimes × 3 seeds) | total added | what it buys |
|---|---|---|---|---|
| `cosmx_nsclc_3d` | **180 core-h** | 48 core-h | **≈ 228 core-h** | non-brain **and** widest uniform gaps (30 µm) in one row. But `paper_2_4` holds out **2** sections, so the per-section Wilcoxon runs at n = 2 |
| `merfish_thick_cortex` | 29 core-h | 8 core-h | ≈ 37 core-h | a second brain volume at STARmap's scale. bench3's README warns this is exactly the shape that *"adds breadth without adding discrimination"* — **let the §1c screen decide it** |

Adding `cosmx_nsclc_3d` **nearly doubles the campaign for one row**. `specs/10` §12 already said so
from a model; §2's anchors now say it from measurements.

### 4e. Ablations — a decision, not a line item

The closeout has already spent these and the answers are in.

| arm | state | recommendation |
|---|---|---|
| **A2** (metric-aware weights) | **A9 ran: 6 fits, 3 seeds, UNINFORMATIVE.** Worst primary envelope **0.4323** against the condition's 0.067 — **6.5x over** — and both autocorrelation primaries had signs disagreeing across seeds | **Do not re-run at 3 seeds.** §9: *"more seeds is not a cheap path — this design was already too noisy at three."* Instead adopt the standing recommendation: **set all three weights to 0** for the next campaign, which refits anyway and adopts it at zero cost |
| **A7** (SEFL) | **REFUTED at 3 seeds**, re-measured under today's code and agreeing more strongly — SEFL costs on all six metrics, 4 of them 3/3 on sign; on seed 3 the ON arm's `morans_pearson` goes **negative** | ⚠️ measured at **1200 steps only**, not the selected 2400. Re-running at 2400 costs 6 fits/dataset. **Not recommended** — that arm did not underperform, it *collapsed* (`i_gen` at 1.6–2.4 % of target, 218 alarms from step 250 of 1200), and the burden is on the claim that more steps recover a dead field |
| **A1, A3, A4, A5, A6, A8** | 1-seed arms per `specs/10` §12 except A8 | ~8–14 fits/dataset. Price when the driver exists and the §1c screen has fixed the dataset list |

🚨 **One thing that is not optional and costs nothing.** The three metric-aware weights ship **on at
0.5**, selected on the **synthetic fixture** by aggregate rank with per-metric margins of
0.0052 / 0.0101 / 0.0018 inside a 0.0335 envelope, on **one seed** — and they are inside the baseline
every number in this project was measured against. Whichever way you go, the paper has to say so.
Note also that with all four SEFL weights at zero, `check_collapse` **never runs on the shipped
model** (it is gated on `sefl_teacher is not None`), so every empty alarm list in every SEFL-off
campaign means *never armed*, not *did not fire*. Fixed 2026-09-07 with a test; verify the fix is in
the tree you fit from.

---

## 5. GPU — there is nothing to qualify yet, because there is no GPU path

You asked what it would take to qualify GPU first. The honest answer is that qualification is
preceded by **implementation**, and the three tests you named cannot fail on GPU today because
nothing in this codebase can put a tensor on one.

### 5a. The evidence, from the tree

| check | result |
|---|---|
| occurrences of the string `device` in `spatialcpav25_gen/` | **6 total** — 2 in `config.py` (the field and its validator), 4 as `device=<tensor>.device` propagation inside ops |
| `Config.device` read by any code path | **none.** It is declared, validated against `DEVICES = {"auto","cpu","cuda"}`, hashed into the config — and never consulted. `device="cuda"` validates, then fits on CPU |
| `torch.cuda`, `torch.device(...)`, `.cuda()` in `spatialcpav25_gen/`, `tests/`, `scripts/` | **zero occurrences** |
| every `.to(...)` in the package | a **dtype** cast (`torch.float32`, `torch.long`, …). Not one device move |
| `torch.Generator()` construction sites | **23**, across `field.py`, `embeddings.py`, `expression.py`, `layout.py`, `retrieval.py`, `spatialcpav25_gen.py`, `train/loso.py` — **all CPU-default** |

`Config.device` is a dead field, and its docstring (*"`auto` resolves to cuda when available, else
cpu"*) describes behaviour that does not exist. That is a Convention 6 problem in its own right: a
field that silently does nothing is worse than one that raises.

### 5b. The port, and why the three tests you named are the right ones

**Each of the 23 generator sites is a determinism-relevant port point**, and the design choice made
there decides whether your tolerance criterion is even meaningful:

* **Generators moved to CUDA.** A CUDA generator with the same seed produces a **different draw
  sequence** from a CPU one. A GPU fit is then a *different model*, not a nearby one, and "agree to
  a stated tolerance" has nothing to measure — the difference is not numerical error.
* **Generators kept on CPU, draws transferred.** The draws stay bitwise identical and the residual
  difference is reduction order alone. **This is the only design under which your tolerance
  criterion is well-posed** — and it costs a host↔device transfer on every stochastic op in the step.

Your qualification statement silently assumes the second. Make it explicit before anyone writes code.

**`test_batch_shape_does_not_change_a_single_value` is the most exposed, and it is load-bearing.**
Read its docstring: the guarantee is *engineered*, not incidental — `forward` zero-pads the last
chunk so every matmul in every query is `(grf_chunk_points, M) @ (M, d_h)`, *"the same shape,
therefore the same kernel, blocking and reduction order"*. Without the padding a one-row query is
dispatched to a matrix-**vector** kernel and lands **~1e-5** away. **On cuBLAS that argument does not
carry**: kernel selection is a heuristic over shape *and* device occupancy, and split-K reductions
can vary with runtime state. The companion test already concedes the limit on CPU —
`test_chunk_boundaries_do_not_change_a_single_value` asserts bitwise only *within* a chunk size and
falls back to `< 1e-5` across chunk sizes, because *"that is a property of BLAS, not something
padding can remove."*

What breaks if it breaks: this is **G1.2**, and it is what makes **intersection consistency exact by
construction** — two crossing sections emitting **bitwise identical** expression on an *untrained*
model, at every checkpoint, with no consistency loss applied. That is `reports/t09_closeout.md`
§2.3, the property no competing method has, and one of the three surviving positive claims. A GPU
backend that degrades it from *exact* to *within 1e-5* does not cost precision — it costs the claim.

**`test_resumed_fit_is_bitwise_identical`.** `train/checkpoint.py` saves the model, optimiser,
scheduler, EMA teacher and the SEFL **numpy** `bit_generator.state` — and **no torch RNG state**,
because Convention 3 means no global torch RNG is ever used. Under the CPU-generator design that
stays correct unchanged. Under the CUDA-generator design, CUDA RNG state must enter the checkpoint
payload — a schema change to a file that already carries a portability guard naming its fields
(`train_steps: checkpoint 1200 -> this run 2400`).

**A step nobody has listed, and it comes before any GPU work.** `pyproject.toml` pins
`torch==2.2.2` **exactly**, on the project's own reasoning that *"a silent dependency bump that
shifts a metric by 0.01 is not a debuggable event"*, installed by pip into `bench_spatialcpav25`
(the env yml fixes `python=3.12` and nothing else). If the installed wheel is CPU-only, enabling
CUDA means a **different build of the same version** — which can change CPU kernel dispatch too.
**Re-run the bitwise suite on CPU under the new wheel before touching the GPU**, or you will not
know which backend moved a number.

### 5c. How much of a fit could even run on GPU — unmeasured, and structurally bounded

The per-step path is **not** torch-dominated. In `numpy`/`scipy` `float64`, on CPU, by construction:

| component | what runs per step |
|---|---|
| retrieval (`model/retrieval.py`) | `cKDTree` per section, blocked `query`, `argsort`, `einsum`, the z-window derivation — **all numpy float64** |
| layout head (`model/layout.py`) | `cKDTree(coords).query(k+1)`, `count_neighbors` for the Strauss/hard-core repulsion |
| metric-aware losses (`losses/metric_aware.py`) | `cKDTree` kNN weight graph per LOSO step |
| SEFL (`losses/sefl.py`) | `cKDTree` |

None of that moves to GPU without a rewrite far larger than a device port. The torch half is small
MLPs at `(batch_cells, genes_per_step=128, decoder_hidden)`. **No profile of the split has ever been
taken**, so no speedup can be quoted — and one cannot be taken here (this container has no torch).

**Direct answer to your fallback question: nothing silently falls back to CPU.** Two separate facts:

1. **Under a memory cap, PyTorch raises.** `torch.cuda.OutOfMemoryError` — it does not degrade to
   CPU. Convention 6 is on your side.
2. **The real issue is the opposite of a fallback.** The numpy/scipy half never leaves CPU *by
   construction*, and nothing announces it — no cap, no warning, no error. A "GPU fit" would be a
   partly-GPU fit whose CPU fraction is unmeasured.

### 5d. Capping the card so it stays shareable

In-process, in code, and the only cap that lives with the job:

```python
torch.cuda.set_per_process_memory_fraction(frac, device)   # e.g. 0.25 of total
```

Three caveats:

* It caps **the PyTorch caching allocator only**. The **CUDA context** — roughly 300–600 MB per
  process — sits outside it, so N concurrent processes pay N context overheads on top of N × frac.
* It must run **after** the CUDA context exists (`torch.cuda.init()` / first CUDA call).
* Exceeding it **raises**, per §5c.

Stronger isolation, if the card is genuinely contended: **MIG** partitioning (hard, hardware-enforced,
if the card supports it), or **MPS** with `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`. `CUDA_VISIBLE_DEVICES`
selects a card; it caps nothing.

### 5e. Qualification cost, and the recommendation

| step | cost |
|---|---|
| 0 · Confirm driver/wheel; **re-run the bitwise suite on CPU under the CUDA wheel** | ~0.5 day |
| 1 · **Write the port** — resolve `Config.device`, decide RNG placement, thread a device through model construction, all 23 generator sites, `train_ctfflow`, `forward_train`, generation, checkpoint | **3–5 days**, touching every model module — i.e. re-opening the code every committed number was produced by |
| 2 · Qualification measurements — your three, plus the GRF chunk-shape test, plus a CPU/GPU tolerance stated as a **distribution** (§4.2i applies to this tolerance too: three seeds each backend, ≈ 6 tier-1 fits ≈ 6 core-h CPU + GPU time) | ~1–2 days |
| 3 · If the tolerance is non-zero, **every claim-bearing number must come off one backend** — and this project's existing numbers are CPU | all-or-nothing |

### 🚨 Recommendation: **CPU. Do not qualify GPU for this campaign.**

Four reasons, in order of weight:

1. **The workload is throughput-bound, not latency-bound, and it already parallelises 192-wide.**
   Every fit is independent per (dataset, arm, seed) at one thread each. A 24-way slice of your box
   turns 256 core-h into ~11 h. One shared, memory-capped GPU runs far fewer concurrent processes.
   GPU buys latency on a *single* fit; the campaign does not need that.
2. **The speedup is unmeasured and structurally bounded.** Retrieval, layout, metric-aware and SEFL
   are scipy/numpy on CPU per step, and no profile of the split exists. Spending a week to accelerate
   an unknown fraction is not a costed decision.
3. **It puts the project's strongest surviving guarantee at risk for no claim.** Bitwise refit
   agreement (36/36, difference exactly 0) and exact intersection consistency are two of the three
   things that *work*. A second numerical backend introduced mid-campaign means every number must
   state its backend, and by the project's own §4.2a logic a figure measured in one setting may not
   be applied to another.
4. **The port re-opens frozen code.** ~1 engineering week, across every model module, on a tree whose
   campaign just closed — against ~2–3 weeks of driver work (§0) that is on the critical path and
   buys the campaign directly.

**Qualification costs more than the speedup buys on this campaign.** Revisit it only if the
`deep_starmap` RSS probe (§2) shows concurrency is memory-bound below ~8-way, which is the one
scenario where the CPU box stops being the cheaper parallel machine.

---

## 6. What to do first — five things, none of them a fit

1. **Check whether a predictions tree exists on the box** (§4a). Minutes. Forks the comparator line
   item by an order of magnitude.
2. **Run `survey_datasets.py`** across the built datasets and read the `flank_r` column (§1c).
   No fits.
3. **Run the ceiling + bootstrap screen** on the five unmeasured survivors (§1c). No fits. Extend
   `eval/ceiling.py` to all six metrics while doing it (1–2 days).
4. **One `deep_starmap` fit under `/usr/bin/time -v`** for peak RSS (§2). One fit, ~4 core-h, and it
   sets the whole campaign's wall clock.
5. **Start the T10 driver** (§0). It is the critical path and it is 2–3 weeks.

Only then fix the dataset list and commit the ~256 core-h.

---

## Sources

`reports/t09_closeout.md` · `reports/advisor_report.md` · `specs/10_TASK_benchmark_and_baselines.md`
§0a §1 §2 §3 §4.1 §4.2 §4.2a §4.2b §4.2i §4.4 §5 §11 §12 §13.1 · `specs/09` §3 ·
`reports/pilot.md` §3 §5 · `reports/config_selection_synthetic.md` ·
`progress/t09_inference_and_calibration.md` (measured fit costs) ·
`reports/t09_depth_ceiling_{starmap,deep}.md` · `reports/t09_ceiling_bootstrap_{starmap,deep}.md` ·
`reports/t10_a9.md` · `benchmark-pbya-v3/README.md` · `benchmark-pbya-v3/src/bench3/survey_datasets.py` ·
`spatialcpav25_gen/config.py` · `tests/test_noise.py` · `tests/test_checkpoint.py` ·
`spatialcpav25_gen/train/checkpoint.py`
