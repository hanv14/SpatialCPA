# v20/v21 — the evidence base, audited under the conventions v25 was held to

**2026-09-09.** The direction is that v20/v21 is the method and v25's field representation plus its
negative result is the methodological contribution. This audits the v20/v21 evidence against
`specs/10` §4.2a–j, §4.6 and §5 — the rules v25's own claims were withdrawn under — and does not
assume the older numbers are sound because they are older. **v25's numbers looked sound until they
were audited, four times.**

---

## 0. What the evidence base actually is

Everything this project holds about v20/v21 is in `specs/10` §13.2–13.4, derived from a
`per_section_metrics.csv` recovered from an earlier campaign: 829 rows, 18 datasets, 8 methods.

🚨 **That file has never been in this repository.** Searched the working tree and the full history
including reflog-only commits (`git log --all --reflog --diff-filter=AMR`): **no
`per_section_metrics.csv`, no `all_metrics.csv`, no `summary_by_method.csv`, at any commit.** So
§13's three tables — the entire v20/v21 case — are **attested by the spec document alone**. That is
§8b (measured but unrecoverable) for the project's new headline claim, and §4.2a-iv's shape: a set
of numbers carried in prose with no artifact behind them.

This is the first thing to fix and it is nearly free: the re-scored tree makes §13 reproducible, and
until it does, **no number in §13 may enter the paper.**

✅ **What does exist**: v20 (`reference/learn_spatialcpav20.py`) and v21 (`learn_spatialcpav21.py`,
at the repository root — not under `reference/`), both with bench3 wrappers and `METHODS` entries,
both marked `available: True`. The methods are runnable. It is the *numbers* that are unsourced.

---

## 1. Convention-by-convention

| convention | v25 was held to | v20/v21 status | verdict |
|---|---|---|---|
| pinned evaluator (§0, §2) | SHA asserted before and after | ✅ **closed for tier-1** by the re-score into `results_rescored/`, six methods, `failed 0` | **survives** |
| per-metric per-arm envelopes (§4.2a) | required; pooled figure retired | **no envelope exists** — every v20/v21 number is a single run | 🚩 **unestablished** |
| three seeds (§4.2, `claim_min_seeds`) | claims withdrawn for failing it | **one seed**, for every method and every row | 🚩 **unestablished** |
| median over sections (§4.6) | enforced in code | preamble says median; §13.2's own header says **cell-count-weighted** | ⚠️ **mislabelled — numbers appear right** |
| tier purity (§1) | enforced by `assert_tier_purity` | §13.3's pooled wide table is already flagged illegitimate in the spec | **survives, because it was already caught** |
| one metric panel (the two-`SIX` defect) | found and recorded | §13 reports **7 metrics** and never names which seven | ⚠️ **unstated** |
| referents beside every number (§2, §4.2c) | floor and ceiling quoted | §13 quotes **no floor and no ceiling** | 🚩 **the biggest reading gap — see §3** |
| artifacts carry their arm (§4.2a-ii) | recovered for r11 | `method_params` is written per prediction — **checkable in the re-scored tree** | ✅ **recoverable** |
| the record's values exist in artifacts (§4.2a-iv) | 0.5 found to be prose-only | §13's tables are **prose-only** until the re-score lands | 🚩 **unestablished** |
| validity screening (§4.2j) | alarms must be armed | v20/v21 fall back to a numpy path that "is not the method under test", caught by `invalid_log_markers` **in the run log** | ⚠️ **see §2** |

---

## 2. Three defects the audit found

### 2.1 ⚠️ §13.2's estimator label contradicts §13's own preamble

§13 states: *"Everything below is the **median over held-out sections**, per §4.6."* §13.2's table
header states: *"cell-count-weighted over the three held-out sections."* A cell-count-weighted
average is a weighted **mean**, which §4.6 forbids for any claim-bearing statistic.

**The numbers are the medians and the header is wrong**, and it can be shown rather than assumed:
§4.6's worked example gives v20's `marker_depth_r` per section as 0.7422 / 0.9704 / 0.9783 —
median **0.9704**, mean **0.8970**. §13.2's table lists **0.9704**. With section sizes of
4187 / 4102 / 4162 a cell-count-weighted average is within a hair of the plain mean, so it cannot
produce 0.9704. The label is a documentation defect, not a numbers defect.

It matters anyway, because `marker_depth_r` is the exact metric §4.6 uses to show that the two
estimators **invert the v20-vs-SpatialZ verdict**. A reader who takes the header at face value reads
the one table where the estimator decides the result as having used the forbidden estimator.

### 2.2 🚩 Zero seed replication — and this is the load-bearing one

Every v20, v21 and SpatialZ number in §13 is a **single run**. There is no across-seed spread for
any method, on any dataset, in the whole prior campaign. So under §4.2a/§4.2b:

* **No margin in §13 is established.** Not v20's `marker_depth_r` win (0.9704 vs 0.9267), not v21's
  `marker_field_r` win (0.8881 vs 0.8522), not SpatialZ's localization lead (+0.061 over v20).
* **§13.2 already half-says this**: *"the margin is inside twice the reproducibility envelope"* —
  reasoned from the **retired 0.0120**, which was a seeding defect rather than run-to-run variation.
  With that figure gone, the margin has no noise scale at all.
* ⚠️ **And v20/v21's determinism is unknown.** R10 established that *v25* refits bitwise-identically
  at a fixed seed after `section_seed` replaced a salted `hash()`. That says nothing about v20/v21,
  which are separate single-file implementations with their own RNG use. Whether their seed-to-seed
  spread is 0.001 or 0.05 has never been measured.

**This is the objection a reviewer will raise first**, and it is self-inflicted: a paper whose
methodological contribution is *how to read a repeated-seed benchmark* cannot report its own headline
at one seed.

### 2.3 ⚠️ The fallback screening is a run-time check, and the re-score cannot re-perform it

v20 and v21 both degrade to a numpy latent-grounded fallback when torch is missing or the flow fails.
bench3 treats that correctly — *"the fallback is not the method under test, so v3 fails the run
rather than scoring it"* — but it detects it from `invalid_log_markers` **in the run log**.

`evaluate_all --force` re-scores `prediction.h5`. It does not read logs. **So the re-scored numbers
inherit whatever validity screening happened at run time**, and if a fallback prediction was ever
scored, the re-score launders it onto the pinned evaluator with a clean provenance trail. That is
§4.2j's shape: a check that cannot be re-armed, on an artifact that now looks better-sourced than
before.

**Cheap and decisive**: the logs either exist beside the predictions or they do not. If they do,
re-run the marker check over them. If they do not, say the screening is attested by the original run
and not re-verifiable — which is a caveat, not a disqualification.

---

## 3. 🚨 The reading gap that matters most: §13 quotes no floor

Every v25 number in this project is reported against `flanking_copy` and `oracle`, because §2 says a
score means nothing until it is placed between them. **§13 has no floor and no ceiling column at
all.** So "v20 0.9704 beats SpatialZ 0.9267" is currently unreadable in the way this project has
spent four rounds insisting numbers must be readable.

⚠️ **And the arithmetic cannot be done yet.** Placing §13's numbers beside today's probe values
(`reports/r11_probes_recheck.json`) would be exactly the cross-instrument comparison §13.1 forbids —
§13's numbers are from the old evaluator revisions, the probes are from today's. **The re-score is
what makes it legitimate**, and it is the first thing to compute once `results_rescored/` is read.

🚩 **State the risk before doing it, so the result cannot be fitted afterwards.** v20 is
`resample` + Bernoulli cross-mix, and R13 established that `cross-mix` under `resample` **is a copy**
— matching a model-free "copy the nearest other section" to 0.001 on `deep_starmap`. So v20 is,
mechanically, a structured copier. Two outcomes, both publishable, and they want different papers:

| if, on one instrument | reading |
|---|---|
| v20/v21 sit **clearly above** the copy floor | the method adds something copying does not, and the paper's claim is sound as framed |
| v20/v21 sit **at or near** the copy floor | the method is a good copier, SpatialZ is a slightly different copier, and the headline becomes *"on this protocol, copying is most of what is achievable and every method is near that ceiling"* — which is v25's finding arriving on v20's numbers |

The second is not a bad paper. It is a **stronger** version of the methodological contribution. But
it is a different claim from "v20/v21 beats SpatialZ", and the difference must be settled by
measurement before the abstract is written, not after.

---

## 4. 🚨 The exposure the audit cannot close: version selection on the reporting protocol

bench3 registers **fourteen** internal versions — v8, v11, v14, v15, v16, v17, v18, v19, v20, v21,
v22, v23, v24, v25 — and its own `METHODS` notes describe how they were arrived at:

> v18: *"H3D-FLA (v14) + **benchmark-driven fixes**"*
> v22: *"H3D-FLA (v21) **recalibrated on the real benchmark**"*
> v20: *"…keeping sparsity/count-ness intact at wide gaps (**fixes v19**)"*

That is the project's own documentation stating that versions were iterated against benchmark
results. **Selecting v20/v21 out of fourteen versions on `starmap_visual_cortex` / `paper_2_4_6` and
then reporting v20/v21 on `starmap_visual_cortex` / `paper_2_4_6` is model selection on the test
protocol** — regardless of the fact that each individual run is a bare invocation with no
`wrapper_args`, which it is, and which is to the project's credit.

✅ **The invocation half is clean and checkable.** Neither v20 nor v21 carries `wrapper_args`; both
run at defaults; and `method_params` records the resolved knobs per prediction, so "was it tuned per
dataset" is answerable from the re-scored tree in one pass. **The version-selection half is not
fixable by any re-scoring** — it is a property of how the method was arrived at.

**Three ways to discharge it, in increasing strength:**

1. **State it.** One paragraph in the methods: fourteen versions were developed against this
   benchmark; v20/v21 are the best of them on it; the headline is therefore a development result and
   not a held-out one.
2. **Report a held-out regime.** The protocol dataset is the development set; `wide_3_4_5`, the
   boundary holdouts, and the other claim-bearing datasets are not. A v20/v21-vs-SpatialZ comparison
   on a regime that did *not* drive development is a genuine test, and §13.3 already says the wide
   tier-1 head-to-head **has never been run by anyone**.
3. **Both.** This is what I would do, and item 2 is the same measurement the pilot's C2 criterion
   already asks for.

---

## 5. What it costs to bring v20/v21 to v25's standard

| item | what it buys | cost |
|---|---|---|
| ✅ pinned-evaluator re-score, tier-1 | one instrument for all methods; every column populated for every method (§13.1's 132/132-vs-0/3 defect) | **done** |
| read the re-scored tree into §13's tables | replaces prose-only numbers with sourced ones; §13 becomes quotable | **zero runs** — an aggregation pass |
| place v20/v21/SpatialZ against `flanking_copy` and `oracle` | makes every number readable; settles §3's two outcomes | **zero runs** — probes already regenerated on today's code |
| check `method_params` and the fallback markers | closes §2.3 and the per-dataset-tuning question | **zero runs** — reads the tree and the logs |
| **three seeds of v20, v21 and SpatialZ on tier-1** | the only thing that makes any margin a claim | **9 runs** ⚠️ no v20/v21/SpatialZ timing exists in this repository — time one, then multiply |
| the same three seeds at `wide_3_4_5` | discharges §4 item 2 **and** answers C2, which has never been measured | **9 runs** |

**Four of six cost nothing but reading.** The two that cost runs are the two that turn §13 from a
directional signal into a result, and they are comparator runs — far cheaper than a v25 fit.

⚠️ **No timing for any comparator exists in this repository**, the same gap §13.1a records for the
scoring pass. Time one v20 tier-1 run before committing to eighteen.

---

## 6. Summary

**Survives:** the instrument (now), the methods themselves (runnable, bare-invoked, self-describing
via `method_params`), §13.3's pooled-wide caveat (already caught), and the *direction* of every
signal in §13 — SpatialZ leads on localization, v20/v21 lead on `marker_depth_r` and
`marker_field_r`.

**Unestablished:** every **margin**, because there is no seed replication anywhere in the prior
campaign; every **level**, because no floor or ceiling is quoted; and every **number**, until the
re-scored tree replaces the prose.

**Not fixable by measurement:** that v20/v21 were selected from fourteen versions on the protocol
they would be reported on.

**Nothing here says the numbers are wrong.** It says they are currently *directional signals* —
which is exactly what §13.1 already calls them — and that promoting them to a headline requires the
same work v25's claims were put through.

---

## 7. Is the paper honest?

**Yes — as a framing. Not yet, as currently evidenced.** The gap is not in the idea; it is that
three specific things must be said out loud, and one of them is not yet measured.

### 7.1 The framing itself is sound

Presenting v20/v21 as the method and v25 as a companion negative is a normal, honest structure:
a working method, and a principled attempt at a more ambitious one that failed with the failure
localised. Nothing about it misrepresents what happened. Two things make it *more* honest than the
alternative, not less:

* v25's negative is being published rather than buried, with its mechanism (R4) and its methodology
  (§4.2a–j).
* v20/v21 are not being presented as the ambitious thing. The paper is not claiming a continuous 3D
  generative field works.

### 7.2 The regime objection, stated precisely — because the usual phrasing is slightly wrong

The concern as posed is *"v20/v21 win on a protocol where copying is nearly optimal, and v25's
negative says generation loses in exactly that regime."* The second half is right. **The first half
is true of `deep_starmap` and only partly true of tier-1**, and the distinction is the paper's.

§0a, measured model-free: copying reaches **98 %** of the achievable ceiling on `deep_starmap` and
**81 %** on tier-1. Tier-1 retains real headroom — **3.3x** its own `marker_depth_r` envelope
(corrected 2026-09-08 from 4.6x). So the protocol the paper would report on is **not** a saturated
one; it is the *informative* one, which is why §0a made it the headline dataset.

**The real objection is narrower and sharper: v20 is itself a copier.** v20 is `resample` +
Bernoulli cross-mix, and R13 established that `cross-mix` under `resample` **is** a copy — matching
a model-free nearest-section copy to 0.001 on `deep_starmap`. So the honest sentence is not "v20 wins
in a regime where copying is optimal" but:

> **The method that wins is a structured copier, on a task where copying reaches 81–98 % of what is
> achievable — and v25's negative result explains why that is the winning design rather than an
> embarrassment.**

That is a coherent, publishable, and genuinely interesting claim. **It is dishonest only if the
paper reports v20/v21 against SpatialZ without the copy floor beside it**, because then a reader
cannot see that both methods are near a bound that neither of them set. The project already owns the
instrument for this (`flanking_copy`, `oracle`), has already re-verified the probes on today's code,
and §3 above shows the computation is now legitimate for the first time.

### 7.3 What a reviewer will actually object to, in order

1. 🚨 **Single-seed headline in a paper about repeated-seed benchmarks.** This is the one that will
   sink it, and it is entirely self-inflicted. §4.2a–j is a contribution about reading margins
   against noise; §13's margins have no noise scale at all. A reviewer who reads the methods section
   and then the results table will notice within a page. **Nine comparator runs fix it.**
2. 🚨 **Version selection on the reporting protocol** (§4). Fourteen versions, "benchmark-driven
   fixes" and "recalibrated on the real benchmark" in the project's own registry, and the winner
   reported on the protocol that drove the iteration. Every method paper does some of this; **this
   paper cannot, because its contribution is evaluation rigour.** Disclose it, and report the
   wide-gap regime that did not drive development — which C2 asks for anyway and which has never
   been run by anyone.
3. ⚠️ **v25 characterised on an ablation.** §5.0: every draft of the six-metric table is
   `text_emb_mode=lookup` at `expr_pca_dim=16`, i.e. **ablation A3**. The shipped-configuration
   numbers now exist (A9, three seeds, zero extra fits) and are **worse** on six of seven metrics.
   The negative result is *stronger* on the real configuration — but if the paper quotes the A3
   numbers as v25's, a reviewer who reads the artifacts will find the label wrong.
4. ⚠️ **Asymmetric rigour.** v25's claims were withdrawn for failing §4.2a–j. If v20/v21's are not
   held to the same rules, the methodology section indicts the results section. This is the
   generalisation of (1) and (2), and it is the thing I would flag hardest.

### 7.4 The one thing that would make it dishonest

Not any of the above individually — each is fixable by a disclosure or nine runs. It is this:

> **Spending v25's audit as credibility while exempting v20/v21 from the audit.**

The negative result's value comes entirely from the rigour that produced it. A paper that says *"we
found six unstated conventions that silently decided verdicts"* and then reports its own headline at
one seed, with no floor, from an unsourced CSV, on a version selected on the reporting protocol, is
using the rigour as a credential rather than applying it. **That is the failure mode this project has
caught itself in five times already** — an instrument that reports a check it did not perform
(§4.2j), one level up, on the paper itself.

The remedy is the audit above, and four of its six items cost nothing but reading.

### 7.5 What I would state in the paper regardless of what the measurements say

* v20/v21 is a **structured copier**, and the copy floor is quoted beside every number.
* The headline protocol **drove the development** of the version being reported; the wide-gap
  comparison is the held-out one.
* v25's numbers are on its **shipped configuration** (A9), not on A3 — and the negative is stronger
  there.
* Margins are reported against **per-metric, per-arm, three-seed envelopes**, or explicitly as
  single-run directional signals. Not silently as claims.
