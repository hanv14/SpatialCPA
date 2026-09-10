# Repairing the emission model — a pre-registered design comparison

**2026-09-09. Nothing built. Read §0 before costing anything.**

The brief: v25 becomes the method paper, so the emission model must be repaired; everything upstream
is verified working and one operation destroys the result. This report costs the candidate repairs —
and opens by saying that **the evidence the brief rests on is the superseded arm**, which changes
which repair is worth doing and possibly whether one is needed at all.

---

## 0. 🚨 STOP — the chain the brief cites is `decoder_mu_link="softplus"`, and the shipped link is `exp`

`reports/chain_2400.md` is one of **four** chain artifacts. It is the arm the mu-link candidate was
run *against*, not the shipped one:

| artifact | cells emitted | I(mu) | **I(counts)** | I(real counts) | retention |
|---|---|---|---|---|---|
| `chain_2400.md` (+ `_calibrated`) | 11 168 | 0.8607 | **0.1297** | 0.4635 | **14.4 %** |
| **`chain_2400_explink.md`** | 48 343 | 0.9008 | **0.5253** | 0.4635 | **60.5 %** |
| **`chain_2400_grid.md`** | 267 567 | 0.9098 | **0.5408** | 0.4635 | **61.0 %** |
| real tissue (same step) | — | 0.5812 / 0.5717 | 0.4635 | — | **79.8 / 81.1 %** |

**Three things establish that `chain_2400.md` is the softplus arm**, and none of them is inference:

1. `scripts/t10_chain_diagnostic.py` carries `--decoder-mu-link {softplus,exp}`, documented as
   *"T10 candidate 1: softplus compresses dynamic range"*.
2. `Config.decoder_mu_link` defaults to **`exp`** (`config.py:1031`), and has since 2026-08-21.
3. The record states the swap directly: counts Moran's I **+0.1297 → +0.4782** against tissue's
   +0.4635 (`progress/t06_expression_head.md`).

So *"0.8607 → 0.1297, one operation destroys the result"* describes a link this project stopped
shipping a month ago. On the shipped link the same step gives **0.9008 → 0.5253**, and the generated
counts are **more** spatially autocorrelated than the real section's.

⚠️ **This does not mean the problem is solved**, and the record is explicit about why — see §1. It
means the brief's headline number is not the shipped decoder's, and the design work should be aimed
at what is.

### 0a. And the "0.09–0.19 against 0.62" gap is a cross-panel comparison the record already rejects

The two halves come from different datasets and different gene panels:

| number | source | panel |
|---|---|---|
| structured share **0.09–0.19** | six `deep_starmap` fits, R4 (v) | **1017 genes**, most with no spatial signal |
| real tissue's **0.62** | tier-1 STARmap `chain_2400_calibrated.md` | **28 marker genes**, every one spatially structured by construction |

`progress/t09_inference_and_calibration.md` names this exact error, twice — once against its own
voided run (*"I transferred a threshold across a panel boundary it does not cross"*) and once,
crucially, against **the record's own numbers**:

> *"The panel-transfer problem that voided the previous run applies to the **record's** numbers too,
> not only to mine."*

A median Moran's I over 1017 mostly-unstructured genes sits near zero on **both** sides, so 0.09–0.19
and 0.62 are not the same quantity measured on two arms. **The gap as stated is not a measured gap.**

---

## 1. What *is* established, panel-matched, on the shipped link

The record does carry a like-for-like comparison, and it is the number the design work should aim at:

> **`deep_starmap`, exp link, the 32 most structured kept genes: the model emits Moran's I of
> 0.071–0.097 where the real section has 0.283–0.291 — retention 25–34 %.**
> *"not the size factor, not the link — `exp` is in place and confirmed by the fit's own config hash
> — but the count draw itself."*

Beside it, unreconciled:

* tier-1 STARmap at exp shows retention **~103 %** (saved model), and the record **forbids quoting
  it**: *"R12's 'candidate 1 recovered it' must not be quoted as a property of the shipped decoder.
  It is a property of one saved model on one panel."*
* that saved model carries two live caveats — **48 343 cells against a ground truth of 4 187**, and
  `sd(log mu)` 0.777 against a tissue figure of 1.213.
* on `deep_starmap` the same statistic is `sd_log_mu` **0.287–0.295**, lower still.
* ⚠️ every chain artifact runs at `expr_pca_dim=16` (`t10_chain_diagnostic.py:345`) — the pilot
  stand-in, not the clamp rule's 28. A fourth arm mismatch, unremarked until now.

**So the honest state of the diagnosis is: the emission loses two thirds to three quarters of the
spatial structure on `deep_starmap`, and appears to lose none on tier-1, and no measurement
separates dataset from saved-model artifact.** The record names that separation as the next thing to
do. It has not been done.

---

## 2. 🚨 An arithmetic bound that rules two candidates out from numbers already in hand

Asked for: *"if any candidate would be ruled out by a measurement I already have, say so rather than
costing it."* Two are, and the calculation takes one line of algebra and no new data.

By the law of total variance, the structured share is `s = V / (V + N)` — `V = Var(mu)` across cells,
`N = E[Var(X | cell)]`. R4 (v) decomposes `N` on the generated cells of six fits:

| term | share of conditional variance |
|---|---|
| overdispersion `(1-pi) mu²/theta` | **0.568–0.614** |
| Poisson floor `(1-pi) mu` | **0.290–0.336** |
| zero inflation `(1-pi) pi mu²` | 0.080–0.101 |

Scale `N` by a factor `f` and the share becomes `s' = s / (s + f(1-s))`. **Any repair that only
touches the noise is a choice of `f`, and `f` has a floor: the Poisson term, ~0.31.**

| intervention | `f` | s = 0.09 → | s = 0.14 → | s = 0.19 → |
|---|---|---|---|---|
| remove overdispersion (theta → ∞), keep ZI | 0.41 | 0.194 | 0.284 | 0.364 |
| **pure Poisson** (also remove ZI) — the floor | 0.31 | **0.242** | **0.344** | **0.431** |
| what reaching 0.62 would require | **0.061–0.144** | — | — | — |

**The required `f` is 2.2× to 5.6× below the Poisson floor.** A conditionally-independent count draw
cannot get there: it would need a Fano factor around 0.3, i.e. a strongly *under*-dispersed emission,
which no standard count law for transcriptomics provides and which would itself be a large modelling
claim.

Turned round, with the noise already at Poisson, the structured component must rise:

> **`Var(mu)` must increase by 2.2×–5.1× *in addition to* eliminating overdispersion.**

✅ **Two independent routes agree.** The variance algebra says 2.2–5.1×; the dynamic-range statistic
says `sd(log mu)` 0.777 against a tissue 1.213, a variance ratio of **2.4×**. Different measurements,
same order. That convergence is the strongest single piece of evidence in this report.

### What this rules out

* 🚫 **Candidate A (theta prior / floor) cannot be sufficient.** Its best case is `f → 0.41`, which
  is short of the target by 3–7×. It is a *partial* measure at most.
* 🚫 **Candidate C (Poisson with an overdispersion floor) cannot be sufficient.** It is the `f = 0.31`
  row by construction — the best any independent draw can do, and still short by 2.2–5.6×.

Neither is worthless: both attack a term that is provably 57–61 % of the conditional variance, and
either would move the share materially. **But neither can close the gap alone, and costing them as
if they might is the mistake this report exists to prevent.**

---

## 3. The candidates

Each: what it changes mechanically · why it would or would not fix the measured failure · cost to
implement and test · what it invalidates · pre-registered success / failure / uninformative.

Costs use the project's own measured units: one tier-1 fit at 2400 steps ≈ **56–59 min**
(A9 `fit_seconds`), one `deep_starmap` fit ≈ **3.5–4.1 h**. Three seeds is the minimum for a claim
(`claim_min_seeds`).

### A. Constrain `theta` toward a per-gene prior from within-type dispersion, or floor it

**Mechanically.** Adds a penalty pulling `log theta_g` toward a per-gene estimate from the data's own
within-cell-type dispersion, or a hard floor `theta ≥ theta_min`. Shrinks the `mu²/theta` term.

**Why it might work.** That term is 57–61 % of the conditional variance, arm-independent across two
trainings with different embeddings — the strongest structural evidence in the project. Removing the
per-cell freedom removes the channel R4's trade runs through.

**Why it will not be enough.** §2: bounded at `f = 0.41`. And ⚠️ **the direction is not obviously
opposite to the failed attempt.** Moment-matching failed because the pooled estimator returned a
value **7.5× smaller** than learned — i.e. the data's own dispersion is *larger* than what the
decoder learned, so a prior built from within-type dispersion may pull `theta` **down**, deepening
the trade rather than removing it. A **floor** avoids this; a **prior** may not. Establish the sign
of the per-gene target before building either.

**Cost.** Implementation ~0.5 day (one loss term, one `Config` field, T01-style). Test: 3 seeds ×
2 arms on tier-1 ≈ **6 h**; `deep_starmap` ≈ **24 h**.

**Invalidates.** Nothing already measured — it is an addition experiment with the term at zero by
default, so every existing number stands as the off arm.

**Pre-registered outcomes**, on `deep_starmap` top-32 structured genes, 3 seeds, per-metric per-arm
envelopes:
* **SUCCESS** — retention rises from 25–34 % to **≥ 45 %**, signs agreeing 3/3, clearing the shared
  envelope, with `Var(mu)` **not decreased** (so the gain is not bought by flattening).
* **FAILURE** — retention change inside the envelope, or `f_overdispersion` unchanged (the term did
  not bite).
* **UNINFORMATIVE** — the fit fails T06's acceptance criteria (detection MAD, mean–variance slope),
  or `check_collapse` fires: the arm is not the model under test.

### B. A structured-variance term in the loss

**Mechanically.** Penalises the generated section's structured share falling below the tissue's,
computed on held-out training sections via T08's LOSO scheduler.

**Why it might work.** It is the only candidate that targets the **binding constraint** identified in
§2 — `Var(mu)` — rather than the noise. R4's root cause is *"nothing in the objective opposes the
trade"*; this is an objective term that opposes it.

**Why it might not.** 🚨 **The share is a ratio, and a ratio can be raised by shrinking its
denominator.** `s = V/(V+N)` improves if `N` falls *or* if `V` rises, and the cheaper gradient is
almost certainly `N`. **This is A9's failure repeating**: the metric-aware autocorrelation term
drove `variance_ratio` to 0.08–0.17 — *"it hits its target statistic by removing the structure the
statistic was meant to certify."* Penalising a **variance-normalised** quantity is the exact shape
that already failed once.

**Mitigation, and it is a design requirement rather than a nicety**: penalise `Var(mu)` against the
tissue's **directly**, not the share; and report `variance_ratio` and `spatial_ratio` beside it every
time, with both collapse alarms armed.

**Cost.** Implementation ~1–1.5 days (a LOSO-scheduled term, and the alarms must be armed for it).
Test as A. ⚠️ Add **1.63×** compute — measured on A9's metric-aware arm.

**Invalidates.** Nothing by default (ships at zero). If adopted at non-zero weight it becomes part of
the shipped configuration and **every absolute number is re-opened** — the exposure §4 of the
close-out describes, arriving for real this time.

**Pre-registered outcomes:**
* **SUCCESS** — retention ≥ 45 % **and** `Var(mu)` up ≥ 1.5× **and** `variance_ratio` ≥ 0.5. All
  three, because any two without the third is the A9 pathology.
* **FAILURE** — retention up while `variance_ratio` falls below 0.25, i.e. the share was bought by
  draining amplitude. **This outcome is a result and must be published as one**: it would be R4's
  sixth instance and the second time a metric-aware term gamed its own target.
* **UNINFORMATIVE** — envelope exceeds the effect, or the alarms fire on both arms.

### C. A different emission where dispersion cannot absorb mean error

**Mechanically.** Poisson with a learned per-gene overdispersion **floor** (or a quasi-Poisson with
capped dispersion): the `mu²/theta` channel is bounded above rather than free.

**Why it might work.** It removes the trade by construction rather than by penalty — no weight to
tune, no gaming surface.

**Why it will not be enough.** §2: this **is** the `f = 0.31` row. It is the best an independent draw
can do and it lands at 0.24–0.43 against a 0.62 target.

**Cost.** Implementation ~1 day (a decoder variant beside `ZINBDecoder`; note `ZIGammaDecoder` is
already implemented and *never validated* — `specs/10` §5.1 — so this adds a second unvalidated
branch unless the calibration path is extended too). Test as A. ⚠️ **Calibration is ZINB-only**
(`infer/calibrate.py:1086`), so a new emission needs its own calibration branch or the calibration
must be disabled and that stated.

**Invalidates.** Every calibration result; the mean–variance slope work; T06's acceptance criteria
would need re-deriving for the new law.

**Pre-registered outcomes:** as A, plus a **hard pre-condition**: if the fitted dispersion sits *at*
the floor for > 80 % of genes, the floor is doing the work and the experiment measures the floor's
value, not the emission's form.

### D. A two-stage draw that preserves structure by construction

**Mechanically.** Draw counts with **spatially correlated** noise — e.g. a copula / latent-Gaussian
layer sharing the GRF's correlation structure — rather than independently per cell.

**Why it works, and why that is a problem.** It is the **only** candidate that escapes §2's bound,
because the bound assumes conditional independence. 🚨 **And it escapes it partly by changing what
the metric measures.** The record's own derivation is explicit: *"Since the draw is spatially
uncorrelated by construction, `I(counts)/I(mu)` is the share of between-cell variance carried by the
structured mean."* Make the draw correlated and **retention stops being a measure of the structured
share** — it can be raised by correlating the noise, with no improvement in per-cell fidelity.

**So D can pass the test without fixing the model**, and any D experiment needs a control that
independent-draw candidates do not: a per-cell fidelity metric that correlated noise cannot inflate
(gene–gene covariance Frobenius against the independent-donor baseline; `duplicate_profile_rate`;
per-cell NLL on held-out counts).

**Cost.** Implementation **3–5 days** — the largest of the four, and it touches sampling,
calibration and the determinism guarantee (Convention 3: the correlated draw needs an explicit
generator and a bitwise test). Test as A plus the control metrics.

**Invalidates.** The retention statistic as currently defined, in every report that quotes it. That
is a documentation cost across `chain_2400*`, R12, R4 and the close-out.

**Pre-registered outcomes:**
* **SUCCESS** — retention ≥ 45 % **and** gene–gene Frobenius **improves** (or holds) against the
  independent-donor baseline **and** `duplicate_profile_rate` unchanged.
* **FAILURE** — retention rises while Frobenius worsens: the noise was correlated, the model was not
  improved. **Pre-register this as the expected failure mode**, because it is the one the design
  makes easy.
* **UNINFORMATIVE** — the sampler's determinism test fails, or the correlated draw changes the
  marginal mean (then it is not the same emission).

### E. ⬆️ Raise `Var(mu)` — the candidate the brief does not list, and the binding constraint

§2 says the noise-side candidates are bounded and `Var(mu)` must rise 2.2–5.1×. **Nothing in A–D
raises `Var(mu)` except B, and B raises it only if it is specified to target `Var(mu)` directly.**

**Mechanically.** Options, cheapest first: (i) check whether the mu head's capacity or the residual
gate is compressing dynamic range; (ii) train `mu` against the tissue's own per-gene variance as an
explicit target; (iii) revisit the flow — I(latent) is 0.8677 but its *variance* against the real
latent's has never been measured, and a variance-preserving flow objective is a different thing from
a structure-preserving one.

**Why it might not work.** `Var(mu)` may be low because the model genuinely cannot predict that much
of the variation — in which case raising it manufactures variance that is not conditioned on
anything, and per-cell fidelity falls while the share rises. **The same gaming surface as B.**

**Cost.** (i) is a **diagnostic, not a fit** — hours. (ii)–(iii) are 1–2 days plus the same test
matrix.

**Pre-registered outcome for the diagnostic (i)**: measure `Var(mu)` on generated cells against
`Var(mu_encoder)` on the real section, per gene, on the shipped configuration and on the same panel.
**SUCCESS = the ratio is established**, in either direction; there is no failure branch, because it
is a measurement.

### F. Change the deliverable rather than the model — emit the conditional mean

**Mechanically.** Emit `mu` (or a minimally-noised draw) instead of sampled counts.

**Why it is worth stating.** The model's `mu` is *already good* — I(mu) = 0.90 against real counts'
0.46. Everything the project wants is present before the draw. Many spatial methods report denoised
expression rather than counts.

**Why it is not a repair.** It changes the object being evaluated. `duplicate_profile_rate` exists
(`specs/10` §2) precisely to catch a method that is not emitting counts, and `specs/10` §5's
raw-counts convention is a project-wide rule. **This is a paper-framing decision, not an engineering
one**, and it should be taken explicitly or not at all — presenting `mu` as counts would be the
worst option available.

### G. Replace the likelihood with a proper scoring rule

**Mechanically.** Fit the emission by an energy score / CRPS on the count distribution instead of
the ZINB NLL.

**Why it addresses the root.** R4's statement is that *"a likelihood that can be reduced by moving
explanatory power out of the structured component into the unstructured one will be, and nothing in
the objective opposes the trade."* A strictly proper scoring rule that penalises the whole predictive
distribution does not reward that move in the same way. **This is the only candidate that attacks the
stated root cause rather than a symptom.**

**Why it is last.** Highest risk, least precedent in this codebase, and it re-opens T06 entirely.
**Cost: 1–2 weeks.** Not a first move; the right move if A–E establish that the trade survives every
constraint placed on it.

---

## 4. What I would do, in order

**Nothing on this list is the first thing to do.** The first two items are measurements that decide
whether the design work is aimed at a real failure, and both are cheap.

| # | action | kind | cost |
|---|---|---|---|
| **0** | **Re-run the chain diagnostic on the shipped configuration** — exp link, `expr_pca_dim=28`, GT-matched density, tier-1 **and** `deep_starmap`, same panel definition on both sides | measurement — settles §0 and §1's contradiction | **generation only, no fits** |
| **1** | **E(i): measure `Var(mu)` against the real latent's, per gene, same panel** | measurement — tests §2's binding constraint directly | **hours, no fits** |
| 2 | **A (floor form)** — cheapest *test of the diagnosis*: if bounding `theta` moves retention by roughly the amount §2 predicts, the variance account is right | cheap test | ~0.5 day + 6 h tier-1 |
| 3 | **B targeting `Var(mu)` directly**, alarms armed | candidate real fix | ~1.5 days + 6 h ×1.63 |
| 4 | D, with the fidelity controls | candidate real fix, largest | 3–5 days |
| 5 | G | root-cause attempt | 1–2 weeks |

**Cheap tests of the diagnosis: 0, 1, 2.** **Candidate real fixes: B, D, G.** **C is neither** — it
is a bounded variant of A with a larger blast radius, and §2 rules it out as sufficient before it is
built.

---

## 5. How each repair fails *even if the diagnosis is right*

| candidate | fails despite a correct diagnosis if… |
|---|---|
| A | the per-gene target pulls `theta` **down** (the moment-matched estimator was 7.5× smaller than learned), deepening the trade; or the floor binds on so few genes it does nothing |
| B | the term raises the share by shrinking `N` — A9's exact failure — or `Var(mu)` rises without conditioning, so fidelity falls while the statistic improves |
| C | the fitted dispersion sits at the floor for most genes, making the experiment a measurement of the floor's value |
| D | correlated noise inflates retention with no per-cell gain — pre-registered as the expected failure |
| E | `Var(mu)` is low because the model cannot predict more, not because it is compressed |
| G | the scoring rule's optimum still admits the trade, differently weighted |

## 5a. What would indicate the **diagnosis** needs revisiting rather than the repair

* 🚩 **Step 0 shows tier-1 retention at 60–100 % on the shipped configuration.** Then there is no
  emission failure on the headline dataset, and the `deep_starmap` number is a panel or density
  artifact. **This is already the most likely outcome on the evidence in §0.**
* **A moves `f_overdispersion` as predicted but retention does not follow.** Then the variance
  account is wrong and something outside the decomposition is destroying structure.
* **Retention depends on the gene embedding.** The record already flags this: if `medcpt` and
  `lookup` differ materially, *"'the emission loses the structure' is the wrong attribution."*
* **`Var(mu)` measures fine in step 1.** Then §2's binding constraint does not exist and the noise
  side is the whole story after all — which would make C sufficient and change the ordering.

---

## 6. My honest view, since you asked for it plainly

**On the 0.09→0.62 gap: it is not a measured gap, and I would not fund work against it.** Its two
halves are different datasets and different gene panels, and the project's own record identifies
that transfer as an error and applies the criticism to these very numbers. §0a.

**On the real gap** — `deep_starmap` top-32, retention 25–34 %, panel-matched, on the shipped link —
**my view is that no single candidate here closes it, and the arithmetic says why.** Every noise-side
repair is bounded at a structured share of ~0.43 and most of them well below; the target needs
`Var(mu)` to rise 2.2–5.1× as well. So the honest answer to *"is any of these likely to close it"* is:

* **A and C: no, and this is provable from data already held.** They are worth doing as *tests of the
  diagnosis*, not as fixes.
* **B: only if it is specified against `Var(mu)` rather than the share** — and even then it inherits
  A9's gaming surface, which is the closest precedent this project has and which failed.
* **D: yes, but partly by changing what the metric measures**, which makes it the least honest win
  available unless the fidelity controls are pre-registered and reported.
* **G: the only candidate aimed at the stated root cause**, and the one I would expect a reviewer to
  ask about — *"you diagnosed an objective that rewards the trade; did you change the objective?"*

**What would actually close it**, if the failure survives step 0: a combination, not a candidate —
**bound the dispersion (A/C) *and* raise `Var(mu)` (B/E)**, because §2 shows neither alone reaches
the target and together they roughly do. That is a two-term change to the emission and its objective,
which is a T06 redesign, and it should be costed as one.

**And the thing I would not do**: start any of it before step 0. The brief's premise is that
*"everything upstream is verified working and one operation destroys the result"*. On the shipped
link that operation takes 0.9008 → 0.5253, above the real section's 0.4635. **Weeks of design work
against a superseded arm is the expensive version of the mistake this project has spent five rounds
catching cheaply.**


---

## 7. The two corrections, recorded

Both premises came from the brief and both were wrong; recorded here because this project's rule is
that a withdrawn premise is a finding, not an embarrassment.

**7.1 `chain_2400.md` was handed over as the diagnosis without checking which arm it was.** It is the
`softplus` arm. The shipped `exp` link gives 0.9008 → 0.5253 at the same step, and the generated
counts are *more* autocorrelated than the real section's. §0.

**7.2 The 0.09-vs-0.62 comparison was repeated across panels** — `deep_starmap`'s 1017 mostly
unstructured genes against tier-1's 28 marker genes — **after this project had already voided a run
for exactly that error**, and after the record had extended the criticism to its own numbers. §0a.

⚠️ **The common shape, and it is worth naming because it is the third instance this week.** Neither
error was a mistake about a number; both were **an arm or a panel inherited with a number**. The
r11 six-metric table was mislabelled the same way (§4.2a-ii), and the 0.5 weights were carried the
same way (§4.2a-iv). `specs/10` §4.2a-ii's rule — *read every fit-time gate out of the artifact, not
just the gate under test* — has a companion this adds: **a number carries its panel and its arm, and
quoting it without them is quoting a different number.**

---

## 8. Step 0 and step 1 — built, and the two commands to run

**Status: built.** `scripts/t10_chain_diagnostic.py` carries the four flags and step 1's block.
Nothing else from §10 is built; gates 1 and 2 come first and either can stop the work.

The reason it needed building: `main()` **hardcoded** `text_emb_mode="lookup"` and
`expr_pca_dim=16` with no CLI override, so every chain artifact in `reports/` is a `lookup` /
`pca16` arm — the same ablation-A3 mislabelling found in the six-metric table, in a second
instrument. Issuing a step-0 command against that script would have been §4.2j a third time.

### 8.1 What was built

| flag | what it does | default |
|---|---|---|
| `--text-emb-mode {medcpt,lookup}` | fits with a **live** text channel at that mode, through `build_entity_embeddings` and the panel's gene-metadata table. Raises if the table or the encoder is unreachable — no fall back to zeros | omitted = zero vectors, as before |
| `--expr-pca-dim N` | overrides `Config.expr_pca_dim` | 16, the pilot's |
| `--match-density` (`--density-seed`) | subsamples the generated cells to the real section's count **before** the neighbour query, the conditioning and the metric, so the kNN graph — the thing density changes — is equalised | off; seed = the run seed |
| `--top-k-by {all,real,model}` (`--top-k`) | restricts the gene-space stages to the top-k genes by Moran's I on the **real** section, or on the model's own counts | `all` |

Five things came with them, none optional in hindsight:

1. **The text channel has three states, not two.** The old default is *neither* A3 arm: A3's
   `lookup` arm carries the MedCPT vectors and zeroes the *projection*, while this carries no
   vectors at all — and `distillation_loss` reads `text_vecs` directly, so the two differ. The
   docstring and the report now say "zero vectors (neither A3 arm)" rather than "the lookup arm".
2. **`specs/10` §0's clamp is applied** (`clamp_config_to_input`), so `expr_pca_dim` is narrowed to
   the panel width by the rule rather than by hand. A no-op at the default 16.
3. **An unknown `--section` now raises before the fit**, listing the ids the file has. It used to
   produce an all-False mask and a NaN reference — several hours downstream of the fit that paid
   for it. `--section section_2` is tier-1's and does not transfer to `deep_starmap` unchecked.
4. **The ground truth's gene order is checked against the training volume's.** Every stage indexes
   `model.embeddings.gene` by column number, so a disagreement is a silently wrong gene.
5. **`--target-z` is checked against the volume** — outside the z bbox (every GRF query clamped to
   a face) or a boundary plane (R3's one-sided evidence) is flagged in the report, not left to a
   `BBoxClampWarning` this script suppresses.

The panel restricts **gene-space stages only** — decoded `mu`, sampled counts, their calibrated
twins, `REF real counts`, and both variance decompositions. Stages 1 and 2 and `REF real latent h1`
are `Config.latent_dim` channels that are not genes; the report says so where it reports them.

`--top-k-by model` is the flattering selection and exists only to bound how far it flatters. It
prints a 🚩 in its own report saying it is not a number to quote.

### 8.2 The two commands

```bash
# tier-1 STARmap, shipped arm, GT-matched density, top-32 by the real section's own I
python scripts/t10_chain_diagnostic.py --steps 2400 \
    --dataset starmap_visual_cortex --decoder-mu-link exp --layout-sampler grid \
    --text-emb-mode medcpt --expr-pca-dim 32 --match-density \
    --top-k-by real --top-k 32 \
    --out reports/chain_shipped_tier1.md --save-model runs/chain/shipped_tier1.pt \
    --fit-checkpoint runs/chain/shipped_tier1.ckpt

# deep_starmap, the same arm and the same panel rule
python scripts/t10_chain_diagnostic.py --steps 2400 \
    --dataset deep_starmap --section <a held-out section id> --target-z <its z> \
    --decoder-mu-link exp --layout-sampler grid \
    --text-emb-mode medcpt --expr-pca-dim 32 --match-density \
    --top-k-by real --top-k 32 \
    --out reports/chain_shipped_deep.md --save-model runs/chain/shipped_deep.pt \
    --fit-checkpoint runs/chain/shipped_deep.ckpt
```

Run `python scripts/t10_chain_diagnostic.py --self-check` first: it asserts the panel and density
logic on synthetic fields in seconds, with no fit and no data. Those two flags decide *what is
compared* rather than what is measured, so a defect in either would move every number in the
report without failing.

Three notes on the commands:

* **`--expr-pca-dim 32` on both, not 28 on tier-1.** The clamp narrows it to the panel width by
  `specs/10` §0's rule and records the narrowing; writing 28 by hand is the hand-picked value that
  rule exists to replace. It lands on the same number if tier-1's panel is 28 genes wide, and on
  the right one if it is not.
* **`deep_starmap` needs its own `--section` and `--target-z`.** The defaults are tier-1's
  `section_2` at z = 30.0 µm. An unknown section id now raises before the fit and lists the
  available ones, so the cheapest way to learn them is to run the command and read the error.
* **`--fit-checkpoint` is safe on a first run** — it is written every
  `Config.checkpoint_every_n_steps` and read only if it exists and matches this config, seed and
  budget. On a 3.5–4 h fit it is the difference between an interruption costing minutes and
  costing the run.

**Cost**: one fit per dataset — **~1 h tier-1, ~3.5–4 h `deep_starmap`** — plus the live MedCPT
encode on a cold text cache (seconds to a couple of minutes; `t09_select_starmap --preflight` warms
it).

**The number step 0 exists to produce** is not the model's retention. It is **`I(model counts)` and
`I(real counts)` on the same top-32 panel, on both datasets** — because that pair, and only that
pair, decides §9.

### 8.3 Step 1 — `Var(mu)` against the real latent's, per gene

Folded into the same run rather than given a `--load-model` reader, as decided: step 1 costs
nothing beyond step 0.

`real_section_reference` now returns `h1 = encoder(real counts)` alongside its two summary rows,
and the same decoder, the same size head and the same panel are applied to it. The "Candidate 2"
block gained a **real latent** column beside the generated one, so the two sides differ in the
latent and in nothing else — that column is the matched tissue-side quantity the record has been
quoting as *"tissue's 1.213"* without a source.

The block also reports `share_shape_bounded` — `Var(shape) / (Var(shape) + Var(log s))` — beside
the unbounded share R12's 15.3 % / 61.4 % / 62.2 % are on. The unbounded one is **not bounded by
1** (a negative covariance makes the total smaller than `Var(shape)`; `t09_structured_share.py`
measured 1.21), and only the bounded one can carry a threshold.

### 8.3a 🚨 STATUS: gate 2 reads NOT EVALUATED, and §10 is SUSPENDED

Measured: **0.983** (tier-1), **1.917** (`deep_starmap`). Both are `>= 0.8`, which on the face of it
fails gate 2 and, under §10's stop rule, would stop half 2 of the redesign.

**It does not fail. It did not evaluate, and the fault is in this pre-registration.** Both sides of
the ratio pass through the same decoder, so a narrow `mu` head pins both columns and the ratio is ~1
by construction; the gate cannot return the informative answer. The proof it was pinned rather than
measuring: generated `sd(log mu)` is **0.7269** on 28 genes at ~100 % detection and **0.7254** on
1017 genes at 1.6 % detection — two datasets with nothing in common, agreeing to three decimals.
Full account in `reports/chain_shipped_review.md` §3; recorded as a second instance of `specs/10`
§4.2's closing rule.

**What NOT EVALUATED does to the stop rule, stated so it cannot be read the other way:**

> **§10 is SUSPENDED, not re-opened.** A gate that could not be read is **not** a gate that passed.
> Neither half of the redesign is built, costed further, or resumed on the strength of this
> reclassification. The suspension lifts only when a *replacement* gate — one whose reference is not
> the model's own decoder — is pre-registered and run, and returns a readable answer.

`reports/a1_preregistration.md` is that replacement's first half: A1's model-free arm (A1c) puts the
tissue's own mean field through a bare Poisson draw, so its reference is counting statistics rather
than this decoder. A1's decision table (§4 there) is what resumes or ends §10, and three of its four
rows end it.

Until then: **gate 1 has not been run, gate 2 did not evaluate, and no part of §10 proceeds.**

**Pre-registered reading for step 1**, as written before the run, and now to be read only alongside
§8.3a:
`Var(log mu_generated) / Var(log mu_encoder_real)`, per gene, median over the top-32 panel.
**≥ 0.8** means the structured component is intact and §2's binding constraint does not exist;
**≤ 0.4** confirms it; between is uninformative and needs the three-seed version.

### 8.4 What could not be verified here

This container has **no torch**, so the script could not be executed end to end. What was verified:

* `ruff check` and `ruff format --check` clean;
* the module imports and all three command lines parse (they stop at path resolution, since the
  built datasets are not in this checkout);
* `--self-check`'s **20 assertions pass** on the pure numpy/scipy logic — panel selection
  (size, ordering, determinism, tie-breaking by column index, constant columns sorting last),
  `rank_normalize` commuting with column selection (so the panel does not itself move a stage's
  `I`), density subsampling (size, uniqueness, seed-determinism, no upsampling), and the variance
  summary (medians over genes, the median ratio rather than the ratio of medians, the unbounded
  share exceeding 1 under negative covariance while the bounded one does not).

Not verified: anything touching torch — the live MedCPT channel, the clamp against a real header,
the encoder path, and the decoder decompositions. Run `--self-check` on the campaign machine before
the fits; it will also exercise the import of the torch-dependent module.

---

## 9. Is a 60 % / 25–34 % split a defect or a dataset property?

**On that evidence: a dataset property to characterise and report, not a defect worth a redesign.**
Four reasons, and one condition that reverses it.

**1. On the headline dataset there is no deficit to repair.** Tier-1 at the shipped link emits counts
at **1.13× the real section's** autocorrelation (0.5253 against 0.4635). §0a establishes tier-1 as
*the informative reconstruction benchmark* — 3.3× its own envelope of genuine headroom. The dataset
the paper is about does not exhibit the failure.

**2. The deficit is on a dataset the spec forbids reconstruction claims on.** §0a, measured
model-free: *"`deep_starmap` cannot carry a reconstruction claim against an optimal copier"* —
copying already reaches **98 %** of the achievable ceiling there. Redesigning the emission on the
strength of a reconstruction deficit measured on that dataset would be justifying a change by a
number the spec says may not carry that kind of claim.

**3. There is a mechanism, and it predicts the split.** `deep_starmap`'s median gene is detected in
**1.6 %** of cells against `cosmx`'s 9.3 % (the density note). A gene present in 1.6 % of cells has
counts that are almost all zero, so the Poisson term dominates and the structured share is bounded
low **for any method — and for the real tissue too**. Tier-1's 28-gene curated panel has a median
detection rate of **0.9999** (§0's data-contract note). **Sparsity, not the emission, is the obvious
difference between the two datasets**, and it is the one nobody has tested.

**4. The cost asymmetry is severe.** Adopting a new emission re-opens **every absolute number in the
project** and re-derives T06's acceptance criteria — for a deficit on a Tier-2 dataset, while the
Tier-1 headline is already at parity with real tissue.

🚩 **The condition that reverses this.** If step 0 shows, on the **same top-32 real-selected panel at
GT-matched density**, that *real tissue's own* `I(counts)` on `deep_starmap` is **high** (say ≥ 0.28,
as the record's top-32 figure suggests) while the model's stays at 0.07–0.10, then sparsity does
**not** explain it and the deficit is the emission's. **That is a defect**, and §10's combination
becomes the right response. The whole judgement turns on one pair of numbers that step 0 produces.

**What I would publish either way**, because it is a result in both branches:

> The emission's retention of spatial structure is **panel-dependent**: at parity with real tissue on
> a 28-gene curated panel with ~100 % detection, and 3–4× below it on a 1017-gene panel whose median
> gene is detected in 1.6 % of cells. Whether that is the sparsity bound or the emission is settled by
> the real section's own retention on the same panel.

That is a **characterisation**, it costs two runs, and it is more useful to a reader than a repair
would be — because it tells them when the method works, which is what a method paper owes.

---

## 10. The combination, costed as one T06 redesign


> # 🛑 CLOSED 2026-09-10 — on a measurement, not on a gate
>
> **The emission programme is aimed at a quantity the benchmark does not score, and taken to its
> limit it makes the scored one worse.** `reports/q15_review.md`:
>
> * `paper_morans_pearson` is a **correlation across genes**; everything below is about a
>   **median**. The chain's reconstruction of that correlation tracks the published score — tier-1
>   `r(stage 4)` = **+0.5076** against **0.5574**, on the same 28-gene set.
> * Removing the emission's noise **entirely** takes tier-1 from **+0.5076 to +0.3878**
>   (`d r = −0.1198`) and triples the per-gene level error, `mae` **0.1018 → 0.3529**.
> * **No dataset returns MOVES IT + CEILING HIGH** under `q15_preregistration.md` §5. Tier-1
>   returns both negatives; `deep_starmap`'s governing all-genes row is PARTIAL on both.
> * The target is the **model-free copy floor at 0.9836**, not SpatialZ's 0.932. A complete
>   emission repair lands at 0.3878 and **widens** the gap by 0.17.
>
> Neither half is built. Gate 1 was never run; gate 2 never evaluated (§8.3a). This section closes
> without either, because a free measurement bounded the whole programme from above and the bound
> is below its own baseline on the dataset the paper leads with. **Everything below stands as the
> costing it was — a record of work that was correctly not done.**
>
> What replaces it: the scored statistic is carried by the **count-generating process**, not by the
> latent (`q15_review.md` §4 — the mean field alone scores *worst* on both datasets), so the open
> question is **which genes** the model structures, not **how much**. `reports/ladder_preregistration.md`
> is the free measurement that decides whether that is closeable at all.

Asked for, because §6's answer is that no single candidate closes it. **The two halves are not equal
partners, and that is the main finding of costing them together.**

### 10.1 What each half contributes

| | half 1 — bound the dispersion | half 2 — raise `Var(mu)` |
|---|---|---|
| mechanism | per-gene `theta` **floor** (candidate A's floor form, not the prior form) | an objective term on `Var(mu)` against the tissue's, or a capacity/flow change (candidate E) |
| what it buys | noise factor `f` **1.0 → 0.41** | signal factor **2.9×–6.8×** (after half 1) |
| share reached alone | s 0.09–0.19 → **0.19–0.36** | not meaningful alone — raising `Var(mu)` without bounding the noise is partly re-absorbed |
| certainty | **high** — the term is a measured 57–61 % of conditional variance, arm-independent | **low** — `Var(mu)` may be small because the model cannot predict more (§5) |
| cost | ~0.5 day | 1–2 days, or open-ended if it is a flow/capacity problem |

**So the combination is one cheap, near-certain half and one expensive, uncertain half — and the
uncertain half does most of the work.** A 2.9–6.8× increase in the structured component is not a
tuning change; it is a claim that the model can predict three to seven times more of the between-cell
variation than it currently does. **Nothing in the record says it can.**

### 10.2 The validation design and its cost

A **2 × 2** — dispersion floor {off, on} × `Var(mu)` term {off, on} — because that is the only design
that attributes the result to a half rather than to the pair.

| | tier-1 | `deep_starmap` |
|---|---|---|
| 12 fits (4 arms × 3 seeds) | **11.5 core-hours** | **45.6 core-hours** |
| ×1.63 if half 2 is a loss term (A9's measured penalty) | **18.7 core-hours** | 74 core-hours |

Implementation **3–4 days**. Total to a claim-bearing answer on tier-1: **~4 days and ~19
core-hours**; on both datasets, ~5 days and ~65–90 core-hours.

### 10.3 What would show the combination is working, before the full thing is built

Three gates, in order, each cheap and each able to stop the work:

1. **After half 1 alone, `f_overdispersion` must fall from 0.57–0.61 to near zero and `s` must land
   in 0.19–0.36.** This is arithmetic, not hope — §2 predicts the interval. **If `s` lands outside
   it, the variance decomposition is wrong and the whole design is mis-aimed.** One tier-1 fit,
   ~1 hour. **This is the single most informative hour in the plan.**
2. **Step 1's `Var(mu)` ratio must be ≤ 0.4.** If it is ≥ 0.8, half 2 has nothing to fix and the
   combination collapses to half 1, which §2 says cannot reach the target — meaning the target is
   wrong, not the model.
3. **After half 2, `Var(mu)` must rise *and* per-cell fidelity must hold** — gene–gene Frobenius
   against the independent-donor baseline, and `variance_ratio` ≥ 0.5. **If `Var(mu)` rises while
   Frobenius worsens, the term is manufacturing unconditioned variance**, which is A9's failure in a
   new place and must be published as such.

**Stop rule, pre-registered**: if gate 1 fails, do not build half 2. If gate 2 returns ≥ 0.8, do not
build half 2. **Either outcome ends the redesign for the price of one fit and one diagnostic**, which
is the point of ordering them this way.

### 10.4 What adopting it costs beyond the fits

Every absolute number in the project is re-opened — the emission is upstream of all of them. T06's
acceptance criteria (detection MAD, mean–variance slope, gene–gene Frobenius) need re-deriving for
the new emission. `infer/calibrate.py` is ZINB-only and would need a branch or an explicit
disabling. And the shipped configuration changes, so §5.1's three-seed table is measured again.
**That is the real cost of the combination, and it is larger than the fits.**
