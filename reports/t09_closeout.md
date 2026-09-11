# T09 close-out — SpatialCPA-v25-Gen

**2026-09-07.** Written after the `cosmx` replication, A7, and the pool-sparsity test. Nothing is
pending measurement. This is the consolidated statement; the working record is
`progress/t09_inference_and_calibration.md`, which carries every number's derivation and every
correction made to it.

---

## 1. The headline I would defend

> A continuous 3D field is a good **representation** of a tissue volume and a bad **generator**
> from it.
>
> Oblique planes reconstruct at **95.5 %** of axis-aligned quality; per-gene spatial
> autocorrelation is controllable through a 3D GRF prior; and two crossing sections emit **bitwise
> identical** expression along their intersection, exactly and without training. But every
> generative component built on that representation loses to **copying a real section** — the
> intensity-field layout, the flow-matching expression head, and the sectioning-equivariant losses
> the method is named for, which make it actively worse. One capability claim survives replication
> on a second dataset: text embeddings place genes the model never saw above a measured floor. The
> generative failure is localised and mechanical rather than diffuse: the decoder reproduces the
> *pattern* of between-cell variation almost perfectly and a fraction of its *amplitude*, and the
> ZINB objective closes the gap with dispersion instead of with the structured mean.

The last sentence is the one I would defend hardest. It is the difference between "our expression
head underperformed" and a finding.

---

## 2. What works

Three claims, all about **encoding**. None is about generation.

**2.1 The correlated prior controls spatial autocorrelation (GATE 1).** Error ratio **0.130**
against an i.i.d. prior, per-gene `I_gen` vs `I_real` correlating at **r = 0.917**, and median
`I_gen` monotone as `ell` sweeps 0.25x–4x. On the synthetic fixture — stated as such.

**2.2 Oblique reconstruction reaches parity (GATE 2).** Depth-matched parity **0.955**,
edge-excluded **0.979**, against a pre-registered **≥ 0.90**.

⚠️ **CORRECTED 2026-09-11.** This read *"a clean, unqualified pass on real data, and the strongest
single result in the project"*. **GATE 2 ran on the synthetic fixture** —
`make_synthetic_volume(seed=0, extent_xy=3000)`, "the same fixture GATE 1 was measured on"
(`reports/gate2.md`) — and its probe is a **linear head on 32 expression PCs**, not the generation
pipeline, because "the full generative heads do not exist yet". `gate2.md` says so and says T10's
**E3 is where it meets real data**; E3 has never run. See `reports/framing_honesty_review.md` §4.

**2.3 Intersection consistency is exact by construction.** Two crossing sections emit **bitwise
identical** expression along their intersection on an **untrained** model, with no consistency loss
applied (`test_generation_is_intersection_consistent_by_construction`). Not approximately, not at a
particular checkpoint: the 3D noise field is continuous and every conditioning pathway — retrieval,
the GRF, the Fourier encoding — is queried at **physical** points while `CTFFlow.generate`
conditions with the identity pose whatever plane it is handed. The property therefore holds for
every model of this architecture, at every checkpoint, exactly.

Three things make it a result rather than an implementation note:

* **It is a stronger claim than the loss written to earn it.** `L_cross` exists in `specs/07` to
  *train* approximate agreement. The architecture already has exact agreement.
* **It survived that loss being shown unnecessary *and* harmful.** `L_cross` is vacuous in v25, and
  the only plane-dependent channel left for it to act on is the augmentation pose — which T04 made
  pose-dependent on purpose — so minimising it destroys the field: per-gene variance falls to
  **0.065** of real at `w_cross = 0.3`, against **0.711** with SEFL off. `w_cross` ships at **0**,
  and the property was unaffected, because it never depended on it.
* **No competing method has it.** Independent per-plane generation gives crossing sections no
  reason to agree anywhere. It is a property of committing to a continuous 3D field — the thesis
  the generative half failed to support.

**2.4 Text beats a gene-blind arm on unseen genes — replicated. *Which* text path does it — not.**
The qualification belongs in the header because this line will be quoted alone, and §3 records the
other half of the same experiment as PARTIAL and DOES NOT REPLICATE.

What replicated, under criteria fixed before the fits, on a second dataset: A2 (`medcpt`, pure
text) clears the `shuffled` floor at **2.52x** on `deep_starmap` and **2.08x** on `cosmx`, and
A2 − A4 — text against an arm that sees no gene identity at all — is **+0.3949, 1.96x** the shared
envelope with signs agreeing on 3/3 seeds and 6/6 folds. The void condition holds at 0.12x. **Two
independent positives on two datasets.**

What did **not**: A2 − A3, the pure-text projection against the distillation head, which is the
contrast that would say *how* the text channel reaches an unseen gene. See §3.

---

## 3. What does not work

**Every generative component loses to copying — the same trivial baseline each time.**

| component | result | measurement |
|---|---|---|
| intensity-field layout | **REFUTED** | `field` 0.6607, `hybrid` 0.6692 against `resample` **0.7546**; the model-free copy floor is 0.7765 and the oracle ceiling 0.9808. Both field modes score **below the floor** on the metric the layout head exists to win, by **0.1158 and 0.1073 raw**. 🚩 **The "3.5x and 3.2x the envelope" this row carried is withdrawn and not replaced**: it divided by the pooled synthetic-fixture 0.0335, and no envelope exists for `paper_celltype_localization` on the `field`/`hybrid` arms — all five r11 arms are **one seed** (`reports/envelope_correction.md` §3). The raw deficits are what the verdict rests on and they are large; the multiple is not available. ✅ **The comparison itself got firmer on 2026-09-09**: the checkpoint recovery shows all five r11 arms are provably **one fit** differing only in generation-time gates, and `layout_mode` is in `FIT_INVARIANT_GATES` — bitwise identical weights across all 96 tensors when only it moves (`test_layout_mode_does_not_enter_the_fit`). So the ordering is a clean within-fit contrast; what it lacks is an across-seed spread, which one checkpoint can never supply. `resample` ships. |
| flow-matching expression head | **REFUTED, both datasets** | on `deep_starmap` `cross-mix` (copying) wins **every live metric**; on tier-1 by **2.2x, 2.3x and 7.4x** their own per-metric per-arm envelopes on three (`gearys`, `morans`, `umap_mixing`; margins +0.1303 / +0.1313 / +0.1415 against 0.0595 / 0.0574 / 0.0190, three seeds). ⚠️ **Corrected 2026-09-08** from "4.6–5.3x the envelope", which was one seed on the pre-frame-fix code state, divided by the pooled fixture 0.0335 — the two autocorrelation metrics carry less than half the weight quoted and `umap_mixing` carries more (`reports/envelope_correction.md` §2.1). The one metric where generation had won was an artifact of the frame defect and reversed when it was fixed. |
| SEFL — the mechanism the method is named for | **REFUTED, 3 seeds** | the SEFL arm **collapses** the anatomical field: `i_gen` at **1.6–2.4 %** of target against the off arm's 95.6–97.5 %; five of seven metrics cost with signs agreeing 3/3 at 1.31x–4.68x. `check_collapse` fired **218 times** across the three ON fits, from step 250 of 1200. All three weights ship at **0**. |
| the mechanism half of the zero-shot claim | **PARTIAL** | A2 − A3 — the pure-text projection against the distillation head — is **+0.0450, 0.22x** the shared envelope, where `deep_starmap` had +0.2514 at 2.7x. `specs/10` §7's mechanism sentence is **withdrawn**. |
| the seen/unseen sign flip | **DOES NOT REPLICATE** | signs reversed on **12 of 12** seed x fold cells across both gene pools, and **both** magnitudes sit inside their pool's envelope (0.22x, 0.51x). The direction replicated; the effect size did not. |

🚨 **And a second sentence, found 2026-09-09: none of the absolute numbers above were measured on
the shipped configuration, and now some of them are.** Every draft of the six-metric table is
`text_emb_mode=lookup` at `expr_pca_dim=16` — **ablation A3**, because the pilot ran in the
container where the MedCPT encoder is unreachable (`specs/10` §3 names that arm by name). The
shipped-configuration measurement existed all along in **A9's fits** (medcpt, `expr_pca_dim=28`,
`resample`, 2400 steps, three seeds, pinned evaluator), filed under A9's verdict. Re-read at zero
fits, it moves the negatives **further from the floor, not closer**: `marker_field_r` 0.203 below →
**0.320**, the two autocorrelation metrics ~0.33 below → **~0.43**, and `gene_mean_spearman` — this
project's "one genuinely solved thing" — no longer clears its copy floor (+0.0038 above → −0.0142
below).

✅ **And these are the first deficits in the project readable against an admissible envelope.** A9's
`w=0` arm *is* the shipped configuration, so its own three-seed spreads decide (§4.2b: the referent
is a fixed probe, envelope zero). `morans` **21.1x**, `gearys` **17.8x**, `marker_field_r`
**10.4x**, `marker_depth_r` **2.1x** — and the retired pooled figure was mis-scaling in *both*
directions, reading the autocorrelation pair at 12.7x/12.8x and `marker_depth_r` at 7.7x. ⚠️ Two
cells do not survive the reading: `gene_mean_spearman`'s deficit is **0.4x its own envelope — a
tie**, so it is not established as worse either, and `celltype_localization` is 🚩 **not readable**
(its `w=0` spread is 0.0009 with bitwise-identical cell counts across seeds, because `resample`
copies the layout — §4.2g's degeneracy, arriving on the arm under test rather than on a referent).
Table and derivation: `reports/advisor_report.md` §5.0–5.2, §5.1a.

**The sentence that follows, and which had not been written down.** v25 ships `resample` +
`zinb-flow`: real positions with generated expression. v20's fallback is `resample` + `cross-mix`:
the same real positions with the donor's counts copied. The difference between them is exactly
v25's contribution to that pairing, and **it is negative on every live metric.** A reader will work
this out; the paper should say it first.

---

## 4. The one component recorded as shipping ON while established by nothing — and no code path produces it

Flagged here rather than left in a table, because it is a different kind of problem from everything
in §3.

⚠️ **Heading corrected 2026-09-09.** It read *"the one component that ships ON"*. The checkpoint
recovery showed that is not what the code does: **`Config` declares all three weights at `0.0`**,
and every real-data fit in the corpus ran at 0.0 except A9's `05` arm. The problem is real and it is
a different one — **a value the selection chose, recorded everywhere as shipped, that nothing
implements**. The correction is stated in full at the end of this section.

`w_autocorr = w_profile = w_distribution = **0.5**` were **selected**. At 1200 steps they lose (rank 3.5
against 3.0, a cost on every metric); at **2400** — the budget T09 selected — they win the
selection on aggregate rank, 1.0 against 2.0, taking four of six metrics. That is why the record
says they ship — though see the correction below: no code path sets them.
The per-metric margins at 2400 are **0.0052 / 0.0101 / 0.0018**, and against the fixture's **own
per-metric** envelopes — `morans_pearson` 0.0160, `gearys_pearson` 0.0335 — every one is still
inside, **by factors of 1.6 to 19**, on **one seed**. ⚠️ **Corrected 2026-09-08** from "against
R10's 0.0335 envelope … by factors of 3 to 19": R10's pooled figure is `gearys_pearson`'s envelope
worn by all six metrics and is up to 5x too strict on the others. The verdict is unchanged — the
weights are still established by nothing — but the safety factor is half what was stated. ⚠️ The
record says these three margins sit *"on the autocorrelation metrics"* and there are only two of
those, so **the third metric is unidentified**, and the table that produced them is §8b
(unrecoverable) so the assignment cannot be checked. Had one of the three been
`celltype_localization` (fixture envelope 0.0068) the 0.0101 margin would **clear** at 1.49x.
`reports/envelope_correction.md` §2.3.

🚨 **And the selection ran on the SYNTHETIC FIXTURE** — 19 fits at ~8 minutes each, against the 56
minutes a real tier-1 fit takes. So the accurate statement is stronger than "one seed with small
margins": these weights ship on **an aggregate rank over the fixture, and have never been measured
on real data at all.** That is the pattern R11 already burned, where a fixture tie-break put
`hybrid` ahead and real data reversed it — its flanking baseline sitting at 58 % of its ceiling
against real tissue's 79 %, so it over-rewards a generative addition at any number of seeds.
⚠️ **The parenthetical "inside the envelope" is withdrawn from that sentence.** Re-read per metric,
the fixture `layout_mode` gate was **not** decided inside the noise: `umap_mixing` separates at
**3.49x** its own envelope toward `resample` and `celltype_localization` at **2.84x** toward
`hybrid`, both 3/3 on sign. The fixture was not underpowered on this gate — it gave two answers that
disagreed, and the pooled reading hid that (`reports/envelope_correction.md` §2.4).

The selection is sound as a selection. It is **not evidence that the losses do what they are named
for**, and `claim_min_seeds = 3` says one seed cannot resolve it either way.

| state | components |
|---|---|
| ships **off**, evidence against | SEFL's three weights, `w_cross`, the intensity-field layout |
| ships **on**, evidence for | `prior_mode="correlated"`, `layout_mode="resample"`, `decoder_mu_link="exp"` |
| 🚨 **recorded as shipping on at 0.5, established by nothing — and no code path produces it** | the three metric-aware weights. `Config` declares them **0.0** and every real-data fit in the corpus ran at 0.0 except A9's `05` arm (corrected 2026-09-09, §4) |

A component that ships **off** with evidence against it is an honest negative: the reader learns
something and the shipped model does not depend on it. A component that ships **on** while
established by nothing is inside the baseline **every number in this project was measured
against** — it is not a claim the paper makes, it is a claim the paper's *setup* makes silently.

**A9 has now run — six fits, three seeds — and returned UNINFORMATIVE.** Pre-registered condition
(a) fired: the worst primary envelope is **0.4323** against the 0.067 the condition names, **6.5x
over**, and both autocorrelation primaries had signs disagreeing across seeds. **The weights remain
unestablished, and a three-seed real-data design could not resolve them** — so "run more seeds" is
not a cheap path either. Two things the run found instead:

* 🚨 **The collapse alarm is disarmed whenever SEFL is off** — `train_ctfflow` gates it on
  `sefl_teacher is not None`, and the teacher exists only when a SEFL weight is above zero. The
  shipped configuration has all four at zero, so **on the shipped model `check_collapse` never
  runs**, and every empty alarm list in every SEFL-off campaign means *never armed* rather than
  *did not fire*. An owed fix, and §4.2f failing in the instrument written to enforce it.
* 🚨 **Applied by hand, that alarm would have fired on all three ON fits.** Median `variance_ratio`
  is **0.86 / 0.77 / 0.81** off and **0.14 / 0.08 / 0.17** on, against a 0.25 collapse threshold —
  and A7's *collapsed* arm settled at 0.105–0.193 on the same statistic. Meanwhile `spatial_ratio`
  goes **up** (1.4–1.6 → 2.2–2.5), so the autocorrelation term hits its target while the amplitude
  drains: R4's shape in a fifth place. **Candidate, not established** — it decides no A9 branch.
* Wall clock: the ON arm costs **1.63x** (93 min against 57).

### 🚩 Standing recommendation — **WITHDRAWN 2026-09-09 as vacuous** (see the resolution at the end of this section): set the three metric-aware weights to **zero**

Not a note. Whoever runs the next campaign should decide this with the evidence in front of them
rather than inheriting a coin-flip rank, so the recommendation is stated and the reasons are
listed:

1. **They were selected on the synthetic fixture**, by an aggregate rank over six metrics, with
   per-metric margins of 0.0052 / 0.0101 / 0.0018 inside their **own** fixture envelopes (0.0160
   and 0.0335 on the autocorrelation metrics, so inside by 1.6x to 19x — ⚠️ corrected 2026-09-08
   from "inside a 0.0335 envelope", §4), on **one seed** — and the fixture is documented to
   over-reward exactly this kind of addition (R11: its flanking baseline sits at 58 % of its
   ceiling against real tissue's 79 %, and real data reversed its verdict). ⚠️ The gloss that the
   fixture was *"underpowered, not wrong"* is withdrawn: read per metric it separated the
   `layout_mode` gate on two metrics that **disagreed** with each other
   (`reports/envelope_correction.md` §2.4). Over-rewarding a generative addition is the surviving
   objection; being too blunt to see a difference is not.
2. **The one real-data test could not resolve them.** A9, three seeds, six fits: UNINFORMATIVE,
   with the worst primary envelope **6.5x** the condition's bound and both autocorrelation
   primaries' signs disagreeing across seeds. "More seeds" is not a cheap path — this design was
   already too noisy at three.
3. **They cost 1.63x the compute** — 93 minutes a fit against 57.
4. **On the diagnostic the shipped model does not watch, they sit in the collapse regime.** Median
   `variance_ratio` **0.14 / 0.08 / 0.17** on against **0.86 / 0.77 / 0.81** off, threshold 0.25,
   where A7's *collapsed* arm settled at 0.105–0.193. Candidate, not established (§5a) — but it is
   evidence pointing one way and there is none pointing the other.

**Nothing here was turned off post-hoc, and that is deliberate.** Disabling them now would re-open
every fitted number in the project for a change that is, by the only real-data test available,
within noise — a threshold moved after seeing where the data fell. The recommendation is for the
**next** campaign, which will refit anyway and can adopt it at zero cost.

**And what the paper must say either way**: the weights were **selected** at 0.5 on a
**fixture-selected** aggregate rank with per-metric margins inside their own envelopes, and a
three-seed real-data test could not resolve their contribution.

🚨 **CORRECTED 2026-09-09 — "every absolute number in this project was produced with them active"
is backwards.** The checkpoint recovery read the gates out of the fits: **`Config` declares all
three at `0.0`**, `base_config` never overrides them, and `t09_ship_starmap.py --w-metric-aware` is
an override flag that defaults to unset. The six-metric table's checkpoint, A7's two arms and A9's
own control arm are all at **0 / 0 / 0**; **A9's `05` arm is the only run in the corpus that had
them on, and it is the experiment that tested them.** So the absolute numbers are *cleaner* than
this section claimed, not dirtier — and the real defect is the other one: **they were not produced
with the configuration the record calls shipped.** The standing recommendation above is now nearly
a no-op, because `Config` already does what it asks; what the next campaign needs is for the record
to stop saying the weights are on. 🚨 **RESOLVED 2026-09-09, and it is neither branch that was pre-registered.** A persisted selection
**does** exist — `runs/select/starmap_visual_cortex/selected.yaml`, added at `3d57725` and deleted
at `5cd1fd6`, still in history — and it carries the three weights at **0.0**, with `train_steps: 20`,
`decoder_mu_link: softplus` and `expr_pca_dim: 32`: a 20-step smoke artifact from the halted pilot,
predating the link fix, the clamp rule and R11. Not a selection anyone would ship, and not 0.5.

**So the record reads: `0.5` appears in NO machine-readable artifact this project produced** — not
`Config`, not any fit's recorded config, not the one persisted selection that exists. It lives only
in prose, in documents citing each other. 🚨 **That is the seventh provenance failure mode and it is
stronger than "never persisted": a value carried through the entire written record, cited as
shipped, reasoned from in two standing arguments, with no artifact anywhere that ever held it.**
Nothing is missing, because nothing was ever written — and no provenance check that compares
artifacts to each other can catch it, because they all agree, at 0.0.

**Two consequences, both on arguments that assumed the 0.5 was real** (`reports/advisor_report.md`
§6c):

* **This standing recommendation is withdrawn as vacuous** — it recommends changing a state that
  does not exist. Its first reason goes too: *"they were selected on the fixture"* rests on the
  §8b-unrecoverable selection table, and the one recoverable selection carries 0.0, so **even the
  claim that a selection chose 0.5 is prose-only**. What replaces it is documentary: the record must
  stop asserting 0.5. And **A9 was framed backwards** — with 0.5 unsourced it is an **addition**
  experiment, its `0` arm the shipped configuration and its `05` arm the addition, the same
  inversion A2, A4 and A7 each went through. The UNINFORMATIVE verdict is untouched.
* **R10's envelope was measured on an arm with no other instance in the project.**
  `scripts/t09_envelope.py` sets all three weights to **0.5** for its nine fits — so the 0.0335 and
  its per-metric decomposition describe a configuration that appears nowhere else. Not "the wrong
  arm" but **an arm with a single instance, created by a script on the strength of a value with no
  source**. ✅ Unaffected, and now the useful figure: **A9's `0` arm is the shipped configuration**,
  so its per-metric spreads are **the only envelope this project has ever measured on what it
  actually ships**, on the pinned instrument at the right design at three seeds.

⚠️ The instrument that answered this reported **NONE FOUND** first: it walked filesystem roots while
the file was missing in *time*, not space, inside the very repository it ran in. §4.2j again, in a
script that quoted §8c's reflog lesson in its own output. Fixed with a git-history scan over
`--all --reflog`, with a regression test on the false negative itself.
⚠️ And the fixture envelope ran the other way round — `scripts/t09_envelope.py` sets all three to
**0.5** for its nine fits, so R10's 0.0335 was measured on the weights-**on** arm while every
number it judged was weights-**off**.

**It does not weaken the negatives.** Each of R11, A7 and the zero-shot arms is a
*within-configuration* contrast with the same weights on both sides. The exposure is to the
**absolute** numbers and to the claim that the shipped model is the best configuration found — not
to any of the differences.

---

## 5. The mechanism under the generative failure

**R12, localised.** The chain measures median Moran's I of 0.9714 at the GRF prior, 0.9015 after
the flow, 0.8607 at the decoded `mu` — then **0.1297** at the sampled counts. Stages 1–3 lose 0.11
in total; the count draw loses **0.73 in one operation**. Real tissue retains **62 %** across the
same latent→counts step; the model retains **14 %**.

**5a. A fifth instance, and it is a candidate rather than a result.** A9's metric-aware arm drove
median `variance_ratio` to **0.140 / 0.083 / 0.171** against the off arm's **0.856 / 0.771 /
0.813** — below the 0.25 collapse threshold on all three seeds, in the regime A7's *collapsed* arm
occupied (0.105–0.193). Over the same fits `spatial_ratio` moved the **other way**: **1.64 / 1.39 /
1.39** off against **2.43 / 2.54 / 2.17** on.

Both at once is not a contradiction, and the reason is the mechanism. **Moran's I is
variance-normalised** — it measures how much of a field's variance is spatially structured, not how
much variance there is. So a field can score better on it while the amplitude it is measuring
drains away, and that is what the autocorrelation term appears to do: it hits its target statistic
by removing the structure the statistic was meant to certify. Same shape as R4 (i)–(iv), fifth
place.

⚠️ **Candidate, not established.** Two diagnostic trajectories, one budget, one dataset; it decides
no A9 branch and the A9 verdict stays UNINFORMATIVE. It is reported because a number that would
have raised an alarm had the alarm been armed must not reach a reader as silence.

**R4, the trade that explains it.** The decoder reproduces the *pattern* of between-cell variation
almost perfectly — `mu`'s Moran's I is **0.8607**, above the tissue's own latent at 0.7449 (`reports/chain_2400.md`) — ⚠️ a comparison **between two latents**, the 0.8607 measured on the encoder's latent for a real section and never on the flow's latent at generated positions, so it says structure is present where a latent is decoded, **not** that the conditional mean is healthy in the generative path — at a
fraction of its *amplitude*, and the ZINB objective closes the gap with dispersion. `theta` carries
**61–63 %** of the conditional variance and correlates with the data's own dispersion at Spearman
**0.068** over 1017 genes. It is absorbing unpredicted mean variation, not estimating noise.

That is a property of the objective, not a tuning error: **no data-derived value of `theta` exists
to match to**, which is why the moment-matching experiment was stopped after one fit. The same
shape was measured four separate ways (R4 i–iv) — in each case a likelihood is reduced by moving
explanatory power out of the structured component into the unstructured one, with nothing in the
objective opposing it.

---

## 6. The methods contribution, and why I think it is the stronger half

Eleven rules (`specs/10` §4.2a–j, plus §4.2a-i and §4.2a-ii from the 2026-09-08 envelope
correction). They are worth a paper not individually but as a pattern:

> Every claim in this literature has the form **"the margin exceeds the noise"**, and that sentence
> hides six independent choices. In this project each one silently decided a verdict before anyone
> noticed it was a choice.

* **§4.2a** which arm's variance is the noise — a pooled envelope was too lenient on three metrics
  and too strict on two, and the worse arm alternates by metric. 🚨 **And this project went on
  dividing by the pooled figure for three weeks after writing the rule** — the per-metric table
  existed in `envelope_synthetic.md` from the day it was measured, with a line in that report saying
  it was the right thing to quote. **This is the strongest instance in the set, because the rule was
  already written and it still did not bite** (`reports/envelope_correction.md`, 2026-09-08).
* 🚨 **§4.2a-i** which **instrument** the envelope came from — **the largest of the set by blast
  radius, and stated here rather than as a footnote to §4.2a because it is a different error.**
  This repository holds two scorers over the same six metric names: `train/select.py::section_scores`
  on internal LOSO over the *interior training* sections, and `bench3.evaluate_paper` on
  `paper_2_4_6`. **Every three-seed envelope this project ever quoted is the first. Every headline
  number — the six-metric table, R11, A7, A9, the marker deficits, the boundary work — is the
  second.** On tier-1 their per-metric envelopes differ by **2.6x to 6.7x**, and on the two
  autocorrelation metrics the real instrument-B envelope is **~9x** the pooled fixture figure that
  was used, in the direction that flatters every clearance. `specs/10` §5 forbade the mixture in
  prose — *"a different quantity … must not be placed beside these"* — while every division in the
  project performed it. It changes no verdict in §3 (each is a within-configuration contrast, so a
  wrong divisor rescales without changing a sign) and it means **no "Nx the envelope" on a `paper_*`
  number was ever a statement about that number's noise.**
* 🚨 **§4.2a-ii** whether the artifact records the arm it describes — the committed, verified,
  bitwise-reproducible files behind the six-metric table carry no `config_hash`, no `text_emb_mode`
  and no metric-aware weights, so a correctly measured envelope could not be matched to them. ⚠️
  **The information was never lost, and that is the sharper version of the finding**: the model file
  those artifacts were scored from carries the whole `Config`, and `t10_rescore_saved.py` already
  reads and prints it. The reporting scripts simply never copied the block into their own output.
  A recovery costs one file read (`scripts/t09_recover_checkpoint_config.py`).
* ⚠️ **A retrieval failure, not a measurement one, and the kind that recurs.** A9's envelope existed
  for a month and nobody looked, because the run was **filed by its verdict rather than by what it
  measured** — UNINFORMATIVE about the metric-aware weights, and simultaneously three seeds of two
  arms of the shipped-shape configuration on the pinned instrument, which is precisely the envelope
  §4.2a-i says the project lacked. "We have no real-data envelope on that instrument" and "A9
  measured one" were both true, and only the first was written down. **The rule: an experiment's
  record states what it measured, not only what it concluded** — a null result's measurements stay
  valid after its verdict stops being interesting. §4.2f's shape one level up: not a diagnostic
  firing where nobody looks, but a measurement filed where nobody will think to look.
* **§4.2b** which envelope a clearance takes — two arms 0.004 apart landed on opposite sides
  because one varied less.
* **§4.2c** which referent is a floor at all — the pre-registered constant-field band had
  bitwise-identical input.
* **§4.2d** how the spread is aggregated — an effect read 1.12x under fold-averaged noise and 0.75x
  under per-fold; reported as standing, withdrawn.
* **§4.2g** which arms may contribute a spread — on the replication the envelope was set on **both**
  pools by a **degenerate** member, at 6–18x every informative arm's variance, so an effect
  consistent on 12 of 12 cells could not clear it.
* **§4.2h** which side of the threshold refutes — a two-sided "within 1.5x" band on a one-sided
  hypothesis returned *false* on the result that refuted it most strongly.

Plus two failures of a different kind — **§4.2e**, a metric that is a correlation on a dead field
(one seed inverted a headline's sign on two of seven metrics); **§4.2f**, an alarm that fired 218
times where no report read it — and one standing requirement that falls out of all of them:

* **§4.2i** — a three-seed envelope is an **estimate**, and none of ours was ever reported as one.
  For a correlation over `n` genes the null has `sd ≈ 1/sqrt(n-1)`; the range of three draws has
  expectation `1.693 sd` and its own `sd = 0.888 sd`, a **CV above 50 %**. All three measured
  envelopes are ordinary draws from that null, so the 2.2x gap between two datasets' envelopes —
  which reads like a fact about the datasets — is noise in the noise estimate.

I think this is the stronger half because it transfers. The negative result is about one method on
two datasets; the methodology is about how anyone reads a repeated-seed benchmark, and none of it
is reported in this literature.

⚠️ **The honest discount.** Every one of these was found *because a criterion in this project nearly
returned the wrong answer*, and in four cases it had already returned one that was recorded and
later withdrawn. That is the right provenance for a methods paper and should be stated as such
rather than presented as foresight: **we found these by getting them wrong first.**

---

## 7. Budget provenance — the objection a reviewer will raise

`Config.train_steps` defaults to **1200**; T09's selection gate chose **2400**; and T08 showed the
metric-aware terms **reverse sign** between the two. So "at which budget was this measured?" is a
live question. Checked:

| result | budget measured at |
|---|---|
| layout (R11) | **2400** for the headline swap; count error confirmed at **both** |
| zero-shot (E1 + replication) | **2400** — both campaigns, all twelve fits |
| metric-aware | **both** — and it is the one case where the budget flips the sign |
| **SEFL (A7)** | 🚨 **1200 only** — the default, not the selected budget |

**A7 re-scored at its own budget (2026-09-08), and the verdict strengthens.** With
`--train-steps 1200` the six checkpoints load, and generation-plus-scoring under today's code gives:

| arm | morans | gearys | umap | marker_field_r | marker_depth_r | localization |
|---|---|---|---|---|---|---|
| SEFL **off**, median of 3 | 0.4179 | 0.4205 | 0.8493 | 0.5316 | 0.6973 | 0.7499 |
| SEFL **on**, median of 3 | 0.1375 | 0.1135 | 0.5090 | 0.1171 | 0.2154 | 0.4167 |
| off > on | 2/3 | 2/3 | **3/3** | **3/3** | **3/3** | **3/3** |

Sources `reports/t10_a7_{off,on}_s{1,2,3}.json`. **SEFL costs on all six metrics**, four of them with
signs agreeing 3/3, and on seed 3 the ON arm's `morans_pearson` goes **negative** (−0.0748): the
generated field is anti-correlated with the real one, not merely uninformative. The recorded
REFUTED verdict was reached on a different scoring pass; this is an independent re-measurement under
today's code and it agrees, more strongly.

⚠️ **These are 1200-step numbers and are not comparable to the shipped row's 2400.** They compare
the two A7 arms to each other, which is the only comparison A7 was built to make.

**Confirmed from the weights themselves, not inferred (2026-09-08).** Re-scoring A7's six
checkpoints at 2400 was refused by the portability guard, and the guard now names the field: all
six report `train_steps: checkpoint 1200 -> this run 2400`. The budget in this row is read out of
the saved configs, not reconstructed from a launch command.

**It touches A7 alone**, and the record already scoped it that way: *"SEFL adds nothing at the
default budget of 1200 steps"*, not "to the shipped model". The metric-aware reversal is proof the
objection is not hypothetical.

**It does not rescue A7.** That arm did not underperform, it **collapsed** — `i_gen` at 1.6–2.4 % of
target, 218 alarms from step 250 of 1200. A budget that ends after 950 steps of a dead field is not
obviously one that escapes the collapse, and the burden is on the claim that more steps recover it.

**It does not make the negatives provisional.** The layout and zero-shot results are the bulk of the
negative column and were measured at the selected budget. Writing the caveat broadly would be the
§4.2 failure in reverse: a discount applied to soften an answer that was measured correctly.

---

## 8. What remains open

1. **R4 / the ZINB trade.** The largest, unchanged. A measured mechanism, a named term, and a
   design change — objective or emission model — as the only route. A follow-up paper, not a
   follow-up measurement.
2. **R12's two questions.** The 46–119x per-gene spread in `s`; why `theta` is uncorrelated with the
   data's dispersion. Both owed to T06.
3. **§4.2g / §4.2i's construction question.** How to take an envelope when some members are
   degenerate by design, and how to report one three seeds cannot pin. **Must be settled before the
   next pre-registration, not after the next verdict.**
3b. 🚩 **The envelope correction's residue** (`reports/envelope_correction.md` §3). Six clearance
   figures have **no admissible envelope** and are now flagged rather than numbered: R11's two
   layout deficits, `marker_field_r`'s tier-1 deficit, the boundary gap, `gene_mean_spearman`'s
   position against its floor, and §13.2's prior-campaign localization lead. The blocker is not
   compute for five of the six: **`runs/pilot/model_exp_2400.pt`'s configuration is not recorded in
   any artifact that cites it** — no `config_hash`, no `text_emb_mode`, no metric-aware weights — so
   A9's three-seed `bench3` envelopes cannot be matched to the arm behind the six-metric table.
   Recovering that one config block closes most of the list from files already committed; if the
   arm turns out to differ, it is a three-seed measurement and should be costed as one.
4. **The metric-aware weights** (§4). Unresolved, shipping on, inside the baseline everything else
   was measured against.
5. **The `cosmx` effect-size shrinkage.** Direction replicated 12/12, magnitude fell 5.6x, nothing
   explains it. Two candidates eliminated (pool sparsity, per-gene sparsity — both point the wrong
   way). The **density difference** (`cosmx` 5.7x denser per gene) is recorded as a candidate with a
   stated test and is **deliberately not being chased**: fourth mechanism hunt of the campaign, and
   the base rate governs.
6. **`marker_field_r` — a weakness that has survived three method generations.** It has no row
   anywhere above and it has appeared **four times**, which makes it an open item rather than a
   table entry:

   | # | where | what |
   |---|---|---|
   | 1–2 | v20 and v21 against SpatialZ | the **pooled** loss across datasets — **withdrawn as evidence**, see below |
   | 3 | v25 at T09, on the fixture | its **single** losing metric |
   | 4 | v25 at T10, tier-1 STARmap | **0.6384 against a `flanking_copy` floor of 0.8857 — 0.247 below it**, the worst of the six. 🚩 **"7.4x the envelope" is withdrawn and not replaced** (§envelope-correction §3): it divided by the pooled fixture 0.0335, the arm is the **superseded `hybrid` pilot row** (the shipped `resample` deficit is 0.203), and the only real-data envelope on this instrument — A9's `paper_marker_field_r`, 0.0596 — is on an arm whose configuration r11's artifacts do not record |

   ⚠️ **State appearances 1–2 at their real strength, which is lower than it sounds.** That loss is
   a **cross-dataset pool**, and §4.2a forbids exactly that: read per dataset it is **9–9**, on
   tier-1 v20 (0.8804) and v21 (0.8881) both **beat** SpatialZ (0.8522), and in the wide regime v20
   wins **7 of 7**. Only the pooled figure favours SpatialZ. So the honest count is **two clean
   appearances in v25 plus a pooled comparison our own methodology rejects** — which is still a
   pattern, and still the metric where this line has been weakest for three generations.

   **What makes it worth opening rather than noting.** The tier-1 deficit is not expression
   magnitude: `gene_mean_spearman` sits **0.0033** off its copy floor **on the A3 arm** — ⚠️ and
   **−0.0142 below** it on the shipped configuration (2026-09-09), which is **0.4x its own
   envelope**, i.e. a tie: the "per-gene magnitude is solved" reading is withdrawn, and no claim
   that it is *worse* replaces it. 🚩 **"inside the envelope"
   is withdrawn** — this metric is not in `METRIC_NAMES` and appears in **no** fixture or
   internal-LOSO envelope at all (the two-`SIX` defect), and the only spreads that exist for it are
   A7's and A9's on `bench3.evaluate_paper`, at **0.0033 to 0.1193** — a range that brackets the
   deficit, so "inside" is not established either way. What survives without an envelope is the
   comparison of magnitudes: 0.0033 against `marker_field_r`'s 0.203, i.e. the residual is
   **spatial arrangement** and not per-gene magnitude, by a factor of 60. And it is one of only
   two **pose-dependent** metrics, with tier-1 rotations of 0.0–1.5 degrees — so alignment is
   measurably **not** confounding it, and a v25 deficit at ~0 degrees points at **T05's intensity
   head**, not at `align.py`. That is a specific place to look, which is what distinguishes an open
   item from a complaint.

   ✅ **DONE 2026-09-07 — and the redirect is the useful half.** The boundary candidate is
   eliminated, but what the elimination *points at* is worth more than the elimination:
   **`resample` does not use the intensity head to place cells at all, and still carries a 0.19
   deficit.** The weakness survives removing the layout head from the picture entirely, so it is
   not T05's — it is in the **expression path**, R12's territory. §8.6 predicted the intensity head
   via the boundary regime; that route is closed and the prediction is **weakened, not confirmed**.
   The remaining search is arrangement in the emission, at every depth.

   **The measurement.** Zero fits and zero generation — both sides were already on disk. On
   `resample-grid`, the
   **shipped** layout mode and the only arm whose sections are density-comparable (spread
   **1.04x**), the deficit below each section's own copy floor is **0.1729 / 0.1877 / 0.2043** —
   a boundary-vs-interior gap of **−0.0231**.

   🚩 **BOUNDARY ELIMINATED is downgraded to NOT READABLE, 2026-09-08.** The verdict rested on that
   gap being **0.69x the envelope**, and `t10_marker_field_boundary.json` hard-codes
   `"envelope": 0.0335` — the pooled synthetic-fixture figure, on a `bench3.evaluate_paper` number.
   Against the only real-data envelope on that instrument (A9's `paper_marker_field_r`, 0.0596) the
   same gap is **0.39x** — still inside — while against instrument A's `marker_field_r` envelope
   (0.0148) it is **1.56x**, outside. ⚠️ **Corrected: an earlier revision of this note attributed
   the 1.56x to A9's envelope; it belongs to instrument A's.** The two candidate divisors give
   **opposite answers**, which is why the verdict is unreadable rather than overturned. That does
   **not** establish a boundary
   effect either: A9's envelope is fold-aggregated where this effect is **per section** (§4.2d says
   the per-section construction is always the larger), and it is on an arm r11's artifacts do not
   identify. So the honest state is that the test cannot be read at all until an envelope exists for
   this metric, on this arm, at this aggregation level. The **raw** finding stands and is what the
   redirect below rests on: the three deficits are 0.1729 / 0.1877 / 0.2043, and the boundary
   section carries the **smallest** of the three, so nothing in the data points at a boundary
   mechanism. `reports/envelope_correction.md` §3.

   The other four arms return INVERTED and **must not be read**: pre-registered condition (c) fires
   on all five, but the field-based modes emit **63.9x** the truth at `section_2` and **0.90x** at
   `section_6`, so the boundary section is scored on a 64x-thinned, better-conditioned draw. The
   "inversion" is the count error read through the metric.

   `reports/t10_marker_field_boundary.md`. **Closed as far as free measurement goes**: nothing else
   about this metric can be settled without a spend.

7. **R14's donor rule.** Costs 0.116 of `marker_depth_r`, deliberately unfixed because fixing it
   makes the negatives **stronger**.

**Closed since 2026-09-01:** the replication (PARTIAL / DOES NOT REPLICATE), A7 (SEFL harmful), the
`decoder_mu_link` refit (not owed — `exp` shipped before every real-data audit), the moment-matched
`theta` (no value exists to match to), and the pool-sparsity guess (refuted, direction backwards).

---

## 9. What would change this assessment — and what would not

**Would:** a change to the emission model or its objective that closes R4, followed by a re-run of
the six-metric comparison against copying. If generated expression beat `cross-mix` on real tissue
after that change, the headline inverts from "a good representation and a bad generator" to a
method paper. This project cannot run it.

**Would not: more seeds on the results already measured.** §4.2i is the reason. The envelopes three
seeds produce are estimates with a CV above 50 %, so a fourth and fifth seed on an existing
comparison buy precision in the **noise estimate**, not evidence about the method. The zero-shot
claim in particular is as strong as three seeds on two datasets can make it, and adding seeds there
would sharpen an envelope rather than change a verdict.

**The two statements below are about different objects and are not in tension.** The paragraph
above is about **precision on comparisons already run**. The one below is about **resolving a
component that has never been measured at three seeds at all** — an unrun comparison, not a rerun
one. More seeds do not help the first; three seeds are the minimum that can settle the second.

**The cheapest thing that would change something real** is §4's six fits: `(2400, on)` against
`(2400, off)` at **three** seeds, where the existing evidence is **one**. That is not extra
precision on a measured result; it is the first measurement that `claim_min_seeds` would accept, on
a component that ships **on** and sits inside the baseline every other number was produced against.
It converts a silent assumption into a stated result — in either direction, since a null there is
as publishable as a positive.

I am **not** recommending it over the R4 work, which is the larger question and which this project
cannot run anyway. I am recording that it is the only small spend on the list that touches the
configuration every other number was measured on.
