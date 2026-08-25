# Open risks carried forward

Part of [PROGRESS.md](../PROGRESS.md).

## Open risks carried forward

| # | Risk | Raised | Owed to | Decision due |
|---|---|---|---|---|
| R8 | ~~**The reduced-epoch selection budget reverses gates with a training-free option**~~ — **RESOLVED at T09.** Measured at both budgets (`reports/r8_budget_grid.md`): `cross-mix` won `expr_mode` at 600 steps under both priors and came **last** at 2400 under both; `iid` won `prior_mode` at 600 on exactly the two expression paths where the prior can act. Cause in one column — morans from 600→2400 gains **+0.3432** for `zinb-flow` and **−0.0180** for `cross-mix`, the copying path needing no training. Fixed by writing the **training-free-option rule** into `specs/09` §3 and enforcing it in code: `TRAINING_FREE_OPTIONS` classifies every gate, `_check_gate_classification` raises for an unclassified one, and the two gate sets are derived from it. `layout_mode` x `prior_mode` x `expr_mode` are now one **18-cell gate at full budget**; all six `cross-mix` cells rank bottom. **All three gates changed answer**, `layout_mode` included — the leg with no observed reversal. New config (`2ce15bbaf5cf2bc1` once `Config` gained R9's two fields), and the definition of done against the independent-donor bar goes **2 of 6 → 4 of 6** | T09 | — | **closed; new residual recorded below** |
| R9 | ~~**A gate can be decided on an unconverged incumbent even when the training-free rule passes it**~~ — **RESOLVED at T09.** `specs/09` §3's rule gained a second condition: a gate is scored at the selected budget when it has a training-free option **or** when the incumbent is unconverged at the reduced budget. Condition (2) is *measured* per run by `incumbent_is_unconverged` — it depends on the incumbent, so no static declaration can predict it — and fires here on the run's own numbers (shortfalls 0.36 / 0.43 / 0.81 against a 0.05 tolerance). Re-scored at 2400, `lookup` **still wins** (rank 1.2 vs 1.8), so the winner does not flip back; but the gate becomes undecidable — see R10 | T09 | — | **closed** |
| R10 | ~~**Scores are reproducible only to ~0.012, wider than several gate margins**~~ — **RESOLVED at T09 as a rule; the envelope is now measured.** 3 cells x 3 seeds (`reports/envelope_synthetic.md`): largest across-seed spread **0.0335**, nearly 3x the n=2 figure, **not score-dependent** (0.0299 / 0.0335 / 0.0270 at score levels 0.95 / 0.94 / 0.83) but strongly **metric-dependent** (0.0068 to 0.0335, a 5x range). Two remedies ship as rules: the **repeated-seed rule** (>= `claim_min_seeds` = 3 for anything reaching a claim, scoped in `specs/10` §3 with a ~2.4x campaign bill) and the **capability tie-break** (`claim_tie_break_envelope` = 0.04, set from the measurement and rounded up). Re-checking the 18 cells found **`layout_mode` decided inside the noise** (margin 0.0344 vs a 0.0335 envelope) — tie-broken to `hybrid` — alongside the known `text_emb_mode` (0.0110 → `medcpt`). `prior_mode` (1.8x) and `expr_mode` vs `cross-mix` (7.3x) stand on measurement. Shipped: **`00ef4a19a2f576b8`** | T09 | T10 | **closed as a rule; the borderline `layout_mode` call is flagged in the report** |
| R1 | ~~**`ell_z` cannot be resolved by a 9-section stack** (fit 561 µm against a 200 µm ground truth)~~ — **DECIDED at T07, 2026-08-17.** Remedy **2** (calibrate `ell_z` at inference against observed between-section correlation) is adopted, **owed to T09**, with remedy **3** (treat the fit as a bracket endpoint and fail loudly) shipped alongside it as the guard; remedy 1 is rejected as the primary. The decision is evidence-based: T07 measured whether `L_cross` can serve as the training-time instrument remedy 2 might have used, and it **cannot** — see the R1 section below | T03 | T07 → T09 | **implemented at T09; risk stays open.** Both remedies ship and are tested (`calibrate_ell_z`, the bracket-endpoint guard, and a new guard that *refuses* a `prior_mode` under which `ell` is inert). On the fixture both return `target_unreachable`, but the diagnosis says that is an **artefact**: under the selected `expr_mode="cross-mix"` the objective is constant to 10 decimal places across a 15× `ell_z` sweep, because `_cross_mix` copies donor counts verbatim and never evaluates the flow. The GRF is monotone (−0.005 → 0.921) and so is the whole pipeline under `zinb-flow` (0.887 → 0.932 against a 0.918 target), so neither the field nor the 400 µm stack is the limit. Both follow-ups are now done. **C30 is closed**: `apply_lengthscale` writes a converged axis into the `Config` and drops a non-converged one with a warning naming both numbers (`specs/09` §2 amended to name the writer). **The live arm** (`correlated` + `zinb-flow`) gives the first real measurement: `ell_xy` **converges** to 86.4 µm (I_gen 0.4051 vs a 0.4102 target, 2 iterations) and is applied; `ell_z` runs to the bracket's **top** (364.6 µm) still undershooting (0.8706 vs 0.9182), so the data support `ell_z ≥ 364.6 µm` — a **lower** bound, which corrects the 'upper bound' phrasing this risk carried since T07 — and the writer drops it. Remedy 3 is intact. What keeps R1 open is no longer mechanism but **the target**: the objective excludes both sections from retrieval so the correlation comes from the field alone, while the observed 0.9182 draws heavily on shared anatomy, so no `ell_z` need exist that closes the gap. That is the spec owner's call. |
| R7 | **SEFL as a whole is unverified and, on every statistic that could be measured, harmful.** With `w_thick = w_prog = 0.2` a model trained at **T06's own budget and config** fails three T06 acceptance tests: detection MAD **0.0551** (< 0.05, T06 recorded 0.0191), mean-variance slope error **0.2838** (< 0.15, recorded 0.0084), Frobenius covariance **20.301** (recorded 9.316, baseline 7.563). All three SEFL weights therefore ship at **0** and A7 becomes an *addition* experiment | T07 | T10 (A7) | **at T10** — SEFL is used only if A7 shows a gain on the six target metrics |
| R6 | **`L_cross` is vacuous in v25 and its residual minimiser destroys the anatomical field.** Two crossing planes already emit bitwise identical expression (by construction, untrained); the only plane-dependent channel left is the augmentation pose, which T04 made pose-dependent on purpose. At `specs/07`'s `w_cross = 0.3` the generated section's per-gene variance falls to **0.065** of the real one's against **0.711** with SEFL off. `w_cross` ships at **0**; `loss_cross` is kept and pinned by a strict xfail | T07 | T10 (A7, E5) | **at T10**, or sooner if the spec's owner takes SPEC_QUESTIONS C19's option 1 |
| R2 | ~~GATE 2's attention is near-uniform (0.987 × log K)~~ — **CLOSED at T06.** With the flow-matching head trained the entropy falls to **0.8563 × log K** (a fall of 0.132, required 0.05) and stays above the 0.5 collapse line. The query is what changed: T04's probe queried with the field feature alone, T06's with `[F(p), fourier(p), type_emb, region_emb, z_embed]`, and a query that knows its own cell type can prefer a donor that shares it. | T04 | T06 | **closed 2026-08-16** |
| R5 | **C14 blocks the paper's headline novelty.** *(Sharpened: the table also has to be *built correctly* — see B19, four defects in `build_gene_meta` that produced four species' genes for a one-species request, all fixed and replayed, but the live mouse-only summary coverage is still unmeasured here.)* Open-vocabulary decoding has no positive evidence: zero-shot is r = −0.368 on the fixture (B18) because arbitrary gene names carry no text signal, and `resources/gene_meta.parquet` cannot be built here — every gene-annotation host is 403'd by the network policy. Needs one command on a networked machine (see C14). | T02, T06 | T10 (E1) | **before the first real run** |
| R4 | **Likelihood/fidelity divergence — one pathology, two heads.** *(T08 ran: the cure is **built and measured**, it is **not** enabled by default, and R4 stays **OPEN**. The covariance criterion is missed on both regimes — 11.02 vs 7.73 and 13.39 vs 11.38 — so the covariance claim is downgraded; but at 2400 steps the arm **with** the terms improves its covariance while the arm without it degrades, which is the first direct evidence that anything in this project can arrest the divergence. Decision owed to T09 §3's joint selection gate and reported by T10's A2, at two budgets.)* Fitting by likelihood alone makes the likelihood better and the *generated section* worse. On the **layout intensity** head (SPEC_QUESTIONS B10, T05): recovered intensity correlation 0.97 → 0.28 over 300 → 1200 steps while the Poisson NLL keeps falling. On the **expression** head (T06): 1200 → 2400 steps lowers the ZINB NLL 1.589 → 1.578 nats/pair while Frobenius covariance error goes 17.7 → 21.3 and detection MAD 0.056 → 0.069. Neither head has a term that could stop it. **`TRAIN_STEPS = 1200` is early stopping fitted to this fixture and will not transfer to real data.** | T05 (B10), T06 | ~~T08~~, T09, T10 (A2) | **T08 ran and did not close it**; the criterion is a strict xfail carrying 11.02 / 13.39, and the enable/disable decision moves to **T09 §3's joint selection gate** (`train_steps` × the metric weights, scored per dataset on internal LOSO), reported by **T10's A2** at both budgets. The 0 is a starting point, not a shipped decision |
| R11 | ~~**The intensity-field layout is worse than copying, on real data**~~ — **DECIDED 2026-08-25: `layout_mode` ships as `resample`, recorded as a negative result.** `Config.layout_mode` defaults to `"resample"`; `field` and `hybrid` stay selectable and are reported as ablation A4 (`specs/10` §6, restated because repulsion-off alone became a no-op under the new default). Written into `specs/05` §4a, `specs/10` §4.5b and §6, `design/v23_design.md` §3.3 and its risk register (the row that predicted this outcome now reads FIRED), and `progress/fixture_limitations.md` §2. The measurement it rests on: Holding everything else fixed on tier-1 STARmap and swapping only `layout_mode`, on the **grid** sampler: `celltype_localization` at ground-truth-matched density is **0.6607** for `field` and **0.6692** for the shipped `hybrid` against **0.7546** for `resample`, where the model-free `flanking_copy` floor is **0.7765** and the `oracle` ceiling **0.9808** — both field-based modes score *below the floor* on the metric the layout head exists to win, by **3.5x and 3.2x** R10's across-seed envelope, while `resample` lands on the floor to within 0.0219 (inside it). The rejection sampler's bias was **not** the explanation: correcting it buys `field` 0.0599 (1.8x the envelope) and `hybrid` 0.0158 (inside it), and leaves the ordering unchanged. It made the **count** worse, by unmasking it — median `cell_count_ratio` 0.895 -> **5.362**, `section_4` 163 cells placed -> 21 993 (`reports/r11_starmap_layout_modes.md`). `cell_count_ratio` 0.894 -> 0.988 in the same swap, and `hybrid` cannot help: its count draw precedes the sliced-Wasserstein polish (`layout.py:889`) and is bit-identical to `field`'s. and the count error is large and erratic: 4332/1372/3866 at 1200 steps against a ground truth of 4187/4102/4162, 11168/146/3788 at 2400, and **48 343 against 4 187** in the `exp`-link arm. It agrees with the fixture, where `resample` was the **rank winner** (3.0), `hybrid` followed (4.2) and **`field`'s best cell ranked 7.0, winning nothing** — `hybrid` ships only on a tie-break whose margin (0.0344) sat inside R10's 0.0335 envelope. The fixture signal was **underpowered, not wrong**. ⚠️ It does **not** explain R12: with real positions the emitted expression still carried only 22 % of the tissue's Moran's I. Coupling to the decoder is **indirect** — `IntensityHead` takes only coordinates and field features, but `_layout_term` reads the **shared** triplane and `forward_train` backpropagates `recon` and `layout` together | T05, T09 | T10, then T05/T09 | **CLOSED 2026-08-25** — `specs/05`'s headline layout claim did not hold and is amended there as a negative result; the `hybrid` tie-break is overturned in `progress/t09_inference_and_calibration.md`; and the inheritance question is answered **no**, with a specific cause rather than a statistical one — the fixture's flanking baseline sits at 58 % of its ceiling against real tissue's 79 %, so it over-rewards a generative layout at any number of seeds. What remains open is **not** R11 but the defect it localised: the intensity integral's scale, unstable 3.7x across refits and link-coupled, which is T05's to fix if a generative layout is ever to ship |
| R3 | **The stack's ends reconstruct far worse than its interior.** Per-section R² was **0.2912** at the first section and **0.3642** at the last, against an interior mean of **0.4474** — a 20–35 % deficit. One-sided evidence at the volume boundary. | T04 | T09, T10 | **at T09** |

### R11 — the layout head, on one page

`layout_mode` field vs hybrid vs resample, on both datasets, on the metric the layout head exists
to win. **Every number is a median over held-out sections** (`specs/10` §4.6).

#### ✅ DECIDED 2026-08-25 — `layout_mode` ships as `resample`, as a negative result

`Config.layout_mode` defaults to `"resample"`. `field` and `hybrid` remain implemented and
selectable and are reported as ablation **A4** (`specs/10` §6, restated: repulsion-off alone became
a no-op under the new default, so A4 is now the ablation of the generative layout as a whole).

The reasoning, in the order it carried weight:

1. **Three measurements agree and the ordering never changed** — the fixture's 18-cell gate, the
   pilot on the biased sampler, and the re-measurement on the grid sampler. Removing the bias moved
   `field` 1.8x the envelope and `hybrid` 0.5x it, and moved neither past `resample`.
2. **`resample` sits on the model-free copy floor, inside the envelope** (−0.0219 against 0.0335),
   which is the consistency check the table needed: `resample` *is* the copy layout, so anything
   else would have meant the instrument was drifting.
3. **The count decides it on its own.** A 3.7x swing between refits of one configuration is not
   shippable whatever the localization says, and no density-matched score can see it.

Written into `specs/05` §4a, `specs/10` §4.5b and §6, `design/v23_design.md` §3.3 and its risk
register, `progress/fixture_limitations.md` §2, and the tie-break note in
`progress/t09_inference_and_calibration.md`.

**The fixture gate was re-run on the corrected sampler (3 seeds x 2400 steps, one fit per seed,
`reports/t09_layout_mode_gate_grid.md`) and it returns no answer at all** — which is a cleaner
result than the one it was expected to give:

| | seed 1 | seed 2 | seed 3 | pooled |
|---|---|---|---|---|
| winner by median rank | `field` | `hybrid` | `resample` | **3-way tie at 2.0** |

**Three seeds, three different winners**, and the pooled ranks tie exactly. On **5 of the 6**
metrics the spread between the three arms is smaller than the arms' own spread across seeds, so
the gate is reading seed variation rather than layout. On `celltype_localization` — the one metric
where the arms separate at all — `resample` leads by 0.0193 (`field`) and 0.0204 (`hybrid`), which
agrees in direction with the real data and is 0.6x the envelope, i.e. still not a difference.

This supersedes the earlier reading of the fixture as *underpowered*. It is not that three seeds
were too few; it is that the answer changes with the seed while the arms' true separation on this
dataset is smaller than that noise. The real-data measurement is the one that decides, and it is
9x further from the envelope.

**One number in that report is a correctness check worth keeping.** `resample`'s
`celltype_localization` has an across-seed spread of **exactly zero** — `_resample_layout` copies
the flanking section's coordinates and types unchanged, so the layout carries no seed, and the
fixture's localization is a per-type field correlation over positions and type labels only. Its
expression-driven metrics *do* vary across seeds, which is how the three fits are known to be
genuinely different rather than accidentally identical.

**What stays open is the defect, not the risk.** The intensity integral's scale — 1.5x-64x against
ground truth, unstable 3.7x across refits, link-coupled through the shared triplane — is T05's, and
it is what a generative layout would have to fix before it could ship. The point *pattern* is not
the problem: `g(r)` over `[0, 3R]` matches the real pair-correlation function at 0.093 against pure
Poisson's 0.994.

#### The measurement it rests on (2026-08-25). The ordering survives; the counts do not.

Everything below this subsection was measured on the **rejection** sampler that
`reports/r11_envelope.md` found biased, so the whole comparison had to be re-taken. It has been:
`reports/r11_starmap_layout_modes.md`, five arms off one refit of the same configuration, on the
same instrument and the same ground-truth-matched density, with `field` and `hybrid` also re-run on
the old sampler as a control. **The referents reproduce `reports/pilot.md` §3 to four decimals**
(`flanking_copy` 0.7008 / 0.7765 / 0.7868, `oracle` 0.9808), which is what says the instrument and
the dataset build are the pilot's and not merely similar.

| arm | `celltype_localization` (median) | per section (2 / 4 / 6) | `cell_count_ratio` (median) | per section |
|---|---|---|---|---|
| `oracle` — ceiling | **0.9808** | 0.9765 / 0.9888 / 0.9808 | 1.000 | 1.000 / 1.000 / 1.000 |
| `flanking_copy` — copy floor | **0.7765** | 0.7008 / 0.7765 / 0.7868 | 0.988 | 0.973 / 1.016 / 0.988 |
| `resample` (sampler-independent) | **0.7546** | 0.7008 / 0.7546 / 0.7868 | 0.988 | 0.973 / 1.016 / 0.988 |
| `hybrid`, **grid** | **0.6692** | 0.6692 / 0.6600 / 0.6948 | **5.362** | 63.90 / 5.36 / 0.895 |
| `field`, **grid** | **0.6607** | 0.6634 / 0.4437 / 0.6607 | **5.362** | 63.90 / 5.36 / 0.895 |
| `hybrid`, rejection (control) | 0.6534 | 0.6534 / 0.3581 / 0.6790 | 0.895 | 42.87 / 0.040 / 0.895 |
| `field`, rejection (control) | 0.6008 | 0.6008 / **0.0000** / 0.6628 | 0.895 | 42.87 / 0.040 / 0.895 |

Emitted / ground truth: `field` and `hybrid` **267 567 / 4 187**, **21 993 / 4 102**, 3 727 / 4 162
on the grid sampler; 179 495, 163, 3 727 on the rejection control; `resample` 4 073, 4 169, 4 110.

**1. The refit is not the confound, and the control is what proves it.** The pilot's checkpoint no
longer exists, so these arms come from a refit on a different machine, where Convention 3's bitwise
determinism does not reach. Re-running the *old* sampler on the *new* weights reproduces the pilot
within R10's 0.0335 envelope on every arm — `field` 0.6008 vs 0.6136 (0.38x), `hybrid` 0.6534 vs
0.6572 (0.11x), `resample` 0.7546 vs 0.7601 (0.16x). So the grid-minus-rejection differences below
are the sampler's, not the refit's.

**2. The bias was real but small, and it does not change the answer.** Same weights, sampler
swapped: `field` **+0.0599** (1.8x the envelope — a real gain), `hybrid` **+0.0158** (0.5x — inside
it, not a difference). Both field-based modes remain **below the model-free copy floor**, now by
**3.5x** (`field`) and **3.2x** (`hybrid`) the envelope, against 4.9x and 3.6x before. `resample`
sits 0.0219 under the floor, *inside* the envelope, exactly as before — it is the copy layout, and
lands on it. **The ordering `resample` > `hybrid` > `field` is unchanged on both samplers.**

**3. The count went the other way, and that is the sharpest new fact.** The grid sampler could not
have fixed the count — `n_target ~ Poisson(n_expected)` is drawn from the slab integral *before*
either sampler is reached — but by removing the starvation it stops the intensity integral's
overshoot from being masked. `section_4` goes from **163** cells placed to **21 993**; the median
`cell_count_ratio` for both field-based modes goes **0.895 -> 5.362**. The reassuring median this
risk warned was "hiding the failure" is gone: the failure is now the median.

**4. The count is also far less reproducible than the scores.** On the *same* sampler and the same
configuration, the pilot emitted 48 343 cells for `section_2` and this refit emits **179 495** —
3.7x — while every density-matched score moved less than 0.013. Whatever the intensity integral is
estimating, it is not stable across a refit that the metrics cannot tell apart. Any future report of
`cell_count_ratio` has to carry that alongside the per-section spread.

**5. `layout_mode` does not move the autocorrelation metrics at all.** `morans_pearson` spans
0.6465-0.6690 across all five arms — a spread of 0.0225, inside the envelope — against a copy floor
of 0.9836. R12 is untouched by any of this, which is the same conclusion the pilot reached from the
other direction.

**6. `marker_field_r` is the standing weakness for the fifth time.** Best arm 0.6830 (`resample`)
against a 0.8857 floor: **6.0x** the envelope, the largest shortfall in the table.

**What is still not settled.** One seed (`claim_min_seeds` = 3 before any of this carries a claim);
still `text_emb_mode="lookup"`, i.e. ablation A3; T09's per-dataset selection and calibration have
still never run on this dataset. And **the `layout_mode` decision is not taken here** — this is the
measurement it was waiting on, not the call.

#### The original measurement, on the biased sampler — superseded by the subsection above

*(Kept because the reasoning below is still the reasoning, and because the fixture half of it was
never re-run. The three STARmap `layout_mode` rows are the ones superseded; do not quote them.)*

**Real STARmap, tier 1 (`paper_2_4_6`).** `flanking_copy` and `oracle` are bench3's own probes,
measured on the pinned instrument (`reports/pilot.md` §3) and model-free, so they bracket what is
achievable.

All three `layout_mode` arms below come from **one set of weights**
(`runs/pilot/model_exp_2400.pt`, `decoder_mu_link=exp`, 2400 steps): `layout_mode` is a
generation-time gate that `check_generation_cfg` permits to differ, so no refit was needed and
nothing but the layout varies between the rows. `celltype_localization` is scored at
**ground-truth-matched density** — each section subsampled to its own true cell count, because a
denser point set puts kNN neighbours closer and inflates every graph-based metric, and `field`
overshoots by 11x on one section. `cell_count_ratio` is from the raw pass, where it means
something. Full six-metric table in `reports/t10_rescore_exp.md`.

| arm | `celltype_localization` | `cell_count_ratio` |
|---|---|---|
| `oracle` — ceiling | **0.9808** | 1.000 |
| `flanking_copy` — copy floor | **0.7765** | — |
| `layout_mode="resample"` | **0.7601** | 0.988 |
| `layout_mode="hybrid"` (shipped) | **0.6572** | 0.894 |
| `layout_mode="field"` | **0.6136** | 0.894 |

**Three things this table says that the ordering alone does not.**

1. **`resample` is the copy floor, and lands on it.** `_resample_layout` reuses the nearest flanking
   section's coordinates and types unchanged, so `resample` and `flanking_copy` are the same layout;
   they differ only in that v25 replaces the expression. The 0.0164 between them is *inside* R10's
   0.0335 envelope, which is the consistency check this table needed. The two field-based modes are
   then **0.1629** (`field`) and **0.1193** (`hybrid`) below that floor — **4.9x and 3.6x** the
   envelope.

2. **`hybrid` cannot fix the count, by construction.** `sample_layout` draws
   `n_target ~ Poisson(n_expected)` from the slab integral and only *then*, at `layout.py:889`,
   applies `hybrid`'s sliced-Wasserstein polish to the positions. With the same seed the count draw
   is bit-identical, so `hybrid`'s emitted counts equal `field`'s exactly —
   `48343/4187, 92/4102, 3719/4162` for both. The docstring already says it: `resample` is "the one
   mode whose count does not come from the intensity". Whatever `hybrid` buys (+0.0436 here) it buys
   with positions alone, on top of the same broken count.

3. **The median `cell_count_ratio` of 0.894 is not reassuring, it is hiding the failure.** The three
   per-section ratios are **11.5x, 0.022x, 0.894x**. The median picks the only sane one. `specs/10`
   §4.6 mandates the median for headline statistics and that mandate stands — but `cell_count_ratio`
   spans 500x across three sections here, and any report of it must carry the per-section spread
   beside it or it misdescribes the layout entirely.

*(An earlier arm, before the `decoder_mu_link` switch and without the density control, put `field`
at 0.4252 and `resample` at 0.7008. Same ordering, wider gap. The numbers above supersede it: they
are one config, one instrument, all three arms.)*

**Synthetic fixture (T09's merged 18-cell gate, at the selected budget).** The gate ranks whole
cells, so the column is the best cell each `layout_mode` appears in:

| arm | best median rank | `celltype_localization` in that cell |
|---|---|---|
| `resample` | **3.0** — the rank winner | −0.0660 |
| `hybrid` | 4.2 | −0.0532 |
| `field` | **7.0** — won nothing | −0.0591 |

*(The fixture's `celltype_localization` column is a leave-one-out delta, not a bench3 score, so it
is not comparable with the STARmap column and is shown only to place the ranks.)*

**Reading the two together.** They agree in direction and differ in power. On the fixture the
preference against `field` was inside R10's reproducibility envelope — margin 0.0344 against
0.0335 — which is why `hybrid` shipped on a tie-break at all. On real data the same preference is
**~29x the envelope**, and `field` is below a model-free copy. The fixture result was underpowered,
not wrong.

**What it is not.** R11 does not explain R12: with `resample`'s real cell positions the emitted
expression still carried only 22 % of the tissue's Moran's I. Two defects, not one.

**Coupling to the decoder is indirect, and its size is being measured.**
`IntensityHead.forward(xyz, field_feat, region)` takes no gene embedding, no size factor and no
`mu`, so the link function has no direct path to the intensity integral. But `_layout_term` computes
`lam = self.intensity(cells_data, self.field(cells_model))` from the **same** triplane the decoder's
conditioning reads, and `forward_train` sums `recon` and `layout` and backpropagates them together.
`scripts/t10_r11_coupling.py` fits both links with everything else matched and instruments the
integral.

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
