# The 0.0335 correction — every clearance figure re-read against its own envelope

**2026-09-08.** `specs/10` §4.2a says an envelope is per-metric, per-arm, per-dataset and per-gate.
The practice throughout this project has been to divide every margin by **0.0335** — a figure that
is none of those things. This is the re-derivation. **Zero fits and zero generation**: every number
below comes from artifacts already in the branch.

Sources: `reports/envelope_synthetic.csv` (R10, 9 fits), `reports/t09_envelope_starmap_seed{2,3,4}.json`,
`reports/t09_tenv_deep_seed{2,3,4}.json`, `reports/t10_a9_{0,05}_s{1,2,3}.json`,
`reports/t10_a7_{off,on}_s{1,2,3}.json`, `reports/r11_starmap_layout_modes.json`,
`reports/t10_marker_field_boundary.json`.

---

## 0. What 0.0335 actually is, and the two separate defects

`reports/envelope_synthetic.md`: nine fits, three cells, three seeds, on the **synthetic fixture**,
scored by `train/select.py::section_scores`. The reported envelope is the **maximum over six
metrics of the maximum over three cells** of the across-seed range. Two defects follow, and they
are independent:

| defect | who it hits | the rule it breaks |
|---|---|---|
| **(1) pooled across metrics** | every figure, fixture and real-data alike | §4.2a — "a 4.0x range, and a pooled figure errs in *both* directions" |
| **(2) measured on the fixture, applied to real data** | every real-data figure | §4.2a — "the envelope is also dataset- and gate-specific" |

Defect (1) was **avoidable from the day the number was measured.** `envelope_synthetic.md` already
carries the per-metric table and already says so: *"A per-metric envelope is available in the table
above and is the right thing for T10 to quote per claim."* It was never used.

Recomputed from `envelope_synthetic.csv`, reproducing that table exactly:

| metric | fixture envelope | 0.0335 is this much too strict |
|---|---|---|
| `morans_pearson` | 0.0160 | **2.10x** |
| `gearys_pearson` | **0.0335** | 1.00x — the pooled figure *is* this metric |
| `umap_mixing` | 0.0115 | **2.92x** |
| `marker_field_r` | 0.0162 | **2.06x** |
| `marker_depth_r` | 0.0299 | 1.12x |
| `celltype_localization` | 0.0068 | **4.96x** |

So on the fixture the pooled figure is never too lenient — it is `gearys_pearson`'s envelope worn
by all six, and it is up to **5x too strict**. Every "inside the envelope" call made on the fixture
was therefore made against a bar up to five times too generous, in the direction that **suppresses**
findings.

---

## 1. 🚨 The blocking finding: the corpus contains two instruments, and the envelopes are on the wrong one

This is why the correction could not be carried out as a renumbering, and it was not visible until
every figure was traced to the script that produced it.

| | instrument A | instrument B |
|---|---|---|
| scorer | `train/select.py::section_scores` (T08 kernels) | `bench3.evaluate_paper`, SHA-pinned |
| design | internal LOSO, folds = **interior training** sections (`section_3`, `section_5`) | **`paper_2_4_6`** held-out (`section_2/4/6`) |
| metric names | `morans_pearson`, … | `paper_morans_pearson`, … |
| scripts | `t09_envelope.py`, `t09_audit_starmap.py`, `t09_depth_ceiling.py` | `t09_ship_starmap.py`, `t10_layout_modes_table.py`, `t10_rescore_saved.py`, `t10_a9_aggregate.py` |

`specs/10` §5 already forbids mixing them by name: *"Numbers elsewhere in the record scored on
internal LOSO with T08 kernels are a different quantity and must not be placed beside these."*

**Every three-seed envelope the project had ever quoted is on instrument A.** The headline
six-metric table, R11's layout-mode comparison, the `marker_field_r` deficits and the boundary
stratification are all on **instrument B**. So the pooled fixture figure was not merely pooled and
not merely off-dataset — for the bench3 numbers it was off-**instrument** as well, a third error
nobody had named.

### 1a. ✅ And the missing envelope existed all along, inside A9

`reports/t10_a9_{0,05}_s{1,2,3}.json` are tier-1 STARmap, `paper_2_4_6`, `bench3.evaluate_paper`,
**2400 steps, three seeds, two arms**, `layout_mode=resample`, `layout_sampler=grid`,
`decoder_mu_link=exp`. Their across-seed spreads are the first real-data envelope this project has
had on instrument B, and they were never extracted, because A9 was filed as UNINFORMATIVE about the
question it was designed to answer.

| metric | env, weights **off** | env, weights **on** (shipped level) | shared |
|---|---|---|---|
| `paper_morans_pearson` | 0.0202 | **0.2894** | 0.2894 |
| `paper_gearys_pearson` | 0.0241 | **0.2861** | 0.2861 |
| `paper_umap_mixing` | 0.0464 | 0.1281 | 0.1281 |
| `paper_marker_field_r` | 0.0307 | 0.0596 | 0.0596 |
| `paper_marker_depth_r` | **0.1225** | 0.1003 | 0.1225 |
| `paper_celltype_localization` | 0.0009 | 0.0061 | 0.0061 |
| `paper_gene_mean_spearman` | 0.0400 | 0.0301 | 0.0400 |

(`paper_marker_depth_r`'s 0.1225 is `reports/t10_a9.md`'s own envelope (a), reproduced to four
decimals — the derivation checks out against the one place it was already computed.)

**Read that table against 0.0335.** On instrument B the real per-metric envelopes span **0.0009 to
0.2894 — a 320x range**, and the two autocorrelation metrics move by **~0.29 between seeds** where
the pooled fixture figure claims 0.0335. A9 also supplies `paper_gene_mean_spearman`, which is in
*no* fixture or instrument-A envelope at all.

⚠️ **It still cannot be substituted into the r11-derived figures**, and this is the §4.2a
distinction doing real work rather than pedantry. A9's arms are `text_emb_mode=medcpt` fitted from
`Config` defaults (`provenance.source = "defaults"`, `selection_path = null`). The arm behind the
six-metric table is `runs/pilot/model_exp_2400.pt`, whose gates the r11 artifacts did not record —
they carried `model`, `decoder_mu_link`, `train_steps` and `seed`, and no `config_hash`, no
`text_emb_mode` and no metric-aware weights. **They now carry a `recovered_config` block on every
arm** (2026-09-09), and §1b reports what it says: the two arms differ on **two fit-time gates**, so
A9 is not an admissible donor. §4.2a's own evidence says `text_emb_mode` is not a gate to wave
through — on the `deep_starmap` gate the two arms' envelopes differ by up to 2.6x and **the worse
arm alternates by metric**.

✅ **CORRECTION 2026-09-08, and it is the useful half — the configuration was never lost.** The
first revision of this section implied the arm's identity was gone. It is not:
`scripts/t10_rescore_saved.py` **writes and reads the model file as**
``{"config": <every Config field>, "state_dict": ...}``, and its own ``--preflight`` branch already
prints ``decoder_mu_link``, ``train_steps``, ``layout_sampler``, ``text_emb_mode`` and
``expr_pca_dim`` from it. The whole `Config` has been sitting in `runs/pilot/model_exp_2400.pt` the
entire time.

**So the defect is narrower and more ordinary than "a landed file missing its own identity": the
reporting scripts never copied the block into their own output.** The recovery is one file read,
not a measurement — `scripts/t09_recover_checkpoint_config.py` prints every envelope-deciding gate
and, with `--patch`, writes a `recovered_config` block into the r11 artifacts so the next reader
does not have to ask:

```
python scripts/t09_recover_checkpoint_config.py runs/pilot/model_exp_2400.pt \
    --patch reports/r11_starmap_layout_modes.json \
    --patch reports/r11_resample_grid.json --patch reports/r11_resample_grid_umap.json \
    --patch reports/r11_determinism_a.json --patch reports/r11_determinism_b.json
```

⚠️ **It must be run where the checkpoint lives** (the campaign machine); this container has no
`torch`. The reader's payload logic is exercised without `torch` by
`python scripts/t09_recover_checkpoint_config.py --self-check`, and the patcher was verified against
a copy of `r11_determinism_a.json`: it is idempotent, refuses to overwrite without `--force`, and
leaves every measured field byte-identical.

### 1b. 🚨 RAN 2026-09-09 — the arms do not match, and the answer is worse than either branch

I predicted that if the recovered gates matched A9's, four of the six §3 rows would become
recomputable. **They do not match, and the recovery closes none of them.** Every gate is a direct
read (`reports/r11_*.json` now carry a `recovered_config` block on every arm):

| gate | r11 checkpoint | A9's arms | applied at | admissible donor? |
|---|---|---|---|---|
| `expr_mode` / `prior_mode` / `decoder_mu_link` / `train_steps` / `layout_sampler` | zinb-flow / correlated / exp / 2400 / grid | same | fit / gen | ✅ agree |
| `layout_mode` | fitted `field`, generated per arm | `resample` | **generation-time, provably fit-invariant** | ✅ irrelevant to the fit |
| **`w_autocorr` / `w_profile` / `w_distribution`** | **0 / 0 / 0** | `0` arm: **0 / 0 / 0** | fit | ✅ agree with A9's `0` arm |
| 🚩 **`text_emb_mode`** | **`lookup`** | **`medcpt`** | fit | ❌ **differs** |
| 🚩 **`expr_pca_dim`** | **16** | **28** | fit | ❌ **differs** |

**Two fit-time gates apart, so A9's envelope is inadmissible for every r11-derived figure** — and
`text_emb_mode` is not a gate that can be waved through: §4.2a's own evidence is that on the
`deep_starmap` `text_emb_mode` gate the two arms' envelopes differ by up to **2.6x** and the worse
arm **alternates by metric**. Matching on the metric-aware weights alone is not matching.

✅ **What the recovery did settle, and it is not nothing.** Each flagged row moves from *"no
envelope, and we do not know whether one exists"* to *"no envelope, and here are exactly the two
gates that would have to agree"* — a determinate negative with a stated remedy, instead of an
open question. And it made the **within-r11** comparison firmer: all five arms are provably one
fit differing only in generation-time gates, one of which (`layout_mode`) is asserted fit-invariant
bitwise across all 96 tensors by `tests/test_select.py::test_layout_mode_does_not_enter_the_fit`.
R11's *ordering* is a clean within-fit contrast; what it lacks is an across-seed spread, which one
checkpoint can never supply.

🚨 **And it surfaced two things the envelope question was not looking for**, both in
`reports/advisor_report.md`: the six-metric table is **not the shipped configuration** (§5 there —
`lookup` is ablation A3 and `expr_pca_dim=16` is the pilot stand-in), and *"every absolute number in
this project was produced with [the metric-aware weights] active"* is **backwards** (§6a there —
`Config` ships them at 0.0 and A9's `05` arm is the only run in the corpus that had them on).

---

## 2. Recomputed — the figures where a legitimate envelope exists

### 2.1 `expr_mode`, tier-1 — the "4.6–5.3x" figure. **Verdict holds. Two of three numbers fall, one rises.**

The quoted 4.6–5.3x is from `reports/t09_audit_expr_mode.md`: **one seed**, and on the
**pre-coordinate-frame-fix** code state (its `marker_field_r` is negative). Its successor is the
three-seed run that produced the envelope files themselves — same instrument, same design, same two
arms, so the envelope is measured on exactly the arms being compared. Margins are `cross-mix` −
`zinb-flow`, median across seeds per §4.6, with the mean beside it because the record's own
recomputation used the mean:

| metric | margin (median) | margin (mean) | env `cross-mix` | env `zinb-flow` | shared | **vs shared** | vs 0.0335 | quoted |
|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.1313 | +0.1242 | 0.0054 | **0.0574** | 0.0574 | **2.29x** | 3.92x | 5.2x |
| `gearys_pearson` | +0.1303 | +0.1311 | 0.0027 | **0.0595** | 0.0595 | **2.19x** | 3.89x | 5.3x |
| `umap_mixing` | +0.1415 | +0.1464 | 0.0068 | **0.0190** | 0.0190 | **7.45x** | 4.22x | 4.6x |
| `marker_field_r` | +0.0201 | +0.0206 | 0.0049 | **0.0148** | 0.0148 | **1.35x** | 0.60x | — |
| `marker_depth_r` | +0.0569 | +0.0711 | 0.0084 | **0.0472** | 0.0472 | **1.21x** | 1.70x | — |
| `celltype_localization` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **inert** | 0.00x | — |

**"4.6–5.3x on three" becomes "2.2x, 2.3x and 7.4x on three."** Copying still beats generating on
every live metric and the verdict is unchanged — but the two autocorrelation metrics carry **less
than half** the weight they were quoted at, and `umap_mixing` carries **60% more**. The
direction of the error is opposite on different metrics, which is §4.2a's whole point.

⚠️ **Estimator note.** The record's recomputation (PROGRESS, R10) used the **mean** across seeds
where §4.6 specifies the median. The two agree here to within 0.3x and no verdict turns on it —
`marker_field_r` 1.35x/1.39x and `marker_depth_r` 1.21x/1.51x are the widest gaps and both stay on
the same side of 1x. Recorded so the estimator is a stated choice rather than an unstated one.

### 2.2 Reconstruction headroom, `specs/10` §0a. **Verdict holds; every multiple falls.**

`scripts/t09_depth_ceiling.py` *"mirrors `section_scores` exactly"* and uses `selection_folds`, so
these are instrument A — the same instrument and design as the envelope files. Metric is
`marker_depth_r` throughout. Tier-1 envelope 0.0472; `deep_starmap` envelope 0.0427 (shared, the
`lookup` arm carrying it).

| figure | value | quoted | **corrected** |
|---|---|---|---|
| tier-1 headroom over the best copy | +0.1551 | 4.6x | **3.3x** |
| `deep_starmap` headroom over the best copy | +0.0160 | 0.5x | **0.37x** |
| `deep_starmap` headroom over the **operational** copy (R14) | +0.0855 | 2.6x | **2.0x** |
| R14's donor-rule cost on `deep_starmap` | 0.1160 | 3.5x | **2.7x** |

Every conclusion §0a draws survives: tier-1 still has real room, `deep_starmap` is still saturated
against an oracle copier and still not against ours, and the reversal between the two datasets is
untouched. What changes is that **tier-1's room is 3.3 envelopes, not 4.6** — the headline number in
§0a's consequence 2.

⚠️ Dividing a model-free headroom by a *fit* envelope is a scale comparison, not a clearance: it
asks whether the room exceeds the noise a fitted method would carry. The referent's own envelope is
zero (§4.2b, fixed layout), so the fitted arm's envelope is the right one — but the quantity is not
a margin between arms and should not be read as one.

### 2.3 The metric-aware weights' selection margins. **Verdict holds; the safety factor shrinks.**

`0.0052 / 0.0101 / 0.0018` against 0.0335 is a **fixture** margin against a **fixture** envelope, so
only defect (1) applies. The record says the three sit *"on the autocorrelation metrics"*, and there
are only two of those, so the third metric is unidentified — and the table that produced them is
`§8b`, **measured but unrecoverable**, so the assignment cannot be checked against an artifact.

Under the fixture's own per-metric envelopes:

| margin | vs 0.0335 | vs `morans` (0.0160) | vs `gearys` (0.0335) | vs the smallest, `ct_loc` (0.0068) |
|---|---|---|---|---|
| 0.0052 | 0.16x | 0.33x | 0.16x | 0.76x |
| 0.0101 | 0.30x | **0.63x** | 0.30x | **1.49x — outside** |
| 0.0018 | 0.05x | 0.11x | 0.05x | 0.26x |

**The verdict does not move: on the autocorrelation reading all three margins remain inside their
own envelopes, so the weights are still established by nothing.** What moves is the stated safety
factor — *"inside it by factors of 3 to 19"* becomes **"by factors of 1.6 to 19"**, because
`morans_pearson`'s real fixture envelope is 0.0160, not 0.0335. And the correction is
assignment-dependent: had one of the three been `celltype_localization`, the 0.0101 margin would
**clear** its envelope at 1.49x. That is not a reason to re-read the verdict — it is a reason the
per-metric assignment should have been recorded.

### 2.4 🚨 The fixture `layout_mode` tie-break. **This verdict changes.**

`envelope_synthetic.md` and `specs/10` §4.5b both state that `resample` beat `hybrid` by **0.0344
against a 0.0335 envelope — "a margin 1.03x the noise floor, which is not a decision"**, and that
this is *why `hybrid` shipped at all*. Both sides of that comparison are pooled: a
maximum-over-metrics margin against a maximum-over-metrics envelope.

Per metric, from the same nine fits (`winner` = `resample`, `closest_rival` = `hybrid`, margin =
resample − hybrid, median over three seeds):

| metric | per-seed margins | median | fixture env | **vs own env** | vs 0.0335 |
|---|---|---|---|---|---|
| `morans_pearson` | +0.0084 / +0.0062 / +0.0061 | +0.0062 | 0.0160 | 0.39x | 0.19x |
| `gearys_pearson` | −0.0114 / +0.0254 / +0.0030 | +0.0030 | 0.0335 | 0.09x | 0.09x |
| **`umap_mixing`** | +0.0300 / +0.0401 / +0.0411 | **+0.0401** | 0.0115 | **3.49x → `resample`** | 1.20x |
| `marker_field_r` | +0.0101 / +0.0046 / +0.0041 | +0.0046 | 0.0162 | 0.28x | 0.14x |
| `marker_depth_r` | +0.0079 / +0.0051 / −0.0320 | +0.0051 | 0.0299 | 0.17x | 0.15x |
| **`celltype_localization`** | −0.0235 / −0.0167 / −0.0193 | **−0.0193** | 0.0068 | **2.84x → `hybrid`** | 0.58x |

**The fixture gate was not "inside the noise". Two of six metrics clear their own envelopes with
3/3 sign agreement, and they point in opposite directions** — `umap_mixing` to `resample` at 3.5x,
`celltype_localization` to `hybrid` at 2.8x. The pooled reading collapsed a genuine two-metric
disagreement into a single number that landed 1.03x from a bar 5x too strict for one of them.

Three things this does and does not do:

* **It does not change what ships.** `layout_mode=resample` was decided on real data (R11), far
  outside any envelope, and this is the fixture.
* **It corrects the account of why the fixture was uninformative.** The recorded account is
  *"underpowered, not wrong"* — a margin too small to read. The per-metric reading says the fixture
  was **not underpowered on this gate at all**; it separated on two metrics and they disagreed. That
  is a different failure and a more interesting one, and it is consistent with the later grid-sampler
  re-run returning *"a different winner for every seed"*.
* ⚠️ **These nine fits predate the grid sampler** (R11, 2026-08-24), so both arms are on the biased
  rejection sampler. The 0.0344-vs-0.0335 claim rests on exactly the same nine fits, so the
  correction is like-for-like — but neither number describes the shipped sampler.

---

## 3. 🚩 Cannot be recomputed — flagged, not substituted

🚩 **STATUS AFTER THE 2026-09-09 RECOVERY: still six. None closed.** The checkpoint recovery was
expected to close four of these; it closed none, because the r11 arm differs from A9's on
`text_emb_mode` and `expr_pca_dim` (§1b). What changed is the *reason* each row is flagged — from
"no candidate donor is known" to "the one candidate donor is measurably the wrong arm" — and rows
1–4 now have a stated, costed remedy instead of an open question. Rows 5–6 never had a candidate.

For each: the metric, why no admissible envelope exists, and the one measurement that would supply
one. **None of these has been given a substitute envelope**, which is the whole point of the
exercise.

| figure | where | metric / instrument | why it cannot be recomputed | what would fix it |
|---|---|---|---|---|
| `field` **3.5x**, `hybrid` **3.2x**, `resample` *"inside"* below the copy floor | close-out §3; advisor §3; `specs/10` §4.5b | `paper_celltype_localization`, instrument B, **one seed** | The `field`/`hybrid` arms have **no across-seed spread at all** — `r11_starmap_layout_modes.json` is `seed: 1` for all five arms. A9 supplies an envelope for `resample` only, on an arm whose `text_emb_mode` is unrecorded | three seeds of the layout-mode comparison on instrument B. ⚠️ **Re-using A9's is now ruled out by measurement** (§1b), not merely unverified |
| `marker_field_r` **7.4x** below its copy floor | close-out §8.6 | `paper_marker_field_r`, instrument B, one seed | same: one seed, and the arm is the **superseded `hybrid` pilot row** the advisor already withdraws (§5). A9's 0.0596 is the right shape and the wrong arm | as above |
| boundary-vs-interior gap **0.69x** → **BOUNDARY ELIMINATED** | close-out §8.6; advisor §7b | `paper_marker_field_r`, instrument B, one seed | ⚠️ **the strongest case for flagging, and the reason is that the two candidate divisors disagree.** `t10_marker_field_boundary.json` hard-codes `"envelope": 0.0335`, giving 0.69x. The same −0.0231 gap is **1.56x** against instrument A's `marker_field_r` envelope (0.0148) — *outside* — and **0.39x** against A9's instrument-B `paper_marker_field_r` (0.0596) — *inside*. Neither is admissible: A is the wrong instrument, B is the wrong arm and is fold-aggregated where the effect is **per section** (§4.2d (b) ≥ (a)). **A verdict that flips with the choice of divisor is not a verdict** | three seeds of the shipped arm, read per section |
| `gene_mean_spearman` **0.0033 off its copy floor, "inside the envelope"** | close-out §8.6 | `paper_gene_mean_spearman` | The metric is **not in `METRIC_NAMES`** and appears in no fixture or instrument-A envelope — the two-`SIX` defect (advisor §5). A7 and A9 do carry it on instrument B (0.0033–0.1193, 0.0301–0.0400) — spanning the claimed deficit — so "inside" is **not established either way** | extract it from A9 once r11's arm is identified |
| SpatialZ's **+0.015** localization lead *"inside twice the reproducibility envelope"* | `specs/10` §13.2 | `paper_celltype_localization`, **prior campaign** | rests on the **retired 0.0120** (§4.2, "that defect, not run-to-run variation"), and §13.1 establishes the two sides were scored by **different `evaluate_paper` revisions** | re-run the comparators (§3), which §13.1 already requires |
| the pilot layout swap at *"~29x the 0.0120 across-seed envelope"* | `specs/10` §4.5b | `paper_celltype_localization` | same retired figure; the row is also superseded by the grid-sampler re-measurement above it | as above |

---

## 4. Already compliant — checked, no change

Verified by recomputation from the artifacts, not assumed:

* **A7 (SEFL), all six ratios.** Recomputing from `t10_a7_{off,on}_s{1,2,3}.json` against the worse
  arm's own per-metric spread reproduces the record **exactly**: `umap_mixing` 4.68x,
  `marker_field_r` 3.76x, `gene_mean_spearman` 2.75x, `celltype_localization` 1.55x,
  `marker_depth_r` 1.31x, and the two unreadable autocorrelation metrics at 0.78x/0.83x with signs
  2/3. A7 measured its envelope inside its own run, on its own arms, on the instrument it was scored
  on. **It is the model of what §4.2a asks for**, and it is the only campaign in the project that
  did it.
* **A9.** Both §4.2d constructions, its own arms, its own instrument. `reports/t10_a9.md`'s envelope
  (a) of 0.1225 reproduces to four decimals.
* **Every zero-shot figure** — 2.52x, 2.08x, 1.96x, 0.22x, 2.7x, 3.3x, 0.12x, 0.24x, 0.27x, 0.51x,
  5.8x, 5.6x, 0.48x, 0.37x. These divide by the run's **own** shared envelopes (0.1273, 0.2015,
  0.0646, 0.0532, 0.0230, 0.0930), per pool and per dataset. They carry a different defect (§4.2g,
  a degenerate member setting the envelope) which is recorded and deliberately not applied
  retroactively. **Nothing here divides by 0.0335.**
* **`specs/10` §4.2d's `retention_top` / `share_shape_bounded` rows.** Own envelopes, both
  constructions.

⚠️ **A correction to the advisor report's own framing.** §8a-bis says *"the 0.0335 every ratio in
this report is divided by"*. That is not true — the zero-shot ratios, which are most of §2 and §6,
divide by their own measured envelopes. The claim is corrected in place: the defect is real and it
is narrower than stated.

---

## 5. What this leaves

**Nothing in the negative column moves.** Copying still beats generating, SEFL is still harmful, the
intensity-field layout still loses, the metric-aware weights are still established by nothing. Every
one of those is a within-configuration contrast, and the correction changes the *scale* a margin is
read against, not the sign.

**Two verdicts do move, and both are about the fixture and the instrument rather than the method:**

1. The fixture `layout_mode` gate was **not decided inside the noise** (§2.4). Two metrics separate
   and disagree.
2. **The boundary elimination is no longer supported at the strength it is stated** (§3). It is not
   refuted either — it is unreadable until an envelope exists on the right arm and the right
   aggregation level.

**And one thing is worse than it looked.** Every `paper_*` clearance in this project — the entire
headline table, R11, the marker deficits, the boundary work — has been read against a figure that is
pooled, off-dataset **and off-instrument**, where the real per-metric envelopes on that instrument
span 0.0009 to 0.2894. That is not a 4x error; on the autocorrelation metrics it is closer to 9x,
in the direction that **flatters** every clearance. The correction cannot be finished from artifacts
alone, and §3 is the list of what it is waiting on.

⚠️ **The cheapest thing was one file read. It was run on 2026-09-09 and it closed nothing** —
the r11 arm is `lookup` at `expr_pca_dim=16` where A9 is `medcpt` at 28, so A9's envelope is
inadmissible for every r11-derived figure (§1b). My prediction that it would close four of six was
wrong, and the way it was wrong is worth keeping: **I assumed the only gate in question was the one
the flag was about** — the metric-aware weights, which do match — and did not ask what *else* would
have to agree. That is §4.2a's own failure mode, committed while writing the correction for it.

**So §3's remaining cost is a measurement, not a lookup**: three seeds of the shipped arm on
instrument B, at `paper_2_4_6`, on the pinned evaluator. A9 shows the shape of that spend — six
fits, ~93 and ~57 minutes each.

🚨 **And the recovery's real yield was elsewhere.** Reading the gates out of the checkpoint showed
that the column this project calls **"v25 shipped" is not the shipped configuration** (`lookup` is
ablation A3; `expr_pca_dim=16` is the pilot stand-in) and that *"every absolute number was produced
with the metric-aware weights active"* is **backwards** (`Config` ships them at 0.0). Both are in
`reports/advisor_report.md` §5 and §6a. Neither is an envelope finding, and neither would have been
found without asking an envelope question — which is the argument for asking provenance questions
of artifacts that are not currently in dispute.

### 5a. The two divisors that would have come back — closed 2026-09-08

A correction that only rewrites prose leaves the defect in the code that generated it. Both
offenders are fixed, and neither now has a default:

* **`scripts/t10_marker_field_boundary.py`** hard-coded `ENVELOPE = 0.0335` and every branch of its
  pre-registered criterion compared against it. `--envelope` is now **required**, a numeric value
  additionally requires `--envelope-source` (§4.2g: an envelope whose owner is never named is one
  nobody checks), and `--envelope unavailable` returns a new **NOT READABLE** outcome instead of
  silently returning `boundary_eliminated` — a criterion that cannot fail must not be scored as
  passed (§4.2j). `reports/t10_marker_field_boundary.{md,json}` are regenerated under it: **every
  measured field is byte-identical** to the superseded pair, and only `envelope` (0.0335 → `null`)
  and `outcome` (`boundary_eliminated` → `not_readable`) moved.
* **`scripts/t09_audit_starmap.py`** printed a `vs 0.0335` column on every gate audit it has ever
  written. The column is now `vs own envelope`, populated only from `--envelopes` — per metric, with
  a required `source` naming the dataset, holdout, instrument, arms and seeds — and printing `—`
  otherwise. `vs fold spread`, which was always the honest column at n = 2, is unchanged.

The eleven `.md` artifacts carrying the old column are annotated in place rather than regenerated
(regenerating them would need fits). The seven **post-frame-fix** seed reports carry the per-metric
per-arm replacement, derivable from those same files; the four `t09_audit_*.md` are marked
**superseded outright** — they are one seed *and* pre-frame-fix, visible in their own tables as a
negative `marker_field_r`, so their margins are as retired as their divisor.

---

## 6. The rule this earns

§4.2a says *measure the envelope per metric, per arm, per gate*. Three additions this exercise
forces, each from a place the existing rule was followed and still gave the wrong number:

* **§4.2a-i — an envelope is per *instrument*.** Two scorers over the same six metric names are two
  quantities. This project has both, and on tier-1 STARmap their per-metric envelopes differ by
  **2.6x to 6.7x** across the five readable metrics (`morans` 0.0574 → 0.2894, `gearys` 0.0595 →
  0.2861, `umap_mixing` 0.0190 → 0.1281, `marker_field_r` 0.0148 → 0.0596, `marker_depth_r` 0.0472
  → 0.1225) — the arms differ too, so the split is not attributable to the instrument alone, which
  is exactly why neither set may stand in for the other. The prohibition existed in `specs/10` §5
  while the practice violated it everywhere.
* **§4.2a-ii — an artifact must record the configuration it describes.** `r11_starmap_layout_modes.json`
  is a landed, verified, reproducible file that does not say which arm it describes, so a correctly
  measured envelope cannot be matched to it **from the artifact**. §8's provenance tiers assume the
  problem is a missing file. This is a present file whose identity lives somewhere else — in the
  checkpoint it was scored from, which had it all along. That makes it cheaper to fix and easier to
  miss: nothing is absent, so nothing looks wrong.
* **§4.2a-iii — a per-metric envelope that has been measured must be quoted per metric.** R10
  measured one and said in its own report that it was the right thing to use. It was pooled anyway,
  for three weeks, across three documents. The failure was not measurement and not reasoning; it was
  that a pooled scalar is easier to carry than a table.
* ⚠️ **And one that is not about envelopes at all: an experiment's record states what it
  *measured*, not only what it *concluded*.** A9's instrument-B envelope existed for a month and
  nobody looked, because the run was filed under its verdict — UNINFORMATIVE about the metric-aware
  weights — while being, at the same time, three seeds of two arms of the shipped-shape
  configuration on the pinned evaluator. "The project has no real-data envelope on that instrument"
  and "A9 measured one" were both true and only the first was written down. **This is a retrieval
  failure, not a measurement one**, and it is the kind that recurs: a null result's *measurements*
  stay valid long after its *verdict* stops being interesting, and indexing a run by the question it
  was designed to answer makes them unfindable to anyone asking a different one. §4.2f's shape one
  level up — not a diagnostic firing where nobody looks, but a measurement filed where nobody will
  think to look.
