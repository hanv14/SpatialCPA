# The ladder — is the 0.426 gap closeable within this architecture?

**Written before the statistic is computed.** Every arm below is **already drawn**; only the
correlation is new. Criteria, bands and the two stated asymmetries are fixed here.

---

## 1. The question

v25 scores **0.5574** on `paper_morans_pearson`; the **model-free copy floor** (`flanking_copy`,
tier-1, three seeds, `advisor_report.md` §5.1) scores **0.9836**. That is a **0.426** gap against a
baseline that copies a neighbouring real section. Q1.5 closed the emission programme —
`reports/emission_repair_options.md` §10 — by showing a complete emission repair *widens* it.

So: **is the gap closeable by anything inside this architecture?** The ladder answers it by walking
the scored statistic from the model up to the best case the architecture class admits.

---

## 2. The ladder, and what each rung isolates

All four arms are drawn at the **real section's own cells**, so `pred_xy == gt_xy`. Correlation
against `REF real counts`' per-gene `I` vector, by the same construction Q1.5 used
(`evaluate_paper.py::_agreement`, ranked counts, `k = 10`, NaN dropped pairwise).

| rung | arm | what it holds at the truth | isolates |
|---|---|---|---|
| **ceiling** | `A1c` — `Poisson(mu_oracle)` | the mean field **and** a lean draw | the best a mean-field-plus-draw generator can do on this statistic |
| | `A1b` — `ZINB(mu_oracle, theta, pi)` | the mean field | ← minus ceiling = **what the emission costs** |
| | `A1a` — `ZINB(decode(h1), theta, pi)` | the **latent** | ← minus `A1b` = what the decode path costs |
| **where we are** | stage `4` — model latent, full decode | nothing | ← minus `A1a` = what the latent costs |

`A1b-t`, `A1b-p` and the permutation null `A1n` are reported too, since they cost nothing. `A1n`
is a null check: its per-gene `I` vector should be ~0 with near-zero variance, so its `r` should be
~0 or NaN. **If `A1n`'s `r` is materially non-zero, the ladder is not measuring what it claims** and
nothing else on it may be read.

### 2a. Two asymmetries, stated in advance

1. **The `A1*` rungs have an advantage stage 4 does not.** They are drawn at the *same cells* as the
   reference, so both sides share one graph. bench3 — and stage 4 — compare a generated cell set
   against the real one, each on its own graph ("alignment-free"). So every `A1*` figure is an
   **upper bound** relative to how bench3 would score a method that emitted those counts at its own
   positions. **A low `A1*` rung is therefore conclusive; a high one is permissive.**
2. **The last rung conflates two changes.** `A1a → 4` changes the latent **and** the cell set. They
   cannot be separated: the model's `mu` exists only at generated positions. The cell-set change is
   the one bench3 makes too, so the rung is the realistic one — but it is not a clean latent
   contrast and is not read as one.

`mu_oracle` is a kNN mean of the real counts, so it is a *smoothed estimate* of the tissue's mean
field, not the field itself. Smoothing distorts the per-gene ordering in an unmeasured direction,
which is a second reason `A1c` is not simply "the copy floor for this architecture".

---

## 3. The pre-registered readings

### 3a. Is there a ceiling at all? — `r(A1c)`

| band | criterion | reading |
|---|---|---|
| **INSTRUMENT REACHES** | `r(A1c) >= 0.90` | a generator with the tissue's own mean field and a lean draw approaches the copy floor. The 0.426 gap **is** a model defect |
| **INSTRUMENT CANNOT** | `r(A1c) <= 0.70` | **even holding the mean field at the tissue's own, this architecture class cannot reproduce the tissue's per-gene `I` ordering.** The gap is not a model defect and no repair inside the architecture reaches the floor |
| PARTIAL | in between | |

`0.90` rather than `0.9836` because `A1c` pays for the smoothing distortion and for being one draw,
neither of which a copy pays. It is deliberately generous to the architecture: **falling below 0.70
with that generosity is the conclusive direction.**

### 3b. What does a perfect latent buy? — `d r = r(A1a) - r(stage 4)`

Bands **reused unchanged from `q15_preregistration.md` §5(a)** — same quantity, same scale, so no new
scale is invented after the fact:

| band | criterion | reading |
|---|---|---|
| **MOVES IT** | `d r >= +0.15` | the latent is the route to the ceiling |
| **DOES NOT** | `d r <= +0.05` | **a perfect latent does not move the scored metric.** No repair to the latent reaches the floor |
| PARTIAL | in between | |

By §2a(1) this is biased **upward** — `A1a` gets the same-cell advantage — so **DOES NOT is
conclusive and MOVES IT is permissive.**

### 3c. The combined verdict, and it is the answer the paper needs

Read `r(A1c)` first.

| `r(A1c)` | `d r` | **verdict** |
|---|---|---|
| `<= 0.70` | — | **NOT CLOSEABLE — and not a model defect.** The metric rewards something a mean-field-plus-draw generator does not produce. The finding is about the benchmark, not about v25 |
| `>= 0.90` | `<= +0.05` | **NOT CLOSEABLE BY THE LATENT.** A ceiling exists but the latent is not the route; and Q1.5 already showed the emission route makes tier-1 worse. What remains is the decode path |
| `>= 0.90` | `>= +0.15` | **CLOSEABLE, and the latent is the route.** The remaining work is the latent's gene-wise spatial fidelity |
| any PARTIAL on the deciding rung | | **no verdict on that rung**; report and escalate to three seeds |

**Both datasets.** Deep's arms are panel-restricted (32 of 1017), and Q1.5 showed the panel figure is
attenuated by range restriction, so an **all-1017-gene** pass runs for the arms at seed 1 and
**governs on deep**, exactly as in Q1.5. Tier-1's panel is all 28 genes and needs no second pass.

### 3d. Reported, not criteria

`A1c → A1b` (what the emission costs given a perfect mean field), `A1b → A1a` (what the decode path
costs), `spearman`, and `mae` at every rung — `mae` because the evaluator documents it as the
over-smoothing detector and it moved 3.5x where `pearson` moved 0.12 in Q1.5.

---

## 4. R1–R3 — what the correlation is made of

Free, same read. **Bands fixed here, before computing.**

`d` = each gene's **detection rate on the real section**, the fraction of cells with a non-zero count.

| # | measurement | band |
|---|---|---|
| **R1** | `corr(I_real, d)` and `corr(I_real, log mean count)` | `\|corr\| >= 0.70` → the tissue's own `I` ordering is **substantially a sparsity ordering** |
| **R2** | the same two on the model's stage-4 counts | reported; the comparison with R1 says whether the model's ordering is *more* sparsity-driven than the tissue's |
| **R3** | **partial correlation** `corr(I_4, I_real \| d)` | see below — **the decisive one** |

**R3's specification, fixed in advance so the control is not chosen after the answer.** Both `I`
vectors are residualised on `[1, d, d^2]` and the residuals correlated. Reported a second time with
`d` replaced by `log(mean count + 1e-9)`. **If the two specifications disagree by more than 0.15 in
the retained fraction, R3 is uninformative** and no reading is taken.

Retained fraction `f = partial_r / r(4)`:

| band | criterion | reading |
|---|---|---|
| **SPATIAL** | `f >= 0.60` | the correlation survives controlling for sparsity; it is measuring spatial fidelity |
| **SPARSITY** | `f <= 0.30` | most of `r(4)` is the model matching which genes are *sparse*, not which are *structured* |
| PARTIAL | in between | |

**Guard:** if `r(4) < 0.20` on a dataset, `f` is unstable and that dataset reads **uninformative**
regardless.

---

## 5. What would have to be true for this to fail

1. **`A1n`'s `r` is materially non-zero.** The null rung is not null, the construction is wrong, and
   nothing on the ladder is read. Checked first.
2. **`r(A1c)` lands between 0.70 and 0.90.** The deciding rung returns PARTIAL and the ladder does
   not answer its question in one pass. Reachable, and §3c says so rather than forcing a verdict.
3. **The two datasets disagree on `r(A1c)`.** Then "this architecture" is not one thing and the
   answer is dataset-dependent — which is itself the finding, given tier-1 and deep have already
   disagreed on the sign of the emission effect.
4. **R3's two control specifications disagree.** §4 says the reading is then not taken.
5. **Every rung comes back near the ceiling**, including stage 4 — which would mean the chain's
   reconstruction has stopped tracking the score, contradicting Q1.5's tier-1 validation
   (+0.5076 against 0.5574). That would invalidate the ladder and is checkable against that anchor.

Outcome 1 is the one that voids the exercise; outcomes 2 and 3 are real possibilities the bands are
written to survive.

---

## 6. Cost

**Free.** Every arm is already drawn inside `emission_ablation`; `summarise` already stores each
stage's per-gene ranked `I` vector. The panel ladder is arithmetic on vectors in hand, at all three
seeds. Deep's all-genes pass adds four blocked Moran's I passes at seed 1 — a couple of minutes.

No fit, no generation: the same `--load-model` reads that produced `a1_tier1.md` and `a1_deep.md`.
