# M3 — the reallocation test: pre-registration

**Written before either arm runs.** The floor's selection rule, the four outcomes, the fidelity
condition and the null-experiment guard are fixed here and are not to be adjusted once the numbers
exist. §5 states what would have to be true for this to fail, per `specs/10` §4.2's closing rule —
which this project has now been caught by twice (gate 2, and R12 read by its name).

---

## 1. The hypothesis, and why it is a *reallocation* and not an increase

A1 measured, model-free, on `deep_starmap`:

| | model | tissue |
|---|---|---|
| `sd(log mu)` implied by the counts (total between-cell spread) | **1.4131** | **1.3699** |
| `sd(log mu)` of the decoded mean (the structured part) | **0.7118** | 1.0994 – 1.3699 |
| count-level structured share | **10.4 %** | **>= 42.5 %** |

**The variance is already there.** The model's total is slightly *larger* than the tissue's, and
roughly 90 % of it sits in `theta` and `pi`, which are spatially independent and therefore dilute
Moran's `I`, instead of in `mu`, which is not. The ZINB likelihood is close to indifferent to that
split at the margin: a narrow `mu` with a wide `theta` fits the marginal counts about as well as
the reverse, and only the reverse carries spatial structure.

**Hypothesis:** flooring `theta` removes the cheap half of the trade, and the likelihood — still
obliged to explain the same counts — puts the variance into `mu`.

If true, §10's two halves are **one** move rather than two, and half 1 delivers half 2 for free.

**Why a floor and not `decoder_theta_mode="moment_matched"`.** The moment-matched estimator came
back **7.5x smaller** than the learned head, so fixing `theta` to it pulls dispersion *up* and
deepens the trade — the failure `reports/emission_repair_options.md` §5 names for candidate A. A
floor can only move dispersion down. `Config.decoder_theta_floor` is a separate field from
`zinb_theta_min` on purpose: one is a numerical guard, the other is an experimental constraint that
enters the content hash and is reported.

---

## 2. The design, and the floor's selection rule

**Tier-1, one new fit.** Tier-1 because its fit converges (silent after step 360 against
`deep_starmap`'s alarm firing at step 2399), it costs ~1 h, and the baseline checkpoint already
exists at `runs/chain/shipped_tier1.pt`. Everything else is held: same seed, same steps, same link,
same panel, same sampler.

Tier-1 is the **insensitive** case — the decoder is already at 0.99x the tissue's lower bound
there, against 0.65x on deep — so a clear movement on tier-1 is strong evidence and no movement is
**not** yet fatal to the hypothesis on deep. That asymmetry is stated now so it cannot be claimed
afterwards.

**The floor is chosen by a rule, not by eye.** It is the **median learned `theta`** over
`(cell, gene)` pairs at the real section's cells on the run's panel, read from the baseline
checkpoint by `t10_chain_diagnostic.py --report-theta --load-model runs/chain/shipped_tier1.pt`.
That is a deterministic read of an existing artifact — no draw, no choice — and it binds on
approximately half the pairs by construction, which rules out the null experiment in §4 before it
happens.

---

## 3. The outcomes — four, and all reachable

Applied by `scripts/t10_reallocation_table.py::m3_verdict`, in this order, not by whoever reads the
table. `d` is floored minus baseline.

| # | outcome | criterion |
|---|---|---|
| 0 | **NULL EXPERIMENT** | the floor binds on **< 5 %** of `(cell, gene)` pairs. Checked **first**, so a favourable-looking result that rests on a floor which did nothing cannot be reported as a win |
| 1 | **CAPACITY-LIMITED — the `mu` head, not the objective** | `d sd(log mu) < 0.05` |
| 2 | **MANUFACTURING UNCONDITIONED VARIANCE** | `sd(log mu)` rose but `d I(counts) <= 0` |
| 3 | **REFUSED — `I` rose while the benchmark fell** | any pinned `paper_*` metric fell by more than **0.02** |
| 4 | **REALLOCATION WORKS** | none of the above: `sd(log mu)` rose, `I(counts)` rose, and no `paper_*` metric fell past the tolerance |

### 3a. Outcome 1 is a **result**, not a failure of the experiment

Stated here so it cannot be read as a disappointment afterwards. If flooring `theta` leaves
`sd(log mu)` where it was, then **the objective is not the constraint — the `mu` head is**, and the
finding gets *stronger*, not weaker:

* it converts "the ZINB objective allows the trade" from a mechanism into a **bound**: the head
  cannot express the tissue's dynamic range even when the likelihood pushes it to;
* it explains, without further work, why `sd(log mu)` comes back at **0.6728** on tier-1 and
  **0.6725** on `deep_starmap` — 28 genes against 1017, agreeing to three decimals — and why a
  *faithful* latent decodes to **less** spread (0.6035) than the model's own over-smooth one;
* it makes every loss-side repair in §10 unnecessary to try, which is a saving of weeks, and points
  the remaining work at the head's parameterisation instead.

**Outcome 1 closes §10 rather than failing it.** No further loss-side candidate is costed if it
fires.

### 3b. The fidelity condition (M4), and it is a condition, not a later step

Every `paper_*` metric in `t10_rescore_saved.py`'s `METRICS` is reported **in the same table** as
`I`, at ground-truth-matched density, from the pinned evaluator. Outcome 3 refuses the repair on a
drop of more than 0.02 in any of them.

The reason is measured, not precautionary: **the model already beats real tissue on tier-1's `I`
(+0.5134 against +0.4635) with a latent 1.28x smoother than the tissue's.** `I` can be raised by
making the model worse, so `I` alone cannot certify this or any other repair, and a table that
reported `I` without the benchmark would hide exactly the failure this test is most likely to
produce.

---

## 4. Guards

* **Null-experiment guard** — the binding fraction is measured (`--report-theta`) and travels with
  the verdict in every artifact. Ordered first among the outcomes.
* **`--load-model` refuses `--theta-floor`** (except under `--report-theta`, where the floor is the
  candidate being measured against rather than fitted under), because the weights carry their own
  config and honouring a command-line override would report one arm over another's weights.
* **Matched density only.** The `paper_*` medians come from the rescorer's matched pass; the raw
  pass is not comparable between arms that emit different cell counts.
* **One `layout_mode` per comparison.** `t10_reallocation_table.py` refuses a rescore sidecar
  carrying more than one arm.

---

## 5. What would have to be true for this to fail

Four reachable ways, and two of them overturn a standing position:

1. **The floor binds on almost nothing** despite being set at the median — which would mean the
   `theta` distribution is far more concentrated than the percentile read suggests. Outcome 0, and
   the experiment returns nothing.
2. **`sd(log mu)` does not move.** Outcome 1 — and §3a says why that is the stronger finding.
3. **`sd(log mu)` moves and `I` does not.** Outcome 2: the floor pushed variance into `mu` without
   conditioning it on position. A9's failure in a new place, and the reason `sd(log mu)` alone
   cannot be the criterion.
4. **Both improve and the benchmark falls.** Outcome 3 — the case §3b exists for, and the most
   likely way a naive reading of this experiment would report a success that is not one.

There is no configuration of the results that returns "the hypothesis is confirmed" by default. The
one outcome that would be a *foregone* pass — `I` up, everything else unmeasured — is precisely what
the fidelity condition removes.

---

## 6. The commands

```bash
# 0. instrument check, seconds, no data
python scripts/t10_chain_diagnostic.py --self-check
python scripts/t10_reallocation_table.py --self-check

# 1. read the baseline's theta distribution; the floor is its MEDIAN (percentile 50)
python scripts/t10_chain_diagnostic.py \
    --dataset starmap_visual_cortex --load-model runs/chain/shipped_tier1.pt \
    --layout-sampler grid --report-theta \
    --out reports/m3_tier1_baseline.md

# 2. the floored refit — substitute the median from step 1 for <FLOOR>.  ~1 h
python scripts/t10_chain_diagnostic.py --steps 2400 \
    --dataset starmap_visual_cortex --decoder-mu-link exp --layout-sampler grid \
    --text-emb-mode medcpt --expr-pca-dim 32 --theta-floor <FLOOR> --report-theta \
    --out reports/m3_tier1_floored.md --save-model runs/chain/tier1_thetafloor.pt \
    --fit-checkpoint runs/chain/tier1_thetafloor.ckpt

# 3. the pinned benchmark on both arms, no refit
python scripts/t10_rescore_saved.py --model runs/chain/shipped_tier1.pt --modes resample \
    --workdir runs/m3/baseline --out reports/m3_bench_baseline.md
python scripts/t10_rescore_saved.py --model runs/chain/tier1_thetafloor.pt --modes resample \
    --workdir runs/m3/floored  --out reports/m3_bench_floored.md

# 4. the one table, with the verdict applied by code
python scripts/t10_reallocation_table.py \
    --baseline-chain reports/m3_tier1_baseline.json \
    --floored-chain  reports/m3_tier1_floored.json \
    --baseline-bench reports/m3_bench_baseline.json \
    --floored-bench  reports/m3_bench_floored.json \
    --out reports/m3_reallocation.md
```

⚠️ Step 2 passes `--expr-pca-dim 32` and lets `specs/10` §0's clamp narrow it to the panel width,
which is the rule rather than a hand-picked 28. Step 1 is a `--load-model` read and takes its config
from the checkpoint.

**Cost: one fit (~1 h) plus two rescores.** Nothing else in §10 is built, and running this does not
lift its suspension — M3 *is* a candidate for the gate that would.
