# The π/θ reversal — N5's verdict superseded on the scored metric only

**Scope of this note, stated first.** N5 measured which of `theta` and `pi` costs the
`A1c -> A1b` loss **in median Moran's I**, and concluded over-dispersion. That verdict stands
**on its own quantity** and is not withdrawn. What follows replaces it on the *scored* statistic,
`paper_morans_pearson`, and only there. The two answer different questions and disagree.

## 1. The numbers

`deep_starmap` section_4, all 1017 genes — the one scope the corrected null check leaves readable
(`reports/null_band_preregistration.md` §2a). Every arm at the real section's own cells.

| contrast | isolates | median I (N5) | **scored r** |
|---|---|---|---|
| `A1c -> A1b-t` (NB, π=0) | **θ** | +0.2289 — **69.7 %** of the total | +0.9699 → +0.9522 = **0.018** |
| `A1c -> A1b-p` (Poisson draw, then π) | **π** | +0.2095 — **63.8 %** of the total | +0.9699 → +0.8681 = **0.102** |
| `A1c -> A1b` (both) | total | +0.3284 | +0.9699 → +0.8242 = **0.146** |
| additivity gap | | 0.1100 (**not decomposable**) | 0.026 (**nearly additive**) |

**On the scored metric π carries ~70 % of the emission's cost and θ ~12 %.** On the median, θ
carries 70 % and π 64 %, and the two do not decompose at all.

`starmap_visual_cortex` has π ≈ 0 (`L_pi` = 0.0000 exactly), so its four oracle-`mu` rungs are flat
within 0.004 of each other. Its ladder fired the null check as written and is not read here; the
direction is consistent and nothing more is claimed from it.

## 2. Why they disagree, and which one the paper's question needs

Median `I` is a **level** statistic: it asks how autocorrelated a typical gene is. Over-dispersion
adds spatially independent noise to every gene at once, so it moves the level a long way — hence
θ's 70 % there.

`paper_morans_pearson` is an **ordering** statistic across genes. Adding the same relative noise to
every gene shifts the whole `I` vector without reordering it, so θ barely moves the correlation.
Zero-inflation does not act uniformly: `pi` is per gene and per cell, so it hits sparse genes far
harder than abundant ones and **reorders** the vector. That is the mechanism, and it predicts the
observed split rather than being fitted to it.

## 3. The rule this is an instance of

`specs/10` §4.2l: *a refutation must restate the claim in the claim's own terms before it counts.*
This is the same rule applied to my own work rather than to a claim of the record's. N5 was
pre-registered, correctly executed and correctly reported — on a median. The paper's question is a
correlation question, and N5 was never asked it.

## 4. What it costs the record

The entire θ programme — M3, `decoder_theta_floor`, the reallocation table, the 75th-percentile
escalation — was aimed at the half of the emission that is worth **0.018** on the scored statistic.
That work is not withdrawn (its numbers are correct on their own quantity) and it is not resumed.
It retroactively justifies closing §10 and not running Q2, for a better reason than the one given
at the time.

**No repair follows from this.** π being the expensive half does not make the emission a route to
the copy floor: `A1c` itself — a model-free Poisson draw from a perfect mean field, no θ and no π
at all — reaches +0.9699, and the architecture's own ceiling with a perfect mean field and its own
emission is **+0.8242**. Removing π entirely lands inside that ceiling, and the ceiling loses to a
copy. This note records a measurement, not a plan.
