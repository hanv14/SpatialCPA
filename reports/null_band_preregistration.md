# Pre-registration — the ladder's null check, replacing the 0.15 threshold

**Committed before the corrected check is implemented or run.** The threshold it replaces was
mine, it fired on tier-1, and I do not get to retune it against the numbers it fired on. This
document says what replaces it and why, so the replacement is judged on its construction rather
than on what it lets through.

## 1. What was wrong with the check being replaced

`ladder_block` compared the permutation arm `A1n`'s median `r` against an absolute **0.15**. Two
defects, both found by reading the run rather than by the check itself:

**1a. It read only the last `A1n` rung.** `null_r` was assigned inside the render loop, so with two
scopes the second overwrote the first. On `deep_starmap` the panel's `A1n` = **+0.2578** (seeds
−0.0106 .. +0.4465) was overwritten by the all-genes `A1n` = +0.0594 and **no alarm fired**. This is
`specs/10` §4.2f-i's family — an alarm that fails to fire — in a check built one round earlier for
that failure mode.

**1b. The threshold was blind to the number of genes.** Under the null, `atanh(r)` is approximately
`N(0, 1/sqrt(n-3))`, so the null's own scale is **0.373** at n=28, **0.349** at n=32 and **0.0615**
at n=1017. A single number cannot serve all three: at panel size 0.15 fires on noise roughly half
the time, and at all-genes it is six times too lax. The three observed nulls sit at 1.24, 1.34 and
1.89 of their own scale — none is a construction failure, and the check had no power to say so.

**Neither defect is repaired retroactively.** Tier-1's ladder fired the check as written and stays
unreadable; deep's panel ladder would have fired under 1a's repair and stays unreadable; deep's
all-genes ladder passes under both the written check and everything below, and is the only scope
the ladder's verdict was read on.

## 2. The replacement — two separate questions, two separate instruments

The old check conflated *"is the construction sound?"* with *"is this rung distinguishable from
that one?"* They need different instruments and are reported separately.

### 2a. Is the construction sound? — the null's centre

`A1n` permutes the real section's **cells** and recomputes per-gene `I`. A sound construction has
that arm's `r` against the tissue centred on **zero**. A single draw does not test a centre: the
same permutation is shared by every gene, so one unlucky permutation moves many genes together and
the per-draw spread is large (tier-1's three seeds spanned 0.26).

- `A1n` is drawn at **`n_null_seeds = 20`** seeds, at every scope the ladder renders. It is a
  permutation and a Moran's I computation — no model, no decoder — so this is cheap.
- Reported per scope: `mean_r`, `sd_r` across seeds, and the empirical 2.5 / 97.5 percentiles.
- **The alarm fires when `|mean_r| > 2 * sd_r / sqrt(20)`** — the null is *systematically* off
  zero, which is what a broken construction looks like. A wide but centred null is not a
  construction failure; it is a resolution limit, and §2b is where it belongs.
- When it fires, the scope's rungs may not be read, and the block says so in those words.
- **Per scope, not per table.** Each rendered scope carries its own `A1n` row, its own band and its
  own verdict. §5(1) of `ladder_preregistration.md` said "this ladder" without saying whether a
  ladder is the table or the scope; it means the scope, and that ambiguity was mine.

### 2b. Is this rung distinguishable from that one? — a paired gene bootstrap

The comparison the verdict leans on is `r(4) - r(A1a)`. Rungs share the reference vector and the
gene set, so their difference is far better determined than either absolute value, and the null's
spread is the wrong scale for it — using it would be over-conservative in the direction that
protects our own conclusion.

- **2000 replicates**, resampling **genes** with replacement, the same resampled index applied to
  every rung (paired), `numpy.random.default_rng(20260910)`.
- Reported for each adjacent rung pair and for `r(4) - r(A1a)`: the point estimate and the 2.5 /
  97.5 percentiles of the difference.
- A difference whose interval contains zero is **not distinguishable** and the block says so
  rather than printing an ordering.
- The bootstrap resamples genes, so it carries the across-gene sampling error only. Draw-to-draw
  error is the arms' own across-seed spread, already in the `across seeds` column (≤ 0.02 on every
  panel arm at 3 seeds); the two are reported side by side and never combined into one number.

### 2c. The all-genes scope runs at one seed, and that is stated where it is read

The all-genes pass currently draws each arm once. The panel's 3-seed spreads bound draw noise at
≤ 0.02, so this is a small term — but it is not zero and it is not measured at all-genes. The block
prints `n_seeds` per rung, and a rung at `n_seeds = 1` carries `—` in the spread column rather than
a fabricated range. **Stage 4 is the rung this matters most for**, because its variance includes
the layout and the flow sample, not only the emission draw; it is being re-run at three generation
seeds in the same round as this change.

## 3. What this cannot do

The bands above are the null's and the bootstrap's. Neither says the *estimator* is right — that
`morans_agreement` reproduces `evaluate_paper.py::_agreement` is a separate claim, checked by the
self-check's exact-match cases and not by anything here. And a centred null does not make a scope
readable if its rungs are indistinguishable from each other: at n=28 the bootstrap interval on any
rung difference is wide enough that tier-1 may simply be **underpowered**, which is a real outcome
and is to be reported in those words rather than as a number.

---

## §2a-bis — AMENDMENT: the seed count is a property of the construction, not of the caller

**Added after the first run, before the second. What it fixes was a gap in this document, and the
number it supersedes is named.**

Three blocks in `scripts/t10_chain_diagnostic.py` measure the **same construction** — permute the
real section's cells, recompute per-gene `I`, correlate against the tissue's own:

1. the ladder's `A1n` rung;
2. this document's §2a null check;
3. `flanking_copy_preregistration.md` §5c's `spatial_scramble` positive control.

§2a fixed a seed count for (2) and said nothing about (1) or (3). They inherited whatever the
caller passed: (1) took `--ablation-seed`, three seeds; (3) took a `[:5]` slice that appears
nowhere in any pre-registration and was mine. The first run therefore reported one quantity three
times at three seed counts:

| | tier-1, 28 genes | deep panel, 32 genes |
|---|---|---|
| `A1n` rung, **3** seeds | −0.2472 | +0.2578 |
| null check, **20** seeds | **−0.0080** | **+0.0382** |
| `spatial_scramble`, **5** seeds | +0.2634 | +0.0214 |

Read as three findings this says the construction is broken, the null is not null, and §6's
prediction 1 failed. Read as one quantity it says the across-permutation spread at n≈30 is ~0.2 and
**twenty seeds is the only one of the three that can see the centre**.

**The rule.** *A statistic's seed count belongs to the statistic, not to whichever flag reaches it.*
All three now run at `--null-seed`'s count, default 20. Where a block reports the quantity, it
carries the 20-seed figure and says which other blocks are the same construction.

**What is superseded.** Tier-1's `spatial_scramble` = **+0.2634** is withdrawn as an estimate of its
own quantity — not because the number is favourable or unfavourable, but because the identical
construction at 20 seeds reads −0.0080 in the same report. §6's prediction 1 is **confirmed** where
it was measured at a seed count able to test it, and tier-1's 5-seed figure is not evidence against
it. Deep's +0.0214 at 5 seeds stands but is re-run at 20 for the same reason.

## §2a-ter — F3's stratum width must leave at least two strata

Same class, found in the same run. `stratified_relabel_r` accepted any width. At tier-1's 28 genes
the pre-registered widths 25 and 50 put the whole panel in **one** stratum, so the "abundance-matched
relabelling" was a full permutation — the null arm wearing F3's label — and printed **+0.0469** and
**−0.0314** as measurements. Both are withdrawn.

**The rule.** A width `w` is refused unless `n_genes >= 2 * w`, and the refusal is rendered in the
table with its reason rather than as a number or a blank. `flanking_copy_preregistration.md` §5a's
requirement that the reading be stable across three widths therefore reads, at small `n`, as *"F3
has no usable width here"* — which is an outcome, and is not evidence in either direction.
