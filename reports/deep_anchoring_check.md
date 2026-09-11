# Anchoring check — `deep_starmap` does NOT anchor

**Result: the reference numbers do not exist.** This is not "the chain disagrees with them"; there
is no published `deep_starmap` `paper_morans_pearson` for v25 or `flanking_copy` to disagree with.

## 1. What was searched

| where | result |
|---|---|
| `benchmark-pbya-v3/results_rescored/` | **does not exist**. `benchmark-pbya-v3/` carries `.gitignore`, `README.md`, `src/` and nothing else; results are gitignored and live only on the operator's machine |
| every `reports/*.json` carrying `paper_morans_pearson` | **23 files, all tier-1** — `r11_*` (headed "layout_mode on STARmap tier 1"), `t10_a7_*` and `t10_a9_*` (headed "tier-1 STARmap"), `t10_rescore_exp` |
| every `reports/*.json` mentioning `deep` | 12 files. The only ones carrying any Moran's correlation are `t09_tenv_deep*`, which report **`morans_pearson`** — the unprefixed internal-LOSO statistic from the text-embedding ablation, on *training* folds — not `paper_morans_pearson` on the held-out design |
| `reports/advisor_report.md` §5.1, the source of every reference score this campaign quotes | tier-1 only: `paper_morans_pearson` 0.5574, floor 0.9836, oracle 1.0000, referents from `r11_starmap_layout_modes.json` |
| `specs/10` line 2435 | the comparator run including `oracle` and `flanking_copy` is scoped **"on tier 1"**, and notes those numbers are not in the repo. There is no deep equivalent |

## 2. The tier-1 anchoring is stronger than recorded — three-for-three, not one median

`reports/pilot.md` §3 carries the published `flanking_copy` per-section `morans_pearson`, from
`selftest`'s probes through the pinned evaluator:

| | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|
| **published** (`pilot.md` §3) | **0.9517** | **0.9913** | **0.9836** | 0.9836 |
| **the chain** (`a1_tier1_section_*`) | **0.9517** | **0.9913** | **0.9836** | 0.9836 |

Every section matches, not merely the median. `reports/scale_anchoring.md` claimed the weaker
version and is corrected.

## 3. What this settles — C4, without qualification

Tier-1's ladder **is** the published statistic. Deep's ladder is the chain's own reconstruction on
its own 1017-gene scope, with no published referent, and `reports/panel_regime.md` establishes that
the two datasets' `paper_morans_pearson` are **not the same quantity** — about half of deep's is
per-gene abundance and almost none of tier-1's is.

So an absolute gap of **0.146** on deep and **0.14** on tier-1 are not in the same units, and
setting them side by side is the cross-scope comparison this campaign has been caught by four times.

**The claim that the two datasets localise the irreducible remainder in different places is
withdrawn from the paper.** Not qualified — withdrawn. `reports/architecture_ceiling.md` §0 is
amended to state the tier-1 decomposition alone, in published units, with deep's figures kept
separately and never beside them.

**What survives, unaffected:** every within-tier-1 statement, because tier-1 is anchored — the
ceiling at 0.837 against 0.984 and the thirds (latent 0.139, mean field 0.138, irreducible 0.147).
And every within-deep *ordering*, which is internally consistent and is what C2 and C5 rest on:
the copy's retained fraction, `spatial_scramble` at the null, and `F3_copy`.

## 4. What would anchor deep, if it is ever wanted

One command on the operator's machine, no fits — `selftest`'s probes already go through the pinned
evaluator:

```
python -m bench3.selftest --dataset deep_starmap --holdout paper_2_4_6 \
    --probes oracle flanking_copy
```

and, for v25's own side, the existing `deep_starmap` fit re-scored through `bench3.evaluate_paper`
on the same holdout. If those produce per-section `paper_morans_pearson` matching the chain's
0.9869 / 0.9886 / 0.9798 and 0.6601 / 0.7306 / 0.7021, deep anchors and §3's withdrawal can be
reversed.

**It is not required for the paper.** The paper's claims are tier-1 claims (C1, C3, C4) plus two
that rest on within-deep orderings (C2, C5). Nothing is blocked. This is recorded so that the
withdrawal is a decision with a stated reversal condition, not an omission.
