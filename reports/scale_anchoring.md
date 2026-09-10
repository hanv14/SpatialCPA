# The chain reproduces bench3's published tier-1 scores

**This retires `reports/q15_preregistration.md` §2's caveat for tier-1.**

## 1. The match

bench3 forms `paper_morans_pearson` as the median over held-out sections 2/4/6 of the across-gene
correlation on all shared genes. On `starmap_visual_cortex` the panel **is** all 28 genes, so the
chain's per-section figure is that construction exactly, and its median is that number:

| | section_2 | section_4 | section_6 | **median** | **published** | Δ |
|---|---|---|---|---|---|---|
| `flanking_copy` | 0.9517 | 0.9913 | 0.9836 | **0.9836** | **0.9836** | **0.0000** |
| v25 (stage 4) | 0.5076 | 0.5600 | 0.5826 | **0.5600** | **0.5574** | 0.0026 |

The copy matching to four decimals is the **expected** result, not a lucky one — the probe is
deterministic (the nearest training section, emitted verbatim), so an exact match is what a correct
reconstruction of `evaluate_paper.py::_agreement` must produce, and getting it is the strongest
available check that `morans_agreement` is that function. v25's 0.0026 is the generation seed:
bench3 ran its own, and the chain's three-generation spread on these sections is sd 0.011–0.018.

## 2. What it retires

`q15_preregistration.md` §2 has qualified every number in this campaign:

> Comparable **between stages**, not to a published `paper_*` number.

**On tier-1 that caveat is withdrawn.** The ladder, the ceiling and the flanking test are on the
scale the paper reports, and `reports/architecture_ceiling.md` §0 states the ceiling in it: a
perfect mean field with the model's own emission reaches **0.837** against a copy at **0.984**.

## 3. What it does NOT retire

- **`deep_starmap` is not anchored.** Its panel is 32 of 1017 genes and its all-genes scope is the
  chain's own construction; no published `paper_*` figure for it has been read from bench3. Deep's
  numbers stay comparable between stages and to each other, and not to a published score.
- **A single generation seed is still a single generation.** The per-section v25 figures carry the
  spread the stage-4 block prints; only the median over three sections is being compared here.
- **Nothing about the other `paper_*` metrics.** This anchors `paper_morans_pearson` and no other
  member of the scored tuple.

## 4. Why it matters more than a validation usually would

The whole campaign has measured a statistic it reconstructed from source and could not check against
the thing it was reconstructing. Every ceiling, every rung and every gap has carried that
uncertainty. It is now closed on the dataset the paper's headline number comes from, and the closure
was produced by a test built for a different purpose — the `flanking_copy` probe was added to
interrogate the benchmark, and it validated the instrument instead.
