# 8. Related work

## 8.1 The competitor this work is measured against

**SpatialZ** is the published method whose protocol this benchmark follows: the STARmap tier-1
build, the `paper_2_4_6` holdout, the six-metric table and the leakage machinery are all its
protocol, taken unmodified so that "measured under the SpatialZ STARmap protocol" stays true
(`specs/10` §1). It generates a section between two flanking slices in two stages — an
optimal-transport layout that pulls a point cloud toward both flanks, then an expression step that
copies from real donor cells of the same cell type.

**It beats this method on axis-aligned reconstruction.** We state that plainly, and we do not need a
head-to-head to know it: SpatialZ's expression step copies real counts from real cells, and §6.1
measures where copying sits. On tier-1 `paper_morans_pearson` an optimal copier of the flanking
sections scores **0.9836** and our method scores **0.5574**, with the architecture ceiling — what the
decoder reaches given oracle means — at **0.8369**. Any method that emits real donor counts inherits
the copy level; a method that emits from a ZINB decoder cannot, and no route inside this architecture
closes the 0.8369 → 0.9836 gap. **This is consistent with §6 because it is §6**: the deficit was
volunteered there as architectural rather than as a tuning failure, and a copier outscoring us is the
prediction that framing makes, not an embarrassment to it.

What follows from that is the shape of the contribution rather than a ranking. A copy has **no
definition at an arbitrary orientation** — SpatialZ's layout interpolates between two flanking slices
and its expression step draws from the nearest same-type cells, and neither construction has a
meaning for a plane that spans the stack. The claim this paper makes is that section generation
becomes well-defined there, and §3–§5 are about what it costs to check that. It is not that
reconstruction is better on-axis. It is not.

## 8.2 Two mechanisms in the competitor, read from its source

Both are properties of `reference/SpatialZ.py` at its published defaults (`syn_mode='default'`,
`k_sam=3`), stated because they bear on what a copy-based construction can and cannot preserve. They
are read from the code and, for the first, measured on our own fixture with a model-free test.

**1. Expression is drawn independently per gene, which costs gene–gene covariance.** In
`synthesize_gene_expression`, each query cell's expression vector is assembled in a loop over genes,
and **each gene independently samples its own donor cell** from among `k_sam` candidates weighted by
microenvironment similarity. A generated cell is therefore a chimera of up to `k_sam` real cells, one
draw per gene, rather than one real cell's profile.

We measure the cost of exactly that, with the positional confound removed: hold the donors, the
positions and the cells fixed and vary **only** whether the draw is one donor per *cell* or one per
*gene* (`tests/test_expression.py::test_per_gene_independence_destroys_covariance`; the numbers are
in `reports/benchmark.md` and `specs/06` §5). Mean off-diagonal gene–gene correlation retained,
against the real section's:

| donors mixed | default holdout | `consecutive-3` |
|---|---|---|
| 1 (a verbatim copy) | 0.978 | 0.955 |
| 2 | 0.920 | 0.818 |
| **3 — the published `k_sam`** | **0.897** | **0.783** |
| 5 | 0.884 | 0.714 |
| 10 | 0.844 | — |

**At its published setting the construction loses 10% of gene–gene covariance at the standard gap
and 22% at the wider one.** The loss grows monotonically with the number of donors mixed and with the
holdout gap. This is a mechanism result and needs no trained model, which is why we report it: it is
a statement about two samplers, not about two fitted systems.

**It is also a limit on our own claim, and we mark it as one.** Our decoder emits every gene of a
cell from one shared latent, so the mechanism above cannot occur — but on the direct measurement our
model's covariance error is **9.316** Frobenius against the independent-donor baseline's **7.783**, a
verbatim copy's 6.743 and an achievable ceiling of 5.601. **We lose that comparison.** The covariance
argument is therefore a claim about the *mechanism* — per-gene independent draws destroy covariance,
a shared latent cannot — and **not** a claim that our expression head reproduces covariance better in
practice. It does not (`specs/06`, where the original criterion is kept as a strict `xfail` so the
shortfall stays on the record).

**2. Cell-type assignment carries no z-proximity term, though the layout does.** `alpha` is the
parameter that says where between the two flanks the generated plane sits. It enters the layout in
two places — the sliced-Wasserstein loss is `alpha · d(·, slice₁) + (1 − alpha) · d(·, slice₂)`, and
the initial point cloud is split `alpha : 1 − alpha` between the two flanks' coordinates. It then
**does not enter the cell-type step at all.**

That step takes each flank's `k_neighbors` nearest cells to the generated position, weights them by
`1 / (in-plane distance + ε)`, concatenates the two flanks' weights and takes the dominant type. The
weights are in-plane distances only; the two flanks enter on equal terms; no distance along z
appears. At the published default `k_neighbors = 1` this reduces to: **take the type of whichever
flank's single nearest cell is closer in the plane** — a comparison between two cells that are, by
construction, different distances away along z, made as though they were not.

So a plane placed a tenth of the way between two sections has its *coordinates* pulled toward the
near slice and its *cell types* decided by an in-plane comparison that does not know which slice is
near. We report this as a reading of the source, not as a measured defect: **we have not run the
ablation that would say what it costs**, and it could plausibly be small on an evenly spaced stack
where the generated plane is usually near the midpoint and the two flanks are near-symmetric — which
is also the regime the published protocol evaluates. It is stated because it is the kind of thing a
continuous formulation makes unnecessary to decide: z-proximity is not a term to remember to add when
position is a coordinate the field is queried at.

*(A third observation, smaller: the per-gene draw calls the global `numpy` random state rather than a
seeded generator, so two runs of the expression step are not reproducible without setting a process-
wide seed. We note it only because determinism is a stated convention of this work.)*

## 8.3 ⛔ What is missing, and what it would take

**The six-method tier-1 table does not exist in this repository, and this paper does not contain
one.** `specs/10` §12 specifies it — SpatialZ, FEAST, isoST, v20, plus `flanking_copy` and `oracle`,
all on the pinned evaluator under `paper_2_4_6` — and `specs/10` step 5 lists that run as **required
and not yet performed**, for the reason that the pre-existing comparator numbers cannot be reused.

They cannot be reused for three reasons, all established before this paper was drafted
(`reports/v20_v21_evidence_audit.md` §3, `reports/pilot.md` §44, `reports/spatialz_claim_struck.md`):

1. **Cross-instrument.** The published comparator numbers were produced by earlier `evaluate_paper`
   revisions; every number in this paper comes from the content-hash-pinned one. Putting them in one
   table is the comparison the benchmark's own rules forbid. SpatialZ's STARmap rows carry **0 of 3**
   of the newer evaluator's columns against v20's 3 of 3.
2. **No floor and no ceiling.** The published rows quote neither `flanking_copy` nor `oracle`, and a
   score on this metric means nothing until it sits between them — which is the rule §6 is built on.
3. **A claim of ours was struck on exactly this.** "v20 beats SpatialZ 5 of 6" was quoted through
   much of this campaign and **could not be located in the record at all**; it is withdrawn in full
   (`reports/spatialz_claim_struck.md`). We mention our own withdrawn claim here rather than in §7
   because this is the section where a reader would otherwise expect to meet it.

**What this paper claims about the competitor is therefore limited to what is sourced above**: the
two mechanisms, read from its code, one of them measured on our fixture with a model-free test; and
that it beats us on reconstruction, which follows from the copy floor in §6.1 without a head-to-head.
**We make no claim about relative performance on any metric**, in either direction.

**What would close it** is one run — the comparators on tier-1, on the pinned evaluator, with the
floor and ceiling columns attached — and it would be a *corroboration* rather than a dependency:
under §6.1's framing it would be expected to show every method near the copy floor, which is itself
the methodological finding. Nothing in §3, §4 or §5 rests on it.
