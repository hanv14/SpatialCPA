# 8. Related work

## 8.1 The competitor this work is measured against

**SpatialZ** is the published method whose protocol this benchmark follows: the STARmap tier-1
build, the `paper_2_4_6` holdout, the six-metric table and the leakage machinery are all its
protocol, taken unmodified so that "measured under the SpatialZ STARmap protocol" stays true
(`specs/10` §1). It generates a section between two flanking slices in two stages — an
optimal-transport layout that pulls a point cloud toward both flanks, then an expression step that
copies from real donor cells of the same cell type.

**It beats this method on axis-aligned reconstruction**, on five of the six tier-1 metrics, and §8.2
is the table. We state it plainly because it is what the measurement says and because §6 predicts it:
SpatialZ's expression step copies real counts from real cells, and §6.1 is the argument that copying
is near the achievable maximum on this protocol while our decoder cannot reach it. A copier
outscoring us is the prediction that framing makes, not an embarrassment to it.

What follows is the shape of the contribution rather than a ranking. A copy has **no definition at
an arbitrary orientation** — SpatialZ's layout interpolates between two flanking slices and its
expression step draws from the nearest same-type cells, and neither construction has a meaning for a
plane that spans the stack. The claim this paper makes is that section generation becomes
well-defined there, and §3–§5 are about what it costs to check that. It is not that reconstruction is
better on-axis. It is not.

## 8.2 The tier-1 table, against the floor and the ceiling

Tier-1 `starmap_visual_cortex`, holdout `paper_2_4_6`, the protocol unmodified. The six comparator
rows — SpatialZ, FEAST, isoST and three SpatialCPA predecessors — were re-scored **together, in one
call**, by `evaluate_all --force`, 0 failures, on the content-hash-pinned evaluator:

```
7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992  src/bench3/evaluate_paper.py
```

**That hash is verified three ways**: it is what the scoring run reported, it is what `specs/10` §0
pins (764 lines), and it is what `benchmark-pbya-v3/src/bench3/evaluate_paper.py` in this repository
hashes to. `tests/test_instrument_pin.py` asserts the last two on every test run, so the claim
"these numbers came from the pinned instrument" is checked rather than stated.

**`flanking_copy` and
`oracle` are model-free probes** — they copy real cells rather than running a model, so they are
arm-independent; they come from the probes tree (`r11_starmap_layout_modes.json`, re-measured and
reproducing to four decimals), as does **this method's own row**, which is the shipped configuration
at medians over the same three held-out sections.

| metric | spatialz | feast | isost | v18 † | v20 † | v21 | **v25 (ours)** | **`flanking_copy`** | **`oracle`** |
|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | 0.9289 | 0.7742 | 0.7884 | 0.9811 | 0.9811 | 0.9768 | **0.5574** | **0.9836** | 1.0000 |
| `gearys_pearson` | 0.9307 | 0.7746 | 0.7981 | 0.9815 | 0.9815 | 0.9781 | **0.5543** | **0.9840** | 1.0000 |
| `umap_mixing` | 0.8013 | 0.7742 | **0.9956** | 0.9238 | 0.9238 | 0.9392 | **0.8318** | ⚠️ *no probe* | ⚠️ *none* |
| `marker_field_r` | 0.8535 | 0.5686 | 0.6344 | 0.8707 | 0.8707 | 0.8757 | **0.5655** | **0.8857** | 0.9997 |
| `marker_depth_r` | 0.9199 | 0.7690 | 0.6984 | 0.8963 | 0.8963 | 0.9580 | **0.7228** | **0.9794** | 1.0000 |
| `celltype_localization` | 0.8175 | ⚠️ 0.0000 | ⚠️ 0.0000 | 0.7766 | 0.7766 | 0.7954 | **0.7591** | **0.7765** | 0.9808 |

† `v18` and `v20` emit **bitwise-identical predictions** under this holdout — the same file, not two runs that agree. See the fourth reading below; the row is printed twice as returned.

### The reading, in the order the floor forces

**1. On four of the five metrics that have a floor, not one of the seven methods reaches it.** Not
SpatialZ, not FEAST, not isoST, not any SpatialCPA version, ours included. The best value in each of
those columns still sits below a model-free copy of the flanking sections: −0.0025 at the closest
(v18/v20 on the two autocorrelation metrics) and −0.0100 at best on `marker_field_r`. **The only
column where anything exceeds the copy is `celltype_localization`**, where SpatialZ clears it by
+0.0410 and v21 by +0.0189.

This is §6.1's finding, generalised past ourselves. We reported it about our own method first
(§6.1, volunteered) and the comparator run says it is a property of the protocol: **on this
benchmark, copying the flanking sections is at or above the state of the art on four of the five
metrics that have a floor.** That is the methodological result this table carries, and it is not a result about any
method in it.

**2. SpatialZ beats this method on five of six, and our single win is on the unreadable metric.**
0.5574 against 0.9289, 0.5543 against 0.9307, 0.5655 against 0.8535, 0.7228 against 0.9199, 0.7591
against 0.8175. The one column we take is **`umap_mixing`, 0.8318 against 0.8013 — the one metric
with no floor and no ceiling**, which by this paper's own rule (§6) means a score on it cannot be
read at all. We do not claim it.

**3. v20 is not competing with SpatialZ; the copy is.** `v20` sits **+0.0001** from `flanking_copy`
on `celltype_localization`, −0.0025 on both autocorrelation metrics. `reports/v20_v21_evidence_audit.md`
established on other grounds that `cross-mix` under `resample` **is** a copy, matching a model-free
nearest-section copier to 0.001 on `deep_starmap`; this table is that finding on the pinned
instrument, on a second dataset, to four decimals.

**This is why we do not quote "v20 beats SpatialZ on 5 of 6" even though the table now supports the
arithmetic.** That claim was struck earlier in this work for two reasons, and the comparator run
resolves only one of them. It is no longer unsourced. It is still **circular**: the number doing the
beating is the copy floor's, and this paper's own §6.1 is the argument that the copy floor is near
the achievable maximum, so quoting it as a method's competitiveness would be claiming a model-free
probe's score as a result of the model (`reports/spatialz_claim_struck.md`, amended with this run).
**What the table licenses is "a copy beats SpatialZ on four of the five readable metrics", which is a
statement about the benchmark.**

**4. Two of the six comparator rows are the same prediction, and that is a result about the
protocol.** `v18` and `v20` agree in all six columns to four decimals. They are not two runs of one
method and not a transcription error: **their prediction files are identical array by array** on
`paper_2_4_6` — `X/data`, `X/indices`, `X/indptr`, `cell_id`, `cell_type`, `section`, `x`, `y`, `z`,
across **12 403 cells and 344 361 non-zero entries**. The only difference between the two files is
`/uns`, which carries provenance and is not scored.

They are not the same method. In the **wide** holdout regime the two differ — and differ **only in
expression**: on `allen_merfish_brain/wide_26_…_34`, `X` differs while every coordinate, `cell_id`,
`cell_type` and `section` is identical. So v20's changes over v18 are **expression-path only, and
they fire only when the section gap exceeds the volume's median spacing.** `paper_2_4_6` never
reaches that gap, so on tier-1 the two versions execute the same code path and emit the same file.

**The headline table of this literature cannot distinguish two released versions of a method** — not
in the weak sense that their scores are close, but in the strong sense that there is nothing to
distinguish: the protocol never activates the code that separates them. The six comparator columns
are **five distinct predictions**. We report the row twice, as returned, because merging it would
hide that.

This is the sharpest available statement of a limit this paper reports in three other places. §6.3
found the metric nearly blind to the expression head (an oracle-position arm's across-seed spread is
exactly zero); §5.6 found the statistic resolves ~0.26–0.30 of the tissue radius; §8.2's first
reading found no method clearing a model-free copy. Here the protocol does not merely fail to resolve
a difference — **on this holdout there is no difference to resolve**, and a reader comparing the two
published versions on tier-1 would be comparing one file with itself.

### Three things in the table that are not measurements, flagged rather than smoothed

- **FEAST and isoST score exactly `0.0000` on `celltype_localization`.** An exact zero on a
  Sinkhorn-divergence-against-null statistic is the value returned when nothing is scorable — no
  cell type clearing `min_gt_cells`, or no type column — not a measurement of poor localization.
  **These two cells should be read as blank.** They are shown as returned because a silent blank
  would hide a run that did not do what it was asked (Convention 6).
- **isoST's `0.9956` on `umap_mixing` is the largest number in the table**, returned by the method
  that is otherwise last or second-last in four of the other five columns. A metric on which the
  weakest reconstruction scores highest is not ordering reconstructions — and it is the same metric
  that has no floor and no ceiling, and the same one our own single win is on. We take all three
  facts together as a reason to **read nothing from the `umap_mixing` column**, ours included.
- **FEAST returns 0.7742 on `morans_pearson` and 0.7742 on `umap_mixing`** — two statistics of
  different construction agreeing to four decimals. Noted, unexplained.

## 8.3 Two mechanisms in the competitor, read from its source

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

## 8.4 What is missing from §8.2, and what it would take

**The run happened.** An earlier draft of this section said the comparator re-score had not been
performed, citing `specs/10` step 5 and `reports/pilot.md` §44. That was true when those were written
and is not true now: `evaluate_all --force` produced `results_rescored/` for
`starmap_visual_cortex/paper_2_4_6`, all six methods, 0 failures, on the pinned evaluator. §8.2 is
that run. What remains missing is narrower and it is about the **reference columns**, not the run.

**1. The floor and ceiling were not produced by the same invocation as the comparator rows.** The six
comparator rows come from `results_rescored/`; `flanking_copy`, `oracle` and this method's own row
come from the probes tree. Both are the same pinned evaluator, the same build and the same
`paper_2_4_6` holdout, and the probes are **model-free and arm-independent** — they copy real cells
and run no model, which is exactly why they transfer and why the join is legitimate. But it is a
**join of two invocations, not one table**, and we say so rather than presenting it as one. Closing
it means re-running the probes inside `evaluate_all` so every cell in §8.2 comes from one call.

**2. `paper_umap_mixing` has no probe at all** — neither floor nor ceiling exists for it, on any run.
One of the six columns is therefore unreadable by this paper's own rule, and §8.2's fourth flag gives
two further reasons not to read it. It is printed because omitting a column of the protocol's own
six would be a silent edit, not because we draw anything from it.

**3. Two cells are not measurements** — FEAST's and isoST's exact `0.0000` on
`celltype_localization` — and establishing what those runs actually did would take reading their
outputs, which we have not done.

**4. The `results_rescored/` tree is not in this repository**, and the table in §8.2 was transcribed
from it. What *is* here is the instrument: `evaluate_paper.py` is committed and hashes to the pin the
run reported, asserted by `tests/test_instrument_pin.py`. So the measuring device is reproducible from
this repository and the measurement is not — what is missing is the tree of scored outputs and the
comparator methods' own predictions, not the thing that scored them. Committing the tree, or a
manifest of it, closes the gap; transcription is the only step in §8.2 that a reader has to take on
trust, and it is one table of 42 numbers.

**What this section claims about the competitor.** The two mechanisms of §8.3, read from its source,
one of them measured on our fixture with a model-free test; that SpatialZ beats this method on five
of the six tier-1 metrics, from §8.2 and consistent with §6; and that no method in the table reaches
the copy floor on four of the five metrics that have one. **We make no claim that any SpatialCPA
version beats SpatialZ as a method**, in either direction, for the reason given in §8.2's third
reading.
