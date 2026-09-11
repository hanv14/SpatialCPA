# Is any unbuilt dataset worth building, so the demonstration replicates?

**The question.** `merfish_thick_hypothalamus` clears 90°. One specimen carrying the paper's central
claim is one specimen. If a second clears too, the capability replicates; if none can, that is worth
knowing before anyone spends a build.

**Two of the candidates can be ruled out by arithmetic, before any build.** That screen is below and
it is checkable now.

## 1. Why this specimen cleared, stated mechanically

`merfish_thick_hypothalamus` is **a 200 µm block cut into 7 slabs of ~27 µm** (`specs/10` §8). It is
not a stack of thin sections — it is a block that happens to be sliced. That is the whole reason its
in-plane : depth is **10.3 : 1** where tier-1's is 21.6 : 1, and 10.3 : 1 is what lets a 90° plane
retain 1988 cells instead of tier-1's 304.

**So the screen is not "which dataset is biggest" — it is "which dataset is a block".**

## 2. The pre-build screen: G2 is an absolute floor, and most volumes fail it on total cell count

`celltype_localization` subsamples every type to `max_n` = 250, so **G2 requires a single cell type
with ≥ 250 cells inside the oblique strip**. The strip retains a fraction `f(θ)` of the volume, and
`f(90°)` is measurable on the two specimens already read:

| specimen | cells at 90° | cells in volume | `f(90°)` |
|---|---|---|---|
| `starmap_visual_cortex` | 304 | ~16 600 | **1.8%** |
| `merfish_thick_hypothalamus` | 1988 | **47 189** | **4.2%** |

*(CORRECTED — `retractions.md` **R6**. The second row first read 79 000, which is the **full**
dataset against a **training-volume** strip count. Mixing the two scopes is the error §4.2a exists
for, committed in the document that was ruling other candidates out for being unmeasured.)*

Taking `f(90°) ≈ 2–4%` — read off two measured specimens, not assumed — a dataset can clear G2 at
90° only if its **largest type holds roughly ≥ 8 300 cells in the whole volume**. Which needs a
volume of order 10⁵ cells unless the types are extraordinarily unbalanced.

⚠️ **And this screen is now secondary.** `reports/the_comb_limit.md` shows that at 90° an oblique
ground truth from `N` serial sections is `N` parallel lines whatever its cell count, so **no**
volume in this table clears 90° meaningfully. The screen still ranks candidates for a 30–45°
demonstration, which is where the claim now sits.

**That rules out two candidates on their own headline numbers**, with no build and no judgement:

| dataset | cells in volume | verdict |
|---|---|---|
| `exseq_visual_cortex` | **1 130**, 5 sections, 28% unannotated | ❌ cannot clear G2 at any oblique angle. Its *whole volume* holds fewer cells than the strip would need from one type. |
| `exseq_breast_cancer` | **1 979**, min 57 cells/section | ❌ same arithmetic. `specs/10` §5.4 already flags it "a small-n row". |

Neither is worth building for this purpose. Both may still be worth building for other reasons —
§5.2's non-brain requirement — but they cannot replicate a 90° demonstration.

## 3. The ranking, for what remains

**① `merfish_thick_cortex` — build it. The clear first choice.**
28.8 k cells, 254 genes, `raw_counts`, **`paper_2_4_6`** — the same holdout design as the headline —
and **13 µm slabs**, which is the same *thick-slab* construction as the specimen that cleared, at
half the thickness. `specs/10` §5.4 already calls it "the **cheapest** claim-bearing analogue". It is
the only candidate that is simultaneously cheap, claim-bearing, same-design, and the same *kind of
preparation* as the volume that worked. Its 28.8 k cells put it near the G2 screen's edge rather
than past it, so the budget read is a genuine test rather than a formality.

**② `allen_merfish_brain` — build it if ① does not clear.**
**59 sections** across a whole mouse brain: on depth alone it is the most likely thing in the table
to clear, and possibly at a better aspect ratio than the hypothalamus block. 1.17 M cells is a
*fitting* cost, and `specs/10` §5.4 excludes it from the campaign on exactly that basis — but the
**angle budget needs only coordinates and cell-type labels**, so the geometry read is cheap even if
the demonstration on it is not. Two caveats, stated rather than discovered later: an oblique plane
through a whole brain cuts across gross anatomy in a way a plane through one nucleus does not, so
`celltype_localization`'s within-tissue null may behave differently; and its 59 sections may be
spaced far enough apart that the "volume" is not spatially contiguous in z at all — which the budget
will show immediately.

**③ `cosmx_nsclc_3d` — worth building for a different claim, not this one.**
340 k cells clears the G2 screen comfortably, and it would make the oblique demonstration **non-brain
as well as replicated**, which discharges `design/v23_design.md` §7's reviewer defence on the very
claim that defence exists for. But 57 k cells per section makes it the most expensive row in the
table, and its depth is unknown. **Read its budget first** — it is free — and build only if it
clears.

**④ `imc_breast_cancer` — read its budget, do not build it for this.**
`fluorescence_intensity` → `zigamma`, which `specs/10` §5.1 bars from carrying a claim-bearing
number. It cannot replicate claim B. Its *geometry* is still a free and useful row in the budget
table — 15 real serial sections — and the table is a contribution in its own right.

## 4. The replication that costs nothing and should be done first

`specs/10` §9 already specifies re-partitioning **`merfish_thick_hypothalamus`'s own 200 µm block
into 14 slabs of ~13.5 µm** instead of its shipped 7 of ~27 µm — same volume, same cells, same panel,
**thickness the only variable** (that is V4a). Running the angle budget on both partitions is free
and answers a question the single 90° result cannot:

**does the clearance survive halving the slab thickness, or is it an artefact of thick slabs?**

That is not a second specimen and must not be reported as one. But it is a robustness result on the
claim's own volume, it costs one budget read, and if the thin partition stops clearing then the
paper's claim is about *thick-slab preparations* specifically — which is a materially different and
more careful statement than "oblique sections are generable", and one we would rather make ourselves
than have a reviewer make for us.

## 5. Recommendation

1. **Free, now:** budget the 14-slab re-partition of `merfish_thick_hypothalamus` (§4), and read the
   budgets of `cosmx_nsclc_3d` and `imc_breast_cancer` if their inputs can be built cheaply.
2. **Build `merfish_thick_cortex`.** One build, the cheapest claim-bearing volume in the table, the
   same holdout design, and the same thick-slab preparation at half the thickness.
3. **Only if that fails to clear, build `allen_merfish_brain`** for its geometry read.
4. **Do not build `exseq_visual_cortex` or `exseq_breast_cancer` for this purpose.** Ruled out on
   arithmetic in §2, and the arithmetic is checkable before the spend.
