# 2. Method

The system carries a volume as a **continuous field** rather than as an indexed stack, so that every
component can be queried at a coordinate. A section is then specified by a plane — an origin, a
normal and a thickness — and generation is the same operation whatever that plane's orientation.

## 2.1 What a plane needs

To emit a section at a plane, four things must be defined at arbitrary coordinates:

| | component | what it supplies |
|---|---|---|
| 1 | **correlated noise prior** | spatially structured randomness, so neighbouring cells are not independent draws |
| 2 | **anatomical field + retrieval** | where in the specimen a coordinate is, and which real cells resemble it |
| 3 | **layout head** | where the cells of this section go |
| 4 | **expression head + emission** | what each cell expresses, as raw counts |

Components 1, 2 and 4 are functions of position and are orientation-agnostic by construction.
**Component 3 is where orientation enters**, and it is the one this paper is about.

## 2.2 The continuous noise prior

Independent per-cell noise produces sections with no spatial autocorrelation, which is the first
thing a spatial statistic detects. The prior is instead a **Gaussian random field over the volume**,
sampled at the cells' 3D coordinates, so that two cells close in space receive correlated noise
whether or not they lie in the same section. Its correlation lengths are fitted, and a section's
noise is a slice of one 3D field rather than an independently drawn 2D one — which is also what makes
two overlapping generated planes mutually consistent.

**This was gated before anything downstream was built** (`reports/gate1.md`, verdict PASS).
Substituting the correlated prior for an i.i.d. one in the same generative map cuts the median
per-gene Moran's I error to **13%** of the i.i.d. prior's (0.0552 against 0.4233, against a
threshold of 50%) and raises the per-gene correlation between generated and real Moran's I from
**r = 0.38 to r = 0.92** (threshold 0.7) — so the correlated prior survives the generative map and
shows up as preserved spatial autocorrelation in the counts, rather than being smoothed away. That
report also records the limit of the mechanism: `I_gen(ell)` is not monotone, turning over at 2.52×
the fitted length-scale, which is a property of the statistic and bounds what the T09 calibration
loop can be asked to hit.

## 2.3 The anatomical field and retrieval

A learned field maps a coordinate to a representation of *where in the specimen* it is, trained with
rotation augmentation over several plane orientations so that the representation does not depend on
the sectioning axis. Retrieval conditions generation on real cells from elsewhere in the volume,
excluding the section being generated.

**This was gated too** (`reports/gate2.md`, verdict PASS): reconstruction quality on oblique planes
reaches **0.955** of axis-aligned quality on held-in sections, worst angle 30°, against a required
0.90 — with both arms depth-matched, which matters because an oblique strip necessarily draws cells
from the stack's poorly-reconstructed ends and a single central coronal baseline flatters the
denominator by 8.5%. Two qualifications travel with that number and are in the report rather than
only in its appendix. It is measured on a **synthetic fixture 3000 µm across and 400 µm deep —
7.5 : 1**, a geometry §4 shows no built dataset has; and the criterion **cannot resolve 0.886 from
0.90** at this sample size, so the margin is real but not precise. §5 is where oblique reconstruction
becomes a statement about tissue, and §4 is why that statement is available on one specimen.

## 2.4 The layout: three modes, and the one that ships

The layout head is the only component for which orientation is a substantive question, and we report
three modes.

**`field`** samples positions from a learned continuous intensity. It is the mode the architecture
was designed around, and it is **refuted**: it scores below a model-free copy baseline on the metric
it exists to win, and supplying the cell count externally does not rescue it (§6.2). It is reported
as an ablation.

**`resample`** copies the real coordinates of a donor section. This is the configuration that
**ships**, because on real tissue copying real positions beats generating them. Its conventional form
selects the donor by `|Δz|` — a *z*-distance, which is undefined for a plane spanning the stack — and
copies that donor's entire face.

**`plane-distance`** is the generalisation this paper introduces: donors are selected by
**perpendicular distance to the plane**, per cell. It is defined at any orientation, and at a coronal
plane it returns exactly the cells `|Δz|` selects — we assert this **bitwise**, so every number
previously measured under the old rule stands unchanged. The generalisation lives entirely in the
oblique case, where a flat section meets a tilted plane in a line and the cells the plane passes
through are drawn from several sections at once.

## 2.5 Expression and emission

Expression is generated in a latent space by a flow-matching head conditioned on the field and on
retrieval, and emitted as **raw counts** through a zero-inflated negative-binomial decoder. Any
normalisation is an input-side transform only; normalised values never reach the decoder target, so
the model's output is on the same scale as the measurement.

## 2.6 Consistency losses

Three losses tie the representation to the fact that a section is an arbitrary cut of a volume rather
than a thing in itself: sections generated at nearby planes must agree where they overlap, the same
plane approached from different orientations must give the same answer, and a plane's content must
vary smoothly with its position. One further loss is built **to fail** and is reported as such — an
ablation whose job is to show that the consistency it enforces is doing work.

## 2.7 What the shipped configuration is

**`resample` layout with `zinb-flow` expression**: real donor coordinates, generated expression. This
is a deliberate and reported choice, not a default — the generative layout loses to copying on real
tissue (§6.2), and saying so is more useful than shipping the more novel component.

The contribution of this paper's layout work is therefore not that generated positions beat copied
ones. It is that **copying becomes well-defined at an arbitrary orientation**, where the conventional
rule has no definition and emits an object that is not a section of the requested plane (§5.3).

## 2.8 Evaluation

All numbers are computed by an external, content-hash-pinned benchmark harness, unmodified. The
primary statistics are `paper_celltype_localization` — per cell type, an optimal-transport divergence
between predicted and ground-truth point clouds, calibrated against a within-tissue null — and
`paper_morans_pearson`, the across-gene correlation of per-gene spatial autocorrelation with each
side computed on its own graph. §3.2 and §5.6 characterise what the first of these can and cannot
resolve; those characterisations apply to every number in this paper, including the ones that favour
us.

Held-out sections are never touched by training, calibration or configuration selection; the
separation is enforced by type rather than by convention.
