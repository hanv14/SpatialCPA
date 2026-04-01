# SpatialCPA — Methodology Explained
### A step-by-step guide to understanding the method

---

## The problem in one sentence

We have a few thin tissue slices measured at scattered positions along the z-axis,
and we want to predict what the tissue looks like *everywhere in between* — at any
position, any angle, with full gene expression and cell-type information.

---

## What goes in (Input)

SpatialCPA takes **sparsely sampled 2D spatial transcriptomics sections** as input.
Each section is a thin physical slice of tissue (e.g., a brain) that has been profiled
with a technology like MERFISH or STARmap.

For each section, we need four things:

```
┌──────────────────────────────────────────────────────────────────┐
│                         PER SECTION                              │
│                                                                  │
│  1. Gene expression matrix ─── cells × genes (e.g., 30,000 × 1,100)
│     What: raw or normalized transcript counts for each cell      │
│     Why:  this is what we want to predict for virtual cells      │
│                                                                  │
│  2. Spatial coordinates ────── (x, y) per cell                   │
│     What: the 2D position of each cell within its slice          │
│     Why:  tells us where cells are located in the tissue plane   │
│                                                                  │
│  3. Cell-type labels ─────── one label per cell                  │
│     What: e.g., "glutamatergic neuron", "astrocyte", "microglia" │
│     Why:  expression depends heavily on what type of cell it is  │
│                                                                  │
│  4. Section metadata ──────── z-position and thickness           │
│     What: where in the tissue this slice was cut (in μm)         │
│     Why:  defines the 3D coordinate of each cell                 │
│                                                                  │
│  Optional extras:                                                │
│     • Region annotations (e.g., "isocortex", "hippocampus")     │
│     • Subclass / supertype labels (finer cell typing)            │
│     • Confidence scores (how reliable is the cell-type label?)   │
└──────────────────────────────────────────────────────────────────┘
```

**Key point:** The sections are *sparse* — there are large physical gaps (50–300 μm)
between consecutive slices. A typical brain atlas might have 129 sections spanning
the entire hemisphere, with ~100 μm gaps between each pair.

---

## What comes out (Output)

Given any 3D coordinate (x, y, z), SpatialCPA returns:

```
┌──────────────────────────────────────────────────────────────────┐
│                     PER QUERIED POSITION                         │
│                                                                  │
│  1. Predicted cell type ───── "What kind of cell lives here?"    │
│     A probability distribution over all cell types               │
│     e.g., 85% glutamatergic neuron, 10% astrocyte, 5% other     │
│                                                                  │
│  2. Predicted region ──────── "What tissue region is this?"      │
│     e.g., isocortex layer 4, hippocampus CA1, thalamus           │
│                                                                  │
│  3. Predicted expression ──── "What genes are active here?"      │
│     A full gene expression profile (all 1,100+ genes)            │
│     Generated as a sample from a learned distribution,           │
│     so each virtual cell has realistic cell-to-cell variability  │
└──────────────────────────────────────────────────────────────────┘
```

**The critical feature:** The z-coordinate can be **anything** — it does not have to
fall between two specific observed slices. You can query at z = 547.3 μm even if the
nearest observed slices are at z = 500 μm and z = 600 μm. You can even define a
cutting plane at an arbitrary angle (sagittal, horizontal, oblique) and the model will
generate a complete virtual section along that plane.

---

## The pipeline, step by step

### Step 1: Data harmonization

**What happens:** Raw datasets are converted into a common format with physically
meaningful coordinates.

**Why it matters:** Different datasets store coordinates differently (some in obsm,
some in obs columns), use different units (μm vs. pixels), and have different
annotation depths. We need everything in one consistent format before training.

**The key operation in this step:**

```
Raw section label  ──→  Physical z-coordinate (in μm)

Example:
  "C57BL/6J-1.050"  ──→  z = 1050 μm
  "C57BL/6J-1.051"  ──→  z = 1051 μm    (1 μm gap — nearly adjacent)
  "C57BL/6J-1.076"  ──→  z = 1076 μm    (25 μm gap — moderate)
  "C57BL/6J-1.098"  ──→  z = 1098 μm    (22 μm gap — moderate)

Each cell now has a full 3D coordinate: (x, y, z) in physical units.
```

We also compute the gap sizes between consecutive sections and record the physical
thickness of each section. These are used later for gap-aware training.

**Output of this step:** A clean AnnData object per dataset with columns for
`x_coord`, `y_coord`, `z_physical`, `section_thickness`, `cell_type`, and
optional annotations.

---

### Step 2: Fourier feature encoding

**What happens:** The raw 3D coordinates (x, y, z) are transformed into a
high-dimensional representation that the neural network can work with effectively.

**Why it matters:** Neural networks are bad at learning sharp spatial boundaries
from raw coordinates. If you feed (x, y, z) directly into a network, it tends to
produce blurry, smeared-out predictions — it cannot represent the sharp transition
between, say, cortical layer 4 and layer 5. Fourier features fix this by encoding
coordinates as a set of sine and cosine waves at multiple frequencies.

**Analogy:** Think of a radio. Raw coordinates are like a single channel. Fourier
features split the signal into many channels at different frequencies — some
capturing large-scale structure (which brain region are we in?), others capturing
fine-grained detail (which cortical layer?).

**The key design choice:**

```
Within a slice (x, y):
  Cell-to-cell distance is ~1–10 μm
  → We need HIGH frequencies to resolve cellular neighborhoods

Between slices (z):
  Section-to-section distance is ~50–300 μm
  → We need LOWER frequencies (but still need some high ones for region boundaries)

Solution: Use SEPARATE frequency bands for x/y vs z,
          calibrated to the actual spacing in the data.
```

**Output of this step:** Each 3D coordinate becomes a vector of ~256 numbers
(128 sine values + 128 cosine values) that encodes its position at multiple
spatial scales.

---

### Step 3: Spatial backbone network

**What happens:** The Fourier-encoded coordinates pass through a deep neural network
(8 layers, 512 units per layer) that learns to extract a "spatial context" — a rich
summary of what the tissue looks like at that 3D position.

**Why it matters:** The backbone is the shared foundation that all three prediction
heads (cell type, region, expression) build on. It learns the tissue's 3D architecture:
where regions are, how cell types are distributed, and how expression varies through space.

**Architecture details:**

```
Fourier features (256-dim)
    │
    ▼
  Layer 1 ──→ Layer 2 ──→ Layer 3 ──→ Layer 4
                                         │
                                    (skip connection:
                                     re-inject original
                                     Fourier features)
                                         │
                                         ▼
                                 Layer 5 ──→ Layer 6 ──→ Layer 7 ──→ Layer 8
                                                                       │
                                                                       ▼
                                                            Spatial context h(x,y,z)
                                                            (256-dimensional vector)
```

The skip connection at the midpoint is important — it ensures the network retains
access to fine-grained coordinate information even after 4 layers of processing.
Without it, subtle spatial details can get lost.

**Output of this step:** A 256-dimensional "spatial context vector" h(x, y, z) for
each queried position. This vector encodes everything the model knows about that
location — which region it's in, what the local niche looks like, what expression
patterns are expected there.

---

### Step 4: Prediction heads (three parallel outputs)

The spatial context vector feeds into three specialized prediction heads
simultaneously:

#### Head A: Cell-type classifier

```
h(x, y, z)  ──→  [hidden layer]  ──→  softmax  ──→  P(cell_type | x, y, z)

Example output at position (340, 520, 750):
  glutamatergic neuron:  0.72
  astrocyte:             0.15
  oligodendrocyte:       0.08
  microglial cell:       0.03
  endothelial cell:      0.02
```

This tells us what kind of cell is most likely to exist at this position.

**Handling datasets with different annotation depths:** The classifier has separate
output layers for each annotation level (cell_type, class, subclass, supertype,
region). During training, we only compute loss for annotation levels that exist
in the current batch's dataset. When a batch comes from a dataset that only has
`cell_type` labels, only the cell_type head gets trained. When a batch comes from
a richly annotated dataset, all heads get trained.

#### Head B: Region classifier

```
h(x, y, z)  ──→  [hidden layer]  ──→  softmax  ──→  P(region | x, y, z)

Example output:
  isocortex:    0.91
  hippocampus:  0.05
  fiber tracts: 0.03
  thalamus:     0.01
```

Same architecture as the cell-type head, but predicting tissue regions.

#### Head C: Expression decoder

This is the most complex head. It generates a full gene expression profile
conditioned on both the spatial context AND the cell type:

```
h(x, y, z)  ─┐
              ├──→  [decoder network]  ──→  ZINB parameters  ──→  sample expression
cell type  ───┘                              (mean, dispersion, dropout)

Why condition on cell type?
  The same spatial position contains multiple cell types with very different
  expression programs. A glutamatergic neuron at position (340, 520, 750)
  expresses different genes than an astrocyte at the exact same position.
  The cell-type conditioning tells the decoder WHICH expression program to use.
```

**Why ZINB (Zero-Inflated Negative Binomial)?**

Single-cell gene expression data has two statistical quirks:
- Many genes are "detected" in only a fraction of cells (excess zeros)
- The variance of detected genes exceeds the mean (overdispersion)

The ZINB distribution captures both properties. Instead of predicting a single
expression value per gene, the decoder predicts three parameters:
- **Mean:** expected expression level
- **Dispersion:** how much cell-to-cell variability to expect
- **Dropout probability:** chance that the gene is an excess zero

We then *sample* from this distribution, so each virtual cell gets a unique,
realistic expression profile — not just the average.

---

### Step 5: Training with z-marginalization (gap-aware)

This is where SpatialCPA handles the key challenge of non-uniform gaps and
section thickness.

#### The slab module: modeling section thickness

**The problem:** Each physical section has a real thickness (~10 μm). Cells within
a single slice don't all sit at exactly the same z. If we assign them all the same z,
the model sees an artificial discontinuity at section boundaries.

**The solution:** During training, instead of evaluating the model at a single z,
we sample 5 random z-values within the section thickness and average the predictions:

```
Real section at z = 500 μm, thickness = 10 μm

Standard approach (what SpatialZ does):
  All cells assigned z = 500
  → Model sees a sharp wall of cells at z = 500

Our approach (z-marginalization):
  Cell 1 evaluated at z = 497.2, 499.1, 500.5, 502.3, 504.8
  Cell 2 evaluated at z = 496.5, 498.7, 501.2, 503.0, 505.1
  ...predictions averaged across the 5 z-samples

  → Model learns that cells in this section could be anywhere
    between z = 495 and z = 505
  → No artificial discontinuity at section boundaries
  → Smoother, more realistic z-variation
```

At **inference** time, when generating virtual slices, we query at the exact
z-coordinate — no averaging needed.

#### Leave-one-out self-supervision (gap-aware)

**The problem:** We never have ground truth for virtual slice positions during
training. How do we know the model's interpolation is accurate?

**The solution:** Randomly hold out one observed section, train on the rest, and
then test whether the model can predict the held-out section from its neighbors.

**The gap-aware twist:** Not all held-out sections are equally informative. Holding
out a section that's close to its neighbors is easy — holding out one with large gaps
on both sides is hard but more valuable for learning. So we:

```
1. Sample which section to hold out with probability PROPORTIONAL to gap size
   (larger gaps → more likely to be held out → model practices hard cases more)

2. Weight the LOO loss PROPORTIONAL to gap size
   (if the model gets a large-gap prediction right, that counts for more)

3. During LOO, query at the EXACT z-position (no z-marginalization)
   (we're testing precise interpolation ability)
```

This directly trains the model to interpolate well across the specific gap sizes
present in the data.

---

### Step 6: Inference (generating virtual slices)

Once trained, generating a virtual slice at any z-value is straightforward:

```
Query: "Generate a virtual slice at z = 547.3 μm"

Step A: Determine tissue boundary
  → Query cell-type head on a coarse 100×100 grid at z = 547.3
  → Positions where the model is confident about some cell type = tissue
  → Positions where the model is uncertain about everything = empty space

Step B: Sample cell positions
  → Within the tissue boundary, place cells using Poisson disk sampling
  → This gives realistic spacing (not a regular grid)

Step C: Assign cell types
  → For each sampled position, get P(cell_type | x, y, z = 547.3)
  → Sample a cell type from this distribution

Step D: Assign region labels
  → Same as C, but using the region head

Step E: Generate expression
  → For each cell, feed its position + assigned cell type into the decoder
  → Sample from the predicted ZINB distribution
  → Result: a complete gene expression profile for each virtual cell

Output: an AnnData object identical in structure to a real section,
        containing positions, cell types, regions, and expression for
        every virtual cell.
```

**In silico sectioning at arbitrary angles:** To cut at a sagittal or horizontal
angle, simply define the cutting plane as a normal vector + point, generate a grid
of 3D positions along that plane, and query the model. No coordinate rotation
needed — the model takes any (x, y, z) and returns predictions.

**Expression gradients (unique capability):** Because the model is a differentiable
function of (x, y, z), we can compute ∂expression/∂z analytically via
backpropagation. This gives spatial expression gradients — showing exactly where
and how gene expression changes through 3D space. High gradient magnitudes
indicate tissue boundaries or laminar transitions.

---

## How we evaluate performance

### Metrics (same as SpatialZ for fair comparison)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  METRIC                    │  WHAT IT MEASURES                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Gene-wise Pearson r        For each gene, correlate predicted vs real  │
│                             expression across all cells. Report mean.  │
│                             Higher = better expression prediction.      │
│                                                                         │
│  Moran's I correlation      Compute Moran's I (spatial autocorrelation)│
│                             per gene on both real and virtual slices.   │
│                             Correlate the two. Higher = better spatial  │
│                             pattern preservation.                       │
│                                                                         │
│  Geary's C correlation      Same idea as Moran's I but measures local  │
│                             spatial variation. Higher = better.         │
│                                                                         │
│  Cell-type accuracy         % of virtual cells assigned the correct    │
│                             cell type, compared to ground truth.        │
│                                                                         │
│  Region accuracy            Same for region labels.                     │
│                                                                         │
│  SSIM                       Structural similarity: rasterize gene      │
│                             expression to a grid and compare images.    │
│                             Higher = more spatially faithful.           │
│                                                                         │
│  Top SVG overlap            Do real and virtual slices agree on which   │
│                             genes are most spatially variable?          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Evaluation protocol

**Experiment 1 — Leave-one-out (direct comparison with SpatialZ):**

```
Start with 7 consecutive sections from a real 3D dataset (STARmap).
Remove sections 2, 4, and 6.
Train on sections 1, 3, 5, 7.
Predict sections 2, 4, 6.
Compare predicted vs real on all metrics above.
Run the same protocol with SpatialZ.
Compare results side by side.
```

**Experiment 2 — Gap-stratified evaluation (our key advantage):**

```
Take a dataset with many sections (e.g., 129 BICCN sections).
Artificially create variable gaps by removing sections at different rates.
Evaluate held-out sections grouped by gap size:

  Small gap  (<50 μm):   Both methods should do well
  Medium gap (50–150 μm): Our method should start pulling ahead
  Large gap  (>150 μm):  Our method should clearly outperform
  Very large (>300 μm):  SpatialZ degrades severely; ours degrades gracefully

WHY we expect this advantage:
  SpatialZ only uses the TWO adjacent slices for interpolation.
  SpatialCPA uses ALL observed slices — the entire tissue's spatial structure
  informs every prediction, even for positions far from any observed section.
```

**Experiment 3 — Novel capabilities (SpatialZ cannot do these):**

```
a) Query at arbitrary z-values (not just midpoints between pairs)
b) Generate sections at arbitrary angles (sagittal, horizontal, oblique)
c) Compute spatial expression gradients ∂expression/∂z
d) Validate against Allen Brain Atlas ISH reference images
```

---

## How SpatialCPA differs from SpatialZ

```
┌─────────────────────┬──────────────────────┬────────────────────────────┐
│  ASPECT             │  SpatialZ            │  SpatialCPA (ours)         │
├─────────────────────┼──────────────────────┼────────────────────────────┤
│                     │                      │                            │
│  Core idea          │  Pairwise            │  Learn one continuous      │
│                     │  interpolation       │  3D function               │
│                     │  between slice pairs │  over entire tissue        │
│                     │                      │                            │
│  Cell locations     │  Wasserstein         │  Density field             │
│                     │  barycenter          │  from neural network       │
│                     │                      │                            │
│  Cell types         │  k-NN from the two   │  Learned spatial           │
│                     │  adjacent slices     │  classifier                │
│                     │                      │                            │
│  Gene expression    │  Copy from real      │  Generate new profiles     │
│                     │  cells (sampling)    │  from learned distribution │
│                     │                      │                            │
│  Training           │  No training         │  End-to-end supervised +   │
│                     │  (heuristic rules)   │  self-supervised           │
│                     │                      │                            │
│  Query positions    │  Only between        │  Any (x, y, z)             │
│                     │  adjacent pairs      │  in the volume             │
│                     │                      │                            │
│  Cutting angles     │  Needs separate      │  Native — just query       │
│                     │  rotation module     │  along any plane           │
│                     │                      │                            │
│  Gap handling       │  Each pair is        │  All slices inform         │
│                     │  independent         │  every prediction          │
│                     │                      │                            │
│  Section thickness  │  Ignored             │  Modeled as slabs          │
│                     │                      │  with z-marginalization    │
│                     │                      │                            │
│  Expression         │  Not possible        │  Analytic via              │
│  gradients          │                      │  backpropagation           │
│                     │                      │                            │
│  Joint optimization │  No — 3 separate     │  Yes — end-to-end         │
│                     │  sequential stages   │  differentiable            │
│                     │                      │                            │
└─────────────────────┴──────────────────────┴────────────────────────────┘
```

---

## Quick summary

```
INPUT:
  Sparsely sampled 2D tissue sections with gene expression,
  cell coordinates, cell-type labels, and section z-positions.

METHOD:
  1. Harmonize data → assign physical z-coordinates to every cell
  2. Encode (x,y,z) with adaptive Fourier features (separate xy/z scales)
  3. Pass through spatial backbone → 256-dim spatial context vector
  4. Three parallel heads predict cell type, region, and expression
  5. Train with z-marginalization (slab model) + gap-aware leave-one-out

OUTPUT:
  For any query (x, y, z) → cell type + region + full expression profile.
  Enables: virtual slices at any z, arbitrary-angle sectioning,
  spatial expression gradients, dense 3D atlas construction.

EVALUATE:
  Gene-wise Pearson r, Moran's I/Geary's C correlation, cell-type accuracy,
  SSIM, top SVG overlap. Key experiment: gap-stratified evaluation showing
  graceful degradation vs SpatialZ at large inter-slice gaps.
```