#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
================================================================================
 learn_spatialcpav21.py   (v20 + COHERENT MIX + FIELD-ALIGNED GROUNDING)
 A single-file, from-scratch, heavily annotated re-implementation of the
 **SpatialCPA-v14 / H3D-FLA** virtual-slice generator, driven end-to-end by the
 **benchmark-pbya-v3** experiment (the SpatialZ STARmap protocol).
================================================================================

V21 CHANGES (each targets a measured finding, from the 18-dataset benchmark's
per_section_metrics or from the two-regime synthetic testbed built to
reproduce those findings; every mechanism keeps every emitted value a REAL
measurement, and every knob that trades regimes is GAP-ADAPTIVE via the v19
alpha)
-----------------------------------------------------------------------------
 BENCHMARK FINDINGS driving v21 (132 head-to-head sections vs SpatialZ):
   (a) v20's cross-mix REGRESSED the spatial-agreement metrics on the wide
       (consecutive-holdout) designs: Moran's Pearson 0.905 vs v18 0.923
       (win-rate vs SpatialZ 39% vs v18's 82%); marker Moran's MAE 0.097 vs
       v18 0.050.  Root cause: the mix mask is i.i.d. PER CELL PER GENE, so
       every mixed gene acquires salt-and-pepper spatial noise - neighbouring
       cells disagree about which flank each gene came from, which decorrelates
       neighbour values and scrambles the per-gene autocorrelation ranking.
   (b) At the SAME time v20's cross-mix WON the field metrics at wide gaps
       (marker field_r 0.571 / SSIM 0.572 vs SpatialZ 0.559; 61-64% win-rate):
       the interpolation signal is right, only its spatial granularity is wrong.
   (c) On the narrow (alternating "paper") designs BOTH v18 and v20 lose the
       gene-spatial-pattern metrics to SpatialZ (field_r 0.534/0.543 vs 0.569,
       depth_r 0.665/0.673 vs 0.697, win-rate 32-40%) while winning Moran's.
       Cumulative ablation + ceiling diagnostics on the synthetic testbed
       LOCALIZED the loss to two sources: (i) the FLOW RE-GROUNDING SWAPS
       (verbatim patchwork scores 0.951 Moran / 0.877 field / 0.025 Moran-MAE;
       adding the swaps at blend 1.0 / margin 1.0 drops it to 0.937 / 0.866 /
       0.034 - at narrow gaps the inherited exemplar already IS the local
       truth and latent disagreement is model noise), and (ii) COARSE
       SINGLE-FLANK LAYOUT PATCHES (a no-replacement flank-union scores 0.894
       marker-field vs 0.877 for the patchwork: when the flank fields are
       near-identical, giving each analysis bin only one flank discards half
       the local information).
 The v21 mechanisms:
 1. COHERENT MIX (cfg.coherent_mix, cfg.mix_field_freq; fixes (a), keeps (b)):
    every stochastic per-gene mixing mask (v20 cross-mix AND v18 gene-mix) is
    drawn from per-gene SMOOTH RANDOM FIELDS over the plane, rank-normalized
    per gene to an EXACT uniform marginal.  Marginal mixing probability per
    cell is unchanged (E[output] is still the interpolation; sparsity and
    detection hold by construction), but neighbours flip the SAME genes
    together, so Moran's/Geary's survive the mix.  Wide-regime ablation:
    i.i.d. masks 0.869 Moran's Pearson -> coherent 0.885+, at equal rates.
    mix_field_freq is the granule dial (finer = more per-bin flank averaging,
    less Moran headroom).
 2. PER-GENE MULTI-PARTNER DRAWS (cfg.mix_cand_k): each mixed (cell, gene)
    entry takes its value from ONE of mix_cand_k local same-type candidates,
    chosen independently PER GENE (SpatialZ-style per-gene stochasticity,
    restricted to real same-type values).  A single shared partner correlates
    the mixed genes' residuals within an analysis bin so bin noise fails to
    average out; per-gene draws decorrelate it (wide: Moran's-MAE 0.055 ->
    0.046 in the final config's ablation).
 3. Z-WEIGHTED, DISTANCE-WEIGHTED GENE-MIX (cfg.mix_dist_scale): gene-mix
    candidates are sampled with weight [(1-t) or t by flank] x
    exp(-d / (mix_dist_scale x median NN spacing)) - the redraw follows the
    z-interpolation at wide gaps and concentrates on the NEAREST same-type
    cells (SpatialZ weights references by distance too).
 4. GAP-ADAPTIVE RE-GROUNDING MARGIN (cfg.ground_gap_adapt,
    cfg.ground_margin_narrow; fixes (c)(i)): the flow re-grounding swap margin
    ramps from ground_margin_narrow (5.0) at alpha=0 down to the configured
    ground_keep_margin at alpha=1, so the swaps only run where they carry
    signal.  Single largest narrow-regime gain: 0.939/0.032 (Moran r / MAE)
    -> 0.951/0.024 at fixed layout.
 5. GAP-ADAPTIVE LAYOUT GRANULARITY (cfg.layout_gap_adapt,
    cfg.layout_freq_narrow; fixes (c)(ii)): the coherent-patch frequency
    ramps from layout_freq_narrow (8.0: granules ~ analysis-bin scale, so
    bins average both flanks while nearest neighbours still mostly share a
    flank) at alpha=0 down to coherent_freq (coarse patches - the v14
    salt-and-pepper protection that matters when the flanks differ) at
    alpha=1.  Measured narrow frontier (freq -> Moran r / field_r):
    4 -> 0.950/0.875, 8 -> 0.943/0.881, 24 -> 0.931/0.882, interleaved ->
    0.928/0.884.
 6. FIELD-ALIGNED RE-GROUNDING (cfg.field_align; alpha-gated by
    cfg.field_align_gap_only): bounded refinement that swaps a cell's
    exemplar to a spatially-local SAME-TYPE real cell whose whole profile
    better matches the z-interpolated local mean field of BOTH flanks,
    eligibility gated by a NOISE FLOOR (sigma0 = the real pool cells' own
    median deviation from their local field).  A swapper, so - like 4 - it
    runs at wide gaps only by default.
 7. GAP-ADAPTIVE GENE-MIX RATE (cfg.gene_mix_wide): the per-gene redraw
    fraction ramps from gene_mix_frac at alpha=0 to gene_mix_wide (0.7) at
    alpha=1 - the wide regime is where per-gene synthesis beats whole-profile
    copies on the field metrics (+0.010 field_r in ablation), and the
    z-weighted candidates keep its expectation on the interpolation.
 8. TAIL-RATE-MATCHED PER-GENE FIELD REPAIR with a GAP-ADAPTIVE TAIL SCALE
    (cfg.field_repair, cfg.repair_*): per gene, a REAL noise band is
    estimated from the pool cells' own residuals against their local field
    (threshold = repair_noise_mult x the repair_q quantile) TOGETHER WITH the
    real TAIL RATE - the fraction of real entries beyond the threshold.
    Among the prediction's beyond-threshold entries, an allowance of
    ts_eff x real-tail-rate (smallest first) is LEFT IN PLACE and only the
    surplus is repaired, worst first, capped at repair_frac; replacements are
    the same gene's value from a random local same-type candidate INSIDE the
    band (random, not closest, preserving the in-band spread).  ts_eff ramps
    from 1.0 at alpha=0 (exact matching: self-calibrating, never
    over-smooths - at narrow gaps the surplus is ~0 and almost nothing is
    touched) to repair_tail_scale (0.30) at alpha=1, deliberately repairing
    into the real tail to buy field_r where the Moran's-MAE headroom over
    per-gene-sampling methods affords it.  (A naive always-repair variant
    lifted field_r to 0.892 but blew Moran's-MAE 0.033 -> 0.13; tail-rate
    matching is what makes the repair safe.)
 9. MIX ORDER FIX: gene-mix now runs BEFORE cross-mix, so where the masks
    overlap the cross-flank value wins; v18-v20's reverse order diluted the
    effective cross-flank fraction to alpha * w_other * (1 - gene_mix_frac).
    The interpolation expectation is now exact with gene-mix on.  (Also
    fixed: a divide-by-zero in the shared dedup counter when
    composition-match swaps preceded later passes.)
 All mechanisms are generation-time only: training is IDENTICAL to v18/v20.
 Setting coherent_mix=False, field_align=False, field_repair=False,
 ground_gap_adapt=False, layout_gap_adapt=False, mix_cand_k=1,
 mix_dist_scale=0, gene_mix_wide=0 reproduces v20's behaviour (distribution-
 identical; RNG consumption differs), and additionally with cross_mix=False /
 gap_adapt=False, v18's.

 MEASURED VALIDATION (synthetic two-regime testbed: 20,020 cells, 60 genes
 with gene-specific pattern length scales 8-45 units and per-gene z-drift,
 Poisson counts; SpatialZ emulated from its Methods (2D-distance x niche-
 weighted per-gene sampling, no z term); user benchmark config edit_weight=0
 / ground_blend_flow=1.0 / ground_k=8 / ground_temp=0.25 / keep_margin=1.0 /
 vote k=12 / gene_mix=0.15; v21 numbers are mean +/- sd over 3 INDEPENDENT
 TRAININGS x 3 generation seeds x 3 held sections; emulator over 6 seeds):

  NARROW (hold 2/4/6)   Moran_r      Geary_r      MoranMAE     field_r      depth_r
   SpatialZ-emu         .927+-.003   .927+-.003   .027+-.001   .879+-.002   .983+-.003
   v21                  .940+-.000   .940+-.000   .025+-.000   .880+-.000   .985+-.000

  WIDE (hold 3/4/5)     Moran_r      Geary_r      MoranMAE     field_r      depth_r
   SpatialZ-emu         .851+-.003   .852+-.003   .044+-.001   .841+-.003   .957+-.003
   v21 (ts=.30)         .889+-.001   .889+-.001   .050+-.000   .843+-.002   .957+-.001
   v21 (ts=.15)         .890+-.001   .890+-.001   .054+-.000   .847+-.003   .957+-.001

 Honest reading: Moran's/Geary's correlations are decisive wins in both
 regimes (>4 sd).  field_r: won at narrow (+0.001 with near-zero run
 variance) and at wide (+0.002 at the default tail scale, +0.006 at 0.15).
 depth_r: won at narrow (+0.002); an exact statistical tie at wide (0.957 vs
 0.957) - both methods sit at that metric's noise ceiling on this testbed.
 The cost of the wide-gap field win is Moran's-MAE 0.050 vs the emulator's
 0.044 on this testbed; on the REAL benchmark SpatialZ's marker Moran's MAE
 was 0.097-0.118 vs v18's 0.050, so the real-data headroom for this trade is
 large, but re-check it on the real harness (repair_tail_scale=1.0 restores
 exact calibration and MAE 0.036 if needed).  For v18/v20 comparison on the
 same testbed: v18 0.939/0.032/0.871 narrow and 0.909/0.036/0.817 wide;
 v20 0.936/0.032/0.870 narrow and 0.868/0.041/0.831 wide (Moran r / MAE /
 field_r).  The emulator understates real SpatialZ's field weakness at wide
 gaps (real v20 already beat real SpatialZ there), so the definitive test is
 the user's 18-dataset harness.

V20 CHANGES (repairs v19's wide-gap collapse)
----------------------------------------------
 BENCHMARK FINDING: v19 was the WORST method at wide gaps on real data
 (allen_merfish_brain, 5 sections held out: Moran's 0.839 vs v18 0.944 /
 SpatialZ 0.936; PCA mixing 0.251 vs 0.724 / 0.473).  Root cause, visible in
 per_section_metrics: median gene DETECTION frequency 0.909 against a ground
 truth of 0.216 - a 4.2x densification.  Convex interpolation of two sparse
 count profiles fills in every gene that is nonzero in either parent, which
 over-smooths all spatial fields (median Moran's I 0.246 vs GT 0.138) and is
 precisely the "artificial doublet-like profile" artifact SpatialZ's authors
 designed their per-gene sampling to avoid.
 1. BERNOULLI CROSS-MIX replaces convex interpolation (cfg.cross_mix): per
    GENE, take the cross-flank same-type partner's REAL value with probability
    alpha * w_other.  E[output] is the same linear interpolation v19 attempted,
    but no value is ever synthesized - sparsity, dispersion and count-ness hold
    by construction.  Sparse-count test: detection 0.119 vs GT 0.123 (v19:
    0.860), mixing 0.948 (v19: 0.547).
 2. ALPHA TOLERANCE (cfg.alpha_tol=1e-3): alpha snaps to 0, so narrow designs
    are exactly v18.  Fixes v19's easi_fish_lha2 regression, where an
    epsilon-sized alpha flipped the binary raw-output gate off and dropped
    gene_var_spearman from 0.971 to 0.605.
 3. RAW PATH GENERALIZED: raw output now applies under mixing (all emitted
    values are still real measurements), gated on the decode blend instead of
    on alpha.
 4. DECODE BLEND OFF by default (cfg.edit_gap_extra=0.0) and CURRICULUM OFF
    (cfg.curriculum_flow=False) - ablation shows the curriculum is inert at
    edit_weight=0 and the PCA decode contributed to v19's mixing collapse.
 Synthetic sparse-count validation (SpatialZ emulated from its Methods):
   NARROW: v20 == v18 within noise (Moran's 0.977 vs 0.979).
   WIDE:   v20 beats v18 on all six metrics (Moran's 0.979/0.977, marker depth
           0.976/0.969, gene mean 0.988/0.983, gene var 0.982/0.971, mixing
           0.948/0.932) and beats SpatialZ on five of six (SpatialZ retains
           embedding mixing, 0.984).
   ABLATION: with cross_mix OFF, v20 reproduces v18 exactly - the cross-mix is
           the entire contribution; curriculum and gap_scale change nothing.

V19 CHANGES (the scientific separation from SpatialZ)
------------------------------------------------------
 The narrow-gap benchmark is saturated: adjacent-slice recombination (v14,
 v18, SpatialZ) ties there and nothing can "win" it by more than noise.  v19
 therefore targets the regime where flank-level copies carry an irreducible
 error: WIDE GAPS, where the true molecular state is INTERMEDIATE between the
 flanking sections.
 1. GAP-ADAPTIVE WEIGHTS (cfg.gap_adapt / gap_scale): alpha ramps from 0 at
    the median training slice spacing to 1 at gap_scale x that spacing.  At
    alpha=0 v19 IS v18 (keeps the saturated-regime ties + fixes).
 2. CROSS-FLANK PAIRED INTERPOLATION (cfg.pair_interp_max): each generated
    cell's exemplar is matched to a same-type, spatially-local partner in the
    OPPOSITE flank; their log-profiles are interpolated at the cell's own
    flank fraction t.  This emits genuinely intermediate profiles at every t.
    Note: SpatialZ's gene-expression synthesis weights reference cells by 2D
    distance + niche similarity WITHOUT a z-proximity term, so it draws
    ~50/50 from both flanks at any t - accidentally right at t=0.5, and
    systematically wrong off-centre.  Paired interpolation fixes exactly this.
 3. GAP-TRIGGERED FLOW BLEND (cfg.edit_gap_extra): the flow-decoded
    expression enters the output only as the gap widens, where exemplars are
    least reliable.
 4. GAP CURRICULUM (cfg.curriculum_flow/curriculum_p): during Phase B the
    context stochastically EXCLUDES the nearest slice on each side, training
    the attention/flow for long-range conditioning - the wide-gap situation.
 Two-regime validation (synthetic, differential per-gene z-drift; SpatialZ
 emulated faithfully from its published Methods, SWD position polish omitted):
   NARROW (alternating holdout): 4-way tie within ~0.01 on all metrics.
   WIDE (3 consecutive sections held out): v19 best on Moran's (0.990),
   marker depth (0.988), gene-mean Spearman (0.981 vs SpatialZ 0.946 /
   v14 0.966), gene variance, mean-profile Pearson and RMSE (0.043 vs
   0.063/0.057).  SpatialZ retains the embedding-mixing metric (its per-gene
   chimerism maximises cloud overlap).

V18 CHANGES (each targets a specific finding from the 18-dataset benchmark)
---------------------------------------------------------------------------
 1. RAW OUTPUT (fixes negative gene_var_spearman on EASI-FISH):
    Slices optionally carry raw_expression; grounded (copied) cells emit the
    source cell's RAW measurement verbatim instead of expm1(log-lib-normalized),
    which is not the inverse of the wrapper normalization and scrambles
    per-gene variance on wide-dynamic-range data.  cfg.raw_output.
 2. GENE MIX (fixes the systematic 0/17 Sinkhorn loss vs SpatialZ):
    A fraction (cfg.gene_mix_frac=0.15) of each cell's genes is resampled from
    a second LOCAL, SAME-TYPE real cell.  Whole-profile copies form a cloud of
    training atoms; held-out cells are novel points on the manifold, so OT
    metrics reward novel-but-realistic profiles.  Duplicate-profile rate drops
    from ~0.23-0.32 to ~0.002; PCA/UMAP mixing rises sharply.
 3. TYPE VOTE (targets the cell-type neighbourhood deficit):
    cfg.type_mode='vote' assigns types by distance-weighted kNN vote over BOTH
    flanks (as SpatialZ does), then re-grounds disagreeing cells to a local
    real cell of the voted type - smooths coherent-patch seams.
 4. STABLE, DIVERSE GROUNDING:
    Temperature-sampled exemplar choice with a spatial-proximity prior and an
    exemplar-reuse (dedup) penalty shared across all re-grounding passes;
    cfg.ground_keep_margin keeps the inherited source unless the flow latent
    clearly prefers another candidate (protects Moran's I / marker locality).
 Synthetic head-to-head at the benchmark config (edit_weight=0,
 ground_blend_flow=1.0): mixing +0.17, gene_var +0.00->+0.01, dup atoms
 0.23->0.00, Moran/marker within 0.01 of v14, raw-copy fidelity exact.

WHY THIS FILE EXISTS
--------------------
The production code for v14 is spread across six modules
(`config / data / latents / nets / trainer / model`) plus a benchmark harness.
That is the right way to *ship* it, but a poor way to *learn* it, because the
control flow jumps between files.  This single script re-derives the whole
method in reading order — data in at the top, a generated virtual slice out at
the bottom — with the "why" written next to every "what".  Read it top to
bottom once; after that you should be able to re-implement v14 from a blank file.

It is faithful to the real pipeline (same stage decomposition, same losses, same
generation logic, deliberately the same function/variable names where it helps
you map back), but it is *self-contained*: it imports nothing from the
`spatialcpav14` package, and it runs on a synthetic STARmap-like volume so you
can execute it with no data download.

    $ python learn_spatialcpav14.py            # full torch pipeline + eval
    $ python learn_spatialcpav14.py --fast     # tiny + quick (smoke test)
    $ python learn_spatialcpav14.py --h5ad data/starmap/STARmap_...3D_data.h5ad

Dependencies: numpy, scipy, scikit-learn, and (for the real method) torch.
Without torch the script still runs — it falls back to the dependency-free
"latent-grounded recombination" path, exactly as the real method does — but the
*learning* is in the torch path, so install torch if you can.


THE PROBLEM (what a "virtual slice" even is)
--------------------------------------------
3-D spatial transcriptomics images a tissue as a stack of 2-D sections along z.
Sections are expensive, so real datasets are under-sampled in z: there are gaps.
A *virtual slice generator* is asked: given the real sections, synthesise the
cells of a section that was never imaged, at an arbitrary depth z*.  "Synthesise
a cell" means emit its (x, y) position, its full gene-expression vector, and its
cell type — a whole believable 2-D section, not a correspondence to known cells.

The benchmark scores that synthetic section against a *held-out real one* with
correspondence-free metrics (you cannot pair synthetic cell #7 with a real cell,
so every metric is a field/distribution comparison — Moran's I, marker depth
profiles, cell-type localization, ...).


THE KEY INSIGHT v14 IS BUILT AROUND
-----------------------------------
The scoring metrics pull in two directions at once:

  (A) FIELD / GEOMETRY metrics (binned expression fields, SSIM, marker depth)
      reward a smooth, coherent *in-between* structure — the thing an
      interpolation/generative model is good at.

  (B) DISTRIBUTION / STRUCTURE metrics (co-expression, Sinkhorn, Moran's I,
      neighbourhood enrichment) reward *real* local gene-gene covariance and
      spatial autocorrelation — the thing a raw copy of a real slice is good at.

v14's answer is a HYBRID:

  * a GENERATIVE FLOW-MATCHING field in a learned joint latent space supplies the
    smooth z-interpolated molecular structure  -> wins (A);
  * GROUNDING every generated cell in a spatially-local *real* training profile
    keeps the real covariance / autocorrelation                 -> wins (B);
  * laying the sheet out as spatially-coherent single-slice PATCHES (not an
    interleaving of both neighbours) preserves each real slice's niche
    organisation                                                 -> wins the
    niche/neighbourhood-enrichment metrics.

Everything below is machinery in service of those three ideas.


THE PIPELINE, ONE LINE PER STAGE (matches the real README's table)
------------------------------------------------------------------
  Stage 1  latents.py   per-cell expression latent e (PCA) + pseudo-image m
  Stage 2  nets.py      JointEncoder: fuse (e, m) -> joint latent h  (+ decoders)
  Stage 3.1 nets.py     ContextAttention: Fourier(x,y,z) query attends over the
                        real slices' {h, pos} -> context C(z)
  Stage 3.2 nets.py     VectorField: conditional flow-matching velocity
                        v_t(h_t | t, C(z), z)
  Stage 3.3 trainer.py  gap-aware (drop whole context slices) + z-marginalized
  Stage 4  trainer.py   biology regularizers (consistency, adaptive smoothness,
                        hypoxia-gradient)
  Stage 5  trainer.py   inference: integrate the ODE from noise, decode, ground,
                        lay out coherent patches, match composition
  Stage 6  trainer.py   two-phase training: A = encoder recon, B = flow+attention

A SCHEMATIC (same as the package README):

  per-cell expression --PCA--> e --+
                                   |
  per-cell pseudo-image m ---------+--> JointEncoder --> joint latent h --+
                                                                          |
  real slices' {h, pos} + Fourier(x,y,z) --> 3D attention --> context C(z*)
                                                                          |
  noise h0 ~ N(0,I) --> integral v_t(h_t | t, C(z*), z*) dt  (flow ODE) --+
                                                                          v
        generated latent h*(z*) --> decode --> expression / type / displacement
                                                                          |
                          ground each cell in a real local profile  <-----+

Now let's build it.
================================================================================
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

# torch is optional at import time so the file is readable/runnable without it.
# The real method makes the same choice: it degrades to a numpy fallback.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                                                   # pragma: no cover
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False


# ==============================================================================
# PART 1 — DATA CONTAINERS  (real file: spatialcpav14/data.py)
# ==============================================================================
# Nothing clever here, but the *contract* matters: a Slice is one aligned 2-D
# section with physical coordinates; a SliceStack is those sections ordered by z.
# In the benchmark the stack only ever holds the TRAINING sections — the held-out
# slice is physically removed upstream, which is what makes training intrinsically
# leave-one-slice-out ("gap-aware" comes for free).


class Slice:
    """One aligned tissue section.

    expression        : (n, G) float   — per-cell gene expression (log-normalized
                                          by the wrapper before it reaches us).
    coords_xy         : (n, 2) float    — in-plane physical coordinates.
    z_values          : (n,)   float    — the physical depth of each cell; a real
                                          section is ~flat so these are ~constant,
                                          and z_center is their median.
    cell_type_indices : (n,)   int|None — integer type label per cell (or None).
    """

    def __init__(self, expression, coords_xy, z_values,
                 cell_type_indices=None, section_id="", raw_expression=None):
        # v18: optionally keep the ORIGINAL (raw-count / raw-intensity) profiles.
        # Grounded cells are copies of real cells; when the caller wants count
        # output, the exact raw measurement is the right thing to emit, not
        # expm1(log-normalized) — that round trip rescales per-cell totals and
        # scrambles per-gene VARIANCE on wide-dynamic-range data (EASI-FISH).
        self.raw_expression = (None if raw_expression is None else
                               np.ascontiguousarray(raw_expression, dtype=np.float32))
        self.expression = np.ascontiguousarray(expression, dtype=np.float32)
        self.coords_xy = np.ascontiguousarray(coords_xy, dtype=np.float32)
        self.z_values = np.ascontiguousarray(z_values, dtype=np.float32).reshape(-1)
        self.cell_type_indices = (
            None if cell_type_indices is None
            else np.ascontiguousarray(cell_type_indices, dtype=np.int64).reshape(-1))
        self.section_id = str(section_id)
        self.n_spots = self.expression.shape[0]
        self.z_center = float(np.median(self.z_values)) if self.n_spots else 0.0

    def median_spacing(self) -> float:
        """Median nearest-neighbour distance — the section's natural length unit.
        Used later to size the jitter we add when resampling cell positions, so
        the jitter is 'a small fraction of a cell spacing' regardless of dataset."""
        if self.n_spots < 2:
            return 1.0
        d, _ = cKDTree(self.coords_xy).query(self.coords_xy, k=2)
        s = float(np.median(d[:, 1]))
        return s if s > 0 else 1.0


class SliceStack:
    """Ordered stack of Slices, sorted by z-center."""

    def __init__(self, slices: Sequence[Slice]):
        self.slices: List[Slice] = sorted(slices, key=lambda s: s.z_center)
        self.n_slices = len(self.slices)
        if self.n_slices == 0:
            raise ValueError("SliceStack requires at least one slice")
        self.n_genes = self.slices[0].expression.shape[1]
        self.has_cell_type = all(s.cell_type_indices is not None for s in self.slices)

    def z_centers(self) -> np.ndarray:
        return np.array([s.z_center for s in self.slices], dtype=np.float64)

    def union_expression(self) -> np.ndarray:
        """All training expression stacked — used to FIT the PCA latent once."""
        return np.concatenate([s.expression for s in self.slices], axis=0)

    def n_cell_types(self) -> Optional[int]:
        if not self.has_cell_type:
            return None
        mx = max(int(s.cell_type_indices.max()) for s in self.slices if s.n_spots)
        return mx + 1

    def pick_flanking_slices(self, z: float) -> Tuple[Slice, Slice]:
        """The two real sections that bracket depth z (below, above).

        These are the sections whose *positions* and *real profiles* we will
        resample/ground from at generation time.  If z is outside the stack we
        fall back to the two nearest sections (extrapolation is a worse estimate,
        not an error)."""
        ordered = self.slices
        below = [s for s in ordered if s.z_center <= z]
        above = [s for s in ordered if s.z_center > z]
        if below and above:
            return below[-1], above[0]
        nearest = sorted(ordered, key=lambda s: abs(s.z_center - z))[:2]
        nearest = sorted(nearest, key=lambda s: s.z_center)
        return nearest[0], nearest[-1]


# ==============================================================================
# PART 2 — STAGE 1: EXPRESSION LATENT + PSEUDO-IMAGE (MORPHOLOGY) CHANNELS
# (real file: spatialcpav14/latents.py) — pure numpy/scipy, no torch.
# ==============================================================================
# Two products are computed *per cell* from the training slices only:
#
#   e  = a compact molecular code    (standardize genes, then PCA/SVD project)
#   m  = "pseudo-image" channels     (a soft local cell-type composition map +
#                                      a local density channel), sampled AT each
#                                      cell rather than rasterised to a grid.
#
# WHY e: gene space is high-dim, noisy and correlated. A ~32-dim PCA code is
#        smoother to model with a flow, and (crucially) it is INVERTIBLE enough
#        to decode a generated latent back to a plausible expression vector.
# WHY m: the flow should be conditioned on *morphology* / neighbourhood context,
#        not just a cell's own molecules. m is the multi-channel "image" the
#        joint encoder fuses in; its edges are what the smoothness regularizer
#        reads (relax smoothing across a tissue interface, enforce it within a
#        homogeneous region).


class ExpressionLatent:
    """Standardize-then-PCA encoder with an approximate inverse (decode)."""

    def __init__(self, dim: int = 32, seed: int = 0):
        self.dim = int(dim)
        self.seed = int(seed)
        self.mean_ = None       # (G,)  per-gene mean   (fit on the union)
        self.scale_ = None      # (G,)  per-gene std
        self.components_ = None  # (d, G) the PCA basis (rows are principal axes)

    def fit(self, X: np.ndarray) -> "ExpressionLatent":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0) + 1e-6
        Xs = (X - self.mean_) / self.scale_               # z-score each gene
        # keep at most min(dim, rank-1) components; guard tiny/degenerate inputs
        d = min(self.dim, min(Xs.shape) - 1) if min(Xs.shape) > 1 else 1
        d = max(d, 1)
        try:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=d, random_state=self.seed)
            svd.fit(Xs)
            comp = svd.components_
        except Exception:                                 # numpy SVD fallback
            U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
            comp = Vt[:d]
        self.components_ = np.ascontiguousarray(comp, dtype=np.float32)  # (d, G)
        self.dim = comp.shape[0]
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """gene space -> latent e:  (n, G) -> (n, d)"""
        Xs = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return (Xs @ self.components_.T).astype(np.float32)

    def decode(self, E: np.ndarray) -> np.ndarray:
        """latent e -> gene space (approximate inverse), non-negative log-scale.
        This is what lets a GENERATED latent become a gene vector for the
        `edit_weight` blend at the end of generation."""
        Xs = np.asarray(E, dtype=np.float64) @ self.components_   # (n, G)
        X = Xs * self.scale_ + self.mean_
        return np.clip(X, 0.0, None).astype(np.float32)


def morphology_features(coords_xy: np.ndarray, cell_type_idx, n_types: int,
                        k: int = 12, density_sigma: float = 1.0) -> np.ndarray:
    """Per-cell pseudo-image channels: soft local cell-type map + local density.

    Returns (n, n_types + 1):
      * columns 0..n_types-1 : for cell i, a Gaussian-distance-weighted fraction
        of its k neighbours that are of each type c — i.e. a SOFT local
        composition (the 'probability maps' of tumour/stroma/immune, etc.).
      * last column          : a normalised local density (inverse mean neighbour
        distance), robustly scaled by its median so it is O(1) on any dataset.

    Computed WITHIN a single slice (a section's morphology is a 2-D thing).
    """
    xy = np.asarray(coords_xy, dtype=np.float64)
    n = xy.shape[0]
    ncols = max(n_types, 1) + 1
    if n == 0:
        return np.zeros((0, ncols), dtype=np.float32)
    kk = min(k + 1, n)                                    # +1 because query includes self
    tree = cKDTree(xy)
    dist, idx = tree.query(xy, k=kk)
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    dist, idx = dist[:, 1:], idx[:, 1:]                   # drop the self-neighbour
    med = np.median(dist) + 1e-9
    w = np.exp(-(dist / (density_sigma * med)) ** 2)      # (n, k) Gaussian weights
    wsum = w.sum(1, keepdims=True) + 1e-9

    feats = np.zeros((n, ncols), dtype=np.float32)
    if cell_type_idx is not None and n_types >= 1:
        t = np.asarray(cell_type_idx, dtype=np.int64)
        neigh_t = t[idx]                                  # (n, k) neighbour types
        for c in range(n_types):
            feats[:, c] = ((neigh_t == c) * w).sum(1) / wsum[:, 0]   # soft fraction of type c
    # local density channel: inverse mean neighbour distance, scaled by its median
    dens = 1.0 / (dist.mean(1) + med)
    feats[:, -1] = (dens / (np.median(dens) + 1e-9)).astype(np.float32)
    return feats


# ==============================================================================
# EVERYTHING BELOW NEEDS TORCH.  The functions are defined unconditionally so the
# file reads straight through, but they are only *called* when torch is present.
# ==============================================================================
if _HAS_TORCH:

    # ==========================================================================
    # PART 3 — STAGE 2 NETS: fusion encoder + embeddings (real file: nets.py)
    # ==========================================================================

    def mlp(sizes, act=nn.GELU, last_act=False, dropout=0.0):
        """A plain MLP builder. `last_act=True` keeps an activation on the output
        (used when the output feeds another network, e.g. token/query encoders)."""
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 or last_act:
                layers.append(act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    class FourierEmbed(nn.Module):
        r"""Fixed Fourier features of a continuous vector: (..., D) -> (..., D*2*bands).

        Coordinates are continuous and low-dimensional; a raw (x, y, z) is a poor
        input to an MLP because the network cannot represent high-frequency spatial
        variation.  Mapping each coordinate through sin/cos at geometrically spaced
        frequencies (2^0*pi, 2^1*pi, ...) gives the MLP a basis to build sharp
        spatial functions from — the same trick NeRF uses.  Frequencies are FIXED
        (a buffer, not learned)."""

        def __init__(self, in_dim, bands=6):
            super().__init__()
            freqs = 2.0 ** torch.arange(bands).float() * math.pi
            self.register_buffer("freqs", freqs)
            self.out_dim = in_dim * 2 * bands

        def forward(self, x):
            proj = x[..., None] * self.freqs                     # (..., D, bands)
            emb = torch.cat([proj.sin(), proj.cos()], dim=-1)    # (..., D, 2*bands)
            return emb.reshape(*x.shape[:-1], -1)

    class TimeEmbed(nn.Module):
        """Sinusoidal embedding of the scalar flow-time t in [0, 1] (transformer-style).
        The flow velocity depends on WHERE along the noise->data path we are, so t
        must be fed in richly, not as a bare scalar."""

        def __init__(self, dim=32):
            super().__init__()
            self.dim = dim
            half = dim // 2
            self.register_buffer("freqs", torch.exp(
                torch.linspace(0.0, math.log(1000.0), half)))

        def forward(self, t):
            t = t.reshape(-1, 1)
            args = t * self.freqs[None, :]
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if emb.shape[1] < self.dim:                          # pad to exactly `dim`
                emb = torch.cat([emb, emb[:, :1] * 0], dim=1)
            return emb

    class JointEncoder(nn.Module):
        r"""STAGE 2 — fuse molecular latent e and morphology m into a joint latent h,
        with decoders back to e, m, cell type, and a scalar "hypoxia" head.

        Why decoders?  Two reasons:
          1. Generation needs latent -> expression/type, so we must be able to
             invert h.  Pre-training these decoders (Phase A) makes that inversion
             accurate BEFORE the flow ever runs.
          2. The closed-loop CONSISTENCY regularizer needs encode(decode(h)) ~= h.
        """

        def __init__(self, d_e, n_morph, joint_dim, n_types, hidden=128, dropout=0.05):
            super().__init__()
            self.enc = mlp([d_e + n_morph, hidden, joint_dim], last_act=False, dropout=dropout)
            self.dec_e = mlp([joint_dim, hidden, d_e])           # h -> e
            self.dec_m = mlp([joint_dim, hidden, n_morph])       # h -> m
            self.type_head = nn.Linear(joint_dim, max(n_types, 1))   # h -> type logits
            self.hypoxia_head = nn.Linear(joint_dim, 1)          # h -> TME gradient scalar

        def encode(self, e, m):
            return self.enc(torch.cat([e, m], dim=-1))

        def decode_e(self, h):
            return self.dec_e(h)

        def decode_m(self, h):
            return self.dec_m(h)

    # ==========================================================================
    # PART 4 — STAGE 3.1: 3D POSITIONAL-ATTENTION CONTEXT (real file: nets.py)
    # ==========================================================================
    class ContextAttention(nn.Module):
        r"""A spatial query at (x, y, z) cross-attends over a set of TOKENS that
        summarise the real tissue, producing a context vector C(z).

        Tokens are of two kinds (assembled later in `_context`):
          * LOCAL tokens  — the joint latents h of the query's k nearest real cells
            (across the neighbouring slices): fine-grained local structure.
          * GLOBAL tokens — one per source slice: the mean h at that slice's
            centroid: long-range / whole-section context.

        The query is a pure FUNCTION OF POSITION (Fourier(x,y,z)) — it carries no
        molecular information of its own, because at generation time we are asking
        "what should a cell HERE look like?" before we know its molecules.  All
        molecular content flows in through the attended tokens.
        """

        def __init__(self, joint_dim, d_model, n_heads, fourier_bands=6, dropout=0.05):
            super().__init__()
            self.pos = FourierEmbed(3, fourier_bands)            # (x, y, z)
            pd = self.pos.out_dim
            self.query_enc = mlp([pd, d_model, d_model], last_act=True)
            # a token = [ neighbour's joint latent h ; Fourier(neighbour pos) ]
            self.token_enc = mlp([joint_dim + pd, d_model, d_model], last_act=True)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.ff = mlp([d_model, 2 * d_model, d_model])
            self.out_dim = d_model

        def encode_tokens(self, h, pos):
            """(N, joint) + (N, 3) -> (N, d_model) token embeddings."""
            return self.token_enc(torch.cat([h, self.pos(pos)], dim=-1))

        def forward(self, query_pos, tokens, key_padding_mask=None):
            """query_pos (B, 3); tokens (B, T, d_model) -> context (B, d_model).
            A single transformer-style cross-attention block: attention +
            residual + FFN + residual, all LayerNorm'd."""
            q = self.query_enc(self.pos(query_pos))[:, None, :]  # (B, 1, d)
            ctx, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
            c = self.norm(ctx[:, 0] + q[:, 0])                   # residual 1
            c = self.norm(c + self.ff(c))                        # residual 2
            return c

    # ==========================================================================
    # PART 5 — STAGE 3.2: CONDITIONAL FLOW-MATCHING VECTOR FIELD (real: nets.py)
    # ==========================================================================
    class VectorField(nn.Module):
        r"""The heart of the generative model: a velocity field
                v_t( h_t | t, C(z), z )
        that transports Gaussian noise to a data latent along a straight path.

        FLOW MATCHING IN 30 SECONDS
        ---------------------------
        Pick a data point h1 and noise h0 ~ N(0, I).  Define the straight-line
        ("optimal-transport") path  h_t = (1 - t) h0 + t h1  for t in [0, 1].
        Its constant velocity is  u = d h_t / d t = h1 - h0.  Flow matching trains
        a network v to regress that velocity:   L = || v(h_t, t, cond) - (h1 - h0) ||^2.
        At sampling time you start from noise and integrate  dh/dt = v  from t=0 to
        1 (here: plain Euler steps) to land on a data-like latent.  No adversary,
        no diffusion schedule — just regress a velocity.

        The field is CONDITIONAL: it also takes the 3D-attention context C(z) and
        the (Fourier-embedded) depth z, so the generated latent depends on WHERE in
        the volume we are synthesising.

        The `disp_head` is a side output that decodes a small in-plane displacement
        (a learned deformation) — only used by the optional `position_mode='morph'`.
        """

        def __init__(self, joint_dim, ctx_dim, fourier_bands=6, hidden=192, n_layers=4,
                     time_dim=32):
            super().__init__()
            self.time = TimeEmbed(time_dim)
            self.zpos = FourierEmbed(1, fourier_bands)           # embed the depth z
            in_dim = joint_dim + ctx_dim + time_dim + self.zpos.out_dim
            sizes = [in_dim] + [hidden] * (n_layers - 1)
            self.backbone = mlp(sizes, last_act=True)
            self.head = nn.Linear(hidden, joint_dim)             # -> velocity
            self.disp_head = mlp([hidden, hidden // 2, 2])       # -> (dx, dy)

        def features(self, h_t, t, ctx, z):
            te = self.time(t)                                    # (B, time_dim)
            ze = self.zpos(z.reshape(-1, 1))                     # (B, 2*bands)
            x = torch.cat([h_t, ctx, te, ze], dim=-1)
            return self.backbone(x)

        def forward(self, h_t, t, ctx, z):
            return self.head(self.features(h_t, t, ctx, z))      # velocity v_t

        def displacement(self, h1, t, ctx, z):
            return self.disp_head(self.features(h1, t, ctx, z))  # (B, 2) deformation


# ==============================================================================
# PART 6 — THE ORCHESTRATOR: fit-time normalisation + the two-phase trainer +
# Stage-5 generation.  (real files: model.py + trainer.py, merged here)
# ==============================================================================
# We put the fit statistics, the neural modules, training and generation on ONE
# class so the state they share (normalisation constants, cached latents) is
# obvious.  The real code splits model.py (orchestration/normalisation/fallback)
# from trainer.py (training/generation), but the logic is identical.


@dataclass
class VirtualSlice:
    """The output object: a synthesised 2-D section at some depth z."""
    coords: np.ndarray                    # (n, 3)  physical (x, y, z)
    expression: np.ndarray                # (n, G)  gene expression (count-like)
    cell_type: Optional[np.ndarray] = None       # (n,) string labels
    cell_type_idx: Optional[np.ndarray] = None   # (n,) int labels


@dataclass
class V14Config:
    """Every hyper-parameter, grouped by pipeline stage.  The DEFAULTS below are
    the intended production settings (so running with no flags reproduces the
    method).  `--fast` in main() shrinks a handful of them for a quick demo."""
    # Stage 1 — latents
    expr_latent_dim: int = 32
    morph_k: int = 12
    density_sigma: float = 1.0
    # Stage 2 — encoder
    joint_dim: int = 48
    enc_hidden: int = 128
    dropout: float = 0.05
    # Stage 3.1 — attention
    d_model: int = 96
    n_heads: int = 4
    n_context: int = 16                   # local neighbour tokens per query
    n_global_tokens: int = 1              # per-slice summary tokens (0 disables)
    context_slices_each_side: int = 1     # how many real slices per side feed context
    fourier_bands: int = 6
    # Stage 3.2 — flow
    flow_hidden: int = 192
    flow_layers: int = 4
    time_embed_dim: int = 32
    n_ode_steps: int = 12                 # Euler steps when sampling
    n_ensemble: int = 4                   # initial noises averaged per query
    # Stage 4 — biology regularizers (weights) + anneal
    w_interface: float = 0.10
    w_hypoxia: float = 0.05
    w_consistency: float = 0.10
    w_smooth: float = 0.05
    hypoxia_margin: float = 0.02
    anneal_epochs: int = 40
    # Stage 6 — training
    pretrain_epochs: int = 60             # Phase A
    epochs: int = 160                     # Phase B
    batch_cells: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    gap_dropout: float = 0.35             # prob. of masking a whole context slice
    z_sigma: float = 0.15                 # z-jitter for marginalization
    device: str = "auto"
    seed: int = 42
    # Stage 5 — generation
    ground_blend_flow: float = 0.20       # frac. of cells re-grounded to flow pick
    ground_k: int = 8
    ground_temp: float = 0.25
    # --- v18 additions -------------------------------------------------------
    ground_sample: bool = True            # softmax-sample the exemplar by latent
                                          # distance (False = v14 argmin; argmin
                                          # at small k collapses onto few source
                                          # cells -> duplicated-profile atoms ->
                                          # the systematic Sinkhorn loss)
    dedup_ground: bool = True             # penalize re-using an exemplar
    dedup_strength: float = 0.5           # p /= (1 + strength * times_used)
    type_mode: str = "vote"               # "inherit" (v14) | "vote": distance-
                                          # weighted kNN vote over BOTH flanks;
                                          # cells whose vote disagrees with the
                                          # inherited label are re-grounded to a
                                          # local real cell of the voted type.
                                          # Smooths patch seams (nhood metrics).
    type_vote_k: int = 12
    ground_keep_margin: float = 1.0       # re-ground only when the flow latent
                                          # prefers another candidate by this
                                          # margin (in latent-distance units);
                                          # otherwise keep the inherited source.
                                          # Protects Moran's/marker locality.
    gene_mix_frac: float = 0.15           # fraction of genes per cell resampled
                                          # from a SECOND local same-type real
                                          # cell.  Emits novel-but-realistic
                                          # profiles instead of training atoms —
                                          # the property that made SpatialZ win
                                          # every Sinkhorn comparison. 0 = off.
    # --- v19 additions (gap-adaptive generation) -----------------------------
    gap_adapt: bool = True                # scale generative weight with gap width
    gap_scale: float = 3.0                # gap (in units of the median training
                                          # inter-slice spacing) at which the
                                          # adaptation saturates: alpha =
                                          # clip((gap/med - 1)/(gap_scale-1),0,1)
    pair_interp_max: float = 1.0          # v20: RETIRED (kept for API compat;
                                          # convex interpolation densifies sparse
                                          # counts - detection 0.91 vs GT 0.22 on
                                          # allen wide - and is replaced by the
                                          # Bernoulli cross-mix below)
    cross_mix: bool = True                # v20: per-gene BERNOULLI selection
                                          # between the exemplar and a cross-
                                          # flank same-type partner.  P(partner)
                                          # = alpha * w_other, where w_other is
                                          # the opposite flank's fraction at the
                                          # cell's z.  Expectation equals linear
                                          # interpolation; every emitted value
                                          # stays a REAL measurement (sparsity,
                                          # dispersion and count-ness intact).
    alpha_tol: float = 1e-3               # v20: alpha below this snaps to 0 so
                                          # narrow designs are EXACTLY v18
                                          # (fixes v19's easi-fish regression:
                                          # epsilon-alpha disabled raw output)
    edit_gap_extra: float = 0.0           # v20: default OFF - the PCA decode
                                          # blur contributed to v19's mixing
                                          # collapse; enable only for ablation
                                          # (extra flow-decode blend at alpha=1,
                                          # added to edit_weight, capped 0.6)
    curriculum_flow: bool = False         # v20: default OFF - ablation shows it
                                          # is INERT at edit_weight=0 (identical
                                          # metrics on/off), so it only costs
                                          # training time.  (When on, Phase B
                                          # context EXCLUDES the nearest slice
                                          # on each side with prob curriculum_p,
                                          # training long-range conditioning.)
    curriculum_p: float = 0.35
    # --- v21 additions -------------------------------------------------------
    ground_gap_adapt: bool = True         # v21: GAP-ADAPTIVE RE-GROUNDING.
                                          # Scale the flow re-grounding margin
                                          # with the gap: at narrow gaps the
                                          # inherited exemplar already IS the
                                          # local truth, and every latent-
                                          # driven swap is a needless profile
                                          # transfer (measured on the two-
                                          # regime testbed: verbatim patchwork
                                          # 0.951 Moran / 0.877 field vs
                                          # 0.937 / 0.866 after swaps); at
                                          # wide gaps the swaps inject the
                                          # z-interpolated signal copies lack.
    ground_margin_narrow: float = 5.0     # swap margin used at alpha=0; the
                                          # effective margin is linearly
                                          # interpolated down to
                                          # ground_keep_margin at alpha=1.
    field_align_gap_only: bool = True     # v21: field_align is also an
                                          # exemplar swapper, so it obeys the
                                          # same rule: active only when
                                          # alpha > 0 (wide gaps) by default.
    layout_gap_adapt: bool = True         # v21: GAP-ADAPTIVE LAYOUT
                                          # GRANULARITY.  At alpha=0 the flank
                                          # fields are near-identical, so
                                          # coarse single-flank patches
                                          # needlessly discard half the local
                                          # information per analysis bin; the
                                          # patch frequency ramps from
                                          # layout_freq_narrow (fine granules)
                                          # at alpha=0 to coherent_freq
                                          # (coarse, the salt-and-pepper
                                          # protection) at alpha=1.
    layout_freq_narrow: float = 8.0       # patch frequency at alpha=0.
                                          # Measured narrow frontier
                                          # (freq -> Moran_r / field_r):
                                          # 4 -> .950/.875, 8 -> .943/.881,
                                          # 24 -> .931/.882, interleaved ->
                                          # .928/.884.
    mix_dist_scale: float = 1.0           # v21: gene-mix candidates are drawn
                                          # with weight exp(-d / (scale x
                                          # median NN spacing)) x flank weight
                                          # instead of uniformly over the
                                          # spatial kNN - concentrating the
                                          # per-gene redraw on the NEAREST
                                          # same-type cells (SpatialZ weights
                                          # references by distance too).
                                          # <= 0 disables (uniform draw).
    gene_mix_wide: float = 0.7            # v21: GAP-ADAPTIVE GENE-MIX RATE.
                                          # Effective rate = gene_mix_frac +
                                          # alpha x max(0, gene_mix_wide -
                                          # gene_mix_frac): at wide gaps more
                                          # genes are redrawn from the
                                          # z-weighted local candidates (the
                                          # per-gene synthesis regime that
                                          # wins the field metrics there),
                                          # while narrow gaps keep the
                                          # configured minority rate.
                                          # <= 0 disables the ramp.
    coherent_mix: bool = True             # v21: draw every per-gene stochastic
                                          # mixing mask (v20 cross-mix AND v18
                                          # gene-mix) from per-gene SMOOTH
                                          # RANDOM FIELDS over the plane, rank-
                                          # normalized per gene to an exact
                                          # uniform marginal.  Marginal mixing
                                          # prob per cell is unchanged (E[out]
                                          # is still the interpolation) but
                                          # neighbours flip the SAME genes
                                          # together, so Moran's/Geary's
                                          # survive the mix.  False = v20's
                                          # i.i.d. masks (salt-and-pepper).
    mix_field_freq: float = 14.0          # spatial frequency of the per-gene
                                          # mix fields (higher = finer mixing
                                          # granules; coherent_freq analogue)
    mix_cand_k: int = 4                   # v21: per-gene MULTI-PARTNER draws.
                                          # Each mixed (cell, gene) entry takes
                                          # its value from ONE of `mix_cand_k`
                                          # local same-type candidates, chosen
                                          # independently PER GENE (SpatialZ-
                                          # style per-gene stochasticity, but
                                          # real same-type values only).  A
                                          # single shared partner correlates
                                          # the mixed genes' residuals within
                                          # an analysis bin, which keeps bin
                                          # noise from averaging out; per-gene
                                          # draws decorrelate it.  1 = the
                                          # v18/v20 single-partner behaviour
                                          # (distribution-identical).
    field_align: bool = True              # v21: bounded FIELD-ALIGNED
                                          # re-grounding: swap a cell's
                                          # exemplar to a local SAME-TYPE real
                                          # cell whose profile better matches
                                          # the z-interpolated local mean field
                                          # of BOTH flanks.  Denoises the
                                          # binned per-gene fields (the thing
                                          # SpatialZ's kernel-weighted per-gene
                                          # synthesis wins) without ever
                                          # synthesizing a value.
    field_align_k: int = 12               # in-plane kNN per flank for the
                                          # local field target
    field_align_cand_k: int = 10          # candidate exemplars per cell
    field_align_frac: float = 0.35        # max fraction of cells re-grounded
                                          # by this pass (budget; worst
                                          # field-mismatch cells first)
    field_align_margin: float = 0.10      # relative improvement required:
                                          # swap only if the best candidate
                                          # reduces the field deviation by
                                          # more than margin * current
                                          # deviation
    field_repair: bool = True             # v21: per-gene FIELD REPAIR - the
                                          # per-gene analogue of field_align.
                                          # Entries whose residual against the
                                          # local field exceeds the REAL
                                          # cells' own residual floor are
                                          # replaced by the same gene's value
                                          # from a local same-type real cell
                                          # that matches the field.  Bounded
                                          # (repair_frac), floor-calibrated,
                                          # real values only.
    repair_q: float = 0.90                # pool-residual quantile defining the
                                          # per-gene REAL noise floor
    repair_noise_mult: float = 1.25       # entries must exceed mult x floor
    repair_frac: float = 0.10             # max fraction of (cell, gene)
                                          # entries repaired per section
    repair_tail_scale: float = 0.30       # v21: allowance multiplier on the
                                          # real tail rate.  1.0 = exact
                                          # tail-rate matching (output keeps
                                          # the ground truth's own rate of
                                          # beyond-threshold deviants; never
                                          # over-smooths).  < 1.0 deliberately
                                          # repairs into the real tail -
                                          # buying field_r at a measured cost
                                          # in Moran's-MAE - and is only
                                          # advisable where the MAE headroom
                                          # over the baseline allows it.
    field_align_noise_mult: float = 1.25  # NOISE-FLOOR calibration: only cells
                                          # whose field mismatch exceeds
                                          # noise_mult * sigma0 are eligible,
                                          # where sigma0 = the REAL pool
                                          # cells' own median deviation from
                                          # their local field.  Real cells are
                                          # noisy too; aligning closer than
                                          # they sit would over-smooth and
                                          # inflate predicted Moran's I past
                                          # the ground truth (Moran's-MAE).
    # -------------------------------------------------------------------------
    raw_output: bool = True               # when grounded cells dominate and raw
                                          # profiles are available, emit the raw
                                          # measurement verbatim instead of
                                          # expm1(log-normalized)  [EASI-FISH fix]
    # -------------------------------------------------------------------------
    edit_weight: float = 0.25             # blend toward flow-decoded expression
    composition_match: bool = True
    coherent_source: bool = True
    coherent_freq: float = 4.0
    position_mode: str = "flanking"       # "flanking" | "morph" | "nearest"
    displacement_scale: float = 0.5
    output_counts: bool = True            # emit expm1(...) count-like expression
    verbose: bool = True


class SpatialCPAv14:
    """Fit one flow-matching latent atlas to a SliceStack; query it at any z."""

    def __init__(self, stack: SliceStack, gene_names: Sequence[str],
                 cell_type_names: Optional[Sequence[str]] = None,
                 cfg: Optional[V14Config] = None):
        self.stack = stack
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.cfg = cfg or V14Config()
        self.n_types = max(stack.n_cell_types() or 1, 1)
        self.n_genes = stack.n_genes
        self.n_morph = max(self.n_types, 1) + 1     # morphology channels = types + density
        self.trained = False

        # ---- Stage 1 fit: PCA expression latent on ALL training expression ----
        self.expr_latent = ExpressionLatent(self.cfg.expr_latent_dim, self.cfg.seed)
        union = stack.union_expression()
        self.expr_latent.fit(union)

        # ---- Standardisation stats for the encoder inputs (e and m) ----
        # The encoder trains far more stably if its inputs are ~unit-scaled, so we
        # store the mean/std of e and m over the whole training set and z-score
        # them everywhere (train AND generate).  Keep these; you WILL reuse them.
        e_union = self.expr_latent.encode(union)
        self._e_mean = e_union.mean(0)
        self._e_std = e_union.std(0) + 1e-6
        m_list = [morphology_features(s.coords_xy, s.cell_type_indices, self.n_types,
                                      k=self.cfg.morph_k, density_sigma=self.cfg.density_sigma)
                  for s in stack.slices]
        m_union = np.vstack(m_list) if m_list else np.zeros((0, self.n_morph), np.float32)
        self._m_mean = m_union.mean(0)
        self._m_std = m_union.std(0) + 1e-6

        # ---- Spatial normalisation for the (x, y, z) query encoder ----
        # Fourier features expect roughly [-1, 1] inputs, so we centre and scale
        # xy by half its extent, and z by half the stack's z-range.
        allxy = np.concatenate([s.coords_xy for s in stack.slices], 0).astype(np.float64)
        self._xy_c = allxy.mean(0)
        self._xy_s = (allxy.max(0) - allxy.min(0)) / 2 + 1e-6
        zc = stack.z_centers()
        self._z_c = float(zc.mean())
        self._z_s = float((zc.max() - zc.min()) / 2 + 1e-6)

        self._fit()

    # --- normalisation helpers (physical <-> normalised coordinates) ---
    def _nxy(self, xy):
        return (np.asarray(xy, np.float64) - self._xy_c) / self._xy_s
    def _dxy(self, nxy):
        return np.asarray(nxy, np.float64) * self._xy_s + self._xy_c
    def _nz(self, z):
        return (float(z) - self._z_c) / self._z_s

    # --------------------------------------------------------------------------
    # FITTING: choose device, build nets, run Phase A then Phase B.
    # --------------------------------------------------------------------------
    def _device(self):
        if not _HAS_TORCH:
            return "cpu"
        d = self.cfg.device
        if d != "auto":
            return d
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _fit(self):
        if not _HAS_TORCH:
            if self.cfg.verbose:
                print("[v14] torch unavailable -> latent-grounded numpy fallback only.")
            return
        try:
            self._train_model()
            self.trained = True
        except Exception as e:                          # keep the method usable
            import traceback
            print(f"[v14] training failed ({e}); using numpy fallback.")
            traceback.print_exc()
            self.trained = False

    # ==========================================================================
    # PART 7 — DATA PREP FOR TRAINING (real file: trainer.py `_prep`, helpers)
    # ==========================================================================
    def _prep(self, dev):
        """Precompute, per slice, the tensors the trainer needs:
            e     standardized expression latent   (n, d_e)
            m     standardized morphology channels (n, n_morph)
            types integer cell types               (n,)
            nxy   normalised in-plane coords       (n, 2)   [numpy]
            nz    normalised slice depth           scalar
            radius per-cell distance to the slice centroid (for the hypoxia term)
        """
        S = []
        for s in self.stack.slices:
            e = self.expr_latent.encode(s.expression)
            mo = morphology_features(s.coords_xy, s.cell_type_indices, self.n_types,
                                     k=self.cfg.morph_k, density_sigma=self.cfg.density_sigma)
            es = (e - self._e_mean) / self._e_std
            ms = (mo - self._m_mean) / self._m_std
            types = (s.cell_type_indices.astype(int) if s.cell_type_indices is not None
                     else np.zeros(s.n_spots, int))
            nxy = self._nxy(s.coords_xy).astype(np.float32)
            c = nxy.mean(0) if nxy.shape[0] else np.zeros(2, np.float32)
            radius = (np.linalg.norm(nxy - c, axis=1).astype(np.float32)
                      if nxy.shape[0] else np.zeros(0, np.float32))
            S.append(dict(
                e=torch.tensor(es, dtype=torch.float32, device=dev),
                m=torch.tensor(ms, dtype=torch.float32, device=dev),
                types=torch.tensor(types, dtype=torch.long, device=dev),
                nxy=nxy, nz=float(self._nz(s.z_center)), radius=radius,
            ))
        return S

    @staticmethod
    def _knn(query_xy, pool_xy, k):
        k = min(k, pool_xy.shape[0])
        _, idx = cKDTree(pool_xy).query(query_xy, k=k)
        if idx.ndim == 1:
            idx = idx[:, None]
        return idx.astype(np.int64)

    @staticmethod
    def _build_pool(S, src_idx):
        """Concatenate several source slices' cells into one candidate pool, and
        remember which slice ('owner') each pooled cell came from — the owner tag
        is what lets us mask a whole slice for the gap-aware training."""
        nxy, z, owner = [], [], []
        for j, si in enumerate(src_idx):
            s = S[si]
            nxy.append(s["nxy"])
            z.append(np.full(s["nxy"].shape[0], s["nz"], np.float32))
            owner.append(np.full(s["nxy"].shape[0], j, np.int64))
        return (np.vstack(nxy).astype(np.float32),
                np.concatenate(z).astype(np.float32),
                np.concatenate(owner).astype(np.int64))

    def _context(self, h_pool, pool_nxy, pool_z, pool_owner, src_idx, S,
                 query_nxy, query_z, ctxmod, dev, drop_owner=None):
        """Build the 3D-attention context vectors for a batch of query positions.

        Tokens = (local kNN cells from the pool) + (one global summary per source
        slice).  `drop_owner` implements GAP-AWARE training: mask out one source
        slice's local AND global tokens, forcing the flow to reconstruct the field
        from the REMAINING slices — exactly the situation at test time, where the
        held-out slice is absent.
        """
        cfg = self.cfg
        Q = query_nxy.shape[0]
        n_local = min(cfg.n_context, pool_nxy.shape[0])
        nbr = self._knn(query_nxy, pool_nxy, n_local)              # (Q, n_local)

        pool_pos = np.column_stack([pool_nxy, pool_z]).astype(np.float32)
        pool_pos_t = torch.tensor(pool_pos, device=dev)
        local_h = h_pool[torch.tensor(nbr, device=dev)]           # (Q, n_local, d_joint)
        local_pos = pool_pos_t[torch.tensor(nbr, device=dev)]     # (Q, n_local, 3)
        local_tok = ctxmod.encode_tokens(
            local_h.reshape(-1, local_h.shape[-1]),
            local_pos.reshape(-1, 3)).reshape(Q, n_local, -1)     # (Q, n_local, d_model)

        # global per-slice summary tokens (mean h at each slice centroid)
        g_h, g_pos = [], []
        src_iter = list(enumerate(src_idx)) if cfg.n_global_tokens > 0 else []
        for j, si in src_iter:
            s = S[si]
            idx = np.where(pool_owner == j)[0]
            if idx.size == 0:
                continue
            gh = h_pool[torch.tensor(idx, device=dev)].mean(0, keepdim=True)   # (1, d)
            cen = s["nxy"].mean(0)
            gp = torch.tensor([[cen[0], cen[1], s["nz"]]], dtype=torch.float32, device=dev)
            g_h.append(gh); g_pos.append(gp)
        if g_h:
            g_h = torch.cat(g_h, 0); g_pos = torch.cat(g_pos, 0)
            glob_tok = ctxmod.encode_tokens(g_h, g_pos)[None].expand(Q, -1, -1)
            tokens = torch.cat([local_tok, glob_tok], dim=1)
            n_glob = glob_tok.shape[1]
        else:
            tokens = local_tok
            n_glob = 0

        # gap-aware padding mask (True = "ignore this token")
        pad = None
        if drop_owner is not None:
            local_owner = pool_owner[nbr]                          # (Q, n_local)
            pad_local = torch.tensor(local_owner == drop_owner, device=dev)
            pad_glob = torch.zeros((Q, n_glob), dtype=torch.bool, device=dev)
            if n_glob:
                present = [j for j in range(len(src_idx)) if np.any(pool_owner == j)]
                for gi, j in enumerate(present):
                    if j == drop_owner:
                        pad_glob[:, gi] = True
            pad = torch.cat([pad_local, pad_glob], dim=1)
            # never allow a fully-masked row (attention would return NaN)
            allmask = pad.all(dim=1)
            if allmask.any():
                pad[allmask, -1] = False

        query_pos = torch.tensor(
            np.column_stack([query_nxy, np.full(Q, float(query_z), np.float32)]),
            dtype=torch.float32, device=dev)
        return ctxmod(query_pos, tokens, key_padding_mask=pad)

    # ==========================================================================
    # PART 8 & 9 — TWO-PHASE TRAINING (real file: trainer.py)
    # ==========================================================================
    def _train_model(self):
        cfg = self.cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        dev = self._device()
        if cfg.verbose:
            print(f"[v14] training device: {dev}")

        self.encoder = JointEncoder(self.expr_latent.dim, self.n_morph, cfg.joint_dim,
                                    self.n_types, cfg.enc_hidden, cfg.dropout).to(dev)
        self.ctxmod = ContextAttention(cfg.joint_dim, cfg.d_model, cfg.n_heads,
                                       cfg.fourier_bands, cfg.dropout).to(dev)
        self.vfield = VectorField(cfg.joint_dim, self.ctxmod.out_dim, cfg.fourier_bands,
                                  cfg.flow_hidden, cfg.flow_layers, cfg.time_embed_dim).to(dev)
        self.dev = dev
        self.S = self._prep(dev)

        self._phase_a()            # encoder + decoders (reconstruction)
        self._phase_b()            # attention context + flow field (CFM + bio)
        self.encoder.eval(); self.ctxmod.eval(); self.vfield.eval()

    # ---- Phase A: reconstruction pre-training of the joint encoder ------------
    def _phase_a(self):
        r"""Teach the encoder to fuse (e, m) -> h AND the decoders to invert it,
        by reconstructing e, m and cell type.  We do this FIRST and then FREEZE
        the encoder, so the flow (Phase B) has FIXED targets h1 to aim at — a
        moving target would make flow matching unstable."""
        cfg = self.cfg
        params = list(self.encoder.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        rng = np.random.default_rng(cfg.seed)
        idxs = [i for i, s in enumerate(self.S) if s["e"].shape[0] > 0]
        for ep in range(cfg.pretrain_epochs):
            tot, nb = 0.0, 0
            for i in idxs:
                s = self.S[i]; n = s["e"].shape[0]
                B = min(cfg.batch_cells, n)
                b = torch.tensor(rng.choice(n, B, replace=False), device=self.dev)
                e, mo, ty = s["e"][b], s["m"][b], s["types"][b]
                h = self.encoder.encode(e, mo)
                loss = F.mse_loss(self.encoder.decode_e(h), e) \
                     + F.mse_loss(self.encoder.decode_m(h), mo)
                if self.n_types >= 2:
                    loss = loss + F.cross_entropy(self.encoder.type_head(h), ty)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                tot += float(loss.detach()); nb += 1
            if cfg.verbose and (ep % 20 == 0 or ep == cfg.pretrain_epochs - 1):
                print(f"    [A] epoch {ep:3d}  recon={tot / max(nb, 1):.4f}")

        self.encoder.eval()
        for p in self.encoder.parameters():       # FREEZE for the flow phase
            p.requires_grad_(False)
        # cache the frozen joint-latent target h per slice (these are the h1's)
        with torch.no_grad():
            for s in self.S:
                s["h"] = (self.encoder.encode(s["e"], s["m"]) if s["e"].shape[0]
                          else s["e"].new_zeros((0, self.cfg.joint_dim)))

    @staticmethod
    def _neighbor_slices(order, pos, k_each_side):
        """Indices of the k nearest slices on each side of position `pos` (in
        z-order), EXCLUDING pos itself.  During flow training the target slice is
        treated as held-out, so its context is built only from its neighbours —
        this is the leave-one-slice-out signal baked into training."""
        lo = [int(order[j]) for j in range(max(0, pos - k_each_side), pos)]
        hi = [int(order[j]) for j in range(pos + 1, min(len(order), pos + 1 + k_each_side))]
        src = lo + hi
        if not src:                               # degenerate (<2 slices)
            src = [int(order[j]) for j in range(len(order)) if j != pos] or [int(order[pos])]
        return src

    # ---- Phase B: conditional flow matching + biology regularizers -----------
    def _phase_b(self):
        cfg = self.cfg
        order = np.argsort(self.stack.z_centers())
        # only INTERIOR slices are flow targets: an interior slice has real
        # neighbours on both sides, so "reconstruct me from my neighbours" is
        # well-posed and mirrors the benchmark (a held-out section is bracketed).
        interior = order[1:-1] if len(order) >= 3 else order[:1]
        k_side = max(cfg.context_slices_each_side, 1)

        # Precompute, per interior target slice, its context source pool and the
        # per-cell DISPLACEMENT target (offset from its neighbours' centroid).
        plan = []
        for i in interior:
            i = int(i); pos = int(np.where(order == i)[0][0])
            src = self._neighbor_slices(order, pos, k_side)
            pool_nxy, pool_z, pool_owner = self._build_pool(self.S, src)
            nbr = self._knn(self.S[i]["nxy"], pool_nxy,
                            min(cfg.n_context, pool_nxy.shape[0]))
            cen = pool_nxy[nbr].mean(1)                            # (n_i, 2)
            disp_t = (self.S[i]["nxy"] - cen).astype(np.float32)
            entry = dict(i=i, src=src, pool_nxy=pool_nxy, pool_z=pool_z,
                         pool_owner=pool_owner, disp=disp_t, far=None)
            # v19 GAP CURRICULUM: a second context that skips the immediate
            # neighbours (distance >= 2 in z-order) when enough slices exist,
            # emulating wide-gap conditions during training.
            if cfg.curriculum_flow and len(order) >= 5:
                lo2 = [int(order[j]) for j in range(max(0, pos - 2 * k_side), max(0, pos - k_side))]
                hi2 = [int(order[j]) for j in range(min(len(order), pos + 1 + k_side),
                                                    min(len(order), pos + 1 + 2 * k_side))]
                far_src = lo2 + hi2
                if far_src:
                    f_nxy, f_z, f_own = self._build_pool(self.S, far_src)
                    entry["far"] = dict(src=far_src, pool_nxy=f_nxy, pool_z=f_z,
                                        pool_owner=f_own)
            plan.append(entry)

        params = list(self.ctxmod.parameters()) + list(self.vfield.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        rng = np.random.default_rng(cfg.seed + 1)

        for ep in range(cfg.epochs):
            # anneal the biology weights in over the first `anneal_epochs`, so the
            # flow first learns to hit its targets, THEN gets nudged to be smooth.
            anneal = min(1.0, (ep + 1) / max(cfg.anneal_epochs, 1))
            tot, nb = 0.0, 0
            for pl in plan:
                i = pl["i"]; s = self.S[i]; n = s["h"].shape[0]
                if n == 0:
                    continue
                B = min(cfg.batch_cells, n)
                b = rng.choice(n, B, replace=False)
                bt = torch.tensor(b, device=self.dev)
                h1 = s["h"][bt]                                    # (B, d) FROZEN target
                query_nxy = s["nxy"][b]
                zq = s["nz"] + rng.normal(0, cfg.z_sigma)         # z-MARGINALIZATION

                use = pl
                if pl.get("far") is not None and rng.random() < cfg.curriculum_p:
                    use = pl["far"]                                # v19 long-range context
                pool_h = torch.cat([self.S[j]["h"] for j in use["src"]], 0)
                drop = None
                if rng.random() < cfg.gap_dropout and len(use["src"]) > 1:
                    drop = int(rng.integers(len(use["src"])))      # GAP-AWARE dropout
                ctx = self._context(pool_h, use["pool_nxy"], use["pool_z"], use["pool_owner"],
                                    use["src"], self.S, query_nxy, zq, self.ctxmod, self.dev,
                                    drop_owner=drop)

                # ---- CONDITIONAL FLOW MATCHING on the OT straight-line path ----
                h0 = torch.randn_like(h1)                          # noise endpoint
                t = torch.rand(B, device=self.dev)                 # random time in [0,1]
                ht = (1 - t)[:, None] * h0 + t[:, None] * h1        # point on the path
                zt = torch.full((B,), float(zq), device=self.dev)
                v = self.vfield(ht, t, ctx, zt)                    # predicted velocity
                u = h1 - h0                                        # true velocity
                loss = F.mse_loss(v, u)                            # the flow-matching loss

                # ---- STAGE 4 biology regularizers (decode the PREDICTED endpoint)
                # From (ht, v) we can extrapolate the endpoint: h1_hat = ht + (1-t) v.
                h1_hat = ht + (1 - t)[:, None] * v
                if cfg.w_consistency > 0:
                    e_hat = self.encoder.decode_e(h1_hat)
                    m_hat = self.encoder.decode_m(h1_hat)
                    h_re = self.encoder.encode(e_hat, m_hat)
                    loss = loss + anneal * cfg.w_consistency * F.mse_loss(h_re, h1_hat)
                if cfg.w_smooth > 0 or cfg.w_interface > 0:
                    loss = loss + anneal * self._smoothness(s, b, h1_hat)
                if cfg.w_hypoxia > 0 and self.n_types >= 2:
                    loss = loss + anneal * cfg.w_hypoxia * self._hypoxia(s, b, h1_hat, rng)

                # ---- displacement head (only used by position_mode='morph') ----
                disp_pred = self.vfield.displacement(h1, torch.ones(B, device=self.dev), ctx, zt)
                disp_tgt = torch.tensor(pl["disp"][b], device=self.dev)
                loss = loss + 0.5 * F.mse_loss(disp_pred, disp_tgt)

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                tot += float(loss.detach()); nb += 1
            if cfg.verbose and (ep % 40 == 0 or ep == cfg.epochs - 1):
                print(f"    [B] epoch {ep:3d}  cfm+bio={tot / max(nb, 1):.4f}")

    # ==========================================================================
    # PART 10 — BIOLOGY REGULARIZERS (real file: trainer.py)
    # ==========================================================================
    def _smoothness(self, s, b, h1_hat):
        r"""EDGE-AWARE latent total variation.

        Neighbouring cells should have similar latents — BUT NOT across a tissue
        interface (tumour vs stroma), where a sharp change is real.  So we weight
        the smoothness penalty by exp(-4 * |morphology difference|): strong
        smoothing inside a homogeneous region, RELAXED across a morphological
        edge.  The morphology m is exactly the "image" whose edges we read."""
        cfg = self.cfg
        xy = s["nxy"][b]
        if xy.shape[0] < 4:
            return h1_hat.new_zeros(())
        kk = min(5, xy.shape[0])
        _, idx = cKDTree(xy).query(xy, k=kk)
        idx = idx[:, 1:]                                            # drop self
        mo = s["m"][b]                                             # morphology channels
        with torch.no_grad():
            edge = (mo[:, None, :] - mo[torch.tensor(idx, device=self.dev)]).abs().mean(-1)
            w = torch.exp(-4.0 * edge)                            # small at edges
        diff = (h1_hat[:, None, :] - h1_hat[torch.tensor(idx, device=self.dev)]).pow(2).mean(-1)
        tv = (w * diff).mean()
        return (cfg.w_smooth + cfg.w_interface) * tv

    def _hypoxia(self, s, b, h1_hat, rng):
        r"""A soft, biology-motivated directional prior: a scalar "hypoxia" head
        should INCREASE from the tissue periphery toward its core (as O2 drops).
        We enforce it only as an ORDER constraint with a margin (a ranking hinge
        loss over random cell pairs), never a hard value — it nudges, not dictates.
        `radius` is each cell's distance to the slice centroid (inner = small r)."""
        r = torch.tensor(s["radius"][b], device=self.dev)
        sh = self.encoder.hypoxia_head(h1_hat)[:, 0]
        B = sh.shape[0]
        perm = torch.tensor(rng.permutation(B), device=self.dev)
        r2, sh2 = r[perm], sh[perm]
        inner = r < r2                                             # first cell more inner?
        d = torch.where(inner, sh2 - sh, sh - sh2)                # want outer - inner >= margin
        return F.relu(d + self.cfg.hypoxia_margin).mean()

    # ==========================================================================
    # PART 11 — STAGE 5: GENERATION (real file: trainer.py `generate_slice`)
    # ==========================================================================
    def generate_virtual_slice(self, z: float) -> VirtualSlice:
        """Public entry point: synthesise the section at depth z."""
        if self.trained:
            try:
                return self._generate(z)
            except Exception as e:
                print(f"[v14] generation failed ({e}); numpy fallback.")
                import traceback; traceback.print_exc()
        return self._fallback(z)

    @staticmethod
    def _tp_frac(z, zl, zh):
        """How far z sits between the two flanking slices, in [0, 1]."""
        if zh == zl:
            return 0.5
        return float(np.clip((float(z) - zl) / (zh - zl), 0.0, 1.0))

    def _coherent_source_mask(self, lo_xy, hi_xy, t, freq, rng):
        r"""Per-cell keep-probabilities that form spatially-COHERENT PATCHES.

        This is the trick that preserves niche organisation.  Instead of drawing
        each generated cell independently from 'lower with prob 1-t, upper with
        prob t' (which INTERLEAVES the two slices and scrambles neighbourhoods),
        we build a smooth random field f(x, y) in [0, 1] (a sum of a few random
        sinusoids) and let the LOWER slice supply the region {f >= t} and the
        UPPER slice the region {f < t}.  The area of the upper region is ~t, so
        composition is still interpolated correctly — but now a generated cell's
        whole neighbourhood comes from ONE real slice (a coherent patch), which
        is what keeps Moran's I and neighbourhood-enrichment high."""
        allxy = np.vstack([lo_xy, hi_xy])
        span = (allxy.max(0) - allxy.min(0)) + 1e-6
        xn = (allxy - allxy.min(0)) / span                        # -> [0, 1]^2
        f = np.zeros(allxy.shape[0])
        for _ in range(3):                                        # sum of 3 random sinusoids
            kx, ky = rng.uniform(0.5, freq, 2)
            ph = rng.uniform(0, 2 * np.pi)
            f += np.sin(2 * np.pi * (kx * xn[:, 0] + ky * xn[:, 1]) + ph)
        f = (f - f.min()) / (f.max() - f.min() + 1e-9)            # -> [0, 1]
        n_lo = lo_xy.shape[0]
        is_lower = np.zeros(allxy.shape[0], bool); is_lower[:n_lo] = True
        keep = np.where(is_lower, (f >= t).astype(float), (f < t).astype(float))
        keep += 1e-3                                              # floor so no region is empty
        return keep / keep.sum()

    def _resample_layout(self, lower, upper, t, n_target, rng, alpha=1.0):
        """Draw the generated cells' POSITIONS by resampling the two flanking
        slices' coordinates (coherent patches by default), with a little jitter.
        Returns normalised anchor xy AND, for each drawn cell, its index into the
        concatenated [lower; upper] pool (so grounding can inherit its real profile).

        v21 GAP-ADAPTIVE LAYOUT GRANULARITY: at NARROW gaps the two flank
        fields are near-identical, so large single-flank patches needlessly
        discard half the local information - every analysis bin then reads one
        flank instead of averaging both (ceiling diagnostic: a no-replacement
        flank-union scores 0.894 marker-field vs 0.877 for coarse patches on
        the synthetic testbed).  The patch frequency therefore ramps from
        `layout_freq_narrow` (fine granules ~ analysis-bin scale: bins average
        both flanks, nearest neighbours still mostly share a flank) at alpha=0
        down to `coherent_freq` (coarse patches, the v14 salt-and-pepper
        protection that matters when the flank fields genuinely differ) at
        alpha=1.  Measured narrow frontier (layout freq -> Moran_r/field_r):
        4 -> 0.950/0.875, 8 -> 0.943/0.881, 24 -> 0.931/0.882, interleaved ->
        0.928/0.884; freq 8 keeps the Moran's win while lifting field_r above
        the SpatialZ emulation."""
        lo_xy = self._nxy(lower.coords_xy).astype(np.float32)
        hi_xy = self._nxy(upper.coords_xy).astype(np.float32)
        props = np.vstack([lo_xy, hi_xy])
        if self.cfg.coherent_source:
            freq_eff = self.cfg.coherent_freq
            if self.cfg.layout_gap_adapt:
                f_hi = max(self.cfg.layout_freq_narrow, self.cfg.coherent_freq)
                freq_eff = (1.0 - alpha) * f_hi + alpha * self.cfg.coherent_freq
            w = self._coherent_source_mask(lo_xy, hi_xy, t, freq_eff, rng)
        else:                                                    # legacy interleaving
            w = np.concatenate([np.full(len(lo_xy), max(1 - t, 1e-3)),
                                np.full(len(hi_xy), max(t, 1e-3))])
            w = w / w.sum()
        # sample without replacement when possible (avoids coincident cells that
        # would spike the local density and distort neighbourhood metrics)
        replace = n_target > props.shape[0]
        sel = rng.choice(props.shape[0], size=n_target, replace=replace, p=w)
        med = np.median([lower.median_spacing(), upper.median_spacing()])
        jit = rng.standard_normal((n_target, 2)) * (0.05 * med / self._xy_s.mean())
        return (props[sel] + jit).astype(np.float32), sel.astype(np.int64)

    def _generate(self, z):
        cfg = self.cfg
        lower, upper = self.stack.pick_flanking_slices(z)
        t = self._tp_frac(z, lower.z_center, upper.z_center)
        n_target = max(int(round((1 - t) * lower.n_spots + t * upper.n_spots)), 1)
        rng = np.random.default_rng(cfg.seed)

        li = self.stack.slices.index(lower)
        ui = self.stack.slices.index(upper)
        n_lo = lower.n_spots

        # ---- v19/v21: GAP-ADAPTIVE generative weight (computed FIRST: v21
        # uses it to shape the layout, the re-grounding margin, the gene-mix
        # rate and the mixes) ------------------------------------------------
        # alpha = 0 for gaps at (or below) the median training slice spacing
        # (narrow-gap regime: v18 copy behaviour, which the saturated metrics
        # reward) and ramps to 1 as the gap reaches gap_scale x that spacing
        # (wide-gap regime: paired interpolation + stronger flow decode).
        zc_all = self.stack.z_centers()
        med_gap = float(np.median(np.diff(np.sort(zc_all)))) if len(zc_all) > 1 else 1.0
        this_gap = float(upper.z_center - lower.z_center)
        alpha = 0.0
        if cfg.gap_adapt and med_gap > 0:
            alpha = float(np.clip((this_gap / med_gap - 1.0) /
                                  max(cfg.gap_scale - 1.0, 1e-6), 0.0, 1.0))
        if alpha < cfg.alpha_tol:
            alpha = 0.0                                # v20: exact v18 at narrow gaps

        # ---- 1. choose the generated cells' POSITIONS (position_mode) ----
        if cfg.position_mode == "nearest":
            near = lower if t <= 0.5 else upper
            near_xy = self._nxy(near.coords_xy).astype(np.float32)
            pick = (rng.choice(near_xy.shape[0], n_target, replace=False)
                    if near_xy.shape[0] > n_target else np.arange(near_xy.shape[0]))
            anchor = near_xy[pick]
            anchor_src = (pick if t <= 0.5 else pick + n_lo).astype(np.int64)
            use_disp = False
        else:
            anchor, anchor_src = self._resample_layout(lower, upper, t, n_target,
                                                       rng, alpha=alpha)
            use_disp = (cfg.position_mode == "morph")

        # ---- 2. build the 3D-attention CONTEXT at depth z ----
        # Context pool = the k nearest real slices on each side of z (wider than
        # the two flanking slices we ground from), so the flow sees local AND
        # long-range structure.
        order = np.argsort(self.stack.z_centers())
        zc = self.stack.z_centers()
        k_side = max(cfg.context_slices_each_side, 1)
        below = [int(j) for j in order if zc[j] <= z]
        above = [int(j) for j in order if zc[j] > z]
        ctx_src = (below[-k_side:] if below else []) + (above[:k_side] if above else [])
        for extra in (li, ui):                                    # ensure flanks are in
            if extra not in ctx_src:
                ctx_src.append(extra)
        ctx_pool_nxy, ctx_pool_z, ctx_pool_owner = self._build_pool(self.S, ctx_src)
        ctx_pool_h = torch.cat([self.S[j]["h"] for j in ctx_src], 0)
        zn = self._nz(z)

        with torch.no_grad():
            ctx = self._context(ctx_pool_h, ctx_pool_nxy, ctx_pool_z, ctx_pool_owner,
                                ctx_src, self.S, anchor, zn, self.ctxmod, self.dev)
            Q = anchor.shape[0]
            zt = torch.full((Q,), float(zn), device=self.dev)

            # ---- 3. SAMPLE the flow ODE, marginalizing over initial noise ----
            # Integrate dh/dt = v from t=0..1 by plain Euler, starting from
            # several independent noises, and AVERAGE the endpoints.  Averaging
            # marginalises the z / noise uncertainty and denoises the latent.
            h_acc = torch.zeros((Q, cfg.joint_dim), device=self.dev)
            n_ens = max(cfg.n_ensemble, 1)
            steps = max(cfg.n_ode_steps, 1)
            for _ in range(n_ens):
                h = torch.randn((Q, cfg.joint_dim), device=self.dev)
                for si in range(steps):
                    tt = torch.full((Q,), si / steps, device=self.dev)
                    h = h + (1.0 / steps) * self.vfield(h, tt, ctx, zt)
                h_acc += h
            h_star = h_acc / n_ens                                # (Q, d) generated latent

            e_hat = self.encoder.decode_e(h_star)                # standardized latent
            type_logits = self.encoder.type_head(h_star) if self.n_types >= 2 else None
            if use_disp:                                          # morph mode only
                disp = self.vfield.displacement(h_star, torch.ones(Q, device=self.dev), ctx, zt)
                disp = torch.clamp(disp * cfg.displacement_scale, -0.15, 0.15)
                anchor = anchor + disp.cpu().numpy().astype(np.float32)

        e_hat_np = e_hat.cpu().numpy()

        # ---- 4. GROUND each generated cell in a real training profile ----
        # Grounding pool = the two immediate flanking slices (most spatially
        # relevant).  anchor_src indexes into this [lower; upper] pool.
        lo_st = self.S[li]; hi_st = self.S[ui]
        pool_nxy = np.vstack([lo_st["nxy"], hi_st["nxy"]]).astype(np.float32)
        pool_expr = np.vstack([np.asarray(lower.expression), np.asarray(upper.expression)])
        pool_type = np.concatenate([
            lower.cell_type_indices if lower.cell_type_indices is not None else np.zeros(lower.n_spots, int),
            upper.cell_type_indices if upper.cell_type_indices is not None else np.zeros(upper.n_spots, int)])
        pool_e = np.vstack([lo_st["e"].cpu().numpy(), hi_st["e"].cpu().numpy()])  # standardized e

        # v21 GAP-ADAPTIVE RE-GROUNDING MARGIN (alpha computed above, before
        # the layout).  The narrow-regime cumulative ablation (synthetic
        # two-regime testbed) shows flow re-grounding is NET-HARMFUL at narrow gaps: the inherited exemplar already IS the
        # local truth there, latent disagreement is model noise, and every
        # swap is a needless profile transfer (verbatim patchwork scores
        # 0.951 Moran / 0.877 field / 0.025 Moran-MAE vs 0.937 / 0.866 /
        # 0.034 once swaps run).  At wide gaps the same swaps inject the
        # z-interpolated molecular signal the copies lack (v18's wide-gap
        # Moran advantage).  So the swap margin ramps from
        # `ground_margin_narrow` at alpha=0 down to the configured
        # `ground_keep_margin` at alpha=1.
        margin_eff = cfg.ground_keep_margin
        if cfg.ground_gap_adapt:
            m_hi = max(cfg.ground_margin_narrow, cfg.ground_keep_margin)
            margin_eff = (1.0 - alpha) * m_hi + alpha * cfg.ground_keep_margin

        expr, ct_idx, pick = self._ground(anchor, anchor_src, e_hat_np, pool_nxy, pool_e,
                                          pool_expr, pool_type, rng,
                                          keep_margin=margin_eff)
        if cfg.type_mode == "vote" and self.n_types >= 2 and ct_idx is not None:
            ct_idx, pick, expr = self._vote_types(anchor, pick, ct_idx, expr, pool_nxy,
                                                  pool_type, pool_expr, pool_e,
                                                  e_hat_np, rng)
        if cfg.composition_match and self.n_types >= 2:
            ct_idx, expr, pick = self._match_composition(lower, upper, t, ct_idx, expr,
                                                         anchor, pool_nxy, pool_e,
                                                         e_hat_np, pool_type,
                                                         pool_expr, rng, pick)

        # ---- v21: FIELD-ALIGNED RE-GROUNDING ------------------------------
        # Bounded refinement: for each cell, compare its exemplar profile with
        # the z-interpolated LOCAL MEAN FIELD of both flanks; where the mismatch
        # is large and a local same-type real cell clearly fits better, swap the
        # exemplar.  Profiles stay verbatim real cells - this pass only reduces
        # the mismatch between a profile and its LOCATION, which is exactly what
        # the binned field / depth / SSIM metrics read (the metrics SpatialZ's
        # kernel-weighted per-gene synthesis was winning).  Like the flow
        # re-grounding above, it is an exemplar SWAPPER, and the same ablation
        # shows swappers subtract accuracy at narrow gaps - so it is alpha-
        # gated by default (field_align_gap_only).
        if cfg.field_align and (alpha > 0.0 or not cfg.field_align_gap_only):
            pick, expr, ct_idx = self._field_align(anchor, pick, ct_idx, expr,
                                                   pool_nxy, pool_expr, pool_type,
                                                   t, n_lo, rng)

        # ---- 5. blend the flow-DECODED expression into the real profile ----
        # This is where the generative model genuinely shapes the molecular output
        # (not just selects an exemplar): expr <- (1-w)*real + w*decode(flow latent).
        if cfg.edit_weight > 0.0:
            dec = self.expr_latent.decode(e_hat_np * self._e_std + self._e_mean)
            expr = (1 - cfg.edit_weight) * expr + cfg.edit_weight * dec

        # ---- v18: partial gene resampling for distributional novelty ----
        # v21 ORDER FIX: gene-mix runs BEFORE the cross-mix, so where the two
        # masks overlap the CROSS-FLANK value wins.  In v20 the order was
        # reversed and the same-flank gene-mix overwrote part of the cross-mix,
        # diluting the effective cross-flank fraction to alpha * w_other *
        # (1 - gene_mix_frac) - i.e. E[output] equalled the interpolation only
        # with gene-mix off.  With this order the expectation is exact.
        srcg = None; gmask = None
        if cfg.gene_mix_frac > 0.0 and cfg.edit_weight == 0.0:
            # v21 GAP-ADAPTIVE GENE-MIX RATE: ramp the per-gene redraw fraction
            # toward gene_mix_wide as the gap widens - the wide regime is where
            # per-gene synthesis (vs whole-profile copies) wins the field
            # metrics, and the z-weighted candidates keep its expectation on
            # the interpolation.
            gmf_eff = cfg.gene_mix_frac
            if cfg.gene_mix_wide > 0.0:
                gmf_eff = cfg.gene_mix_frac + alpha * max(
                    0.0, cfg.gene_mix_wide - cfg.gene_mix_frac)
            candsg, gmask = self._gene_mix(anchor, pick, ct_idx, pool_expr,
                                           pool_nxy, pool_type,
                                           pool_expr.shape[1], rng,
                                           t=t, n_lo=n_lo, mix_frac=gmf_eff)
            G = pool_expr.shape[1]
            choice = rng.integers(0, candsg.shape[1], size=(anchor.shape[0], G))
            srcg = np.take_along_axis(candsg, choice, axis=1).astype(np.int64)
            colg = np.arange(G)[None, :]
            expr = np.where(gmask, pool_expr[srcg, colg], expr)

        cross_mask = None; src2 = None
        if cfg.cross_mix and alpha > 0.0:
            cands2, ok2 = self._pair_across_flanks(anchor, pick, ct_idx, n_lo,
                                                   pool_nxy, pool_e, pool_type,
                                                   rng, kc=cfg.mix_cand_k)
            # v20 BERNOULLI CROSS-MIX: per gene, take a cross-flank partner's
            # REAL value with probability alpha * w_other (w_other = fraction of
            # the opposite flank at this cell's z).  E[output] equals the linear
            # interpolation v19 attempted, but no value is ever synthesized, so
            # detection frequency / sparsity match real data by construction.
            in_lower = pick < n_lo
            w_other = np.where(in_lower, t, 1.0 - t) * alpha
            # v21 COHERENT MIX: per-gene smooth random fields with an exact
            # uniform marginal replace the i.i.d. mask.  Marginal P(mix) per
            # cell is IDENTICAL to v20 (E[output] is still the interpolation),
            # but neighbouring cells flip the same genes together, so the mix
            # no longer injects per-gene salt-and-pepper noise - this is what
            # repairs v20's wide-gap Moran's/Geary's regression while keeping
            # its field-metric wins.
            if cfg.coherent_mix:
                U = self._coherent_gene_fields(anchor, expr.shape[1], rng,
                                               cfg.mix_field_freq)
                cross_mask = U < w_other[:, None]
            else:                                     # v20 i.i.d. behaviour
                cross_mask = rng.random(expr.shape) < w_other[:, None]
            cross_mask[~ok2] = False
            # v21 per-gene MULTI-PARTNER draw: each mixed gene independently
            # picks one of the kc cross-flank candidates (kc=1 -> v20's single
            # latent-nearest partner).  Decorrelates within-bin residuals.
            G = expr.shape[1]
            choice = rng.integers(0, cands2.shape[1], size=(anchor.shape[0], G))
            src2 = np.take_along_axis(cands2, choice, axis=1).astype(np.int64)
            colg = np.arange(G)[None, :]
            expr = np.where(cross_mask, pool_expr[src2, colg], expr)

        if cfg.edit_gap_extra > 0.0 and cfg.edit_weight == 0.0 and alpha > 0.0:
            dec = self.expr_latent.decode(e_hat_np * self._e_std + self._e_mean)
            w = min(alpha * cfg.edit_gap_extra, 0.6)
            expr = (1 - w) * expr + w * dec            # ablation-only path

        # ---- v21: per-gene FIELD REPAIR (after all mixes, so it also fixes
        # any out-of-range values the mixes introduced at granule seams) ----
        rep_rows = rep_cols = rep_srcs = None
        if cfg.field_repair and cfg.edit_weight == 0.0:
            expr, rep_rows, rep_cols, rep_srcs = self._field_repair(
                anchor, expr, ct_idx, pool_nxy, pool_expr, pool_type,
                t, n_lo, rng, alpha=alpha)

        expr = np.clip(expr, 0.0, None)
        # ---- v18 output path -------------------------------------------------
        # When every emitted profile is a verbatim copy (edit_weight == 0) and
        # the raw measurements are available, emit the RAW profile of the picked
        # source cell.  v14 instead emitted expm1(log-lib-normalized), which is
        # NOT the inverse of the wrapper normalization (per-cell totals were
        # rescaled to the median library size before log1p): on wide-dynamic-
        # range data this nonlinear round trip inverts the per-gene VARIANCE
        # ranking (the negative gene_var_spearman on all three EASI-FISH sets).
        raw_ok = (cfg.raw_output and cfg.edit_weight == 0.0
                  and cfg.edit_gap_extra == 0.0
                  and lower.raw_expression is not None
                  and upper.raw_expression is not None)
        if raw_ok:
            pool_raw = np.vstack([lower.raw_expression, upper.raw_expression])
            expr = pool_raw[pick]
            colg = np.arange(pool_raw.shape[1])[None, :]
            if srcg is not None:                         # gene-mix on raw scale
                expr = np.where(gmask, pool_raw[srcg, colg], expr)
            if cross_mask is not None:                   # cross-mix on raw scale
                expr = np.where(cross_mask, pool_raw[src2, colg], expr)
            if rep_rows is not None and rep_rows.size:   # v21 field repair, raw
                expr[rep_rows, rep_cols] = pool_raw[rep_srcs, rep_cols]
        elif cfg.output_counts:                                  # the evaluator wants counts
            expr = np.expm1(np.clip(expr, 0.0, 20.0))
        expr = expr.astype(np.float32)

        coords = np.column_stack([self._dxy(anchor), np.full(len(anchor), float(z))]).astype(np.float32)
        labels = self._labels(ct_idx)
        return VirtualSlice(coords, expr, labels, ct_idx)

    def _ground(self, anchor, anchor_src, e_hat, pool_nxy, pool_e, pool_expr,
                pool_type, rng, keep_margin=None):
        r"""Emit one REAL exemplar profile per generated cell.

        DEFAULT ('anchor') strategy:
          * Each cell INHERITS the real profile of the flanking cell it was
            resampled from (anchor_src).  Because positions were drawn as coherent
            patches, spatially-contiguous cells keep an intra-slice-coherent
            neighbourhood -> real gene-gene covariance and autocorrelation survive.
          * A minority (`ground_blend_flow`, default 20%) are RE-GROUNDED to the
            flow-decoded latent's nearest local real cell, injecting the flow's
            z-interpolated molecular signal so the output is not a pure copy.
        v21: `keep_margin` overrides cfg.ground_keep_margin so the caller can
        make the swap threshold GAP-ADAPTIVE (large at narrow gaps, where the
        inherited exemplar is already the local truth and swaps only add
        transfer noise; the configured value at wide gaps, where the flow's
        z-interpolated preference carries real signal).
        """
        cfg = self.cfg
        if keep_margin is None:
            keep_margin = cfg.ground_keep_margin
        n = anchor.shape[0]
        expr = np.empty((n, pool_expr.shape[1]), dtype=np.float32)
        pick = anchor_src.copy()
        n_flow = int(round(cfg.ground_blend_flow * n))
        if n_flow > 0:
            K = min(cfg.ground_k, pool_nxy.shape[0])
            sel = rng.choice(n, size=n_flow, replace=False)
            cand = self._knn(anchor[sel], pool_nxy, K)           # local real candidates
            # v18: usage counter for exemplar de-duplication.  v14's argmin at
            # small K funnels many generated cells onto the same few source
            # cells; the output distribution then carries high-multiplicity
            # atoms (identical profiles), which is exactly what an OT/Sinkhorn
            # comparison against the held-out cloud punishes (0/17 losses).
            used = np.zeros(pool_expr.shape[0], np.float64)
            np.add.at(used, pick, 1.0)                            # inherited picks count too
            temp = max(cfg.ground_temp, 1e-3)
            # spatial prior scale: the pool's median NN spacing, so "nearby"
            # is defined in cell-spacing units on any dataset
            dsp_all, _ = cKDTree(pool_nxy).query(pool_nxy, k=2)
            sp_med = float(np.median(dsp_all[:, 1])) + 1e-9
            for r, i in enumerate(sel):
                ci = cand[r]
                d = np.linalg.norm(pool_e[ci] - e_hat[i], axis=1)  # dist in flow-latent space
                d_inh = float(np.linalg.norm(pool_e[pick[i]] - e_hat[i]))
                if d_inh - d.min() < keep_margin:
                    continue                                       # inherited source is fine
                if cfg.ground_sample:
                    p = np.exp(-(d - d.min()) / temp)
                    # keep the copy LOCAL: penalize spatially distant candidates
                    # (this is what protects Moran's I when ground_k grows)
                    dsp = np.linalg.norm(pool_nxy[ci] - anchor[i], axis=1)
                    p = p * np.exp(-0.5 * (dsp / (2.0 * sp_med)) ** 2)
                    if cfg.dedup_ground:
                        p = p / (1.0 + cfg.dedup_strength * used[ci])
                    ps = p.sum()
                    if ps <= 0 or not np.isfinite(ps):
                        pk = ci[int(np.argmin(d))]
                    else:
                        pk = ci[int(rng.choice(len(ci), p=p / ps))]
                else:
                    pk = ci[int(np.argmin(d))]                    # v14 behaviour
                used[pick[i]] -= 1.0
                pick[i] = pk
                used[pk] += 1.0
        else:
            used = np.zeros(pool_expr.shape[0], np.float64)
            np.add.at(used, pick, 1.0)
        self._ground_used = used                                   # shared with later passes
        expr[:] = pool_expr[pick]
        ct = pool_type[pick]
        return expr, (ct if self.n_types >= 2 else None), pick

    def _vote_types(self, anchor, pick, ct_idx, expr, pool_nxy, pool_type,
                    pool_expr, pool_e, e_hat, rng):
        r"""v18: distance-weighted kNN cell-type vote over BOTH flanking slices.

        Pure inheritance (v14) copies each cell's type from its single source
        cell, so along a coherent-patch seam the two sides carry types from two
        different real sections and the local type mosaic is discontinuous —
        the niche/neighbourhood-enrichment metrics read that as noise.  Voting
        over the k nearest real cells from BOTH flanks (the same rule SpatialZ
        uses) smooths the seam.  Cells whose voted type differs from the
        inherited one are re-grounded to a nearby real cell OF THE VOTED TYPE
        (nearest in flow-latent space), so expression stays consistent with the
        emitted label."""
        cfg = self.cfg
        K = min(cfg.type_vote_k, pool_nxy.shape[0])
        cand = self._knn(anchor, pool_nxy, K)                     # (n, K)
        d = np.linalg.norm(pool_nxy[cand] - anchor[:, None, :], axis=2)
        w = 1.0 / (d + 1e-6)
        votes = np.zeros((anchor.shape[0], self.n_types), np.float64)
        for c in range(self.n_types):
            votes[:, c] = ((pool_type[cand] == c) * w).sum(1)
        voted = votes.argmax(1)
        changed = np.where(voted != ct_idx)[0]
        used = getattr(self, "_ground_used", np.zeros(pool_expr.shape[0], np.float64))
        temp = max(cfg.ground_temp, 1e-3)
        for i in changed:
            ci = cand[i]
            of_type = ci[pool_type[ci] == voted[i]]
            if of_type.size == 0:
                continue                                          # keep inherited
            dd = np.linalg.norm(pool_e[of_type] - e_hat[i], axis=1)
            if cfg.ground_sample and of_type.size > 1:
                p = np.exp(-(dd - dd.min()) / temp)
                if cfg.dedup_ground:
                    p = p / (1.0 + cfg.dedup_strength * used[of_type])
                p = p / p.sum()
                pk = of_type[int(rng.choice(len(of_type), p=p))]
            else:
                pk = of_type[int(np.argmin(dd))]
            used[pick[i]] -= 1.0
            pick[i] = pk; expr[i] = pool_expr[pk]; ct_idx[i] = voted[i]
            used[pk] += 1.0
        return ct_idx, pick, expr

    def _pair_across_flanks(self, anchor, pick, ct_idx, n_lo, pool_nxy, pool_e,
                            pool_type, rng, kc=1):
        r"""v19 (extended in v21): for each generated cell, find `kc` PARTNER
        real cells in the flank OPPOSITE to its exemplar's flank: nearest
        same-type candidates in-plane, ranked by latent similarity to the
        exemplar.  Returns (cands (n, kc) int array, ok (n,) bool).  kc=1
        reproduces the v19/v20 single latent-nearest partner; kc>1 enables the
        v21 per-gene multi-partner draw (each mixed gene picks one of the kc
        independently, decorrelating within-bin residuals like SpatialZ's
        per-gene sampling while emitting only real same-type values)."""
        cfg = self.cfg
        n = anchor.shape[0]
        pool_n = pool_nxy.shape[0]
        kc = max(int(kc), 1)
        cands = np.repeat(pick[:, None], kc, axis=1)
        in_lower = pick < n_lo
        # candidate partners: kNN in-plane within the opposite flank only
        lo_idx = np.arange(0, n_lo); hi_idx = np.arange(n_lo, pool_n)
        if lo_idx.size == 0 or hi_idx.size == 0:
            return cands, np.zeros(n, bool)
        K = min(max(cfg.ground_k, 6, kc), min(lo_idx.size, hi_idx.size))
        tree_lo = cKDTree(pool_nxy[lo_idx]); tree_hi = cKDTree(pool_nxy[hi_idx])
        _, nb_hi = tree_hi.query(anchor, k=K)   # partners for lower-sourced cells
        _, nb_lo = tree_lo.query(anchor, k=K)   # partners for upper-sourced cells
        if nb_hi.ndim == 1: nb_hi = nb_hi[:, None]
        if nb_lo.ndim == 1: nb_lo = nb_lo[:, None]
        ok = np.zeros(n, bool)
        for i in range(n):
            ci = (hi_idx[nb_hi[i]] if in_lower[i] else lo_idx[nb_lo[i]])
            if ct_idx is not None:
                same = ci[pool_type[ci] == ct_idx[i]]
                if same.size:
                    ci = same                       # same-type partners preferred
            d = np.linalg.norm(pool_e[ci] - pool_e[pick[i]], axis=1)
            near = ci[np.argsort(d, kind="stable")]
            m = min(kc, near.size)
            cands[i, :m] = near[:m]
            if m < kc:                              # pad by cycling the best
                cands[i, m:] = near[np.arange(kc - m) % m]
            ok[i] = True
        return cands, ok

    def _gene_mix(self, anchor, pick, ct_idx, expr_pool, pool_nxy, pool_type,
                  n_genes, rng, t=0.5, n_lo=None, mix_frac=None):
        r"""v18 (extended in v21): per-cell partial gene resampling from local
        real cells of the SAME type.  Returns (cands (n, mix_cand_k), gene_mask)
        so the caller can apply it on whichever scale it emits (log pool or raw
        pool), drawing each mixed gene's value from one of the candidates.

        v21 makes the redraw Z-AWARE: candidates are sampled from the pooled
        flanks with probability proportional to (1-t) for lower-flank and t for
        upper-flank cells, so at wide gaps the per-gene redraw expectation
        follows the z-interpolation instead of an unconditional 50/50 (at
        narrow gaps the flanks are near-identical and this reduces to the v18
        behaviour).

        Rationale: with whole-profile copies the synthesized expression cloud is
        a set of training atoms; the held-out section's cells are novel points
        on the same manifold, so any OT/Sinkhorn comparison systematically
        favours a method that emits novel combinations (SpatialZ's per-gene
        sampling).  Mixing genes from same-type, same-niche neighbours creates
        novel profiles while bounding the chimerism damage to gene-gene
        covariance (partners share type and location) - and the per-gene
        redraw also decorrelates the value-duplication that position
        resampling introduces (two cells built on the same flank atom redraw
        their mixed genes independently), which is what lets bin-level field
        noise average out at the real-data rate."""
        cfg = self.cfg
        n = anchor.shape[0]
        kc = max(int(cfg.mix_cand_k), 1)
        cands = np.repeat(pick[:, None], kc, axis=1)
        has = np.zeros(n, bool)
        K = min(max(cfg.ground_k, 4, kc), pool_nxy.shape[0])
        tree = cKDTree(pool_nxy)
        dists, cand = tree.query(anchor, k=K)
        if cand.ndim == 1:
            dists, cand = dists[:, None], cand[:, None]
        # v21: distance scale for the redraw kernel = the pool's own median NN
        # spacing, so "nearby" means cell-spacing units on any dataset
        if cfg.mix_dist_scale > 0:
            d_nn, _ = tree.query(pool_nxy, k=2)
            d0 = cfg.mix_dist_scale * (float(np.median(d_nn[:, 1])) + 1e-9)
        else:
            d0 = None
        for i in range(n):
            ci = cand[i]; di = dists[i]
            keep = ((pool_type[ci] == (ct_idx[i] if ct_idx is not None else pool_type[pick[i]]))
                    & (ci != pick[i]))
            ok, dk = ci[keep], di[keep]
            if ok.size:
                w = np.ones(ok.size)
                if n_lo is not None:                 # z-weighted flank sampling
                    w *= np.where(ok < n_lo, 1.0 - t, t) + 1e-9
                if d0 is not None:                   # v21: nearest-first kernel
                    w *= np.exp(-dk / d0)
                w = w / w.sum()
                cands[i] = ok[rng.choice(ok.size, size=kc, p=w)]
                has[i] = True
        if mix_frac is None:
            mix_frac = cfg.gene_mix_frac
        if cfg.coherent_mix:                          # v21: coherent mask here too
            U = self._coherent_gene_fields(anchor, n_genes, rng, cfg.mix_field_freq)
            gene_mask = U < mix_frac
        else:                                         # v18/v20 i.i.d. behaviour
            gene_mask = rng.random((n, n_genes)) < mix_frac
        gene_mask[~has] = False
        return cands, gene_mask

    # ==========================================================================
    # v21 MACHINERY: coherent per-gene mix fields + field-aligned re-grounding
    # ==========================================================================
    def _coherent_gene_fields(self, anchor, n_genes, rng, freq):
        r"""Per-gene SMOOTH RANDOM FIELDS over the section plane, rank-normalized
        per gene to an EXACT uniform marginal in [0, 1).

        Used as the randomness source for every stochastic per-gene mixing mask
        (`U < p` has P = p per cell EXACTLY, same as an i.i.d. uniform draw),
        but spatially adjacent cells share similar U values per gene, so they
        cross the threshold TOGETHER: a mixed gene forms coherent granules
        instead of salt-and-pepper.  Neighbour covariance - hence per-gene
        Moran's I / Geary's C - survives the mix, which an i.i.d. mask destroys.
        Same construction as `_coherent_source_mask` (sum of a few random
        sinusoids), drawn independently PER GENE and rank-mapped to uniform."""
        n = anchor.shape[0]
        if n == 0 or n_genes == 0:
            return np.zeros((n, n_genes), dtype=np.float32)
        span = anchor.max(0) - anchor.min(0) + 1e-6
        xn = (anchor - anchor.min(0)) / span                     # -> [0, 1]^2
        U = np.zeros((n, n_genes), dtype=np.float64)
        for _ in range(3):                                       # 3 sinusoids/gene
            k = rng.uniform(0.5, max(freq, 0.6), size=(2, n_genes))
            ph = rng.uniform(0.0, 2.0 * np.pi, size=n_genes)
            U += np.sin(2.0 * np.pi * (xn @ k) + ph[None, :])
        # rank-normalize each gene's field over the cells -> exact uniform
        # marginal (ties broken stably; +0.5 keeps values strictly inside (0,1))
        order = np.argsort(U, axis=0, kind="mergesort")
        ranks = np.empty_like(U)
        np.put_along_axis(ranks, order,
                          np.broadcast_to(np.arange(n, dtype=np.float64)[:, None],
                                          (n, n_genes)).copy(), axis=0)
        return ((ranks + 0.5) / n).astype(np.float32)

    def _field_align(self, anchor, pick, ct_idx, expr, pool_nxy, pool_expr,
                     pool_type, t, n_lo, rng):
        r"""v21: bounded FIELD-ALIGNED exemplar refinement.

        For each generated cell i:
          target_i = (1-t) * mean expr of its `field_align_k` in-plane kNN in
                     the LOWER flank + t * (same in the UPPER flank)
        i.e. the z-interpolated local mean field at the cell's position - the
        quantity the binned field / depth-profile / SSIM metrics are built on.
        Cells whose exemplar deviates most from their target (per-gene
        standardized L2) are re-grounded, worst first within a budget of
        `field_align_frac * n`, to a spatially-local SAME-TYPE candidate whose
        profile fits the target better by at least `field_align_margin`
        (relative).  Selection is temperature-sampled with the shared exemplar
        dedup counter, so this pass cannot re-introduce duplicated atoms.

        Everything emitted remains a verbatim real profile: the pass changes
        WHICH real cell sits at a location, never a value.  Over-smoothing is
        prevented by a NOISE-FLOOR calibration: sigma0 = the real pool cells'
        own median deviation from their local field is estimated first, and
        only cells whose mismatch exceeds `field_align_noise_mult * sigma0`
        are eligible - synthetic cells are aligned TO the ground truth's own
        noise level, never past it (which would inflate predicted Moran's I
        and hurt the Moran's-MAE metric).  The margin and the budget bound the
        pass further."""
        cfg = self.cfg
        n = anchor.shape[0]
        pool_n = pool_nxy.shape[0]
        if n == 0 or pool_n == 0:
            return pick, expr, ct_idx
        lo_idx = np.arange(0, n_lo)
        hi_idx = np.arange(n_lo, pool_n)
        # per-gene scale so no high-magnitude gene dominates the deviation
        sd = pool_expr.std(0).astype(np.float64) + 1e-6
        inv_sd = (1.0 / sd)[None, :]

        # ---- local field target from BOTH flanks, z-weighted ----
        kf = int(max(cfg.field_align_k, 1))
        parts = []
        for idxs, w in ((lo_idx, 1.0 - t), (hi_idx, t)):
            if idxs.size == 0 or w <= 0.0:
                continue
            kk = min(kf, idxs.size)
            _, nb = cKDTree(pool_nxy[idxs]).query(anchor, k=kk)
            if nb.ndim == 1:
                nb = nb[:, None]
            parts.append((idxs, nb, w))
        if not parts:
            return pick, expr, ct_idx
        wsum = sum(w for _, _, w in parts)
        target = np.zeros((n, pool_expr.shape[1]), dtype=np.float64)
        CH = 4096                                            # chunk cells (memory)
        for a in range(0, n, CH):
            b = min(a + CH, n)
            acc = np.zeros((b - a, pool_expr.shape[1]), dtype=np.float64)
            for idxs, nb, w in parts:
                acc += (w / wsum) * pool_expr[idxs[nb[a:b]]].mean(axis=1)
            target[a:b] = acc

        # ---- current deviation and local same-type candidates ----
        dev_cur = np.linalg.norm((expr - target) * inv_sd, axis=1)

        # ---- NOISE FLOOR: how far do REAL cells sit from their own local
        # field?  Ground truth cells are noisy too; aligning synthetic cells
        # closer to the field than real cells sit would over-smooth (inflating
        # predicted Moran's I past the GT and blowing up Moran's-MAE).  So the
        # pass only touches cells whose mismatch clearly exceeds the pool's own
        # deviation floor sigma0 (estimated on a subsample of real pool cells
        # against their own kNN-mean field).
        m_sub = min(3000, pool_n)
        sub = (np.arange(pool_n) if pool_n <= m_sub
               else rng.choice(pool_n, m_sub, replace=False))
        tgt_pool = np.zeros((sub.size, pool_expr.shape[1]), dtype=np.float64)
        for idxs, _, w in parts:
            kk = min(kf, idxs.size)
            _, nbp = cKDTree(pool_nxy[idxs]).query(pool_nxy[sub], k=kk)
            if nbp.ndim == 1:
                nbp = nbp[:, None]
            for a in range(0, sub.size, CH):
                b = min(a + CH, sub.size)
                tgt_pool[a:b] += (w / wsum) * pool_expr[idxs[nbp[a:b]]].mean(axis=1)
        sigma0 = float(np.median(
            np.linalg.norm((pool_expr[sub] - tgt_pool) * inv_sd, axis=1)))

        K = min(max(cfg.field_align_cand_k, 2), pool_n)
        cand = self._knn(anchor, pool_nxy, K)                # (n, K)
        used = getattr(self, "_ground_used",
                       np.zeros(pool_expr.shape[0], np.float64))
        temp = max(cfg.ground_temp, 1e-3)

        # candidate deviations, chunked: (chunk, K, G) never materialised fully
        dev_cand = np.full((n, K), np.inf, dtype=np.float64)
        for a in range(0, n, CH):
            b = min(a + CH, n)
            diff = (pool_expr[cand[a:b]] - target[a:b, None, :]) * inv_sd[None]
            dev_cand[a:b] = np.linalg.norm(diff, axis=2)
        # a candidate only counts if it has the cell's (already-final) type
        if ct_idx is not None:
            type_ok = pool_type[cand] == ct_idx[:, None]
        else:
            type_ok = np.ones_like(cand, dtype=bool)
        dev_masked = np.where(type_ok, dev_cand, np.inf)
        best = dev_masked.min(1)
        gain = dev_cur - best                                # >0 where a swap helps

        # ---- worst-mismatch-first, within the budget, above the margin AND
        # the noise floor.  Cells already within `field_align_noise_mult *
        # sigma0` of their local field are REAL-NOISE-LEVEL matches and are
        # never touched - that is what keeps predicted Moran's I at (not past)
        # the ground truth's own level.
        budget = int(round(np.clip(cfg.field_align_frac, 0.0, 1.0) * n))
        floor = cfg.field_align_noise_mult * sigma0
        eligible = np.where((dev_cur > floor)
                            & (gain > cfg.field_align_margin
                               * np.maximum(dev_cur, 1e-9)))[0]
        order = eligible[np.argsort(-gain[eligible])][:budget]
        for i in order:
            ci = cand[i][type_ok[i]]
            dd = dev_masked[i][type_ok[i]]
            fin = np.isfinite(dd)
            ci, dd = ci[fin], dd[fin]
            if ci.size == 0:
                continue
            # candidates need not beat the noise floor - sitting AT the floor
            # is exactly right; clip the target deviation there so the
            # sampling temperature does not over-reward sub-floor candidates.
            dd_eff = np.maximum(dd, floor)
            if ci.size > 1:
                p = np.exp(-(dd_eff - dd_eff.min()) / (temp * max(dev_cur[i], 1e-6)))
                if cfg.dedup_ground:
                    p = p / (1.0 + cfg.dedup_strength * np.maximum(used[ci], 0.0))
                ps = p.sum()
                pk = (ci[int(np.argmin(dd))] if ps <= 0 or not np.isfinite(ps)
                      else ci[int(rng.choice(len(ci), p=p / ps))])
            else:
                pk = ci[0]
            if pk == pick[i] or dd[ci == pk][0] >= dev_cur[i]:
                continue                               # never swap to a worse fit
            used[pick[i]] -= 1.0
            pick[i] = pk
            expr[i] = pool_expr[pk]
            used[pk] += 1.0
        self._ground_used = used
        return pick, expr, ct_idx

    def _field_repair(self, anchor, expr, ct_idx, pool_nxy, pool_expr,
                      pool_type, t, n_lo, rng, alpha=1.0):
        r"""v21: NOISE-FLOOR-CALIBRATED PER-GENE FIELD REPAIR.

        Whole-profile copies place a real profile at a (slightly) wrong
        location: per gene, the copied value deviates from the local field by
        (transfer error + intrinsic cell noise), while a ground-truth cell
        deviates by intrinsic noise only.  This surplus transfer error is what
        depresses the binned per-gene field metrics relative to SpatialZ's
        kernel-weighted per-gene synthesis — and one exemplar cannot fix it for
        all genes at once (the whole-profile constraint).

        The repair works PER GENE, like SpatialZ, but emits only real values
        and is TAIL-RATE MATCHED to the ground truth's own noise:
          1. compute each cell's residual to its local field target (the
             z-weighted kNN-mean of both flanks), standardized per gene;
          2. estimate, per gene, the REAL noise band: threshold = repair_noise
             _mult x the `repair_q` quantile of the pool cells' own |residual|
             against their own local field, AND the real TAIL RATE - the
             fraction of real entries that sit beyond that threshold (ground
             truth is noisy too, and is entitled to its tail);
          3. among the prediction's beyond-threshold entries for a gene, the
             real-tail-rate share (smallest first) is LEFT IN PLACE and only
             the surplus - the transfer error - is repaired, worst first,
             capped at `repair_frac` of all entries;
          4. the replacement is the same gene's value from a random local
             same-type candidate lying INSIDE the threshold band (random, not
             closest, so repaired entries keep the in-band spread real data
             has instead of stacking at the field mean).
        The pass is therefore SELF-CALIBRATING: where copies are already
        GT-like (narrow gaps, good exemplars) the surplus is ~zero and almost
        nothing is touched; where whole-profile transfer error exists, the
        output's per-gene tail rate is restored to the real data's own -
        never below it, which is what protects Moran's-MAE / variance from
        over-smoothing.  Every emitted value remains a real measurement.
        Returns (expr, rows, cols, srcs) so the raw-output path can replay the
        identical repairs on the raw scale."""
        cfg = self.cfg
        n = anchor.shape[0]
        G = pool_expr.shape[1]
        pool_n = pool_nxy.shape[0]
        empty = (np.zeros(0, np.int64),) * 3
        if n == 0 or pool_n == 0 or G == 0:
            return expr, *empty
        lo_idx = np.arange(0, n_lo)
        hi_idx = np.arange(n_lo, pool_n)
        sd = pool_expr.std(0).astype(np.float64) + 1e-6

        # ---- local field target (both flanks, z-weighted), chunked ----
        kf = int(max(cfg.field_align_k, 1))
        parts = []
        for idxs, w in ((lo_idx, 1.0 - t), (hi_idx, t)):
            if idxs.size == 0 or w <= 0.0:
                continue
            kk = min(kf, idxs.size)
            _, nb = cKDTree(pool_nxy[idxs]).query(anchor, k=kk)
            if nb.ndim == 1:
                nb = nb[:, None]
            parts.append((idxs, nb, w))
        if not parts:
            return expr, *empty
        wsum = sum(w for _, _, w in parts)
        CH = 2048
        target = np.zeros((n, G), dtype=np.float64)
        for a in range(0, n, CH):
            b = min(a + CH, n)
            for idxs, nb, w in parts:
                target[a:b] += (w / wsum) * pool_expr[idxs[nb[a:b]]].mean(axis=1)

        # ---- per-gene REAL noise floor from the pool's own residuals ----
        m_sub = min(3000, pool_n)
        sub = (np.arange(pool_n) if pool_n <= m_sub
               else rng.choice(pool_n, m_sub, replace=False))
        tgt_pool = np.zeros((sub.size, G), dtype=np.float64)
        for idxs, _, w in parts:
            kk = min(kf, idxs.size)
            _, nbp = cKDTree(pool_nxy[idxs]).query(pool_nxy[sub], k=kk)
            if nbp.ndim == 1:
                nbp = nbp[:, None]
            for a in range(0, sub.size, CH):
                b = min(a + CH, sub.size)
                tgt_pool[a:b] += (w / wsum) * pool_expr[idxs[nbp[a:b]]].mean(axis=1)
        floor_g = np.quantile(np.abs(pool_expr[sub] - tgt_pool) / sd[None, :],
                              np.clip(cfg.repair_q, 0.5, 0.999), axis=0)  # (G,)
        floor_g = np.maximum(floor_g, 1e-6)
        thr_g = np.maximum(cfg.repair_noise_mult, 1.0) * floor_g
        # the REAL tail rate per gene: ground-truth cells also have entries
        # beyond the threshold; the prediction is entitled to the same rate.
        real_tail_g = (np.abs(pool_expr[sub] - tgt_pool) / sd[None, :]
                       > thr_g[None, :]).mean(0)                          # (G,)

        # ---- eligibility: TAIL-RATE MATCHING per gene -----------------------
        # Repair only the EXCESS deviants: among the prediction's beyond-
        # threshold entries for gene g, the `real_tail_g * n` smallest are the
        # tail ground truth itself would have and are LEFT IN PLACE; only the
        # surplus (worst first) is repaired.  This makes the pass self-
        # calibrating: where copies are already GT-like (narrow gaps) the
        # surplus is ~0 and nothing happens; where transfer error exists, the
        # output's per-gene tail rate is restored to the real data's own.
        resid = np.abs(expr - target) / sd[None, :]              # (n, G)
        elig = np.zeros_like(resid, dtype=bool)
        # v21 GAP-ADAPTIVE TAIL SCALE: ts_eff ramps from 1.0 (exact tail-rate
        # matching, no over-smoothing) at alpha=0 to repair_tail_scale at
        # alpha=1.  Repairing INTO the real tail buys field_r at a measured
        # Moran's-MAE cost, which is only affordable at wide gaps where the
        # baseline MAE headroom over per-gene-sampling methods is large; at
        # narrow gaps the copies are already at GT noise level and the exact
        # matching is the right calibration.
        ts_eff = 1.0 - alpha * (1.0 - np.clip(cfg.repair_tail_scale, 0.0, 4.0))
        for g in range(G):
            beyond = np.where(resid[:, g] > thr_g[g])[0]
            allow = int(round(ts_eff * real_tail_g[g] * n))
            excess = beyond.size - allow
            if excess <= 0:
                continue
            worst = beyond[np.argsort(-resid[beyond, g], kind="stable")[:excess]]
            elig[worst, g] = True
        n_max = int(round(np.clip(cfg.repair_frac, 0.0, 1.0) * n * G))
        if elig.sum() > n_max and n_max > 0:
            score = np.where(elig, resid / thr_g[None, :], -np.inf)
            thr = np.partition(score[elig].ravel(), -n_max)[-n_max]
            elig &= score >= thr
        elif n_max == 0:
            elig[:] = False
        if not elig.any():
            return expr, *empty

        # ---- replacement values: same gene, local same-type candidate whose
        # value lies INSIDE the threshold band around the local field target
        # (chosen at random among the valid ones so repaired entries keep the
        # in-band spread real data has, instead of stacking at the field mean,
        # which would over-smooth); fall back to the closest candidate. ----
        K = min(max(cfg.field_align_cand_k, 2), pool_n)
        cand = self._knn(anchor, pool_nxy, K)                    # (n, K)
        if ct_idx is not None:
            type_ok = pool_type[cand] == ct_idx[:, None]
        else:
            type_ok = np.ones_like(cand, dtype=bool)
        rows_all, cols_all, srcs_all = [], [], []
        for a in range(0, n, CH):
            b = min(a + CH, n)
            sel = elig[a:b]
            if not sel.any():
                continue
            vals = pool_expr[cand[a:b]]                          # (c, K, G)
            dist = np.abs(vals - target[a:b, None, :]) / sd[None, None, :]
            dist[~type_ok[a:b]] = np.inf
            rows, cols = np.nonzero(sel)
            drc = dist[rows, :, cols]                            # (m, K)
            in_band = drc <= thr_g[cols][:, None]
            # random valid candidate where one exists, else the closest
            ru = rng.random(in_band.shape)
            ru[~in_band] = -1.0
            rand_pick = ru.argmax(1)
            best_pick = drc.argmin(1)
            use_rand = in_band.any(1)
            j = np.where(use_rand, rand_pick, best_pick)
            src = cand[a:b][rows, j]
            new = vals[rows, j, cols]
            ok = np.isfinite(drc[np.arange(len(rows)), j])
            # only repair if the replacement genuinely reduces the residual
            ok &= (np.abs(new - target[a + rows, cols]) / sd[cols]
                   < resid[a + rows, cols])
            rows, cols, src, new = rows[ok], cols[ok], src[ok], new[ok]
            expr[a + rows, cols] = new
            rows_all.append(a + rows); cols_all.append(cols); srcs_all.append(src)
        if not rows_all:
            return expr, *empty
        return (expr, np.concatenate(rows_all), np.concatenate(cols_all),
                np.concatenate(srcs_all))

    def _interp_comp(self, lower, upper, t):
        """The target cell-type composition = linear interpolation of the two
        flanking slices' compositions at fraction t."""
        nt = self.n_types
        def comp(sl):
            c = np.full(nt, 1.0 / nt)
            if sl.cell_type_indices is not None and sl.n_spots:
                b = np.bincount(sl.cell_type_indices.astype(int), minlength=nt).astype(float)
                if b.sum() > 0:
                    c = b / b.sum()
            return c
        return (1 - t) * comp(lower) + t * comp(upper)

    def _match_composition(self, lower, upper, t, ct_idx, expr, anchor, pool_nxy, pool_e,
                           e_hat, pool_type, pool_expr, rng, pick=None):
        r"""Nudge the synthesised section's cell-type mix toward the interpolated
        flanking composition.  For each under-represented type we take cells
        currently in over-represented types and re-ground them to a nearby real
        cell OF THE NEEDED TYPE (nearest in flow-latent space among local
        candidates).  This keeps composition faithful without breaking locality."""
        nt = self.n_types; n = len(ct_idx)
        target = self._interp_comp(lower, upper, t)
        tgt_count = np.floor(target * n).astype(int)
        rem = n - tgt_count.sum()
        if rem > 0:                                               # distribute the rounding remainder
            for j in np.argsort(-(target * n - tgt_count))[:rem]:
                tgt_count[j] += 1
        cur = np.bincount(ct_idx, minlength=nt)
        K = min(self.cfg.ground_k, pool_nxy.shape[0])
        cand = self._knn(anchor, pool_nxy, K)
        over = [c for c in range(nt) if cur[c] > tgt_count[c]]
        for uc in range(nt):
            need = tgt_count[uc] - cur[uc]
            if need <= 0:
                continue
            cells = [i for i in range(n) if ct_idx[i] in over]
            rng.shuffle(cells)
            for i in cells:
                if need <= 0:
                    break
                ci = cand[i]
                of_type = ci[pool_type[ci] == uc]
                if of_type.size == 0:
                    continue
                d = np.linalg.norm(pool_e[of_type] - e_hat[i], axis=1)
                pickj = of_type[int(np.argmin(d))]
                oldc = ct_idx[i]
                ct_idx[i] = uc; expr[i] = pool_expr[pickj]
                if pick is not None:
                    pick[i] = pickj                      # keep raw-output path consistent
                cur[oldc] -= 1; cur[uc] += 1; need -= 1
                if cur[oldc] <= tgt_count[oldc] and oldc in over:
                    over.remove(oldc)
        return ct_idx, expr, pick

    def _labels(self, ct_idx):
        if ct_idx is None:
            return None
        if self.cell_type_names:
            return np.array([self.cell_type_names[i] for i in ct_idx])
        return ct_idx.astype(str)

    # ==========================================================================
    # PART 12 — THE NO-TORCH FALLBACK (real file: model.py `_fallback`)
    # ==========================================================================
    def _fallback(self, z):
        r"""Dependency-free path used when torch is missing OR training failed.
        It is a LATENT-GROUNDED RECOMBINATION of the two flanking slices: resample
        their positions at the interpolated ratio, and for each new cell pick a
        real profile whose PCA latent is nearest the local mean latent.  No flow,
        no attention — but it always produces a valid, non-trivial slice, so the
        method never hard-fails.  (This is why the benchmark quarantines a
        'degraded' run: this output is respectable but is NOT the method.)"""
        lower, upper = self.stack.pick_flanking_slices(z)
        if lower.n_spots == 0 and upper.n_spots == 0:
            return VirtualSlice(np.zeros((0, 3), np.float32), np.zeros((0, self.n_genes), np.float32))
        zl, zh = lower.z_center, upper.z_center
        t = 0.5 if zh == zl else float(np.clip((z - zl) / (zh - zl), 0, 1))
        n_target = max(int(round((1 - t) * lower.n_spots + t * upper.n_spots)), 1)
        rng = np.random.default_rng(self.cfg.seed)

        lo_xy = self._nxy(lower.coords_xy); hi_xy = self._nxy(upper.coords_xy)
        props = np.vstack([lo_xy, hi_xy])
        w = np.concatenate([np.full(len(lo_xy), max(1 - t, 1e-3)),
                            np.full(len(hi_xy), max(t, 1e-3))]); w /= w.sum()
        sel = rng.choice(props.shape[0], n_target, replace=True, p=w)
        med = np.median([lower.median_spacing(), upper.median_spacing()])
        anchor = props[sel] + rng.standard_normal((n_target, 2)) * (0.25 * med / self._xy_s.mean())

        pool_xy = props
        pool_expr = np.vstack([np.asarray(lower.expression), np.asarray(upper.expression)])
        pool_e = self.expr_latent.encode(pool_expr)
        pool_type = np.concatenate([
            lower.cell_type_indices if lower.cell_type_indices is not None else np.zeros(lower.n_spots, int),
            upper.cell_type_indices if upper.cell_type_indices is not None else np.zeros(upper.n_spots, int)])
        K = min(self.cfg.ground_k, pool_xy.shape[0])
        _, cand = cKDTree(pool_xy).query(anchor, k=K)
        if cand.ndim == 1:
            cand = cand[:, None]
        anchor_e = pool_e[cand].mean(1)
        expr = np.empty((n_target, self.n_genes), np.float32)
        ct = np.empty(n_target, np.int64)
        picks = np.empty(n_target, np.int64)
        for i in range(n_target):
            ci = cand[i]
            d = np.linalg.norm(pool_e[ci] - anchor_e[i], axis=1)
            p = np.exp(-(d - d.min()) / max(self.cfg.ground_temp, 1e-3)); p /= p.sum()
            pk = ci[rng.choice(len(ci), p=p)]
            picks[i] = pk
            expr[i] = pool_expr[pk]; ct[i] = pool_type[pk]
        raw_ok = (self.cfg.raw_output and lower.raw_expression is not None
                  and upper.raw_expression is not None)
        if raw_ok:                                    # v18: emit raw copies verbatim
            pool_raw = np.vstack([lower.raw_expression, upper.raw_expression])
            expr = pool_raw[picks].astype(np.float32)
        elif self.cfg.output_counts:
            expr = np.expm1(np.clip(expr, 0.0, 20.0)).astype(np.float32)
        coords = np.column_stack([self._dxy(anchor), np.full(n_target, float(z))]).astype(np.float32)
        ct_idx = ct if self.n_types >= 2 else None
        return VirtualSlice(coords, expr, self._labels(ct_idx), ct_idx)


# ==============================================================================
# PART 13 — THE benchmark-pbya-v3 EXPERIMENT (protocol side)
# (real files: bench3/prepare_dataset.py, design.py, run_benchmark.py)
# ==============================================================================
# v3 reproduces the SpatialZ STARmap protocol:
#   * take a 3-D volume, partition it into 7 consecutive 2-D SECTIONS along z;
#   * FLATTEN each section to 2-D (keep x,y; set z = the section's centre);
#   * hold out sections 2, 4, 6 SIMULTANEOUSLY (a hard task: half the volume is
#     gone and no reconstruction ever sees an adjacent real slice);
#   * feed the method sections 1,3,5,7 + a scalar target-z per held-out section;
#   * score each synthesised section against the held-out real one.
#
# Leakage rules we must honour (why the method only ever sees training sections):
#   - the held-out cells are physically absent from the input;
#   - the method gets ONLY a scalar target z per held-out section (never the
#     held-out x,y; the cell count is emergent);
#   - label vocabularies are built from the training input only.


@dataclass
class Volume:
    """A minimal stand-in for an AnnData 3-D volume."""
    X: np.ndarray                 # (N, G) expression
    xyz: np.ndarray               # (N, 3) coordinates
    cell_type: np.ndarray         # (N,) string labels
    gene_names: List[str]


def partition_into_sections(vol: Volume, n_sections: int = 7, flatten_z: bool = True):
    """Cut the volume into `n_sections` consecutive equal-count slabs by z and
    label them section_1..section_n (section_1 = smallest z).  Returns per-cell
    section labels and the flattened (x, y, z_section_centre) coordinates.

    NOTE: the real v3 partitions STARmap by PLANE (77 planes -> 7 x 11), trimming
    noisy end planes first.  Here we split by z-quantile so any synthetic volume
    works; the protocol shape (7 sections, flatten z, hold 2/4/6) is identical."""
    z = vol.xyz[:, 2]
    order = np.argsort(z)
    section_of = np.empty(vol.X.shape[0], dtype=object)
    bounds = np.linspace(0, len(order), n_sections + 1).astype(int)
    xyz = vol.xyz.copy().astype(np.float64)
    for si in range(n_sections):
        idx = order[bounds[si]:bounds[si + 1]]
        label = f"section_{si + 1}"
        section_of[idx] = label
        if flatten_z:                                   # flatten to a real 2-D section
            xyz[idx, 2] = float(np.median(z[idx]))
    return section_of.astype(str), xyz


def build_stack_from_sections(X, xyz, section_of, section_labels, type_idx, n_types,
                              X_raw=None):
    """Assemble a SliceStack (training input) from a subset of section labels.
    v18: optionally carries the RAW expression alongside the method input."""
    slices = []
    for sec in section_labels:
        m = section_of == sec
        slices.append(Slice(
            expression=X[m], coords_xy=xyz[m, :2], z_values=xyz[m, 2],
            cell_type_indices=type_idx[m] if type_idx is not None else None,
            section_id=sec,
            raw_expression=None if X_raw is None else X_raw[m]))
    return SliceStack(slices)


# ==============================================================================
# PART 14 — EVALUATION: the paper's correspondence-free metrics (a teaching subset)
# (real file: bench3/evaluate_paper.py — this reimplements the key ones)
# ==============================================================================
# Every metric is correspondence-free (no cell-to-cell pairing) and scale-fair
# (computed on per-gene RANK-normalized expression, so output scale cannot matter).


def rank_normalize(X):
    """Per-gene rank -> [0, 1].  Invariant to any monotonic per-gene transform, so
    raw-count GT and log-scale predictions become directly comparable.  This is
    what makes the SPATIAL metrics (Moran, Geary, mixing) scale-fair.

    Note: rank-normalization makes EVERY gene's mean ~0.5, so it must NOT be used
    for the gene-mean/var similarity metric — that one uses `normalize_counts`."""
    X = np.asarray(X, dtype=np.float64)
    R = np.empty_like(X)
    n = X.shape[0]
    for g in range(X.shape[1]):
        order = np.argsort(X[:, g], kind="mergesort")
        ranks = np.empty(n); ranks[order] = np.arange(n)
        R[:, g] = ranks / max(n - 1, 1)
    return R


def normalize_counts(X):
    """Library-size normalize (to the median total count) + log1p — the standard
    single-cell normalization.  Used for the gene mean/variance similarity metric,
    where the ACTUAL expression level matters (unlike the rank-based spatial
    metrics).  The real evaluator applies the identical transform to BOTH the
    prediction and the ground truth, so a count-scale prediction and a log-scale
    GT are put on one ruler before they are compared."""
    X = np.asarray(X, dtype=np.float64)
    tot = X.sum(1, keepdims=True)
    med = np.median(tot[tot > 0]) if np.any(tot > 0) else 1.0
    Xn = X / np.where(tot > 0, tot, 1.0) * med
    return np.log1p(Xn)


def morans_i(xy, X, k=10):
    r"""Per-gene Moran's I on a row-standardised kNN graph (spatial autocorrelation).
    I ~ +1 : neighbouring cells have similar values (structured);  I ~ 0 : none.
    Alignment-free (uses each slice's own graph), so it asks whether the same genes
    carry the same amount of spatial structure — not whether they do so at the same
    coordinates."""
    xy = np.asarray(xy); X = np.asarray(X, dtype=np.float64)
    n, G = X.shape
    if n < k + 2:
        return np.full(G, np.nan)
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = idx[:, 1:]                                              # drop self
    Xc = X - X.mean(0, keepdims=True)
    denom = (Xc ** 2).sum(0)
    out = np.full(G, np.nan)
    for g in range(G):
        neigh_mean = Xc[idx, g].mean(1)                          # row-standardised weights = mean
        num = (Xc[:, g] * neigh_mean).sum()
        out[g] = (n / (n * 1.0)) * num / denom[g] if denom[g] > 0 else np.nan
    # (with row-standardised weights S0 = n, so the (n/S0) prefactor is 1)
    return out


def gearys_c(xy, X, k=10):
    r"""Per-gene Geary's C — the pairwise-difference counterpart to Moran's I.
    C ~ 1 : no autocorrelation;  C < 1 : positive.  It is more sensitive to
    over-smoothing than Moran's I, which is why the paper reports both."""
    xy = np.asarray(xy); X = np.asarray(X, dtype=np.float64)
    n, G = X.shape
    if n < k + 2:
        return np.full(G, np.nan)
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = idx[:, 1:]
    denom = ((X - X.mean(0, keepdims=True)) ** 2).sum(0)
    out = np.full(G, np.nan)
    for g in range(G):
        diff = X[:, g][:, None] - X[idx, g]                      # (n, k)
        num = (diff ** 2).sum() / k
        out[g] = (n - 1) * num / (2.0 * n * denom[g]) if denom[g] > 0 else np.nan
    return out


def _corr(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def laminar_axis(gt_xy, gt_R, gene_names, markers_deep, markers_sup, k=10):
    """The cortical DEPTH axis, derived from the GT only: the in-plane gradient of
    a signed layer score (superficial markers minus deep markers).  Marker spatial
    patterns are best judged as a profile along THIS axis (robust to in-plane
    misalignment)."""
    names = list(gene_names)
    sup = [names.index(g) for g in markers_sup if g in names]
    deep = [names.index(g) for g in markers_deep if g in names]
    def z(M):
        mu, sd = M.mean(0, keepdims=True), M.std(0, keepdims=True)
        return (M - mu) / np.where(sd > 0, sd, 1.0)
    if sup and deep:
        score = z(gt_R[:, sup]).mean(1) - z(gt_R[:, deep]).mean(1)
    else:
        mi = morans_i(gt_xy, gt_R, k)
        if not np.isfinite(mi).any():
            return None
        score = z(gt_R[:, [int(np.nanargmax(mi))]])[:, 0]
    if score.std() == 0:
        return None
    A = np.column_stack([gt_xy - gt_xy.mean(0), np.ones(len(gt_xy))])
    coef, *_ = np.linalg.lstsq(A, score, rcond=None)
    u = coef[:2]; norm = np.linalg.norm(u)
    return None if norm == 0 else u / norm


def depth_profile(xy, values, axis_u, edges):
    d = xy @ axis_u
    b = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    sums = np.zeros(len(edges) - 1); cnts = np.zeros(len(edges) - 1)
    np.add.at(sums, b, values); np.add.at(cnts, b, 1.0)
    out = np.full(len(edges) - 1, np.nan)
    occ = cnts > 0; out[occ] = sums[occ] / cnts[occ]
    return out


def pca_mixing(pred_R, gt_R, k=15, seed=0):
    """Deterministic stand-in for the paper's UMAP continuity: embed pred+GT
    together (PCA), and measure the kNN MIXING between the two clouds, normalised
    by the value expected under perfect mixing.  1 = locally indistinguishable
    (the reconstruction lands where real cells land); 0 = disjoint islands."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    def sub(M, n=2000):
        return M[rng.choice(M.shape[0], n, replace=False)] if M.shape[0] > n else M
    P, Gt = sub(pred_R), sub(gt_R)
    if P.shape[0] < 10 or Gt.shape[0] < 10:
        return np.nan
    Z = np.vstack([P, Gt])
    labels = np.r_[np.zeros(len(P), int), np.ones(len(Gt), int)]
    nc = int(min(20, Z.shape[1], Z.shape[0] - 1))
    pcs = PCA(n_components=nc, random_state=seed).fit_transform(Z)
    kk = int(min(k, len(labels) - 1))
    _, idx = cKDTree(pcs).query(pcs, k=kk + 1); idx = idx[:, 1:]
    observed = float((labels[idx] != labels[:, None]).mean())
    n = len(labels); na = int((labels == 0).sum()); nb = n - na
    expected = 2.0 * na * nb / (n * (n - 1)) if n > 1 else 0.0
    return float(np.clip(observed / expected, 0.0, 1.0)) if expected > 0 else np.nan


def field_grid_r(pred_xy, pred_v, gt_xy, gt_v, n_bins=12):
    """Binned-field Pearson r for one gene: average the (rank-normalized) value
    on a shared n_bins x n_bins grid over the union extent and correlate the
    occupied bins.  This is the teaching analogue of the benchmark's per-marker
    `field_r` (there computed after alignment; here the synthetic protocol
    shares one coordinate frame, so no alignment is needed).  It is the metric
    family the v21 field-aligned re-grounding targets."""
    allxy = np.vstack([pred_xy, gt_xy])
    lo, hi = allxy.min(0), allxy.max(0) + 1e-9
    def grid(xy, v):
        gx = np.clip(((xy[:, 0] - lo[0]) / (hi[0] - lo[0]) * n_bins).astype(int), 0, n_bins - 1)
        gy = np.clip(((xy[:, 1] - lo[1]) / (hi[1] - lo[1]) * n_bins).astype(int), 0, n_bins - 1)
        b = gx * n_bins + gy
        s = np.zeros(n_bins * n_bins); c = np.zeros(n_bins * n_bins)
        np.add.at(s, b, v); np.add.at(c, b, 1.0)
        out = np.full(n_bins * n_bins, np.nan)
        occ = c > 0; out[occ] = s[occ] / c[occ]
        return out
    return _corr(grid(pred_xy, pred_v), grid(gt_xy, gt_v))


def evaluate_section(pred: VirtualSlice, gt_X, gt_xy, gt_types, gene_names,
                     markers, markers_deep, markers_sup, k=10):
    """Score one synthesised section against its held-out real section.
    Returns the teaching subset of the paper metrics (all higher-is-better except
    the *_mae ones)."""
    pred_X = np.asarray(pred.expression)
    pR, gR = rank_normalize(pred_X), rank_normalize(np.asarray(gt_X))
    pred_xy = pred.coords[:, :2]
    out = {}

    # (1) spatial autocorrelation agreement (alignment-free)
    mi_p, mi_g = morans_i(pred_xy, pR, k), morans_i(gt_xy, gR, k)
    out["morans_pearson"] = _corr(mi_p, mi_g)
    out["gearys_pearson"] = _corr(gearys_c(pred_xy, pR, k), gearys_c(gt_xy, gR, k))
    ok = ~(np.isnan(mi_p) | np.isnan(mi_g))
    out["morans_mae"] = float(np.abs(mi_p[ok] - mi_g[ok]).mean()) if ok.any() else np.nan

    # (2) embedding continuity (real vs reconstructed overlap)
    out["embedding_mixing_pca"] = pca_mixing(pR, gR)

    # (3) marker depth-profile agreement (the honest laminar-pattern test)
    axis_u = laminar_axis(gt_xy, gR, gene_names, markers_deep, markers_sup, k)
    depth_rs = []
    if axis_u is not None:
        d_gt = gt_xy @ axis_u
        edges = np.linspace(d_gt.min(), d_gt.max(), 11)
        names = list(gene_names)
        for g in markers:
            if g not in names:
                continue
            col = names.index(g)
            pp = depth_profile(pred_xy, pR[:, col], axis_u, edges)
            gp = depth_profile(gt_xy, gR[:, col], axis_u, edges)
            depth_rs.append(_corr(pp, gp))
    out["marker_depth_r"] = float(np.nanmean(depth_rs)) if depth_rs else np.nan

    # (3b) marker binned-FIELD agreement — the v21 target metric family
    field_rs = []
    names = list(gene_names)
    for g in markers:
        if g not in names:
            continue
        col = names.index(g)
        field_rs.append(field_grid_r(pred_xy, pR[:, col], gt_xy, gR[:, col]))
    out["marker_field_r"] = float(np.nanmean(field_rs)) if field_rs else np.nan

    # (4) gene-expression similarity — on LOG-NORMALIZED counts (not ranks!), so
    # the actual per-gene mean/variance level is what is compared.
    pL, gL = normalize_counts(pred_X), normalize_counts(np.asarray(gt_X))
    out["gene_mean_spearman"] = _corr(pL.mean(0), gL.mean(0))
    out["gene_var_spearman"] = _corr(pL.var(0), gL.var(0))

    out["cell_count_ratio"] = pred_X.shape[0] / max(gt_X.shape[0], 1)  # diagnostic, not scored
    return out


# ==============================================================================
# PART 15 — SYNTHETIC DATA + THE END-TO-END DEMO (main)
# ==============================================================================
# A STARmap-like synthetic cortex: cells arranged in layers along y (the "depth"
# axis), with layer-marker genes whose expression varies with depth, everything
# drifting slowly with z so interpolation is meaningful.  This gives real spatial
# structure (so Moran's I, marker depth profiles etc. are non-trivial) without a
# data download.  Genes named so a few act as cortical-layer markers.


def make_synthetic_cortex(n_per_plane=260, n_planes=77, n_genes=28, seed=0) -> Volume:
    rng = np.random.default_rng(seed)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    # designate a few genes as layer markers (superficial vs deep) + one vascular
    markers = {"Cux2": 3, "Pcp4": 7, "Flt1": 11}
    for name, col in markers.items():
        gene_names[col] = name
    sup_marker, deep_marker, vasc = 3, 7, 11        # Cux2 superficial, Pcp4 deep, Flt1 vascular

    Xs, XYZ, CT = [], [], []
    for pz in range(n_planes):
        z = float(pz)
        zf = pz / max(n_planes - 1, 1)              # 0..1 progress through the block
        x = rng.uniform(0, 100, n_per_plane)
        y = rng.uniform(0, 100, n_per_plane)        # y = cortical depth (0 pial .. 100 deep)
        depth = y / 100.0
        # cell type by depth band (6 layers), band edges drift a little with z
        edges = np.linspace(0, 1, 7) + 0.03 * np.sin(2 * np.pi * zf)
        ct = np.clip(np.digitize(depth, edges[1:-1]), 0, 5)
        # expression: baseline noise + depth-dependent markers + a spatial blob
        X = rng.gamma(1.0, 0.4, (n_per_plane, n_genes)).astype(np.float64)
        X[:, sup_marker] += 3.0 * np.exp(-((depth - 0.15) ** 2) / 0.02)   # superficial
        X[:, deep_marker] += 3.0 * np.exp(-((depth - 0.85) ** 2) / 0.02)  # deep
        blob = np.exp(-(((x - 50) ** 2 + (y - 50) ** 2) / (2 * 22.0 ** 2)))
        X[:, vasc] += 2.5 * blob                                          # vascular blob
        # per-type molecular signature so cell types are molecularly distinct
        for c in range(6):
            X[ct == c, (c * 2) % n_genes] += 1.5
        X = X * (1.0 + 0.2 * np.sin(2 * np.pi * zf))                      # gentle z-drift
        # emit RAW-COUNT-like values (Poisson draws) — this mirrors real STARmap,
        # whose uns['expression_type'] == 'raw_counts'.  The method's wrapper is
        # what log-normalizes; the evaluator is scale-fair either way.
        X = rng.poisson(np.clip(X, 0, None) * 4.0).astype(np.float32)
        Xs.append(X)
        XYZ.append(np.column_stack([x, y, np.full(n_per_plane, z)]))
        CT.append(np.array([f"L{c+1}" for c in ct]))
    return Volume(np.vstack(Xs), np.vstack(XYZ).astype(np.float64),
                  np.concatenate(CT), gene_names)


def load_real_volume(path) -> Volume:
    """Load the real STARmap h5ad if anndata is available (optional)."""
    import anndata as ad
    a = ad.read_h5ad(path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    if "spatial" in a.obsm:
        xyz = np.asarray(a.obsm["spatial"], float)[:, :3]
    else:
        xyz = np.column_stack([a.obs["x"], a.obs["y"], a.obs["z"]]).astype(float)
    ct = (a.obs["cell_type"].astype(str).values if "cell_type" in a.obs
          else a.obs.get("leiden", np.zeros(a.n_obs)).astype(str).values)
    # keep raw counts (as in the file); the method-input normalization below
    # mirrors what the real benchmark wrapper does.
    return Volume(np.asarray(X, float), xyz, ct, list(a.var_names))


def main():
    ap = argparse.ArgumentParser(description="Learn SpatialCPA-v14 end to end.")
    ap.add_argument("--fast", action="store_true",
                    help="tiny model + few epochs for a quick smoke test")
    ap.add_argument("--h5ad", default=None,
                    help="path to the real STARmap 3D h5ad (needs anndata); "
                         "otherwise a synthetic cortex is used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="2,4,6",
                    help="comma-separated held-out section numbers (of 7). "
                         "'2,4,6' = the alternating paper protocol (narrow "
                         "gaps); '3,4,5' = a consecutive holdout (wide gap), "
                         "the regime the v19-v21 machinery targets")
    args = ap.parse_args()

    print("=" * 78)
    print("SpatialCPA-v14 (H3D-FLA) — end-to-end learning run over the v3 protocol")
    print("=" * 78)
    print(f"torch available: {_HAS_TORCH}"
          + ("" if _HAS_TORCH else "  -> running the numpy fallback (install torch "
                                   "to exercise the real flow-matching pipeline)"))

    # ---- data: real volume if given, else synthetic ----
    if args.h5ad:
        print(f"\n[data] loading real volume from {args.h5ad}")
        vol = load_real_volume(args.h5ad)
    else:
        print("\n[data] building a synthetic STARmap-like cortex block")
        vol = make_synthetic_cortex(
            n_per_plane=140 if args.fast else 260,
            n_planes=35 if args.fast else 77, seed=args.seed)
    print(f"       {vol.X.shape[0]} cells x {vol.X.shape[1]} genes; "
          f"z in [{vol.xyz[:,2].min():.0f}, {vol.xyz[:,2].max():.0f}]")

    # ---- v3 protocol: 7 sections, flatten z, hold out --holdout ----
    section_of, xyz = partition_into_sections(vol, n_sections=7, flatten_z=True)
    all_labels = [f"section_{i}" for i in range(1, 8)]
    held_out = [f"section_{int(s)}" for s in args.holdout.split(",") if s.strip()]
    bad = [s for s in held_out if s not in all_labels]
    if bad or len(held_out) >= len(all_labels) - 1:
        raise SystemExit(f"--holdout invalid: {args.holdout}")
    train_labels = [s for s in all_labels if s not in held_out]

    # label vocabulary is TRAINING-ONLY (leakage rule): map types to ints from
    # the training cells, then apply that map everywhere.
    train_mask = np.isin(section_of, train_labels)
    type_vocab = sorted(np.unique(vol.cell_type[train_mask]))
    type_to_idx = {t: i for i, t in enumerate(type_vocab)}
    type_idx_all = np.array([type_to_idx.get(t, 0) for t in vol.cell_type], int)
    n_types = len(type_vocab)

    print(f"\n[protocol] sections: {all_labels}")
    print(f"[protocol] held out (reconstruct these): {held_out}")
    print(f"[protocol] input to the method          : {train_labels}")
    print(f"[protocol] cell-type vocabulary (train-only): {type_vocab}")

    # target z per held-out section = its median depth (a scalar — the ONLY thing
    # about a held-out section the method is allowed to know).
    targets = {sec: float(np.median(xyz[section_of == sec, 2])) for sec in held_out}

    # ---- wrapper-style normalization of the METHOD INPUT ----
    # The real benchmark wrapper log-normalizes raw counts before building the
    # SliceStack (the method wants log-normalized expression; the evaluator stays
    # scale-fair by rank-normalizing).  We do the same here.  GT for evaluation is
    # kept as the RAW counts (vol.X), exactly as the held-out file holds them.
    X_method = normalize_counts(vol.X).astype(np.float32)

    # ---- build the training stack and FIT the method ONCE ----
    stack = build_stack_from_sections(
        X_method, xyz, section_of, train_labels, type_idx_all, n_types,
        X_raw=vol.X.astype(np.float32))

    cfg = V14Config(seed=args.seed, verbose=True)
    if args.fast:
        cfg.pretrain_epochs, cfg.epochs = 8, 20
        cfg.joint_dim, cfg.d_model, cfg.flow_hidden = 24, 48, 96
        cfg.n_ensemble, cfg.n_ode_steps = 2, 8
    else:
        # modest defaults so the demo finishes in a couple of minutes on CPU;
        # the true production defaults are pretrain=60 / epochs=160 (see V14Config).
        cfg.pretrain_epochs, cfg.epochs = 25, 60

    print(f"\n[fit] training on {stack.n_slices} sections "
          f"({sum(s.n_spots for s in stack.slices)} cells) ...")
    model = SpatialCPAv14(stack, gene_names=vol.gene_names,
                          cell_type_names=type_vocab, cfg=cfg)
    print(f"[fit] trained flow-matching model: {model.trained}")

    # ---- generate each held-out section and EVALUATE it ----
    markers = ["Flt1", "Pcp4", "Cux2"]
    markers_sup, markers_deep = ["Cux2"], ["Pcp4"]
    print("\n[generate + evaluate] each held-out section vs its real ground truth")
    print("-" * 78)
    header = f"{'section':>10}  {'z':>5}  {'morI_r':>7} {'gearC_r':>7} {'morImae':>7} " \
             f"{'mix_pca':>7} {'mrk_dep':>7} {'mrk_fld':>7} {'gmean_r':>7} " \
             f"{'gvar_r':>7} {'n_ratio':>7}"
    print(header)
    rows = []
    for sec in held_out:
        z = targets[sec]
        vs = model.generate_virtual_slice(z)
        gm = section_of == sec
        scores = evaluate_section(
            vs, vol.X[gm], xyz[gm, :2], vol.cell_type[gm], vol.gene_names,
            markers, markers_deep, markers_sup, k=10)
        rows.append(scores)
        print(f"{sec:>10}  {z:5.1f}  "
              f"{scores['morans_pearson']:7.3f} {scores['gearys_pearson']:7.3f} "
              f"{scores['morans_mae']:7.3f} "
              f"{scores['embedding_mixing_pca']:7.3f} {scores['marker_depth_r']:7.3f} "
              f"{scores['marker_field_r']:7.3f} "
              f"{scores['gene_mean_spearman']:7.3f} {scores['gene_var_spearman']:7.3f} "
              f"{scores['cell_count_ratio']:7.2f}")

    # pooled means (weighted equally over sections here, for simplicity)
    def _mean(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")
    print("-" * 78)
    print(f"{'MEAN':>10}  {'':>5}  "
          f"{_mean('morans_pearson'):7.3f} {_mean('gearys_pearson'):7.3f} "
          f"{_mean('morans_mae'):7.3f} "
          f"{_mean('embedding_mixing_pca'):7.3f} {_mean('marker_depth_r'):7.3f} "
          f"{_mean('marker_field_r'):7.3f} "
          f"{_mean('gene_mean_spearman'):7.3f} {_mean('gene_var_spearman'):7.3f}")

    print("\nHow to read these (all higher = better except that n_ratio ~1.0 is the")
    print("target, not a score):")
    print("  morI_r / gearC_r : does the per-gene RANKING of spatial structure survive?")
    print("  morImae          : |pred - GT| Moran's I per gene (LOWER is better; catches")
    print("                     over-smoothing that the correlation metrics forgive).")
    print("  mix_pca          : do reconstructed cells overlay real ones in embedding space?")
    print("  mrk_dep          : is each marker's cortical DEPTH profile reproduced?")
    print("  mrk_fld          : binned per-marker FIELD correlation - the metric family")
    print("                     the v21 field-aligned re-grounding targets.")
    print("  gmean_r          : per-gene mean expression agreement.")
    print("\nNext steps to cement the learning:")
    print("  * re-run with --fast off and compare; watch the [B] cfm+bio loss fall.")
    print("  * set cfg.coherent_source=False and see Moran/mixing typically drop —")
    print("    that isolates the coherent-patch idea.")
    print("  * set cfg.edit_weight=0.0 (pure real exemplars) vs 0.5 to feel the")
    print("    field-vs-distribution trade-off the whole method is balancing.")
    print("  * read PART 8/9 again with the loss printout in front of you.")


if __name__ == "__main__":
    main()
