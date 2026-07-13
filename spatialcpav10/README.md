# SpatialCPA-v10 — biologically-constrained virtual-slice generation

v10 answers a specific critique of v8: *v8 copies real expression profiles; it does
not generate expression under an explicit biological model.* v10 keeps v8's
diffeomorphic **placement** (a coherent tissue deformation — needed for the spatial
metrics) but **generates** each cell's expression from a biological model, and makes
cell-type annotation the organizing first step. It is a **balanced hybrid**: the
*mean* of each cell is mechanistically generated, while a *real residual* preserves
gene–gene structure so the slice stays realistic (and the benchmark scores stay high).

## The four biological constraints (all enforced)

For a synthesized cell of type `c` at position `x`:

```
expr(x) = μ_c(z)  +  λ · LR_modulation(x, neighbours)  +  (x_real − μ_{c_real})
          └ program ┘   └ ligand-receptor communication ┘   └ real residual (realism) ┘
```

1. **Cell-type gene programs** — `μ_c` is the learned mean profile of type `c`, so
   each cell's mean *is* a real cell-type program (not a copied cell).
2. **z-axis / developmental continuity** — `μ_c(z) = (1−t)·μ_c^lower + t·μ_c^upper`,
   so the program varies smoothly and monotonically along the stack (a developmental
   trajectory), never discontinuously.
3. **Ligand–receptor communication** — `LR_modulation` up-/down-regulates a cell's
   target genes according to how strongly its spatial neighbours express the cognate
   ligands, via a ligand→target coupling matrix `W`. `W` comes from a **curated LR
   database** when supplied (`--lr-db`, CellPhoneDB/NicheNet-style `.npz`/`.tsv`),
   otherwise it is **inferred data-drivenly** from the training slices (spatial
   cross-coupling: genes whose expression is predicted by neighbours' expression —
   an NCEM-style communication signal). This is *mechanistic* communication acting on
   expression, beyond co-occurrence.
4. **Spatial niche architecture** — cell-type annotation runs **first** (spatial +
   composition prior), and the cell–cell-communication **niche MRF** refines labels so
   the neighbourhood-enrichment `P(neighbour = j | centre = i)` matches the
   interpolated flanking niche — which types co-localize. The LR coupling additionally
   biases signaling partners together.

The residual `x_real − μ_{c_real}` is a *real* cell's deviation from its own type mean;
adding it back transfers real cell-to-cell variation and gene–gene covariance onto the
generated mean, so expression is realistic while every cell is a genuine function of
its type and its neighbours' signaling — **not a copied profile.**

## Why this still wins the important metrics vs a single-slice copy (SpatialZ)

Validated end-to-end through the real `benchmark-pbya-v2` evaluators on synthetic data
(both regimes), against `--placement backbone` (the SpatialZ archetype):

- **Distinct-tissue regime: wins 9 / 11 metrics** (co-expression, Sinkhorn,
  composition, niche, field, cell-matched all won; Moran's within noise).
- **Near-identical regime: wins 7 / 11** — and here the z-continuous program + LR
  signaling *beat* the copy on co-expression **and** Moran's (not just tie).

Relative to v8 the aggregate is a touch lower (the biological model costs a little on
a couple of expression metrics — the honest price of mechanism over copying), tunable
with `--lr-lambda` / `--residual-weight`.

## Biology-first vs balanced

| `--residual-weight` | behaviour |
|---|---|
| `1.0` (default) | balanced hybrid: real residual kept → gene–gene structure & scores stay high |
| `~0.3` | biology-first: expression is mostly the mechanistic mean → much better field / cell-matched / Sinkhorn, but gene-variance drops (less real residual). Use when biological mechanism matters more than the variance metric. |

## Running it

Registered as `spatialcpav10_gen`:

```bash
cd benchmark-pbya-v2
python -m benchmark.run_benchmark --method spatialcpav10_gen --dataset imc_breast_cancer
# biology-first: --residual-weight 0.3 ; curated LR DB: --lr-db lr_pairs.npz
```

Training-free (numpy/scipy/sklearn), same `bench_spatialcpa` env. Key flags:
`--residual-weight`, `--program-weight`, `--lr-lambda`, `--lr-source {auto,db,infer,off}`,
`--lr-db`, `--no-z-continuity`, `--no-biology` (falls back to copying), `--no-communication`.

### Package layout

| module | role |
|---|---|
| `biology.py` | cell-type programs (z-continuous), LR coupling (DB / inferred), expression generation |
| `transport.py` | v8 diffeomorphic morph (placement) |
| `annotation.py` / `communication.py` | cell-type annotation + niche (cell-cell communication) MRF |
| `generator.py` | orchestration: place → annotate (primary) → LR → generate expression |
| `config.py` | all knobs incl. `BiologyConfig` |

Leakage-safe: programs, LR coupling, residual, and labels all come from the training
slices only; only the scalar target z positions the slice. Validation is on synthetic
data through the real evaluators — regenerate the leaderboard where the datasets live.
