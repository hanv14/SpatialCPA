# Method Wrappers (benchmark-pbya-v2, generation-only)

Each wrapper synthesizes a virtual slice at a target z from a **training-only,
already-re-registered** input. No wrapper receives the held-out `(x, y)`. See
`../../../README.md` for the full leakage policy.

## Shared interface (`_v2_io.py`)

```bash
conda run -n <env> python src/benchmark/methods/run_<method>.py \
    --input   <train_registered.h5ad>   # training-only, no held-out cells \
    --target-section  S1 [S2 ...]        # held-out label(s), for output/eval join \
    --target-z        Z1 [Z2 ...]        # target z per section (parallel list) \
    --output  prediction.h5 \
    --seed    42
```

`_v2_io` provides `add_v2_args`, `load_targets`, `guard_no_holdout` (defense in
depth — hard-fails if a target section is present in the input), and
`write_prediction_h5` (standard output format; `uns/holdout_sections` =
target sections so the evaluator joins correctly).

## Methods

| Wrapper | Method | Env | Status | Notes |
|---------|--------|-----|--------|-------|
| `run_spatialcpav10.py` | SpatialCPA-v10 (gen) | bench_spatialcpa | available | **Biologically-constrained**: v8 diffeomorphic placement + *generated* expression = z-continuous cell-type program + ligand-receptor communication + real residual (balanced hybrid). Cell-type annotation is the organizing step; niche MRF for co-localization. Beats the SpatialZ archetype on the important metrics without copying profiles. See `spatialcpav10/README.md`. |
| `run_spatialcpav9.py` | SpatialCPA-v9 (gen) | bench_spatialcpa | available | **Learned** neural cross-slice flow-matching bridge (rectified-flow / OT-CFM) conditioned on z + neighbouring slices; PyTorch, trains per holdout, falls back to the v8 OT morph. Runs end-to-end; on small per-holdout data it does not beat the training-free morph — see `spatialcpav9/README.md` (honest eval). |
| `run_spatialcpav8.py` | SpatialCPA-v8 (gen) | bench_spatialcpa | available | Training-free coherent-OT bridge: adaptive smoothed-OT morph (near-identical slices) vs real-cell interpolation (distinct slices) + niche annotation. Smoothing the OT displacement field preserves local structure while morphing the global field, resolving the structure-vs-field trade-off. Verified end-to-end + ablated on both regimes. See `spatialcpav8/README.md`. |
| `run_spatialcpav6.py` | SpatialCPA-v6 (gen) | bench_spatialcpa | available | Training-free OT: adaptive barycentric morph vs interpolation; FM-prior + niche-MRF annotation. Verified end-to-end. |
| `run_spatialcpav4.py` | SpatialCPA-v4 (gen) | bench_spatialcpa | available | Transformer + occupancy head; grid over flanking training bbox; emergent count. Verified end-to-end. |
| `run_spatialz.py` | SpatialZ | bench_spatialz | available | Reuses `run_method` synthesis unchanged; only I/O swapped. Needs runtime pass. |
| `run_feast.py` | FEAST | bench_feast | available | Reuses PASTE2 + interpolation unchanged; only I/O swapped. Needs runtime pass. |
| `run_isost.py` | isoST | bench_isost | available | Reuses SDE generation unchanged; only I/O swapped. Needs runtime pass. |
| `run_stvgp.py` | stVGP | bench_stvgp | **disabled** | Coordinate-query; needs held-out (x,y). Not generation-native. |
| `run_spateo_gp.py` | SVGP (Spateo) | bench_spateo | **disabled** | Coordinate-query; same reason. |

Disabled methods are kept for reference but `available=False` in `config.py`, so
`run_benchmark` skips them before invocation.

## Leakage safeguards inside every wrapper

1. Input is training-only (built by `run_benchmark`); `guard_no_holdout` re-checks.
2. Any label vocabulary / clustering is built on the (all-training) input only.
3. Expression normalization is per-cell (`normalize_total` + `log1p`) — no pooled
   statistic.
4. Only a scalar target z is used to place the synthesized slice.
