# Method Wrappers

Each file is a standalone script that runs one z-interpolation method on one holdout configuration. Invoked by `run_benchmark.py` via `conda run -n <env>`.

## Interface

All wrappers share the same CLI:

```bash
conda run -n <env> python src/benchmark/methods/run_<method>.py \
    --input data/processed/<dataset>/data.h5ad \
    --holdout-sections <section_label> \
    --output results/<method>/<dataset>/loo_<section>/prediction.h5 \
    --seed 42
```

## Methods

| Wrapper | Method | Conda env | Paper | Notes |
|---------|--------|-----------|-------|-------|
| `run_feast.py` | FEAST | bench_feast | Chen et al. 2025 | PASTE1 alignment, sigma=0 |
| `run_spatialz.py` | SpatialZ | bench_spatialz | Lin et al. 2025, Nat Methods | Needs cell_type (Leiden fallback) |
| `run_isost.py` | isoST | bench_isost | Li et al. 2025 | SDE-based, batch_num=5, slow (~100 min/holdout) |
| `run_stvgp.py` | stVGP | bench_stvgp | Wang et al. 2026, Adv Sci | DLPFC defaults, skips >200K cells, GPU required |
| `run_spateo_gp.py` | SVGP (Spateo) | bench_spateo | Qiu et al. 2024, Cell | SVGP per-gene, n_genes_max=2000, GPU |

## Output format

Each wrapper writes `prediction.h5` with:
- `X/` — CSR sparse expression matrix (data, indices, indptr, shape)
- `obs/` — cell_id, x, y, z, section, cell_type
- `var/` — gene_name
- `uns/` — method_name, holdout_sections, method_params, wall_time_seconds

## Key parameters (after paper audit)

**stVGP** (DLPFC tutorial): hidden=[512,24], epochs=1500, n_hvg=5000, all_gat=True, Rbf=512, n_neighbors=10
**isoST**: K=8, delta_d=0.01, hidden_dim=64, gene_dim=50, epochs=[100,100,100], batch_num=5
**FEAST**: sigma=0, PASTE1 alignment
**Spateo GP**: SVGP, training_iter=50, batch_size=1024, inducing_num=512
**SpatialZ**: Default paper parameters
