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
| `run_spatialcpa.py` | SpatialCPA | bench_spatialcpa | (this repo) | Coordinate neural field; no flanking slices needed; predicts at held-out cell coords |
| `run_spatialcpav4.py` | SpatialCPA-v4 | bench_spatialcpa | (this repo) | Transformer; learns {Slice(i-1),Slice(i+1)}->Slice(i); k-NN neighbor tokens + CLS; expression/label/occupancy heads |

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
**SpatialCPA**: n_freq_xy=48, n_freq_z=32, backbone 512x8 -> 256, MSE mode, epochs=50; inference k-NN refine k=5, z_weight=3, alpha=0 (pure cell-type k-NN). Cell types predicted by the model (not leaked) unless `--use-true-celltypes`.
**SpatialCPA-v4**: transformer hidden=256, layers=4, heads=8, dropout=0.1, neighbors=10/side (20 tokens + CLS), epochs=100, lr=1e-3; MSE+Pearson expression loss, cross-entropy label loss, BCE occupancy loss. Predicts each held-out section from its nearest lower/upper reference slices. Two inference regimes: default *coordinate-matched* (predict at the held-out cells' real x,y — count == held-out) and `--generate-mode` *de-novo synthesis* (grid over the flanking slices' XY bbox at the target z, keep occupancy>threshold — emergent count, like SpatialZ/FEAST/isoST; `--grid-points`, `--grid-type`, `--occupancy-threshold`). All hyperparameters are CLI flags (see `run_spatialcpav4.py --help`).
