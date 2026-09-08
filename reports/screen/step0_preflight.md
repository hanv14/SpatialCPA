# Step 0 — preflight

Generated 2026-09-08T19:15:03+00:00 on vm

## Trees

| what | path | present |
|---|---|---|
| v25 repo | `/home/user/SpatialCPA` | yes |
| bench3 | `/home/user/SpatialCPA/benchmark-pbya-v3` | yes |
| bench3 package | `/home/user/SpatialCPA/benchmark-pbya-v3/src/bench3` | yes |
| v2 wrappers (`_v2_io.py`) | `/home/user/SpatialCPA/benchmark-pbya-v2/src/benchmark/methods/_v2_io.py` | yes |
| v1 processed tree | `/home/user/SpatialCPA/benchmark-pbya/data/processed` | **NO** |
| built datasets | `/home/user/SpatialCPA/benchmark-pbya-v3/data/processed` | **NO** |
| results / _inputs | `/home/user/SpatialCPA/benchmark-pbya-v3/results` | **NO** |

## Environment

```
python   Python 3.11.15
torch    ModuleNotFoundError: No module named 'torch'
v25 pkg  ModuleNotFoundError: No module named 'spatialcpav25_gen'
threads  OMP_NUM_THREADS=<unset>
```

## Per-dataset build state — what step 3 needs

| dataset | holdout | built `data.h5ad` | `train_registered.h5ad` | raw source |
|---|---|---|---|---|
| `cosmx_nsclc_3d` | paper_2_4 | **NO** | **NO** | **NO** |
| `merfish_thick_hypothalamus` | paper_2_4_6 | **NO** | **NO** | **NO** |
| `merfish_thick_cortex` | paper_2_4_6 | **NO** | **NO** | **NO** |
| `exseq_breast_cancer` | paper_2_4_6 | **NO** | **NO** | **NO** |
| `exseq_visual_cortex` | paper_2_4_6 | **NO** | **NO** | **NO** |
| `starmap_visual_cortex` | paper_2_4_6 | **NO** | **NO** | **NO** |
| `deep_starmap` | paper_2_4_6 | **NO** | **NO** | **NO** |

Missing a built `data.h5ad`:
`cd /home/user/SpatialCPA/benchmark-pbya-v3 && python -m src.bench3.prepare_dataset --dataset <name>`

Built, but missing `train_registered.h5ad`: it is created by the first method run or by
`cd /home/user/SpatialCPA/benchmark-pbya-v3 && python -m src.bench3.selftest --dataset <name>`. Both are cheap next to a fit,
but neither is free and neither was costed in §4 — report what is missing rather than building it.
