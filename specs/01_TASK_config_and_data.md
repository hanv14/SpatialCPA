# T01 — Config and data contracts

**Goal.** Establish the single source of truth for configuration and the validated data schema
everything else depends on. Get this right and the rest of the project stops arguing with itself.

**Files:** `spatialcpav25_gen/config.py`, `spatialcpav25_gen/data/schema.py`, `spatialcpav25_gen/data/loaders.py`,
`tests/test_config.py`, `tests/test_schema.py`, `tests/fixtures/synthetic.py`

**Dependencies:** none.

---

## 1. `spatialcpav25_gen/config.py`

One frozen dataclass, grouped by module, with docstrings on every field. Fields required so far
(add more as later tasks need them, never inline constants):

```python
@dataclass(frozen=True)
class Config:
    # --- general
    seed: int = 0
    device: str = "auto"
    debug_shapes: bool = False

    # --- data
    coord_key: str = "spatial"          # obsm key for (x, y) or (x, y, z)
    z_key: str = "z"                    # obs key for section depth if not in obsm
    celltype_key: str = "cell_type"
    region_key: str | None = "region"   # optional
    counts_layer: str | None = None     # layer holding raw counts; None => .X
    min_cells_per_section: int = 50

    # --- text embeddings (T02)
    text_model: str = "ncbi/MedCPT-Query-Encoder"
    text_dim_in: int = 768
    gene_emb_dim: int = 128
    ctx_emb_dim: int = 64
    text_cache_dir: str = ".cache/spatialcpav25_gen_text"
    residual_gate_warmup_frac: float = 0.3   # gamma anneal 0->1 over this frac of training

    # --- noise field (T03)
    matern_nu: float = 1.5
    n_rff: int = 4096                   # random Fourier features
    ell_xy: float = 100.0               # micrometres; calibrated at inference
    ell_z: float = 100.0
    latent_dim: int = 64                # d_h

    # --- anatomical field (T04)
    triplane_res_xy: int = 256
    triplane_res_z: int = 32
    triplane_channels: int = 32
    n_plane_orientations: int = 4       # multi-orientation ensemble
    fourier_bands_xy: int = 8
    fourier_bands_z: int = 2
    rotation_aug: bool = True
    field_mlp_hidden: int = 256
    tv_z_weight: float = 1e-3

    # --- retrieval (T04)
    retrieval_k: int = 32
    retrieval_z_window: float = 3.0     # in units of median section spacing
    section_dropout_p: float = 0.3

    # --- retrieval, continued (T04)
    retrieval_w_z: float = 1.0          # z-proximity weight; 0 = ablation A5
    retrieval_w_niche: float = 1.0

    # --- layout (T05)
    layout_mode: str = "field"          # field | hybrid | resample
    potts_beta: float = 0.5
    potts_iters: int = 2
    layout_n_mc: int = 4096             # MC points for the intensity integral

    # --- selection gates (T09 select_config; also the ablation switches for T10)
    prior_mode: str = "correlated"      # correlated | iid          (A1)
    expr_mode: str = "zinb-flow"        # zinb-flow | cross-mix | auto-blend
    text_emb_mode: str = "medcpt"       # medcpt | lookup           (A3)
    decoder: str = "zinb"               # zinb | zigamma | gaussian (A6)
    repulsion: bool = True              # False = Poisson layout    (A4)

    # --- SEFL (T07)
    sefl_every_n_steps: int = 3
    sefl_patch_cells: int = 2000
    sefl_n_line_points: int = 256
    sefl_min_angle_deg: float = 20.0
    sefl_max_angle_deg: float = 160.0
    thickness_ratio: int = 3            # thick section = this many thin ones

    # --- expression (T06)
    cfm_sigma_min: float = 1e-4
    ode_steps: int = 24
    genes_per_step: int = 128           # G' gene subsample during training
    zinb_eps: float = 1e-6

    # --- losses
    w_recon: float = 1.0
    w_layout: float = 1.0
    w_autocorr: float = 0.5
    w_profile: float = 0.5
    w_distribution: float = 0.5
    w_cross: float = 0.3
    w_thick: float = 0.2
    w_prog: float = 0.2
    w_distill: float = 0.1
    sefl_warmup_frac: float = 0.2
    ema_decay: float = 0.999

    # --- training
    epochs: int = 200
    batch_cells: int = 2048
    lr: float = 3e-4
```

Provide `Config.from_yaml(path)` and `cfg.replace(**kw)`. Add `validate()` raising on
contradictory settings (e.g. `layout_mode` not in the allowed set, `fourier_bands_z > 4` with fewer
than 8 training sections).

## 2. `spatialcpav25_gen/data/schema.py`

```python
@dataclass
class Section:
    """One physical tissue section."""
    z: float                   # depth in micrometres
    thickness: float           # slab thickness in micrometres  (see note below)
    coords: np.ndarray         # (N, 2) float32, in-plane (x, y) micrometres
    cell_type: np.ndarray      # (N,) int32 codes
    region: np.ndarray | None  # (N,) int32 codes or None
    counts: sparse.csr_matrix  # (N, G) raw counts / intensities
    section_id: str
    thickness_is_assumed: bool = False   # True if defaulted, not measured

@dataclass
class Volume:
    """A stack of sections from one specimen, plus shared metadata."""
    sections: list[Section]        # sorted by z ascending
    gene_names: list[str]          # length G
    celltype_names: list[str]
    region_names: list[str] | None
    specimen_id: str

    # derived, computed once in __post_init__
    median_spacing: float          # median |z_{i+1} - z_i|
    median_nn_dist: float          # median in-plane nearest-neighbour distance, pooled
    bbox: np.ndarray               # (2, 3) min/max in x, y, z
```

**On `thickness`.** It is load-bearing: T05 integrates cell intensity over the *slab volume*, and
T07's `L_thick` is defined entirely in terms of it. Read it from `adata.uns["section_thickness"]`
or an obs column when present. When absent, default to `median_spacing`, set
`thickness_is_assumed=True`, and **log a warning once per volume**. Any report or figure produced
from a volume with assumed thickness must carry that caveat — do not let an assumed value silently
become a quantitative claim about cell density per unit volume.

`validate_volume(vol, cfg) -> None` must check and raise informatively on:

- sections sorted, strictly increasing z, ≥ 3 sections
- consistent gene set and gene order across sections
- coords finite, 2-D, float32; no duplicate coordinates within a section
- counts non-negative; **integer-valued if `counts_layer` is set** (warn loudly if not — a
  non-integer "counts" matrix means someone passed log-normalised data, which silently breaks the
  ZINB decoder)
- cell type codes within range of `celltype_names`
- `min_cells_per_section` satisfied
- `thickness > 0` for every section; warn if any `thickness > median_spacing` (overlapping slabs,
  which makes the thickness-consistency loss in T07 ill-posed)

Also `to_xyz(section) -> (N,3)` concatenating the section's `z`.

## 3. `spatialcpav25_gen/data/loaders.py`

```python
def load_volume(h5ad_path: str, cfg: Config, specimen_key: str | None = None) -> Volume | list[Volume]
def split_holdout(vol: Volume, mode: str, fold: int) -> tuple[Volume, list[Section]]
```

`mode ∈ {"alternating", "consecutive"}`:

- `alternating` — hold out every other section (fold selects the parity). Flanking gap ≈ 1×
  `median_spacing` on each side.
- `consecutive` — hold out a contiguous run of `k` sections (`k ∈ {1, 3, 5}` configurable), so the
  gap to the nearest real section is up to `(k+1)/2 ×` spacing.

Returns the training `Volume` (with held-out sections removed and `median_spacing` recomputed on
the *training* sections — important, the model must never see the true spacing of the holdout) and
the list of held-out `Section`s for evaluation.

Also `loso_folds(vol) -> Iterator[tuple[Volume, Section]]` — leave-one-section-out over the
**training** volume only. Used by T08 and T09; it must be impossible for it to touch held-out data,
so it takes a `Volume` that has already had holdouts removed.

## 4. `tests/fixtures/synthetic.py`

A synthetic volume generator, used by every subsequent task. Must produce a tissue with **known
ground-truth structure** so that later tests can assert recovery:

```python
def make_synthetic_volume(
    n_sections: int = 9,
    spacing: float = 50.0,
    n_cells_per_section: int = 1500,
    n_genes: int = 200,
    n_types: int = 6,
    autocorr_length: float = 120.0,   # ground-truth spatial correlation length
    z_gradient_genes: int = 20,       # genes with a monotone z-dependence
    seed: int = 0,
) -> Volume
```

Construction:
- Cells placed by a hard-core point process in a rectangular slab (so nearest-neighbour spacing is
  realistic, not Poisson-clumped).
- A smooth 3D latent field (sum of a few low-frequency 3D sinusoids + a Matérn GRF with
  `autocorr_length`) drives both cell type (argmax of type-specific intensities → gives genuine
  spatial cell-type localisation) and expression.
- Expression sampled from ZINB with mean driven by (cell type, latent field), dispersion and
  dropout drawn per gene → realistic sparsity.
- `z_gradient_genes` have means varying monotonically with z, so marker depth profiles have a
  ground truth to recover.
- **Anisotropy:** make the latent field's correlation length differ in x/y vs z (e.g. 120 µm
  in-plane, 200 µm along z). Needed by T04/T07 tests for oblique behaviour.

Return the ground-truth field callable alongside the `Volume` (as `vol.uns["gt_field"]` or a
second return value) so tests can compare against truth.

## Acceptance tests

- `test_config_roundtrip` — yaml → Config → yaml is idempotent; `validate()` raises on bad values.
- `test_schema_rejects_lognormalised` — a Volume built from log1p data with `counts_layer` set
  raises/warns.
- `test_schema_rejects_unsorted_or_duplicate` — both cases raise with a message naming the field.
- `test_synthetic_has_structure` — on the fixture: mean Moran's I over genes > 0.25; per-gene
  detection rate distribution spans (0.05, 0.95); at least 15 genes have |corr(mean expr, z)| > 0.5;
  cell types are spatially clustered (neighbourhood type-purity > chance by ≥ 3σ).
- `test_split_holdout_no_leakage` — held-out section ids never appear in the training Volume;
  `median_spacing` of the training volume is recomputed, not inherited.
- `test_loso_folds_excludes_holdout` — asserted structurally.
- `test_thickness_defaults_and_flags` — a volume with no thickness metadata gets
  `thickness == median_spacing`, `thickness_is_assumed is True`, and emits exactly one warning.
- `test_config_gates_validate` — every gate field (`prior_mode`, `expr_mode`, `text_emb_mode`,
  `decoder`, `layout_mode`) rejects an out-of-set value; the tuple
  `(layout_mode="resample", expr_mode="cross-mix", prior_mode="iid")` is accepted, since T09's
  selector must be able to reach the previous version's behaviour.

## Definition of done

`pytest tests/test_config.py tests/test_schema.py` green; `mypy --strict spatialcpav25_gen/config.py
spatialcpav25_gen/data/` clean; `PROGRESS.md` entry written.

## Do NOT

- Do not add model code here.
- Do not read real datasets yet; the synthetic fixture is what everything is developed against.
- Do not put any constant outside `Config`.
