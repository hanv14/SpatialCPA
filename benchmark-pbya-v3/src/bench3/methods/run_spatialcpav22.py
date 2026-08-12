"""SpatialCPA-v22 (v21 recalibrated on the real 18-dataset benchmark) — v3 wrapper.

Like the v16/v18/v19/v20/v21 wrappers, this lives in ``benchmark-pbya-v3`` rather
than beside the others in ``benchmark-pbya-v2/src/benchmark/methods/``, because v2
is frozen and never ran v22. It speaks the identical ``_v2_io`` contract as every
other wrapper — same CLI, same ``prediction.h5`` format, same leakage guard — so
``run_benchmark`` invokes it exactly as it invokes the rest and the evaluator
reads its output unchanged.

v22 is the single-file re-implementation ``learn_spatialcpav22.py``. It keeps
every v21 mechanism (coherent per-gene mix masks, multi-partner draws, gap-
adaptive re-grounding margin and layout granularity, alpha-gated field-align,
tail-rate-matched per-gene repair) and only recalibrates them for real tissue
after the v21 benchmark run: gene-mix at wide gaps is off by default (heavy
per-gene redraw made off-manifold chimeras on covariance-rich real profiles), the
cross-mix probability is capped, mix/layout granule sizes are specified in
median-NN-spacing units (so cell-density differences across datasets no longer
mis-scale them), and a mild narrow-gap tail repair spends some of the measured
Moran's-MAE headroom on the marker field metric. It is a *module file at the
repository root*, not an installed package, so — like v18-v21 — this wrapper loads
the file directly. The public API is v14's (``Slice`` / ``SliceStack`` /
``SpatialCPAv14`` / ``V14Config`` / ``VirtualSlice``); ``Slice`` carries an
optional ``raw_expression`` and ``V14Config`` gains the v18/v19/v20/v21/v22 knobs.

Generation-only: the input file physically excludes the held-out sections and the
method receives one scalar target z per section. Nothing else.
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

# v2's shared I/O helpers. Imported, never modified — the contract is v2's and
# reusing it is what keeps v3's rows comparable with the rest of the benchmark.
# (Same wiring as the v16/v18/v19/v20/v21 wrappers: this file sits four parents
# below the repo root, so parents[4] is SpatialCPA/ and benchmark-pbya-v2 is
# beside v3.)
_V2_BENCH = (Path(__file__).resolve().parents[4]
             / "benchmark-pbya-v2" / "src" / "benchmark")
sys.path.insert(0, str(_V2_BENCH / "methods"))   # _v2_io
sys.path.insert(0, str(_V2_BENCH))               # leakage_guard
import _v2_io                      # noqa: E402
import leakage_guard               # noqa: E402


# The v22 method is a single file, not a package: name the FILE, or the directory
# it sits in, or let the parent walk find it (it lives at the repository root,
# beside benchmark-pbya-v3/, which parents[4] reaches).
FILE_ENV = "SPATIALCPAV22_FILE"      # full path to learn_spatialcpav22.py
ROOT_ENV = "SPATIALCPAV22_ROOT"      # directory that CONTAINS the file
_MODULE_NAME = "learn_spatialcpav22"
_FILENAME = "learn_spatialcpav22.py"


def _candidate_files():
    """Paths that might be ``learn_spatialcpav22.py``, most-specific first."""
    f = os.environ.get(FILE_ENV)
    if f:
        yield Path(f).expanduser()
    r = os.environ.get(ROOT_ENV)
    if r:
        yield Path(r).expanduser() / _FILENAME
    for cand in Path(__file__).resolve().parents:
        yield cand / _FILENAME


def _load_module():
    """Import ``learn_spatialcpav22.py`` from disk under a stable module name.

    Returns ``(module, path, tried)``; ``module`` is None if no candidate exists.
    Importing the file never runs its ``main()`` (that is under ``__main__``), so
    this only defines the classes we need.
    """
    tried = []
    for cand in _candidate_files():
        tried.append(str(cand))
        if cand.exists():
            spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(cand))
            module = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE_NAME] = module
            spec.loader.exec_module(module)
            return module, str(cand), tried
    return None, None, tried


_V22, _V22_PATH, _TRIED = _load_module()


def check_environment():
    if _V22 is None:
        env_f = os.environ.get(FILE_ENV)
        env_r = os.environ.get(ROOT_ENV)
        print(f"ERROR: could not locate {_FILENAME}.", file=sys.stderr)
        print("  Looked in:", file=sys.stderr)
        for t in _TRIED:
            print(f"    {t}", file=sys.stderr)
        if env_f:
            print(f"  ${FILE_ENV}={env_f!r}; it must name the file itself.",
                  file=sys.stderr)
        elif env_r:
            print(f"  ${ROOT_ENV}={env_r!r}; {Path(env_r) / _FILENAME} must exist.",
                  file=sys.stderr)
        else:
            print(f"  Set ${FILE_ENV} to the file, or ${ROOT_ENV} to the directory "
                  f"holding it, or keep {_FILENAME} at the repository root beside "
                  f"benchmark-pbya-v3/ where the parent walk finds it.",
                  file=sys.stderr)
        return False
    try:
        import torch
        t = f"torch {torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})"
    except Exception:
        # v22 degrades to the numpy latent-grounded fallback without torch — a
        # sensible library default but not the method under test, so config.py
        # matches this marker and fails the run rather than scoring it.
        t = "torch UNAVAILABLE -> latent-grounded numpy fallback (degraded)"
    print(f"spatialcpav22 (learn_spatialcpav22.py) from {_V22_PATH}; {t}")
    return True


def _to_dense_f32(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def _normalize_expression(adata):
    """Log-normalize the METHOD INPUT, respecting the dataset's expression_type.

    Identical policy to the v14/v16/v18/v19/v20/v21 wrappers, so every method
    receives the same apples-to-apples input. Returns the resolved expression_type.
    """
    import scanpy as sc
    et = str(adata.uns.get("expression_type", "raw_counts"))
    if et in ("log1p_normalized", "log2_normalized", "normalized"):
        return et
    if et in ("fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(adata)
        return et
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return et


def build_stack(adata, X_log, X_raw, ct_all):
    """Assemble the training SliceStack.

    ``X_log`` is the log-normalized expression the model trains on; ``X_raw`` is
    the pre-normalization measurement carried alongside so raw_output / the mixing
    and field-repair passes can emit real profiles verbatim (see
    ``learn_spatialcpav22`` ``main()``, which does the same with ``X_raw=vol.X``).
    """
    Slice, SliceStack = _V22.Slice, _V22.SliceStack
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    labels = sorted(np.unique(sections),
                    key=lambda s: float(np.median(coords[sections == s, 2])))
    slices = []
    for sec in labels:
        m = sections == sec
        slices.append(Slice(
            expression=X_log[m],
            coords_xy=coords[m, :2],
            z_values=coords[m, 2],
            cell_type_indices=(ct_all[m] if ct_all is not None else None),
            section_id=str(sec),
            raw_expression=(X_raw[m] if X_raw is not None else None)))
    return SliceStack(slices)


def _build_config(args):
    """A ``V14Config`` whose defaults reproduce v22; flags only override knobs.

    The argparse defaults below mirror ``V14Config``'s own defaults, so running
    the wrapper with just the shared generation-only arguments runs v22 as
    intended and cannot silently drift from this file.
    """
    cfg = _V22.V14Config()
    cfg.seed = args.seed
    cfg.verbose = True
    cfg.device = args.device
    # training
    cfg.epochs = args.epochs
    cfg.pretrain_epochs = args.pretrain_epochs
    # latents / encoder / flow
    cfg.expr_latent_dim = args.latent_dim
    cfg.joint_dim = args.joint_dim
    cfg.n_ode_steps = args.ode_steps
    cfg.n_ensemble = args.ensemble
    if args.context_slices is not None:
        cfg.context_slices_each_side = args.context_slices
    # generation
    cfg.position_mode = args.position_mode
    cfg.ground_blend_flow = args.ground_blend_flow
    cfg.ground_k = args.ground_k
    cfg.ground_temp = args.ground_temp
    cfg.edit_weight = args.edit_weight
    cfg.coherent_source = not args.no_coherent_source
    cfg.output_counts = not args.no_output_counts
    cfg.composition_match = not args.no_composition_match
    # ── v18 additions ──
    cfg.type_mode = args.type_mode
    cfg.type_vote_k = args.type_vote_k
    cfg.gene_mix_frac = 0.0 if args.no_gene_mix else args.gene_mix_frac
    cfg.raw_output = not args.no_raw_output
    cfg.ground_sample = not args.no_ground_sample
    cfg.dedup_ground = not args.no_dedup_ground
    cfg.ground_keep_margin = args.ground_keep_margin
    # ── v19 gap-adaptive weights (alpha; drives cross-mix + the v21/v22 ramps) ──
    cfg.gap_adapt = not args.no_gap_adapt
    cfg.gap_scale = args.gap_scale
    cfg.pair_interp_max = args.pair_interp_max   # retired since v20; kept for compat
    # ── v20 Bernoulli cross-mix ──
    cfg.cross_mix = not args.no_cross_mix
    cfg.alpha_tol = args.alpha_tol
    cfg.edit_gap_extra = args.edit_gap_extra
    cfg.curriculum_flow = args.curriculum_flow
    cfg.curriculum_p = args.curriculum_p
    # ── v21 coherent mix + field-aligned grounding ──
    cfg.coherent_mix = not args.no_coherent_mix
    cfg.mix_field_freq = args.mix_field_freq
    cfg.mix_cand_k = args.mix_cand_k
    cfg.mix_dist_scale = args.mix_dist_scale
    cfg.gene_mix_wide = args.gene_mix_wide
    cfg.ground_gap_adapt = not args.no_ground_gap_adapt
    cfg.ground_margin_narrow = args.ground_margin_narrow
    cfg.layout_gap_adapt = not args.no_layout_gap_adapt
    cfg.layout_freq_narrow = args.layout_freq_narrow
    cfg.field_align = not args.no_field_align
    cfg.field_align_gap_only = not args.no_field_align_gap_only
    cfg.field_align_k = args.field_align_k
    cfg.field_align_cand_k = args.field_align_cand_k
    cfg.field_align_frac = args.field_align_frac
    cfg.field_align_margin = args.field_align_margin
    cfg.field_align_noise_mult = args.field_align_noise_mult
    cfg.field_repair = not args.no_field_repair
    cfg.repair_q = args.repair_q
    cfg.repair_noise_mult = args.repair_noise_mult
    cfg.repair_frac = args.repair_frac
    cfg.repair_tail_scale = args.repair_tail_scale
    # ── v22 recalibration (cap, spacing-relative granules, narrow tail scale) ──
    cfg.cross_mix_cap = args.cross_mix_cap
    cfg.granule_adaptive = not args.no_granule_adaptive
    cfg.mix_granule_spacings = args.mix_granule_spacings
    cfg.layout_granule_spacings_narrow = args.layout_granule_spacings_narrow
    cfg.layout_granule_spacings_wide = args.layout_granule_spacings_wide
    cfg.repair_tail_scale_narrow = args.repair_tail_scale_narrow
    return cfg


def run_method(adata, targets, gene_names, X_log, X_raw, args):
    SpatialCPAv14 = _V22.SpatialCPAv14

    # The input is training-only, so the whole matrix is training data: the
    # leakage-safe vocabulary is built over all of it (train_mask all-ones),
    # exactly as the v14/v16/v18/v19/v20/v21 wrappers do.
    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}

    cfg = _build_config(args)
    print(f"  epochs(A={cfg.pretrain_epochs},B={cfg.epochs}), "
          f"latent(d_e={cfg.expr_latent_dim},h={cfg.joint_dim}), "
          f"flow(steps={cfg.n_ode_steps},ens={cfg.n_ensemble}), "
          f"type_mode={cfg.type_mode}, gene_mix={cfg.gene_mix_frac}"
          f"(wide={cfg.gene_mix_wide}), raw_output={cfg.raw_output}, "
          f"gap_adapt={cfg.gap_adapt}(scale={cfg.gap_scale}), "
          f"cross_mix={cfg.cross_mix}(cap={cfg.cross_mix_cap}), "
          f"coherent_mix={cfg.coherent_mix}(cand_k={cfg.mix_cand_k}), "
          f"granule_adaptive={cfg.granule_adaptive}"
          f"(mix={cfg.mix_granule_spacings},lay_n={cfg.layout_granule_spacings_narrow},"
          f"lay_w={cfg.layout_granule_spacings_wide}), "
          f"ground_gap_adapt={cfg.ground_gap_adapt}(narrow={cfg.ground_margin_narrow}), "
          f"layout_gap_adapt={cfg.layout_gap_adapt}, "
          f"field_align={cfg.field_align}(gap_only={cfg.field_align_gap_only}), "
          f"field_repair={cfg.field_repair}(tail_w={cfg.repair_tail_scale},"
          f"tail_n={cfg.repair_tail_scale_narrow}), "
          f"edit_w={cfg.edit_weight}, blend={cfg.ground_blend_flow}, k={cfg.ground_k}")

    gen = SpatialCPAv14(stack, gene_names=gene_names,
                        cell_type_names=cell_type_names, cfg=cfg)
    # Matched by config.invalid_log_markers: a False here means v22 fell back to
    # the numpy path and is not the method under test.
    print(f"  flow-matching model trained: {gen.trained}")

    results = {}
    for sec, z in targets:
        print(f"  {sec}: flow-generating virtual slice at z={z:.2f} ...")
        try:
            vs = gen.generate_virtual_slice(z=z)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        n = vs.coords.shape[0]
        if n == 0:
            continue
        print(f"    -> {n} cells synthesized")
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64),
                        "cell_type": cell_type}
    return results


def main():
    p = argparse.ArgumentParser(
        description="SpatialCPA-v22 wrapper (v21 recalibrated on the real benchmark)")
    _v2_io.add_v2_args(p)
    # Training / architecture (defaults = V14Config production defaults).
    p.add_argument("--epochs", type=int, default=160, help="Phase B flow epochs")
    p.add_argument("--pretrain-epochs", type=int, default=60, help="Phase A epochs")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--latent-dim", type=int, default=32,
                   help="expression PCA latent dim (V14Config.expr_latent_dim)")
    p.add_argument("--joint-dim", type=int, default=48)
    p.add_argument("--ode-steps", type=int, default=12,
                   help="Euler steps when sampling the flow ODE")
    p.add_argument("--ensemble", type=int, default=4,
                   help="initial noises marginalized per query")
    p.add_argument("--context-slices", type=int, default=None,
                   help="real slices per side feeding the 3D context "
                        "(default: V14Config.context_slices_each_side)")
    # Generation
    p.add_argument("--position-mode", default="flanking",
                   choices=["flanking", "morph", "nearest"])
    p.add_argument("--ground-blend-flow", type=float, default=0.20,
                   help="fraction of cells re-grounded to the flow-latent pick")
    p.add_argument("--ground-k", type=int, default=8,
                   help="local real candidates the flow latent grounds among")
    p.add_argument("--ground-temp", type=float, default=0.25,
                   help="softmax temperature for exemplar sampling")
    p.add_argument("--edit-weight", type=float, default=0.25,
                   help="blend toward the flow-decoded profile (0 = pure exemplar)")
    p.add_argument("--no-coherent-source", action="store_true",
                   help="interleave both flanking slices instead of coherent patches")
    p.add_argument("--no-output-counts", action="store_true",
                   help="emit log1p-normalized (not count-like) expression")
    p.add_argument("--no-composition-match", action="store_true",
                   help="disable cell-type composition matching")
    # ── v18-specific knobs ──
    p.add_argument("--type-mode", default="vote", choices=["inherit", "vote"],
                   help="'vote': distance-weighted kNN type vote over both flanks "
                        "(default); 'inherit': v14 behaviour")
    p.add_argument("--type-vote-k", type=int, default=12)
    p.add_argument("--gene-mix-frac", type=float, default=0.15,
                   help="minority fraction of each cell's genes resampled from a "
                        "local same-type real cell (novelty; 0 disables)")
    p.add_argument("--no-gene-mix", action="store_true",
                   help="ablate gene-mix (equivalent to --gene-mix-frac 0)")
    p.add_argument("--no-raw-output", action="store_true",
                   help="ablate raw output: emit expm1(log-normalized) instead "
                        "of grounded cells' verbatim raw measurement")
    p.add_argument("--no-ground-sample", action="store_true",
                   help="ablate temperature sampling: use the v14 argmin exemplar")
    p.add_argument("--no-dedup-ground", action="store_true",
                   help="ablate the exemplar-reuse penalty")
    p.add_argument("--ground-keep-margin", type=float, default=1.0,
                   help="flow re-grounding swap margin at alpha=1 (wide gaps); the "
                        "effective margin ramps down to this from --ground-margin-narrow")
    # ── v19 gap-adaptive weights (still active: alpha drives cross-mix + ramps) ──
    p.add_argument("--no-gap-adapt", action="store_true",
                   help="ablate gap adaptation: hold alpha=0 so v22 reduces to v18 "
                        "at every gap width")
    p.add_argument("--gap-scale", type=float, default=3.0,
                   help="gap (in units of the median training inter-slice spacing) "
                        "at which the adaptation saturates (alpha -> 1)")
    p.add_argument("--pair-interp-max", type=float, default=1.0,
                   help="RETIRED since v20 (kept for API compat; no effect while "
                        "--cross-mix is on)")
    # ── v20 Bernoulli cross-mix ──
    p.add_argument("--no-cross-mix", action="store_true",
                   help="ablate the Bernoulli cross-mix (per-gene real-value "
                        "selection between exemplar and cross-flank partner)")
    p.add_argument("--alpha-tol", type=float, default=1e-3,
                   help="alpha below this snaps to 0 so narrow-gap designs are "
                        "exactly v18")
    p.add_argument("--edit-gap-extra", type=float, default=0.0,
                   help="extra flow-decode blend at alpha=1 (added to --edit-weight, "
                        "capped 0.6). Default 0; raise only for ablation")
    p.add_argument("--curriculum-flow", action="store_true",
                   help="enable the wide-gap training curriculum (OFF by default — "
                        "inert at edit_weight=0)")
    p.add_argument("--curriculum-p", type=float, default=0.35,
                   help="probability the curriculum excludes the nearest slice "
                        "per side during Phase B (only if --curriculum-flow)")
    # ── v21 coherent mix + field-aligned grounding ──
    p.add_argument("--no-coherent-mix", action="store_true",
                   help="ablate coherent mixing: draw the per-gene mix masks i.i.d. "
                        "(v20's salt-and-pepper) instead of from smooth random fields")
    p.add_argument("--mix-field-freq", type=float, default=14.0,
                   help="spatial frequency of the per-gene coherent mix fields "
                        "(only used when --no-granule-adaptive; otherwise the "
                        "spacing-relative granule below sets it per dataset)")
    p.add_argument("--mix-cand-k", type=int, default=4,
                   help="per-gene multi-partner draws: each mixed (cell, gene) takes "
                        "its value from one of k local same-type candidates (1 = "
                        "the v18/v20 single-partner behaviour)")
    p.add_argument("--mix-dist-scale", type=float, default=1.0,
                   help="distance scale for weighting gene-mix candidates "
                        "(exp(-d / (scale x median NN spacing)); <=0 = uniform)")
    p.add_argument("--gene-mix-wide", type=float, default=0.0,
                   help="gap-adaptive gene-mix rate at alpha=1 (v22 default 0: heavy "
                        "per-gene redraw made off-manifold chimeras on real tissue; "
                        ">0 re-enables the wide-gap ramp for ablation)")
    p.add_argument("--no-ground-gap-adapt", action="store_true",
                   help="ablate gap-adaptive re-grounding: use a fixed swap margin "
                        "at every gap width")
    p.add_argument("--ground-margin-narrow", type=float, default=5.0,
                   help="flow re-grounding swap margin at alpha=0 (narrow gaps); "
                        "high => the inherited exemplar is kept where it is the truth")
    p.add_argument("--no-layout-gap-adapt", action="store_true",
                   help="ablate gap-adaptive layout granularity: use a fixed patch "
                        "granule at every gap width")
    p.add_argument("--layout-freq-narrow", type=float, default=8.0,
                   help="coherent-patch frequency at alpha=0 (only used when "
                        "--no-granule-adaptive)")
    p.add_argument("--no-field-align", action="store_true",
                   help="ablate field-aligned re-grounding (swap a cell's exemplar "
                        "to a local same-type real cell matching the interpolated "
                        "local mean field)")
    p.add_argument("--no-field-align-gap-only", action="store_true",
                   help="run field-align at all gap widths, not only wide gaps")
    p.add_argument("--field-align-k", type=int, default=12,
                   help="in-plane kNN per flank for the field-align local field target")
    p.add_argument("--field-align-cand-k", type=int, default=10,
                   help="candidate exemplars per cell for field-align")
    p.add_argument("--field-align-frac", type=float, default=0.35,
                   help="max fraction of cells re-grounded by the field-align pass")
    p.add_argument("--field-align-margin", type=float, default=0.10,
                   help="relative field-deviation improvement a field-align swap "
                        "must beat before it is taken")
    p.add_argument("--field-align-noise-mult", type=float, default=1.25,
                   help="field-align noise floor: only cells whose field mismatch "
                        "exceeds mult x the real pool's own median field deviation "
                        "are eligible (avoids over-smoothing past the ground truth)")
    p.add_argument("--no-field-repair", action="store_true",
                   help="ablate the per-gene tail-rate-matched field repair")
    p.add_argument("--repair-q", type=float, default=0.90,
                   help="pool-residual quantile defining the per-gene real noise floor")
    p.add_argument("--repair-noise-mult", type=float, default=1.25,
                   help="an entry is a repair candidate only if its residual exceeds "
                        "mult x the per-gene noise floor")
    p.add_argument("--repair-frac", type=float, default=0.10,
                   help="max fraction of (cell, gene) entries repaired per section")
    p.add_argument("--repair-tail-scale", type=float, default=0.50,
                   help="WIDE-gap allowance multiplier on the real tail rate at "
                        "alpha=1 (1.0 = exact matching / never over-smooth; <1 "
                        "repairs into the real tail to buy field_r)")
    # ── v22 recalibration (cap, spacing-relative granules, narrow tail scale) ──
    p.add_argument("--cross-mix-cap", type=float, default=0.35,
                   help="cap on the per-gene cross-mix probability (v20's uncapped "
                        "alpha*w cost Moran's/mixing at wide gaps; <=0 disables the cap)")
    p.add_argument("--no-granule-adaptive", action="store_true",
                   help="ablate spacing-relative granules: fall back to the fixed "
                        "--mix-field-freq / --layout-freq-narrow / coherent_freq "
                        "(v21 behaviour, mis-scaled across real cell densities)")
    p.add_argument("--mix-granule-spacings", type=float, default=5.0,
                   help="per-gene mix granule size in median-NN-spacing units "
                        "(testbed equivalent of --mix-field-freq 14)")
    p.add_argument("--layout-granule-spacings-narrow", type=float, default=10.0,
                   help="layout patch size at alpha=0 in median-NN-spacing units")
    p.add_argument("--layout-granule-spacings-wide", type=float, default=20.0,
                   help="layout patch size at alpha=1 in median-NN-spacing units")
    p.add_argument("--repair-tail-scale-narrow", type=float, default=0.70,
                   help="NARROW-gap tail scale: ts_eff = (1-alpha)*this + "
                        "alpha*--repair-tail-scale. v22 spends a little of the "
                        "measured Moran's-MAE headroom on marker field_r at narrow gaps")
    args = p.parse_args()

    if not check_environment():
        return 1

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    print(f"Loading training-only input {args.input} ...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = list(adata.var_names)
    print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['section'].nunique()} sections")

    # Capture the RAW measurement before normalization (raw_output / mixing /
    # field-repair emit it verbatim), then log-normalize the model input in place.
    X_raw = _to_dense_f32(adata.X)
    et = _normalize_expression(adata)
    X_log = _to_dense_f32(adata.X)
    print(f"  expression_type={et}")

    print(f"Running SpatialCPA-v22 for targets "
          f"{[(s, round(float(z), 2)) for s, z in targets]} ...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, X_log, X_raw, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1

    method_params = {
        "seed": args.seed, "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "latent_dim": args.latent_dim, "joint_dim": args.joint_dim,
        "ode_steps": args.ode_steps, "ensemble": args.ensemble,
        "position_mode": args.position_mode,
        "ground_blend_flow": args.ground_blend_flow, "ground_k": args.ground_k,
        "edit_weight": args.edit_weight,
        "type_mode": args.type_mode, "type_vote_k": args.type_vote_k,
        "gene_mix_frac": 0.0 if args.no_gene_mix else args.gene_mix_frac,
        "raw_output": not args.no_raw_output,
        "ground_sample": not args.no_ground_sample,
        "dedup_ground": not args.no_dedup_ground,
        "ground_keep_margin": args.ground_keep_margin,
        "coherent_source": not args.no_coherent_source,
        # v19 gap-adaptive weights
        "gap_adapt": not args.no_gap_adapt, "gap_scale": args.gap_scale,
        "pair_interp_max": args.pair_interp_max,
        # v20 Bernoulli cross-mix
        "cross_mix": not args.no_cross_mix, "alpha_tol": args.alpha_tol,
        "edit_gap_extra": args.edit_gap_extra,
        "curriculum_flow": args.curriculum_flow, "curriculum_p": args.curriculum_p,
        # v21 coherent mix + field-aligned grounding
        "coherent_mix": not args.no_coherent_mix,
        "mix_field_freq": args.mix_field_freq, "mix_cand_k": args.mix_cand_k,
        "mix_dist_scale": args.mix_dist_scale, "gene_mix_wide": args.gene_mix_wide,
        "ground_gap_adapt": not args.no_ground_gap_adapt,
        "ground_margin_narrow": args.ground_margin_narrow,
        "layout_gap_adapt": not args.no_layout_gap_adapt,
        "layout_freq_narrow": args.layout_freq_narrow,
        "field_align": not args.no_field_align,
        "field_align_gap_only": not args.no_field_align_gap_only,
        "field_align_k": args.field_align_k,
        "field_align_cand_k": args.field_align_cand_k,
        "field_align_frac": args.field_align_frac,
        "field_align_margin": args.field_align_margin,
        "field_align_noise_mult": args.field_align_noise_mult,
        "field_repair": not args.no_field_repair,
        "repair_q": args.repair_q,
        "repair_noise_mult": args.repair_noise_mult,
        "repair_frac": args.repair_frac,
        "repair_tail_scale": args.repair_tail_scale,
        # v22 recalibration
        "cross_mix_cap": args.cross_mix_cap,
        "granule_adaptive": not args.no_granule_adaptive,
        "mix_granule_spacings": args.mix_granule_spacings,
        "layout_granule_spacings_narrow": args.layout_granule_spacings_narrow,
        "layout_granule_spacings_wide": args.layout_granule_spacings_wide,
        "repair_tail_scale_narrow": args.repair_tail_scale_narrow,
        "flow_matching": True, "generation_only": True,
    }
    _v2_io.write_prediction_h5(
        results, gene_names, target_sections, method_params, wall,
        args.output, "spatialcpav22_gen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
