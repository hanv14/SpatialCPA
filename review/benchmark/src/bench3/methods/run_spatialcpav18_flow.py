"""SpatialCPA-v18-flow — benchmark wrapper (method name ``spatialcpav18_gen_flow``).

Retrieval-based virtual-slice synthesis in which the learned flow decides ONE
thing: **which flanking section to retrieve the output section from.**

Built on SpatialCPA-v18 without changing it: ``learn_spatialcpav18.py`` and
``run_spatialcpav18.py`` are imported, never edited. The class below subclasses
``SpatialCPAv18`` and overrides four methods; everything else (training, the
store, the type vote, composition matching, gene splicing, raw output) is v18's
own code.

What differs from v18 at the benchmark configuration
----------------------------------------------------
1. **Layout = one real section, chosen by the flow.** v18's ``nearest`` layout
   copies the cells (exact positions) of the lower flank when ``t <= 0.5``, else
   the upper one. Under the alternating hold-out every target sits at
   ``t = 0.5``, so that rule always copies the section *below* — an arbitrary
   tie-break. Here the flow breaks it: the flow's prediction for depth z* is
   computed at the positions of each flank's real cells, and the flank whose real
   cells agree better with it (lower mean latent distance) is retrieved
   (``_flow_choose_flank``). ``--flank-select rule`` keeps v18's rule instead.
2. **No flow-guided donor reranking.** Re-grounding, the type-vote replacement and
   the composition replacement use each output cell's *own source cell* as the
   query (``pool_e[anchor_src]``) instead of the flow's prediction, so they keep
   the retrieved cell (re-grounding never switches) or pick the replacement most
   like it. This is the "no flow" setting of the v18 ablation, which kept
   continuity (UMAP mixing, Moran's I) that flow reranking lost.

Consequence, by construction: whenever the flow picks the same flank as v18's rule,
the output is **bitwise identical** to v18 run with ``--position-mode nearest`` and
the donor query switched off ("nearest + no flow"); ``--flank-select rule`` is that
variant exactly. The flow can change the result only through the flank it picks.
``tests/`` and REVIEW_NOTES record the check.

The flank choice uses no numpy randomness, so the numpy random stream that drives
the layout, vote, composition and gene splicing is the same as in "nearest + no
flow". The flow's own sampling uses torch's generator, which no later step reads.
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_spatialcpav18 as _V18W    # noqa: E402  (v18's wrapper: loader, I/O helpers)

_V18 = _V18W._V18                    # the loaded learn_spatialcpav18 module
_v2_io = _V18W._v2_io
leakage_guard = _V18W.leakage_guard
torch = _V18.torch

METHOD_NAME = "spatialcpav18_gen_flow"
# Cap on cells per flank scored by the flow when choosing a flank. Deterministic
# (evenly spaced), so the choice consumes no randomness; 4000 covers STARmap's
# sections whole and bounds the cost on million-cell volumes.
FLANK_SCORE_MAX_CELLS = 4000


class SpatialCPAv18Flow(_V18.SpatialCPAv18):
    """v18 with a flow-chosen single-section layout and no flow donor reranking."""

    flank_select = "flow"            # "flow" | "rule" | "lower" | "upper"; set by the wrapper

    # -- remember the target depth: _resample_layout is not given z ------------
    def _generate(self, z):
        self._z_current = float(z)
        self._query = None
        return super()._generate(z)

    # -- 1. layout: one flanking section, exact positions ----------------------
    def _resample_layout(self, lower, upper, t, n_target, rng):
        """Same as v18's ``position_mode="nearest"`` branch, except that the
        flank is chosen by ``_flow_choose_flank`` (or v18's rule)."""
        rule_lower = t <= 0.5
        if self.flank_select == "flow":
            use_lower, info = self._flow_choose_flank(lower, upper, rule_lower)
        elif self.flank_select in ("lower", "upper"):        # diagnostics only
            use_lower, info = self.flank_select == "lower", {"forced": self.flank_select}
        else:
            use_lower, info = rule_lower, {"rule": True}
        info.update(z=self._z_current, t=float(t), rule_lower=bool(rule_lower),
                    chose_lower=bool(use_lower))
        self.flank_log.append(info)

        near = lower if use_lower else upper
        near_xy = self._nxy(near.coords_xy).astype(np.float32)
        pick = (rng.choice(near_xy.shape[0], n_target, replace=False)
                if near_xy.shape[0] > n_target else np.arange(near_xy.shape[0]))
        anchor = near_xy[pick]
        anchor_src = (pick if use_lower else pick + lower.n_spots).astype(np.int64)
        return anchor, anchor_src

    def _flow_choose_flank(self, lower, upper, rule_lower):
        """Pick the flank whose real cells best match the flow's prediction at z*.

        For each flank, the flow is queried at that flank's own cell positions and
        target depth z*; its decoded prediction is compared with those cells' real
        latents (the same standardized PCA latent the store uses). The flank with
        the smaller mean distance is the better retrieval for depth z*. Ties go
        to v18's rule.
        """
        cfg = self.cfg
        z = self._z_current
        li = self.stack.slices.index(lower)
        ui = self.stack.slices.index(upper)

        # context sources: exactly as _generate builds them
        order = np.argsort(self.stack.z_centers())
        zc = self.stack.z_centers()
        k_side = max(cfg.context_slices_each_side, 1)
        below = [int(j) for j in order if zc[j] <= z]
        above = [int(j) for j in order if zc[j] > z]
        ctx_src = (below[-k_side:] if below else []) + (above[:k_side] if above else [])
        for extra in (li, ui):
            if extra not in ctx_src:
                ctx_src.append(extra)
        ctx_pool_nxy, ctx_pool_z, ctx_pool_owner = self._build_pool(self.S, ctx_src)
        ctx_pool_h = torch.cat([self.S[j]["h"] for j in ctx_src], 0)
        zn = self._nz(z)

        dist = {}
        for name, j in (("lower", li), ("upper", ui)):
            st = self.S[j]
            n = st["nxy"].shape[0]
            idx = (np.linspace(0, n - 1, FLANK_SCORE_MAX_CELLS).astype(np.int64)
                   if n > FLANK_SCORE_MAX_CELLS else np.arange(n))
            q_nxy = st["nxy"][idx].astype(np.float32)
            with torch.no_grad():
                ctx = self._context(ctx_pool_h, ctx_pool_nxy, ctx_pool_z, ctx_pool_owner,
                                    ctx_src, self.S, q_nxy, zn, self.ctxmod, self.dev)
                Q = q_nxy.shape[0]
                zt = torch.full((Q,), float(zn), device=self.dev)
                h_acc = torch.zeros((Q, cfg.joint_dim), device=self.dev)
                n_ens = max(cfg.n_ensemble, 1)
                steps = max(cfg.n_ode_steps, 1)
                for _ in range(n_ens):
                    h = torch.randn((Q, cfg.joint_dim), device=self.dev)
                    for si in range(steps):
                        tt = torch.full((Q,), si / steps, device=self.dev)
                        h = h + (1.0 / steps) * self.vfield(h, tt, ctx, zt)
                    h_acc += h
                e_hat = self.encoder.decode_e(h_acc / n_ens).cpu().numpy()
            e_real = st["e"][torch.as_tensor(idx, device=self.dev)].cpu().numpy()
            dist[name] = float(np.mean(np.linalg.norm(e_hat - e_real, axis=1)))

        if dist["lower"] < dist["upper"]:
            use_lower = True
        elif dist["upper"] < dist["lower"]:
            use_lower = False
        else:
            use_lower = rule_lower
        return use_lower, {"dist_lower": dist["lower"], "dist_upper": dist["upper"]}

    # -- 2. no flow donor reranking: query = each cell's own source latent -----
    def _ground(self, anchor, anchor_src, e_hat, pool_nxy, pool_e, pool_expr, pool_type, rng):
        self._query = pool_e[anchor_src].copy()
        return super()._ground(anchor, anchor_src, self._query, pool_nxy, pool_e,
                               pool_expr, pool_type, rng)

    def _vote_types(self, anchor, pick, ct_idx, expr, pool_nxy, pool_type,
                    pool_expr, pool_e, e_hat, rng):
        return super()._vote_types(anchor, pick, ct_idx, expr, pool_nxy, pool_type,
                                   pool_expr, pool_e, self._query, rng)

    def _match_composition(self, lower, upper, t, ct_idx, expr, anchor, pool_nxy, pool_e,
                           e_hat, pool_type, pool_expr, rng, pick=None):
        return super()._match_composition(lower, upper, t, ct_idx, expr, anchor, pool_nxy,
                                          pool_e, self._query, pool_type, pool_expr, rng, pick)


def run_method(adata, targets, gene_names, X_log, X_raw, args):
    """v18's run_method, constructing the subclass."""
    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = _V18W.build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}, []

    cfg = _V18W._build_config(args)
    cfg.position_mode = "flanking"   # routes the layout through _resample_layout
    print(f"  epochs(A={cfg.pretrain_epochs},B={cfg.epochs}), "
          f"latent(d_e={cfg.expr_latent_dim},h={cfg.joint_dim}), "
          f"flow(steps={cfg.n_ode_steps},ens={cfg.n_ensemble}), "
          f"layout=single-section, flank_select={args.flank_select}, "
          f"ctx_slices={cfg.context_slices_each_side}, type_mode={cfg.type_mode}, "
          f"gene_mix={cfg.gene_mix_frac}, raw_output={cfg.raw_output}, "
          f"ground_sample={cfg.ground_sample}, keep_margin={cfg.ground_keep_margin}, "
          f"edit_w={cfg.edit_weight}, blend={cfg.ground_blend_flow}, k={cfg.ground_k}, "
          f"temp={cfg.ground_temp}, device={cfg.device}, donor_query=source-cell")

    SpatialCPAv18Flow.flank_select = args.flank_select
    gen = SpatialCPAv18Flow(stack, gene_names=gene_names,
                            cell_type_names=cell_type_names, cfg=cfg)
    gen.flank_log = []
    print(f"  flow-matching model trained: {gen.trained}")

    results = {}
    for sec, z in targets:
        print(f"  {sec}: retrieving virtual slice at z={z:.2f} ...")
        try:
            vs = gen.generate_virtual_slice(z=z)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        if gen.flank_log:
            f = gen.flank_log[-1]
            chose = "lower" if f["chose_lower"] else "upper"
            rule = "lower" if f["rule_lower"] else "upper"
            extra = (f", dist lower={f['dist_lower']:.4f} upper={f['dist_upper']:.4f}"
                     if "dist_lower" in f else "")
            print(f"    flank: {chose} (rule: {rule}{extra})")
        n = vs.coords.shape[0]
        if n == 0:
            continue
        print(f"    -> {n} cells synthesized")
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_V18W._to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64),
                        "cell_type": cell_type}
    return results, gen.flank_log


def build_parser():
    """v18's flags (identical names and defaults; a test checks it) + ``--flank-select``."""
    p = argparse.ArgumentParser(description="SpatialCPA-v18-flow wrapper")
    _v2_io.add_v2_args(p)
    p.add_argument("--epochs", type=int, default=160, help="Phase B flow epochs")
    p.add_argument("--pretrain-epochs", type=int, default=60, help="Phase A epochs")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--latent-dim", type=int, default=32,
                   help="expression PCA latent dim (V18Config.expr_latent_dim)")
    p.add_argument("--joint-dim", type=int, default=48)
    p.add_argument("--ode-steps", type=int, default=12,
                   help="Euler steps when sampling the flow ODE")
    p.add_argument("--ensemble", type=int, default=4,
                   help="initial noises marginalized per query")
    p.add_argument("--context-slices", type=int, default=None,
                   help="real slices per side feeding the 3D context "
                        "(default: V18Config.context_slices_each_side)")
    p.add_argument("--position-mode", default="flanking",
                   choices=["flanking", "morph", "nearest"],
                   help="not used: this method's layout is the flow-chosen single section")
    p.add_argument("--ground-blend-flow", type=float, default=0.20,
                   help="fraction of cells offered for re-grounding")
    p.add_argument("--ground-k", type=int, default=8,
                   help="local real candidates for re-grounding, composition, gene mix")
    p.add_argument("--ground-temp", type=float, default=0.25,
                   help="softmax temperature for exemplar sampling")
    p.add_argument("--edit-weight", type=float, default=0.25,
                   help="blend toward the decoded profile (0 = pure exemplar)")
    p.add_argument("--no-coherent-source", action="store_true",
                   help="not used by this layout")
    p.add_argument("--no-output-counts", action="store_true",
                   help="emit log1p-normalized (not count-like) expression")
    p.add_argument("--no-composition-match", action="store_true",
                   help="disable cell-type composition matching")
    p.add_argument("--type-mode", default="vote", choices=["inherit", "vote"],
                   help="'vote': distance-weighted kNN type vote over both flanks")
    p.add_argument("--type-vote-k", type=int, default=12)
    p.add_argument("--gene-mix-frac", type=float, default=0.15,
                   help="fraction of each cell's genes taken from a second local "
                        "same-type real cell (0 disables)")
    p.add_argument("--no-gene-mix", action="store_true",
                   help="ablate gene-mix (equivalent to --gene-mix-frac 0)")
    p.add_argument("--no-raw-output", action="store_true",
                   help="emit expm1(log-normalized) instead of raw measurements")
    p.add_argument("--no-ground-sample", action="store_true",
                   help="ablate temperature sampling: use the argmin exemplar")
    p.add_argument("--no-dedup-ground", action="store_true",
                   help="ablate the exemplar-reuse penalty")
    p.add_argument("--ground-keep-margin", type=float, default=1.0,
                   help="keep the inherited cell unless a candidate matches the query "
                        "better by this margin")
    # ── this method's own flag ──
    p.add_argument("--flank-select", default="flow",
                   choices=["flow", "rule", "lower", "upper"],
                   help="'flow': the flow picks which flank to retrieve (default); "
                        "'rule': v18's rule (lower if t <= 0.5) — identical to v18 "
                        "'nearest + no flow'; 'lower'/'upper': force a flank "
                        "(diagnostics: is the flow's choice the better one?)")
    return p


def main():
    args = build_parser().parse_args()
    if args.position_mode != "flanking":
        print("ERROR: --position-mode is not used by spatialcpav18_gen_flow; its layout "
              "is the flow-chosen single section", file=sys.stderr)
        return 2
    if not _V18W.check_environment():
        return 1

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    print(f"Loading training-only input {args.input} ...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = list(adata.var_names)
    print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['section'].nunique()} sections")

    X_raw = _V18W._to_dense_f32(adata.X)
    et = _V18W._normalize_expression(adata)
    X_log = _V18W._to_dense_f32(adata.X)
    print(f"  expression_type={et}")

    print(f"Running SpatialCPA-v18-flow for targets "
          f"{[(s, round(float(z), 2)) for s, z in targets]} ...")
    t0 = time.time()
    results, flank_log = run_method(adata, targets, gene_names, X_log, X_raw, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1

    method_params = {
        "seed": args.seed, "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "latent_dim": args.latent_dim, "joint_dim": args.joint_dim,
        "ode_steps": args.ode_steps, "ensemble": args.ensemble,
        "position_mode": "single_section", "flank_select": args.flank_select,
        "donor_query": "source_cell",
        "ground_blend_flow": args.ground_blend_flow, "ground_k": args.ground_k,
        "ground_temp": args.ground_temp, "edit_weight": args.edit_weight,
        "type_mode": args.type_mode, "type_vote_k": args.type_vote_k,
        "gene_mix_frac": 0.0 if args.no_gene_mix else args.gene_mix_frac,
        "raw_output": not args.no_raw_output,
        "ground_sample": not args.no_ground_sample,
        "dedup_ground": not args.no_dedup_ground,
        "ground_keep_margin": args.ground_keep_margin,
        "flank_choices": [{k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in f.items()} for f in flank_log],
        "flow_matching": True, "generation_only": True,
    }
    _v2_io.write_prediction_h5(
        results, gene_names, target_sections, method_params, wall,
        args.output, METHOD_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
