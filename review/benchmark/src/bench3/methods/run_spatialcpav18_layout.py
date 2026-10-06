"""SpatialCPA-v18-layout — the flow generates the layout's DENSITY
(methods ``spatialcpav18_gen_flow_layout*``).

v18 copies the layout of one flanking section. Here the flow decides how many
cells each region of the output section holds; retrieval still supplies every
cell (real positions, real expression):

1. The rule flank's cells are copied as in v18's ``nearest`` layout.
2. Both flanks' cells are binned on one square grid (side
   ``LAYOUT_BIN_SPACINGS`` median cell spacings). In each occupied bin the
   density source predicts a cell intensity w_b at depth z*.
3. Target counts ``T_b = (1 - rho) * c_b + rho * N * w_b / sum(w)``, where c_b
   is the copied layout's count and N its total (rounded by largest remainder,
   so N is kept). Bins over target drop copied cells; bins under target take
   unused real cells of either flank in that bin. ``rho = 0`` is the copy.

``--layout-source`` sets w_b — the three arms of one experiment:

``flow``       the trained flow: v18 predicts each cell's neighbourhood channels
               ``m`` (type mix + local density) from h*; w_b = (decoded density
               at the bin's cell centroid)^2 — density is an inverse spacing, so
               its square is a number density
``untrained``  the same with the flow re-initialised at random
``interp``     no network: (1 - t) x lower-flank count + t x upper-flank count

rho is chosen per arm by leave-one-training-section-out folds (``LAYOUT_RHO_GRID``,
rho = 0 wins ties), scored with the same 8 pre-set metrics as flow-cv. As in
h-cv, every random draw after the layout comes from a stage-specific seeded
generator, and the relayout draws from its own, so two rho values differ only in
where cells are (common random numbers); the paired baseline is this method at
rho = 0 (``--layout-rho 0``).

Built on ``run_spatialcpav18_flow.py`` and v18, neither of which is edited.
"""

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_spatialcpav18_flow as _F    # noqa: E402  (the flow wrapper; not modified)

_V18W, _v2_io, leakage_guard, torch = _F._V18W, _F._v2_io, _F.leakage_guard, _F.torch
_FlowBase = _F.SpatialCPAv18Flow       # bound once: _layout_cv_folds rebinds the module name

METHOD_NAME = {"flow": "spatialcpav18_gen_flow_layout",
               "untrained": "spatialcpav18_gen_flow_layout_untrained",
               "interp": "spatialcpav18_gen_flow_layout_interp"}
# Bin side in median cell spacings (~25 cells per bin).
LAYOUT_BIN_SPACINGS = 5.0
# Mixing strengths a fold may choose; 0 (= v18's nearest layout) wins ties.
LAYOUT_RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
# Second seed word of the relayout generator.
LAYOUT_STREAM = 301


def bin_ids(xy, origin, side):
    """Square-bin id per point; bins are keyed by (ix, iy) packed into int64."""
    k = np.floor((np.asarray(xy, np.float64) - origin) / side).astype(np.int64)
    return k[:, 1] * 1_000_003 + k[:, 0]


def largest_remainder(x, total):
    """Non-negative integers proportional to ``x`` summing to ``total``."""
    x = np.maximum(np.asarray(x, np.float64), 0.0)
    if x.sum() <= 0 or total <= 0:
        return np.zeros(len(x), dtype=np.int64)
    q = x / x.sum() * total
    n = np.floor(q).astype(np.int64)
    rest = int(total - n.sum())
    if rest > 0:
        n[np.argsort(-(q - n), kind="stable")[:rest]] += 1
    return n


def target_counts(copied, intensity, rho):
    """``(1 - rho) * copied + rho * N * intensity / sum(intensity)`` rounded so
    the total N = sum(copied) is kept. Exactly ``copied`` at rho 0."""
    copied = np.asarray(copied, dtype=np.int64)
    if rho == 0:
        return copied.copy()
    N = int(copied.sum())
    w = np.maximum(np.asarray(intensity, np.float64), 0.0)
    share = w / w.sum() * N if w.sum() > 0 else np.zeros_like(w)
    return largest_remainder((1.0 - rho) * copied + rho * share, N)


class SpatialCPAv18Layout(_FlowBase):
    """v18 nearest + no flow with the per-region cell count set by a density source."""

    flank_select = "layout-cv"       # not a parent mode: the parent lays out the rule flank
    layout_source = "flow"
    layout_rho = 0.0

    def _generate(self, z):
        # as the parent's h-cv: stage-specific generators for every later draw
        self._z_current = float(z)
        self._query = None
        self._stage_rng = {k: np.random.default_rng([int(self.cfg.seed), v])
                           for k, v in _F.STAGE_STREAMS.items()}
        return super(_FlowBase, self)._generate(z)               # v18's own _generate

    def _resample_layout(self, lower, upper, t, n_target, rng):
        anchor, src = super()._resample_layout(lower, upper, t, n_target, rng)  # rule flank
        rho = float(self.layout_rho)
        info = self.flank_log[-1]
        info.update(layout_source=self.layout_source, layout_rho=rho, moved_frac=0.0)
        if rho == 0:
            return anchor, src
        anchor, src, moved = self._relayout(lower, upper, float(t), anchor, src, rho)
        info["moved_frac"] = moved
        return anchor, src

    def _relayout(self, lower, upper, t, anchor, src, rho):
        n_lo = lower.n_spots
        pool_xy = np.vstack([lower.coords_xy, upper.coords_xy]).astype(np.float64)
        side = LAYOUT_BIN_SPACINGS * float(np.median([lower.median_spacing(),
                                                      upper.median_spacing()]))
        origin = pool_xy.min(0)
        pool_bin = bin_ids(pool_xy, origin, side)
        bins = np.unique(pool_bin)
        anc_bin = pool_bin[src]
        copied = np.array([(anc_bin == b).sum() for b in bins])
        w = self._intensity(lower, upper, t, pool_xy, pool_bin, bins, n_lo)
        target = target_counts(copied, w, rho)

        g = np.random.default_rng([int(self.cfg.seed), LAYOUT_STREAM])
        keep = np.ones(len(src), dtype=bool)
        used = np.zeros(len(pool_xy), dtype=bool)
        used[src] = True
        add = []
        for b, c, tb in zip(bins, copied, target):
            if c > tb:                                   # drop copied cells
                here = np.nonzero(anc_bin == b)[0]
                keep[g.choice(here, c - tb, replace=False)] = False
            elif tb > c:                                 # take unused real cells
                cand = np.nonzero((pool_bin == b) & ~used)[0]
                if len(cand):
                    take = g.choice(cand, min(tb - c, len(cand)), replace=False)
                    used[take] = True
                    add.append(take)
        new_src = np.concatenate([src[keep]] + add).astype(np.int64)
        props = np.vstack([self._nxy(lower.coords_xy).astype(np.float32),
                           self._nxy(upper.coords_xy).astype(np.float32)])
        moved = float((~keep).sum()) / max(len(src), 1)
        return props[new_src], new_src, moved

    def _intensity(self, lower, upper, t, pool_xy, pool_bin, bins, n_lo):
        """w_b per bin under ``self.layout_source``."""
        if self.layout_source == "interp":
            lo = np.array([(pool_bin[:n_lo] == b).sum() for b in bins], dtype=float)
            hi = np.array([(pool_bin[n_lo:] == b).sum() for b in bins], dtype=float)
            return (1.0 - t) * lo + t * hi
        if self.layout_source not in ("flow", "untrained"):
            raise ValueError(f"unknown layout source {self.layout_source!r}")
        cen = np.array([pool_xy[pool_bin == b].mean(0) for b in bins])
        li, ui = self.stack.slices.index(lower), self.stack.slices.index(upper)
        pool = self._flow_ctx_pool(li, ui, self._z_current)
        vf = cm = None
        if self.layout_source == "untrained":
            vf, cm = self._cached(("untrained",), self._untrained_flow)
        with torch.no_grad():
            h = self._flow_h(pool, self._nxy(cen).astype(np.float32), self._cv_generator(),
                             vf, cm)
            m = self.encoder.decode_m(h).cpu().numpy()
        dens = m[:, -1] * self._m_std[-1] + self._m_mean[-1]          # de-standardize
        return np.maximum(dens, 1e-3) ** 2


def _layout_cv_folds(adata, gene_names, X_log, X_raw, ct_all, cell_type_names, cfg, source):
    """Folds as in h-cv: every rho in ``LAYOUT_RHO_GRID`` synthesized on each
    left-out training section, gain vs rho = 0."""
    total = {r: 0 for r in LAYOUT_RHO_GRID}
    records = []
    # _F._fold_models builds its fold models from the module name
    # ``SpatialCPAv18Flow``; point it at this class for the folds only, so the
    # flow wrapper itself stays unedited.
    _F.SpatialCPAv18Flow = SpatialCPAv18Layout
    try:
        records = _run_layout_folds(adata, gene_names, X_log, X_raw, ct_all,
                                    cell_type_names, cfg, source, total)
    finally:
        _F.SpatialCPAv18Flow = _FlowBase
    return total, records


def _run_layout_folds(adata, gene_names, X_log, X_raw, ct_all, cell_type_names, cfg,
                      source, total):
    records = []
    for sec, z, fold, score in _F._fold_models(adata, gene_names, X_log, X_raw, ct_all,
                                               cell_type_names, cfg):
        assert isinstance(fold, SpatialCPAv18Layout)
        fold.layout_source = source
        per = {}
        for r in LAYOUT_RHO_GRID:
            fold.layout_rho = r
            per[r] = (score(fold.generate_virtual_slice(z=z)), fold.flank_log[-1])
        rec = {"section": sec, "z": z, "rho": {}}
        for r in LAYOUT_RHO_GRID:
            gain, detail = _F.fold_gain(per[0.0][0], per[r][0])
            total[r] += gain
            mv = per[r][1].get("moved_frac", 0.0)
            rec["rho"][str(r)] = {"gain": gain, "moved_frac": mv,
                                  "metrics": {k: v["other"] for k, v in detail.items()}}
            print(f"    rho={r:<4} {100 * mv:5.1f}% of cells moved: gain {gain:+d} vs rho=0")
        records.append(rec)
    return records


def run_method(adata, targets, gene_names, X_log, X_raw, args):
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", np.ones(adata.n_obs, dtype=bool), seed=args.seed)
    stack = _V18W.build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}, [], None
    cfg = _V18W._build_config(args)
    cfg.position_mode = "flanking"
    print(f"  layout: rule flank, cell count per region from {args.layout_source} "
          f"(bin = {LAYOUT_BIN_SPACINGS:g} spacings)")
    if args.layout_rho is not None:
        rho, cv = float(args.layout_rho), {"layout_source": args.layout_source,
                                           "rho": float(args.layout_rho), "forced": True}
        print(f"  layout-cv: rho={rho} FORCED by --layout-rho (no folds; a diagnostic)")
    else:
        total, records = _layout_cv_folds(adata, gene_names, X_log, X_raw, ct_all,
                                          cell_type_names, cfg, args.layout_source)
        rho, best = _F.calibrate_q(total)
        note = "" if records else " (no interior training section: no folds, rho stays 0)"
        print(f"  layout-cv ({args.layout_source}): total gain by rho "
              f"{ {r: total[r] for r in LAYOUT_RHO_GRID} } -> rho={rho} (gain {best:+d}){note}")
        cv = {"layout_source": args.layout_source, "rho": rho, "validated_gain": int(best),
              "n_folds": len(records),
              "total_gain_by_rho": {str(k): int(v) for k, v in total.items()},
              "folds": records}
    gen = SpatialCPAv18Layout(stack, gene_names=gene_names, cell_type_names=cell_type_names,
                              cfg=cfg)
    gen.flank_log = []
    gen.layout_source, gen.layout_rho = args.layout_source, rho
    print(f"  flow-matching model trained: {gen.trained}")
    results = {}
    for sec, z in targets:
        vs = gen.generate_virtual_slice(z=z)
        f = gen.flank_log[-1]
        print(f"  {sec}: z={z:.2f}, rho={f['layout_rho']}, "
              f"{100 * f['moved_frac']:.1f}% of cells moved -> {vs.coords.shape[0]} cells")
        n = vs.coords.shape[0]
        if n == 0:
            continue
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_V18W._to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64), "cell_type": cell_type}
    return results, gen.flank_log, cv


def main():
    p = _F.build_parser()
    p.description = "SpatialCPA-v18-layout wrapper"
    p.add_argument("--layout-source", default="flow", choices=["flow", "untrained", "interp"],
                   help="per-region cell intensity: the trained flow's predicted density "
                        "(default), an untrained flow's, or the flanks' t-weighted counts")
    p.add_argument("--layout-rho", type=float, default=None,
                   help="force the mixing strength and skip the folds (diagnostics: 0 is "
                        "v18 'nearest + no flow' with this method's random streams)")
    args = p.parse_args()
    if args.flank_select != "flow" or args.position_mode != "flanking":
        print("ERROR: this method's layout is fixed (rule flank + density relayout); "
              "do not pass --flank-select or --position-mode", file=sys.stderr)
        return 2
    if not _V18W.check_environment():
        return 1
    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = list(adata.var_names)
    X_raw = _V18W._to_dense_f32(adata.X)
    _V18W._normalize_expression(adata)
    X_log = _V18W._to_dense_f32(adata.X)
    t0 = time.time()
    results, flank_log, cv = run_method(adata, targets, gene_names, X_log, X_raw, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1
    params = {"seed": args.seed, "layout_source": args.layout_source,
              "layout_bin_spacings": LAYOUT_BIN_SPACINGS, "layout_cv": cv,
              "flank_choices": [{k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in f.items()} for f in flank_log],
              "flow_matching": True, "generation_only": True}
    _v2_io.write_prediction_h5(results, gene_names, target_sections, params, wall,
                               args.output, METHOD_NAME[args.layout_source])
    return 0


if __name__ == "__main__":
    sys.exit(main())
