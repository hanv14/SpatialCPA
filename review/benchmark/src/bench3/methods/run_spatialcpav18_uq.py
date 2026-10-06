"""SpatialCPA-v18-UQ — benchmark wrapper (method name ``spatialcpav18_gen_uq``).

The flow used GENERATIVELY, for what retrieval cannot provide: a per-cell
uncertainty map. The synthesized section itself is unchanged — it is v18's
"nearest + no flow" (``spatialcpav18_gen_flow`` under ``--flank-select rule``,
bitwise) — so the method's paper metrics are those of plain retrieval by
construction. Alongside ``prediction.h5`` it writes ``uncertainty.npz``:

``u_flow``       spread of the trained flow's prediction at each output cell:
                 ``UQ_SAMPLES`` independent draws (ONE noise each — v18 averages
                 its noise ensemble into a conditional mean, which is exactly what
                 discards this information), decoded to the expression latent;
                 u = mean distance of the draws from their mean.
``u_untrained``  the same spread from the same sampler with the flow re-initialised
                 at random — the control for "any noisy sampler has spread".
``d_flank``      disagreement between the two flanking sections at the cell:
                 distance between the mean latent of the ``UQ_KNN`` nearest lower-
                 flank cells and of the upper-flank cells — a retrieval-only proxy
                 that needs no network.

Whether the flow's spread is *calibrated* — high where the output is wrong — is
scored afterwards against the held-out ground truth by
``bench3/evaluate_uncertainty.py`` (evaluation-only). The flow earns a role only
if ``u_flow`` predicts the error better than ``u_untrained``, ``d_flank`` and the
patch cell-count baseline that evaluator adds.

Built on ``run_spatialcpav18_flow.py`` (imported, not edited) and v18 (not edited).
The uncertainty draws use their own seeded torch generator and no numpy
randomness, so the synthesized section is untouched by them.
"""

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_spatialcpav18_flow as _F    # noqa: E402  (the flow wrapper; not modified)

_V18W, _v2_io, leakage_guard, torch = _F._V18W, _F._v2_io, _F.leakage_guard, _F.torch

METHOD_NAME = "spatialcpav18_gen_uq"
SIDECAR = "uncertainty.npz"
# Independent flow draws per output cell (one noise each).
UQ_SAMPLES = 8
# Flank cells averaged into d_flank on each side.
UQ_KNN = 8
# Cells per flow call.
UQ_CHUNK = 4000


class SpatialCPAv18UQ(_F.SpatialCPAv18Flow):
    """v18 "nearest + no flow", plus per-cell flow-sample spread."""

    flank_select = "rule"

    def uncertainty(self, coords_xy, z):
        """``{"u_flow", "u_untrained", "d_flank"}``, each (n,), for output cells at
        physical ``coords_xy`` (n, 2) of the section just synthesized at ``z``."""
        lower, upper = self._flanks
        li, ui = self.stack.slices.index(lower), self.stack.slices.index(upper)
        q = self._nxy(coords_xy).astype(np.float32)
        pool = self._flow_ctx_pool(li, ui, float(z))
        out = {"u_flow": self._spread(pool, q, None, None)}
        vf, cm = self._cached(("untrained",), self._untrained_flow)
        out["u_untrained"] = self._spread(pool, q, vf, cm)
        e_lo, e_hi = self.S[li]["e"].cpu().numpy(), self.S[ui]["e"].cpu().numpy()
        k = min(UQ_KNN, len(e_lo), len(e_hi))
        _, a = cKDTree(self.S[li]["nxy"]).query(q, k=k)
        _, b = cKDTree(self.S[ui]["nxy"]).query(q, k=k)
        a, b = np.asarray(a).reshape(len(q), k), np.asarray(b).reshape(len(q), k)
        out["d_flank"] = np.linalg.norm(e_lo[a].mean(1) - e_hi[b].mean(1), axis=1)
        return out

    def _spread(self, pool, q, vfield, ctxmod):
        """Mean distance of ``UQ_SAMPLES`` single-noise decoded draws from their
        mean, per query: (n,). One seeded generator for all draws."""
        gen = self._cv_generator()
        n_ens, self.cfg.n_ensemble = self.cfg.n_ensemble, 1      # one noise per draw
        try:
            u = np.empty(len(q))
            for a in range(0, len(q), UQ_CHUNK):
                qq = q[a:a + UQ_CHUNK]
                with torch.no_grad():
                    draws = np.stack([self.encoder.decode_e(
                        self._flow_h(pool, qq, gen, vfield, ctxmod)).cpu().numpy()
                        for _ in range(UQ_SAMPLES)])              # (K, m, d)
                u[a:a + len(qq)] = np.linalg.norm(draws - draws.mean(0), axis=2).mean(0)
        finally:
            self.cfg.n_ensemble = n_ens
        return u


def run_method(adata, targets, gene_names, X_log, X_raw, args):
    """Returns ``(results, uncertainty)``: the v18 nearest + no flow sections and,
    per section, the three uncertainty arrays in output-cell order."""
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", np.ones(adata.n_obs, dtype=bool), seed=args.seed)
    stack = _V18W.build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}, {}
    cfg = _V18W._build_config(args)
    cfg.position_mode = "flanking"   # the layout goes through the subclass (rule flank)
    print(f"  layout=nearest + no flow (rule flank); uncertainty: {UQ_SAMPLES} single-noise "
          f"flow draws per cell, untrained-flow and flank-disagreement controls")
    gen = SpatialCPAv18UQ(stack, gene_names=gene_names, cell_type_names=cell_type_names,
                          cfg=cfg)
    gen.flank_log = []
    print(f"  flow-matching model trained: {gen.trained}")

    results, unc = {}, {}
    for sec, z in targets:
        print(f"  {sec}: retrieving virtual slice at z={z:.2f} ...")
        vs = gen.generate_virtual_slice(z=z)
        n = vs.coords.shape[0]
        if n == 0:
            continue
        u = gen.uncertainty(np.asarray(vs.coords)[:, :2], z)
        print(f"    -> {n} cells; median u_flow={np.median(u['u_flow']):.3f} "
              f"u_untrained={np.median(u['u_untrained']):.3f} "
              f"d_flank={np.median(u['d_flank']):.3f}")
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_V18W._to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64), "cell_type": cell_type}
        unc[sec] = u
    return results, unc


def main():
    p = _F.build_parser()
    p.description = "SpatialCPA-v18-UQ wrapper"
    args = p.parse_args()
    if args.flank_select != "flow" or args.position_mode != "flanking":
        print("ERROR: spatialcpav18_gen_uq has a fixed layout (v18 nearest + no flow); "
              "do not pass --flank-select or --position-mode", file=sys.stderr)
        return 2
    if not _V18W.check_environment():
        return 1
    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    print(f"Loading training-only input {args.input} ...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = list(adata.var_names)
    X_raw = _V18W._to_dense_f32(adata.X)
    _V18W._normalize_expression(adata)
    X_log = _V18W._to_dense_f32(adata.X)

    t0 = time.time()
    results, unc = run_method(adata, targets, gene_names, X_log, X_raw, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1
    params = {"seed": args.seed, "layout": "nearest_no_flow", "uq_samples": UQ_SAMPLES,
              "uq_knn": UQ_KNN, "uncertainty_file": SIDECAR, "flow_matching": True,
              "generation_only": True}
    _v2_io.write_prediction_h5(results, gene_names, target_sections, params, wall,
                               args.output, METHOD_NAME)
    # same section order and cell order as prediction.h5
    side = {f"{sec}::{k}": v.astype(np.float32)
            for sec in target_sections if sec in unc for k, v in unc[sec].items()}
    np.savez(Path(args.output).with_name(SIDECAR), **side)
    print(f"Wrote {Path(args.output).with_name(SIDECAR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
