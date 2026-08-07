"""SpatialCPA-v17 (fully-generative VAE + NB decoder) — benchmark-pbya-v2 wrapper.

v17 trains a variational joint autoencoder with a negative-binomial (count-aware) decoder,
freezes it, trains a conditional flow-matching field in its KL-regularized latent space,
and generates virtual slices by **decoding** the flow-generated latent into expression
(retrieval available via ``--retrieval`` for ablation). Unlike v14, the NB decoder needs
raw counts, so this wrapper passes **raw counts** to the method for count datasets (and
falls back to a Gaussian decoder for non-count inputs). Runs in ``bench_spatialcpa``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
import _v2_io  # noqa: E402
import leakage_guard  # noqa: E402


def _add_root():
    for cand in ([Path(os.environ["SPATIALCPAV17_ROOT"])] if os.environ.get("SPATIALCPAV17_ROOT") else []) \
            + list(Path(__file__).resolve().parents):
        if (cand / "spatialcpav17" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_ROOT = _add_root()


def check_environment():
    if _ROOT is None:
        print("ERROR: could not locate the `spatialcpav17` package.", file=sys.stderr)
        return False
    try:
        import spatialcpav17
        from spatialcpav17 import SpatialCPAv17, SpatialCPAv17Config  # noqa: F401
        try:
            import torch
            tinfo = f"torch {torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})"
        except Exception:
            tinfo = "torch UNAVAILABLE -> decode fallback"
        print(f"spatialcpav17 v{getattr(spatialcpav17, '__version__', '?')} from {_ROOT}; {tinfo}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav17: {e}", file=sys.stderr)
        return False


def _to_dense_f32(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def _prepare_expression(adata):
    """Return (is_counts). Count data is passed RAW (for the NB decoder); fluorescence is
    log-transformed to a Gaussian target; already-normalized data is used as-is (Gaussian)."""
    import scanpy as sc
    et = adata.uns.get("expression_type", "raw_counts")
    if et == "raw_counts":
        return True
    if et in ("fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(adata)
        return False
    if et in ("log1p_normalized", "log2_normalized", "normalized"):
        return False
    return True  # default: treat as counts


def build_stack(adata, ct_all):
    from spatialcpav17 import Slice, SliceStack
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    X = _to_dense_f32(adata.X)
    labels = sorted(np.unique(sections), key=lambda s: np.median(coords[sections == s, 2]))
    slices = [Slice(expression=X[sections == sec], coords_xy=coords[sections == sec, :2],
                    z_values=coords[sections == sec, 2],
                    cell_type_indices=ct_all[sections == sec] if ct_all is not None else None,
                    section_id=str(sec)) for sec in labels]
    return SliceStack(slices)


def run_method(adata, targets, gene_names, is_counts, args):
    from spatialcpav17 import SpatialCPAv17, SpatialCPAv17Config

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, ct_all)
    if sum(s.n_spots for s in stack.slices) < 8 or stack.n_slices < 2:
        print("  SKIP: need >=2 sections and >=8 cells")
        return {}

    cfg = SpatialCPAv17Config()
    cfg.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    cfg.train.pretrain_epochs = args.pretrain_epochs
    cfg.train.device = args.device
    cfg.vae.latent_dim = args.latent_dim
    cfg.vae.hidden = args.hidden
    cfg.vae.kl_weight = args.kl_weight
    if args.likelihood != "auto":
        cfg.vae.likelihood = args.likelihood
    cfg.flow.n_ode_steps = args.ode_steps
    cfg.flow.n_ensemble = args.ensemble
    cfg.generation.retrieval = args.retrieval
    cfg.generation.decode_only = not args.retrieval
    cfg.generation.ground_k = args.ground_k
    cfg.generation.coherent_source = not args.no_coherent_source
    cfg.generation.coherent_freq = args.coherent_freq
    cfg.generation.composition_match = not args.no_composition_match

    print(f"  epochs(A={cfg.train.pretrain_epochs},B={cfg.train.epochs}), "
          f"vae(d={cfg.vae.latent_dim},h={cfg.vae.hidden},kl={cfg.vae.kl_weight}), "
          f"flow(steps={cfg.flow.n_ode_steps},ens={cfg.flow.n_ensemble}), "
          f"is_counts={is_counts}, mode={'retrieval' if args.retrieval else 'decode'}")

    gen = SpatialCPAv17(stack, gene_names=gene_names, cell_type_names=cell_type_names,
                        cfg=cfg, is_counts=is_counts)
    print(f"  model trained: {gen.trained} (likelihood={gen.likelihood})")

    results = {}
    for sec, z in targets:
        print(f"  {sec}: generating (decode) virtual slice at z={z:.2f}...")
        try:
            vs = gen.generate_virtual_slice(z=z)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        n = vs.coords.shape[0]
        if n == 0:
            continue
        print(f"    -> {n} cells synthesized")
        cell_type = vs.cell_type.astype(str) if vs.cell_type is not None else np.array(["NA"] * n)
        results[sec] = {"X": sp.csr_matrix(_to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64), "cell_type": cell_type}
    return results


def main():
    parser = argparse.ArgumentParser(description="SpatialCPA-v17 wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    # Defaults mirror spatialcpav17.config TrainConfig (careful-convergence schedule) — keep in sync.
    parser.add_argument("--epochs", type=int, default=260, help="Phase B flow-matching epochs")
    parser.add_argument("--pretrain-epochs", type=int, default=200, help="Phase A VAE epochs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--latent-dim", type=int, default=32, help="VAE latent dim d")
    parser.add_argument("--hidden", type=int, default=256, help="VAE MLP width")
    parser.add_argument("--kl-weight", type=float, default=1.0e-3, help="beta-VAE KL weight")
    parser.add_argument("--likelihood", default="auto", choices=["auto", "nb", "gaussian"])
    parser.add_argument("--ode-steps", type=int, default=12)
    parser.add_argument("--ensemble", type=int, default=4)
    parser.add_argument("--retrieval", action="store_true",
                        help="ABLATION: retrieve real profiles instead of decoding")
    parser.add_argument("--ground-k", type=int, default=2, help="(retrieval mode) candidates per spot")
    parser.add_argument("--no-coherent-source", action="store_true")
    parser.add_argument("--coherent-freq", type=float, default=4.0)
    parser.add_argument("--no-composition-match", action="store_true")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    print(f"Loading training-only input {args.input}...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = adata.var_names.tolist()
    is_counts = _prepare_expression(adata)

    print(f"Running SpatialCPA-v17 for targets {[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, is_counts, args)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "epochs": args.epochs, "pretrain_epochs": args.pretrain_epochs,
        "latent_dim": args.latent_dim, "hidden": args.hidden, "kl_weight": args.kl_weight,
        "ode_steps": args.ode_steps, "ensemble": args.ensemble,
        "mode": "retrieval" if args.retrieval else "decode", "is_counts": bool(is_counts),
        "vae_nb": True, "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav17_gen")


if __name__ == "__main__":
    main()
