"""SpatialCPA-v9 (neural flow bridge) — benchmark-pbya-v2 generation-only wrapper.

v9 is the first *learned* SpatialCPA generator: a conditional flow-matching
(rectified-flow / OT-CFM) model transports one flanking slice's cell distribution
to the other in a joint (position, expression-latent) space, conditioned on the
axial gap and a permutation-invariant summary of the neighbouring slices;
integrating the learned ODE to the fractional depth of the target z gives the
virtual slice. It trains on the training slices only and generalizes to any z. If
PyTorch is unavailable or training fails it falls back to the v8 OT morph.

Runs in the ``bench_spatialcpa`` conda env (PyTorch >= 2.0, as used by v4/v5). The
harness invokes the wrapper with only the shared generation-only arguments, so the
defaults in ``SpatialCPAv9Config`` are the production settings; the extra CLI flags
are for ablation / tuning.

Leakage safeguards (shared with the other v2 wrappers): the input file excludes the
held-out section (``guard_no_holdout`` re-checks), labels/embeddings/training use
the training slices only, expression normalization is per-cell, and only the scalar
target z positions the slice.
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


def _add_spatialcpav9_to_path():
    candidates = []
    env_root = os.environ.get("SPATIALCPAV9_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(Path(__file__).resolve().parents)
    for cand in candidates:
        if (cand / "spatialcpav9" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_SPATIALCPAV9_ROOT = _add_spatialcpav9_to_path()


def check_environment():
    if _SPATIALCPAV9_ROOT is None:
        print("ERROR: could not locate the `spatialcpav9` package.", file=sys.stderr)
        return False
    try:
        import spatialcpav9
        from spatialcpav9 import SpatialCPAv9, SpatialCPAv9Config  # noqa: F401
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            tinfo = f"torch {torch.__version__} ({dev})"
        except Exception:
            tinfo = "torch UNAVAILABLE -> OT-morph fallback"
        print(f"spatialcpav9 v{getattr(spatialcpav9, '__version__', '?')} "
              f"imported from {_SPATIALCPAV9_ROOT}; {tinfo}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav9: {e}", file=sys.stderr)
        return False


def _to_dense_f32(X):
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _normalize_expression(adata):
    import scanpy as sc
    expr_type = adata.uns.get("expression_type", "raw_counts")
    if expr_type == "raw_counts":
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    elif expr_type in ("log1p_normalized", "log2_normalized", "normalized"):
        pass
    elif expr_type in ("fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    return expr_type


def build_stack(adata, ct_all):
    from spatialcpav9 import Slice, SliceStack
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    X = _to_dense_f32(adata.X)
    labels = sorted(np.unique(sections), key=lambda s: np.median(coords[sections == s, 2]))
    slices = []
    for sec in labels:
        m = sections == sec
        slices.append(Slice(expression=X[m], coords_xy=coords[m, :2], z_values=coords[m, 2],
                            cell_type_indices=ct_all[m] if ct_all is not None else None,
                            section_id=str(sec)))
    return SliceStack(slices)


def run_method(adata, targets, gene_names, args):
    from spatialcpav9 import SpatialCPAv9, SpatialCPAv9Config

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, ct_all)
    n_cells = sum(s.n_spots for s in stack.slices)
    if n_cells < 8 or stack.n_slices < 2:
        print(f"  SKIP: need >=2 training sections and >=8 cells "
              f"(have {stack.n_slices} sections, {n_cells} cells)")
        return {}

    cfg = SpatialCPAv9Config()
    cfg.seed = args.seed
    cfg.synthesis.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.ae_epochs = args.ae_epochs
    cfg.train.flow_epochs = args.flow_epochs
    cfg.train.device = args.device
    cfg.model.latent_dim = args.latent_dim
    cfg.flow.n_steps = args.flow_steps
    cfg.flow.morph_prior = args.morph_prior
    cfg.synthesis.expression_mode = args.expression_mode
    cfg.embedding.method = args.embedding
    cfg.embedding.fm_gene_embedding_path = args.fm_gene_embedding
    cfg.annotation.enabled = not args.no_annotation
    cfg.annotation.classifier = args.classifier
    cfg.communication.enabled = not args.no_communication

    print(f"  latent_dim={cfg.model.latent_dim}, ae_epochs={cfg.train.ae_epochs}, "
          f"flow_epochs={cfg.train.flow_epochs}, flow_steps={cfg.flow.n_steps}, "
          f"morph_prior={cfg.flow.morph_prior}, expr={cfg.synthesis.expression_mode}")

    gen = SpatialCPAv9(stack, gene_names=gene_names,
                       cell_type_names=cell_type_names, cfg=cfg)

    results = {}
    for sec, z in targets:
        print(f"  {sec}: synthesizing virtual slice at z={z:.2f} (neural flow bridge)...")
        try:
            vs = gen.generate_virtual_slice(z=z)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        n = vs.coords.shape[0]
        if n == 0:
            print(f"    WARNING: 0 cells synthesized for {sec}")
            continue
        print(f"    -> {n} cells synthesized (mode={getattr(gen, '_last_mode', '?')})")
        cell_type = vs.cell_type.astype(str) if vs.cell_type is not None else np.array(["NA"] * n)
        results[sec] = {"X": sp.csr_matrix(_to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64),
                        "cell_type": cell_type}
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SpatialCPA-v9 generation-only wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    parser.add_argument("--ae-epochs", type=int, default=150)
    parser.add_argument("--flow-epochs", type=int, default=400)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=24,
                        help="Euler steps for the inference ODE integration")
    parser.add_argument("--morph-prior", type=float, default=0.5,
                        help="blend fraction of the coherent OT-morph displacement "
                             "(0=pure learned flow, 1=pure OT morph)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--expression-mode", default="source",
                        choices=["source", "nearest", "decode", "blend"],
                        help="source (real profile of the source cell each output "
                             "flowed from; default, robust), nearest (nearest real "
                             "cell in latent space), decode (AE-decode the flowed "
                             "latent), or blend")
    parser.add_argument("--embedding", default="pca", choices=["pca", "fm_gene", "concat"],
                        help="pca (encoder trained from scratch) or fm_gene "
                             "(warm-start the encoder with a pretrained gene embedding)")
    parser.add_argument("--fm-gene-embedding", default=None)
    parser.add_argument("--classifier", default="spatial",
                        choices=["spatial", "prototype", "knn"])
    parser.add_argument("--no-annotation", action="store_true")
    parser.add_argument("--no-communication", action="store_true")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]

    print(f"Loading training-only input {args.input}...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = adata.var_names.tolist()
    _normalize_expression(adata)

    print(f"Running SpatialCPA-v9 (generation-only) for targets "
          f"{[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, args)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "ae_epochs": args.ae_epochs, "flow_epochs": args.flow_epochs,
        "latent_dim": args.latent_dim, "flow_steps": args.flow_steps,
        "morph_prior": args.morph_prior, "expression_mode": args.expression_mode,
        "embedding": args.embedding, "classifier": args.classifier,
        "annotation": not args.no_annotation, "communication": not args.no_communication,
        "learned": True, "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav9_gen")


if __name__ == "__main__":
    main()
