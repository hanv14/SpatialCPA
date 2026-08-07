"""SpatialCPA-v17 manifold-gap diagnostic runner (benchmark-pbya-v2).

Trains v17 on a training-only h5ad and reports the reconstruction/manifold gap between the
flow-decoded and encode-decoded paths on the interior training slices — i.e. whether the flow's
generated latents land where the decoder was trained. Use it to decide whether the Phase-C
decoder fine-tune (--finetune-decoder) is worth adding. This is a diagnostic only: it does NOT
write predictions and is not part of the benchmark run path.

    python src/benchmark/methods/diagnose_spatialcpav17.py --input <train_only.h5ad>
"""

import argparse
import sys
from pathlib import Path

import anndata as ad

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
import leakage_guard  # noqa: E402
import run_spatialcpav17 as w  # reuse build_stack / _prepare_expression / _add_root  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="SpatialCPA-v17 manifold-gap diagnostic")
    parser.add_argument("--input", required=True, help="training-only h5ad")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--kl-weight", type=float, default=1.0e-3)
    parser.add_argument("--likelihood", default="auto", choices=["auto", "nb", "gaussian"])
    parser.add_argument("--finetune-decoder", action="store_true",
                        help="apply Phase-C decoder fine-tune, then diagnose (to confirm gap closure)")
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not w.check_environment():
        sys.exit(1)
    from spatialcpav17 import SpatialCPAv17, SpatialCPAv17Config, reconstruction_gap

    print(f"Loading training-only input {args.input}...")
    adata = ad.read_h5ad(args.input)
    gene_names = adata.var_names.tolist()
    is_counts = w._prepare_expression(adata)

    import numpy as np
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", np.ones(adata.n_obs, dtype=bool), seed=args.seed)
    stack = w.build_stack(adata, ct_all)
    if stack.n_slices < 3:
        print("Need >=3 sections to have an interior slice to diagnose."); sys.exit(1)

    cfg = SpatialCPAv17Config()
    cfg.seed = args.seed; cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs; cfg.train.pretrain_epochs = args.pretrain_epochs
    cfg.train.device = args.device
    cfg.vae.latent_dim = args.latent_dim; cfg.vae.kl_weight = args.kl_weight
    cfg.train.finetune_decoder = args.finetune_decoder
    cfg.train.finetune_epochs = args.finetune_epochs
    if args.likelihood != "auto":
        cfg.vae.likelihood = args.likelihood

    print(f"Training v17 (A={cfg.train.pretrain_epochs},B={cfg.train.epochs}, is_counts={is_counts})...")
    gen = SpatialCPAv17(stack, gene_names=gene_names, cell_type_names=cell_type_names,
                        cfg=cfg, is_counts=is_counts)
    if not gen.trained:
        print("Model fell back (untrained); cannot diagnose the manifold gap."); sys.exit(1)
    print(f"Trained (likelihood={gen.likelihood}). Measuring manifold gap...\n")
    reconstruction_gap(gen, verbose=True)


if __name__ == "__main__":
    main()
