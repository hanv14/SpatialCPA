"""Fit the zero-shot arms with a gene split held out of *every* channel that could leak it.

Two fits per seed, four evaluation arms: ``use_distill`` is a parameter of
:meth:`~spatialcpav25_gen.model.embeddings.TextGroundedEmbedding.forward_zero_shot`, not a
training setting, so ``medcpt`` serves arms A1/A2 and ``lookup`` serves A3/A4.

**The split is applied in three places, and all three are needed.** The pre-flight found that
holding genes out of training alone leaves two other routes into the model:

1. ``gene_pool`` in :func:`~spatialcpav25_gen.model.spatialcpav25_gen.train_ctfflow` — the batches
   never draw a held-out gene, and the same pool reaches ``reconstruct_hidden`` and the SEFL
   consistency losses.
2. ``gene_pool`` on ``TrainingData.build`` — restricts the **retrieval PCA** *and its size
   factor*. The ``zinb-flow`` path conditions on the retrieved neighbours' PCs, so a PCA fitted
   over the whole panel carries the held-out genes in as conditioning. Zeroing the excluded rows
   of the basis is necessary and **not sufficient**: the library size is a sum over every gene, so
   an excluded gene rescales every kept gene's normalised value. Both are closed, and the
   retrieval suite asserts the invariance rather than the mechanism.
3. ``expr_mode`` is pinned to ``zinb-flow`` and **``cross-mix`` is refused outright** — it reads
   ``model.data.counts``, the full training matrix, filtered by cell and never by gene, so it
   would emit the held-out genes verbatim. It is not a zero-shot method but a lookup of the
   answer, and a run that names it exits rather than producing a number.

Usage::

    python scripts/t09_zeroshot_run.py --dataset deep_starmap --seed 2 \\
        --split reports/t09_gene_split_deep.json --arm medcpt \\
        --workdir runs/zeroshot_s2 --fit-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import (
    base_config,
    clamp_config_to_input,
    describe_text_channel,
    embeddings_factory,
    load_training_volume,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True, choices=["medcpt", "lookup"])
    ap.add_argument("--split", required=True, help="the .json written by t09_zeroshot_ceiling.py")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--expr-mode", default="zinb-flow")
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    if args.expr_mode != "zinb-flow":
        raise SystemExit(
            f"--expr-mode={args.expr_mode!r} is refused. `cross-mix` reads the full training "
            "count matrix, filtered by cell and never by gene, so it emits the held-out genes "
            "verbatim: it is a lookup of the answer, not a zero-shot method. The zero-shot "
            "experiment runs under zinb-flow only."
        )

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")

    split = json.loads(Path(args.split).read_text())
    kept = np.asarray(split["kept"], dtype=np.int64)
    held = np.asarray(split["held_out"], dtype=np.int64)

    overrides: dict = {"train_steps": int(args.train_steps), "text_emb_mode": args.arm}
    if args.expr_pca_dim is not None:
        overrides["expr_pca_dim"] = int(args.expr_pca_dim)
    cfg = clamp_config_to_input(base_config(args.seed, **overrides), paths.input)
    cfg = cfg.replace(expr_mode="zinb-flow", prior_mode="correlated")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    if len(kept) + len(held) != volume.n_genes:
        raise SystemExit(
            f"{args.split} splits {len(kept) + len(held)} genes but the volume has "
            f"{volume.n_genes}. The split was written for a different panel."
        )
    if int(cfg.expr_pca_dim) > len(kept):
        raise SystemExit(
            f"Config.expr_pca_dim={cfg.expr_pca_dim} exceeds the {len(kept)} kept genes the "
            "retrieval PCA may use. Lower it with --expr-pca-dim or hold out fewer genes."
        )

    embeddings = embeddings_factory(volume)
    print(
        f"  arm {args.arm}, seed {args.seed}, {cfg.train_steps} steps, config "
        f"{cfg.content_hash()}; split "
        f"{len(kept)} kept / {len(held)} held out (seed {split['seed']}, "
        f"{split['n_bins']}x{split['n_bins']} strata on {split['reference_section']})"
    )
    print("  text channel: " + json.dumps(describe_text_channel(cfg, volume), default=str))
    print(
        "  held out of: training batches (gene_pool), the retrieval PCA and its size factor "
        "(TrainingData.build gene_pool); cross-mix refused"
    )

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint = workdir / f"fit_zeroshot_{args.arm}_seed{args.seed}.pt"

    t0 = time.time()
    data = TrainingData.build(volume, cfg, gene_pool=kept)
    if not np.all(data.index.expression_pcs.basis[held, :] == 0.0):
        raise SystemExit(
            "the retrieval PCA basis is non-zero on held-out genes after TrainingData.build "
            "was given the kept pool. The exclusion did not take; do not use this fit."
        )
    model = CTFFlow(cfg, data, embeddings(cfg), grf_seed=args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        train_ctfflow(
            model,
            cfg,
            steps=int(cfg.train_steps),
            seed=args.seed,
            gene_pool=kept,
            checkpoint=str(checkpoint),
        )
    print(f"\n── zeroshot {args.arm} seed {args.seed} ({cfg.content_hash()}) ──")
    print(f"  fitted in {time.time() - t0:.0f}s -> {checkpoint}")
    if args.fit_only:
        print("  --fit-only; scoring is a separate pass over these checkpoints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
