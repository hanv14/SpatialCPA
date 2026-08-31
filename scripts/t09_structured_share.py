"""R12's headline number, measured on a fit that exists — no refit.

R12 says the decoder carries **15.3%** of its between-cell variance in the structured mean against
real tissue's **62.2%**. Both halves of that record are stale, in opposite directions:

* the **15.3%** is a *pilot* number, measured on tier-1 STARmap under
  ``decoder_mu_link="softplus"`` — the default until 2026-08-21;
* the **61.4%** that "recovered" it came from a saved model under ``exp`` with caveats stated at
  the time: it emitted 48 343 cells against a ground truth of 4 187, and ``sd(log mu)`` was 0.777
  against tissue's 1.213.

Neither is a measurement of what the **shipped** decoder does on a **current** real-data fit. Six
of those exist — the `deep_starmap` zero-shot campaign's checkpoints — and reading the share off
them costs a checkpoint load and a generation, not four hours of training.

What it reports, per checkpoint and per fold:

``share_shape``
    The fraction of ``Var(log mu)`` carried by the latent-driven shape rather than the size
    factor, from ``t10_chain_diagnostic.mu_variance_decomposition`` — the same estimator the
    15.3% and 61.4% came from, so the numbers are comparable to the record.
``counts_morans`` / ``real_morans``
    Median per-gene Moran's I of the emitted counts and of the real section, on the same
    estimator, so the retention question is answered on this fit rather than inherited.

⚠️ **A share measured on the zero-shot fits is a share for a model trained on 80% of the panel.**
The held-out genes never entered a batch, so their decoder behaviour is not evidence about the
shipped configuration. Every number here is computed over the **kept** genes only, and the script
refuses a pool it was not given.

Usage::

    python scripts/t09_structured_share.py --dataset deep_starmap --holdout paper_2_4_6 \\
        --split reports/t09_gene_split_deep.json --arm medcpt --seed 2 \\
        --workdir runs/zeroshot_s2 --out reports/t09_structured_share_deep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from spatialcpav25_gen.infer.generate import emitted_counts, generate_section
from spatialcpav25_gen.infer.planes import section_plane
from spatialcpav25_gen.train.select import selection_folds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import embeddings_factory, load_training_volume
from t09_zeroshot_run import arm_config, build_and_fit, load_split
from t10_chain_diagnostic import morans_i, mu_variance_decomposition


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True)
    ap.add_argument("--arm", default="medcpt", choices=["medcpt", "lookup"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")
    kept, held, _ = load_split(args.split)
    cfg = arm_config(args.arm, args.seed, paths.input, train_steps=args.train_steps)
    checkpoint = Path(args.workdir) / f"fit_zeroshot_{args.arm}_seed{args.seed}.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"no fit at {checkpoint}; this script scores an existing one only")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    print(f"  {cfg.content_hash()}  decoder_mu_link = {cfg.decoder_mu_link!r}")
    model = build_and_fit(
        cfg, volume, embeddings_factory(volume), kept, held, seed=args.seed, checkpoint=checkpoint
    )

    rows = []
    for index, hidden in enumerate(selection_folds(volume, cfg)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = generate_section(
                model,
                section_plane(hidden),
                volume,
                cfg,
                args.seed + index,
                exclude_z={float(hidden.z)},
            )
        counts = emitted_counts(adata)[:, kept]
        xy = np.asarray(adata.obsm["xyz"], dtype=np.float64)[:, :2]
        real = np.asarray(hidden.counts.todense(), dtype=np.float64)[:, kept]
        real_xy = np.asarray(hidden.coords, dtype=np.float64)
        k = int(cfg.metric_knn_k)
        # The latent the encoder makes of the *real* fold, which is what
        # `mu_variance_decomposition` decomposes — the same input the pilot used, so the share
        # here is comparable to R12's record rather than to a differently-conditioned one.
        full = np.asarray(hidden.counts.todense(), dtype=np.float32)
        with torch.no_grad():
            gene_emb = model.embeddings.gene(torch.arange(full.shape[1], dtype=torch.long))
            totals = torch.from_numpy(full.sum(axis=1))
            size_factor = totals / max(float(model.stats.median_total), 1.0)
            h = model.encoder(torch.from_numpy(full), gene_emb, size_factor)
        row = {
            "fold": hidden.section_id,
            "arm": args.arm,
            "seed": int(args.seed),
            "decoder_mu_link": cfg.decoder_mu_link,
            "config_hash": cfg.content_hash(),
            "n_kept_genes": int(kept.size),
            **mu_variance_decomposition(model, h, cfg),
            "counts_morans": float(np.median(morans_i(xy, counts, k))),
            "real_morans": float(np.median(morans_i(real_xy, real, k))),
        }
        row["retention"] = (
            row["counts_morans"] / row["real_morans"] if row["real_morans"] else float("nan")
        )
        rows.append(row)
        print(
            f"  {hidden.section_id}: share_shape {row['share_shape']:.1%}  "
            f"sd(log mu) {row['sd_log_mu']:.3f}  counts I {row['counts_morans']:+.4f}  "
            f"real I {row['real_morans']:+.4f}  retention {row['retention']:.1%}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    print(
        "\nR12's record for comparison: structured share 15.3% (tier-1 pilot, softplus) and "
        "61.4% (saved model, exp, caveated); tissue 62.2%."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
