"""Step 1 of the T06 handoff: does the structured share of *count* variance govern retention?

R12's recorded mechanism — "`mu` is spatially smooth but too flat in **amplitude** for its
structure to survive sampling" — is unsupported: `sd(log mu)` separates the two `deep_starmap`
arms with **no overlap** while `retention_top` overlaps across half the sample, and the residual
association runs backwards 6/6. Amplitude is not sufficient. This asks what is.

**The candidate.** Sampling noise in a ZINB draw is spatially independent, so it dilutes the
conditional mean's autocorrelation in proportion to how much of the *count* variance is
between-cell structure. By the law of total variance, per gene:

    structured_g = Var_cells( E[X | cell] )      = Var_cells( (1 - pi) * mu )
    sampling_g   = mean_cells( Var(X | cell) )
                 = mean_cells( (1-pi) * mu * (1 + mu/theta + pi*mu) )
    s_g          = structured_g / (structured_g + sampling_g)

so `I(counts) ~ I(E[X|cell]) * s`. `I(mu)` is already near-perfect (0.861 at the pilot, above real
tissue's own latent at 0.745), which is why amplitude alone fails to predict retention: `s` depends
on the learned **dispersion** and **zero-inflation** as much as on `Var(mu)`. A model can widen
`theta` and lose structure at any amplitude — **R4's trade, seen from the emission side**.

**`s` is computed on the generated cells**, not on the real section, because `retention_top` is
measured on the generated section and a share computed elsewhere cannot predict it. The latent is
captured from the generation path itself rather than re-derived: `_flow_counts` is patched for the
duration of one call and the capture is asserted to match the emitted counts' shape. If that
private name ever moves the script fails loudly, which is the right failure — a silently
re-implemented chain would measure a different model.

Pre-registered before the first run (`progress/t09_inference_and_calibration.md`):

* **IDENTIFIED** — |Spearman r(s, retention_top)| >= 0.7 over the 12 arm x seed x fold cells, and
  `s` orders the two arms the same way `retention_top` does.
* **NOT IDENTIFIED** — |r| < 0.4.
* **AMBIGUOUS** — between. Report and do not spend.

Usage::

    python scripts/t09_retention_mechanism.py --dataset deep_starmap --holdout paper_2_4_6 \\
        --split reports/t09_gene_split_deep.json --arm medcpt --seed 2 \\
        --workdir runs/zeroshot_s2 --out reports/t09_retention_medcpt_s2.json

    python scripts/t09_retention_mechanism.py --summarise reports/t09_retention_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from spatialcpav25_gen.infer import generate as gen_mod
from spatialcpav25_gen.infer.generate import emitted_counts, generate_section
from spatialcpav25_gen.infer.planes import section_plane
from spatialcpav25_gen.train.select import selection_folds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import embeddings_factory, load_training_volume
from t09_zeroshot_run import arm_config, build_and_fit, load_split
from t10_chain_diagnostic import morans_i, rank_normalize

IDENTIFIED_R = 0.7
NOT_IDENTIFIED_R = 0.4

LEVER_F = 0.50
NO_LEVER_F = 0.20
RETENTION_ENVELOPE_B = 0.1268
"""Thresholds for the conditional-variance decomposition, set from the *envelope* and not from
where the answer is expected to fall.

`RETENTION_ENVELOPE_B` is `retention_top`'s measured across-seed per-fold spread (§4.2d's stricter
construction) over the 12 `deep_starmap` cells. Removing overdispersion entirely takes `s` to
`s / (s + (1-s)(1-f_od))`; at the `medcpt` baseline the resulting retention gain crosses that
envelope between `f_od` 0.2 and 0.3, so 0.20 brackets "undetectable even in the ideal case" and
0.50 brackets "comfortably detectable" (1.9x). Because the calculation assumes overdispersion is
removed *completely*, it is an upper bound: **a NOT A LEVER verdict rules the real experiment out,
not merely in doubt.**"""


def generate_capturing_latent(model, hidden, volume, cfg, seed):
    """Generate a fold and return ``(adata, h)`` — the latent the counts were actually drawn from.

    ``_flow_counts(model, h, gen, cfg, calibration)`` is the single point every emitted count
    passes through, on both the anchored and unanchored paths. It is patched for the duration of
    one call, the original restored in a ``finally``, and the capture checked against the emitted
    counts. Nothing outside this function sees the patch.
    """
    captured: list[torch.Tensor] = []
    original = gen_mod._flow_counts

    def spy(model_, h, gen, cfg_, calibration):
        captured.append(h.detach().clone())
        return original(model_, h, gen, cfg_, calibration)

    gen_mod._flow_counts = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = generate_section(
                model, section_plane(hidden), volume, cfg, seed, exclude_z={float(hidden.z)}
            )
    finally:
        gen_mod._flow_counts = original
    if len(captured) != 1:
        raise SystemExit(
            f"captured {len(captured)} latents from one generate_section call, expected 1. The "
            "generation path has changed shape; do not read the numbers from this run."
        )
    h = captured[0]
    if int(h.shape[0]) != int(adata.n_obs):
        raise SystemExit(
            f"captured a latent with {int(h.shape[0])} rows against {adata.n_obs} emitted cells. "
            "The capture is not the latent these counts came from."
        )
    return adata, h


def structured_share(model, h, cfg, genes) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(s_g, E[X|cell], variance fractions)`` for ``genes``, on the generated cells."""
    idx = torch.from_numpy(np.asarray(genes, dtype=np.int64))
    with torch.no_grad():
        gene_emb = model.embeddings.gene(idx)
        mu, theta, pi = model.decoder(h, gene_emb, model.size_head.size_factor(h))
    mu, theta, pi = mu.numpy(), theta.numpy(), pi.numpy()
    conditional_mean = (1.0 - pi) * mu
    # The three additive terms of the ZINB conditional variance, kept separate: which one
    # carries it decides whether `theta` is a lever on retention at all. If the Poisson floor
    # `(1-pi) mu` dominates, constraining theta cannot move the structured share.
    poisson = (1.0 - pi) * mu
    overdispersion = (1.0 - pi) * mu * mu / theta
    zero_inflation = (1.0 - pi) * pi * mu * mu
    conditional_var = poisson + overdispersion + zero_inflation
    structured = conditional_mean.var(axis=0)
    sampling = conditional_var.mean(axis=0)
    s = np.asarray(structured / np.maximum(structured + sampling, 1e-30), dtype=np.float64)
    total = np.maximum(sampling, 1e-30)
    fractions = {
        "f_poisson": poisson.mean(axis=0) / total,
        "f_overdispersion": overdispersion.mean(axis=0) / total,
        "f_zero_inflation": zero_inflation.mean(axis=0) / total,
    }
    return s, conditional_mean, fractions


def summarise(paths: list[str]) -> int:
    """Apply the pre-registered criteria over every cell in ``paths``."""
    from scipy.stats import spearmanr

    rows = [r for p in paths for r in json.loads(Path(p).read_text())]
    if len(rows) < 4:
        raise SystemExit(f"{len(rows)} cells is too few to correlate; expected 12")
    s = np.array([r["s"] for r in rows])
    ret = np.array([r["retention_top"] for r in rows])
    r = float(spearmanr(s, ret).statistic)
    print(f"{'arm':<8}{'seed':<6}{'fold':<11}{'s':>9}{'retention':>11}{'mean/real':>11}{'draw':>9}")
    for row in sorted(rows, key=lambda x: (x["arm"], x["seed"], x["fold"])):
        print(
            f"{row['arm']:<8}{row['seed']:<6}{row['fold']:<11}{row['s']:>9.4f}"
            f"{row['retention_top']:>11.4f}{row.get('mean_vs_real', float('nan')):>11.4f}"
            f"{row.get('draw_retention', float('nan')):>9.4f}"
        )
    mv = np.array([row.get("mean_vs_real", np.nan) for row in rows])
    if np.isfinite(mv).all():
        print(
            f"\n  retention = (generated mean vs real) x (draw retention). "
            f"mean/real spans {mv.min():.4f}-{mv.max():.4f}."
        )
        print("  ⚠️ If that first factor is low, retention is not a sampling problem and R12's")
        print("     framing — 'mu is spatially smooth, the draw loses it' — is looking at the")
        print("     wrong stage. The 0.861 behind it was the ENCODER's latent on a real section.")
    # --- the conditional-variance decomposition: is `theta` a lever at all? -----------------
    if all("f_overdispersion" in row for row in rows):
        fod = np.array([row["f_overdispersion"] for row in rows])
        print(
            f"\n{'arm':<8}{'seed':<6}{'fold':<11}{'f_poisson':>11}{'f_overdisp':>12}"
            f"{'f_zeroinfl':>12}{'idealised gain':>16}"
        )
        gains = []
        for row in sorted(rows, key=lambda x: (x["arm"], x["seed"], x["fold"])):
            f = row["f_overdispersion"]
            sv, mvr = row["s"], row["mean_vs_real"]
            # theta -> inf removes the overdispersion term entirely. An UPPER BOUND on what a
            # moment-matched theta could buy, holding the conditional mean fixed.
            s_prime = sv / (sv + (1.0 - sv) * (1.0 - f))
            gain = mvr * (s_prime - sv)
            gains.append(gain)
            print(
                f"{row['arm']:<8}{row['seed']:<6}{row['fold']:<11}{row['f_poisson']:>11.4f}"
                f"{f:>12.4f}{row['f_zero_inflation']:>12.4f}{gain:>16.4f}"
            )
        verdict = (
            "THETA IS A LEVER"
            if fod.min() >= LEVER_F
            else "THETA IS NOT A LEVER"
            if fod.max() < NO_LEVER_F
            else "AMBIGUOUS"
        )
        print(f"\n  f_overdispersion over {len(fod)} cells: {fod.min():.4f}-{fod.max():.4f}")
        print(f"  idealised retention gain if theta -> inf: {min(gains):.4f}-{max(gains):.4f}")
        print(
            f"  against the strict per-fold envelope {RETENTION_ENVELOPE_B:.4f}: "
            f"{min(gains) / RETENTION_ENVELOPE_B:.2f}x-{max(gains) / RETENTION_ENVELOPE_B:.2f}x"
        )
        print(f"\n  -> **{verdict}**")
        if verdict == "THETA IS A LEVER":
            print("     Step 2's 24 core-hours are justified.")
        elif verdict == "THETA IS NOT A LEVER":
            print("     Even removing overdispersion entirely moves retention by less than the")
            print("     envelope it would have to clear. Step 2 cannot produce a detectable")
            print("     effect and must not be run.")
        else:
            print("     Between the pre-registered bounds. Report and do not spend.")

    arms = sorted({row["arm"] for row in rows})
    order = {}
    for name, values in (("s", s), ("retention_top", ret)):
        means = {
            a: float(np.mean([v for row, v in zip(rows, values, strict=True) if row["arm"] == a]))
            for a in arms
        }
        order[name] = max(means, key=lambda a: means[a])
        print(
            f"\n  {name} by arm: "
            + ", ".join(f"{a} {means[a]:.4f}" for a in arms)
            + f"  -> higher: {order[name]}"
        )
    agree = len(arms) < 2 or order["s"] == order["retention_top"]
    verdict = (
        "IDENTIFIED"
        if abs(r) >= IDENTIFIED_R and agree
        else "NOT IDENTIFIED"
        if abs(r) < NOT_IDENTIFIED_R
        else "AMBIGUOUS"
    )
    print(f"\n  Spearman r(s, retention_top) over {len(rows)} cells = {r:+.4f}")
    print(f"  arms ordered the same way by both: {agree}")
    print(f"\n  -> **{verdict}**")
    if verdict == "IDENTIFIED":
        print(
            "     The structured share of count variance governs retention. Step 2 (a "
            "moment-matched theta) is justified."
        )
    elif verdict == "NOT IDENTIFIED":
        print(
            "     Dispersion and zero-inflation do not explain retention either. Step 2 is NOT "
            "justified. Report what this rules out before proposing anything further."
        )
    else:
        print("     Between the pre-registered bounds. Report and do not spend.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--summarise", nargs="*", default=None, help="aggregate the per-run .json files"
    )
    ap.add_argument("--split")
    ap.add_argument("--arm", default="medcpt", choices=["medcpt", "lookup"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workdir")
    ap.add_argument("--out")
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    if args.summarise is not None:
        return summarise(args.summarise)
    missing = [f for f in ("split", "seed", "workdir", "out") if getattr(args, f) is None]
    if missing:
        raise SystemExit(f"--{', --'.join(missing)} required unless --summarise is given")

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
        adata, h = generate_capturing_latent(model, hidden, volume, cfg, args.seed + index)
        counts = emitted_counts(adata)[:, kept]
        xy = np.asarray(adata.obsm["xyz"], dtype=np.float64)[:, :2]
        real = np.asarray(hidden.counts.todense(), dtype=np.float64)[:, kept]
        k = int(cfg.metric_knn_k)
        real_i = morans_i(np.asarray(hidden.coords, dtype=np.float64), rank_normalize(real), k)
        top = np.argsort(real_i)[::-1][: int(cfg.metric_marker_genes)]
        counts_top = float(np.median(morans_i(xy, rank_normalize(counts)[:, top], k)))
        real_top = float(np.median(real_i[top]))
        shares, conditional_mean, fractions = structured_share(model, h, cfg, kept[top])
        # Retention is a PRODUCT of two factors and R12's account assumes the first is ~1. The
        # 0.861 behind "mu is spatially smooth" was measured on the ENCODER's latent for a real
        # section — never on the flow's latent at generated positions. One Moran's call separates
        # them, so the run answers something whichever way the correlation goes.
        mean_top = float(np.median(morans_i(xy, rank_normalize(conditional_mean), k)))
        row = {
            "fold": hidden.section_id,
            "arm": args.arm,
            "seed": int(args.seed),
            "config_hash": cfg.content_hash(),
            "n_top_genes": len(top),
            "s": float(np.median(shares)),
            "s_min": float(shares.min()),
            "s_max": float(shares.max()),
            "counts_morans_top": counts_top,
            "real_morans_top": real_top,
            "cond_mean_morans_top": mean_top,
            "retention_top": counts_top / real_top if real_top else float("nan"),
            # retention_top = mean_vs_real * draw_retention, exactly.
            "mean_vs_real": mean_top / real_top if real_top else float("nan"),
            "draw_retention": counts_top / mean_top if mean_top else float("nan"),
            **{k: float(np.median(v)) for k, v in fractions.items()},
        }
        rows.append(row)
        print(
            f"  {hidden.section_id}: s {row['s']:.4f} "
            f"(range {row['s_min']:.4f}-{row['s_max']:.4f})   "
            f"retention {row['retention_top']:.4f} = mean/real {row['mean_vs_real']:.4f} "
            f"x draw {row['draw_retention']:.4f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
