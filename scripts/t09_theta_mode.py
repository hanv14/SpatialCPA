"""Step 2 of the T06 handoff: does constraining `theta` per gene recover retention?

Step 1 localised the loss. `mu`'s spatial pattern is fine — the generated conditional mean is
2.65-2.85x *more* autocorrelated than the real section's counts — and the structure is thrown
away by the count draw, because only 9-19% of the emitted count variance is between-cell
structure. The decomposition named the term responsible: `mu^2/theta` carries **57-61%** of the
ZINB conditional variance on **both** arms (0.600 vs 0.596, overlapping ranges, two trainings
with different embeddings), against 29-34% for the Poisson floor. That arm-independence is what
makes it a property of the decoder rather than of one fit.

**The change under test.** `Config.decoder_theta_mode="moment_matched"` replaces the learned
per-*(cell, gene)* dispersion with a fixed per-gene vector estimated once from the training
counts. What it removes is the **per-cell degree of freedom** — the optimiser's ability to widen
dispersion cell by cell to absorb between-cell structure it failed to predict, which is R4's
trade seen from the emission side. It does *not* set `theta` to a value chosen to make the answer
come out: the estimator is the marginal moment match, which over-states the variance (it contains
the structure) and so under-states `theta`. `Config.decoder_theta_mode`'s docstring carries that
argument in full.

**Pre-registered before the field was written** (`progress/t09_inference_and_calibration.md`,
"Step 2 — test it causally"), and unchanged since:

* **ANSWERED** — `retention_top` rises by more than the shared across-seed envelope under **both**
  constructions of §4.2d, with signs agreeing on every seed and every fold.
* **NOT ANSWERED** — the rise is inside either envelope, or retention falls.
* **UNINFORMATIVE, and checked first** — `I(mu)` on the constrained fit drops below **0.90x** the
  baseline's, or generation trips `assert_detection_rate`. No retention number from such a fit
  may be read.

⚠️ **The UNINFORMATIVE clause is amended, before any fit was run and with the reason stated.**
As pre-registered it also fired when reconstruction NLL degraded by more than the baseline's own
across-seed spread. Applying the fifth rule — *before pre-registering a test, ask what would have
to be true for it to fail* — that clause fails almost by construction: a constraint that removes a
degree of freedom will cost some likelihood, the baseline's across-seed NLL spread is a few
thousandths, and **a likelihood that gets slightly worse while spatial fidelity improves is R4's
signature, i.e. exactly the outcome this experiment exists to find.** The clause as written would
have declared the finding uninformative. It is replaced by `assert_detection_rate`, which is a
genuine breakage test on the same side of the model and is already enforced in code. **NLL is
still measured and still reported on every fit** — it is no longer a criterion. A dry run of
`--summarise` on fabricated rows is what surfaced this; nothing measured was seen first.

**The diagnostic, and what it is not.** The **`learned` baseline** fit — run it first — also
reports `theta_learned` against `theta_moment_matched`; the constrained fit cannot carry it,
because there the decoder returns the fixed vector and every number would be zero by
construction. If the learned dispersion is already near-constant within a gene and near
the moment estimate, the constraint changes nothing and a null result is a null *change*, not a
null *effect*. It is reported, not thresholded. Making it a gate now — after A LEVER came back
the way the plan wanted — would be a threshold moved after a result I liked, which is the failure
this campaign has already recorded five times.

Full panel, no gene split: the six zero-shot checkpoints are gene-split fits and are not a clean
baseline. A fixed per-gene dispersion also has no legitimate value for a gene held out of the
panel, which is why `ZINBDecoder` raises rather than inventing one.

Usage — **one fit first; the remaining five only after its measured cost is reported**::

    python scripts/t09_theta_mode.py --dataset deep_starmap --holdout paper_2_4_6 \\
        --theta-mode learned --seed 2 --workdir runs/theta_s2 \\
        --out reports/t09_theta_learned_s2.json

    python scripts/t09_theta_mode.py ... --theta-mode moment_matched --seed 2 ...

    python scripts/t09_theta_mode.py --summarise reports/t09_theta_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from spatialcpav25_gen.model.expression import ExpressionError, gene_theta_moments
from spatialcpav25_gen.model.spatialcpav25_gen import TrainingData
from spatialcpav25_gen.train.select import selection_folds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import embeddings_factory, load_training_volume
from t09_retention_mechanism import generate_capturing_latent, structured_share
from t09_zeroshot_run import arm_config, build_and_fit
from t10_chain_diagnostic import morans_i, rank_normalize

from spatialcpav25_gen.infer.generate import emitted_counts  # isort: skip

I_MU_FLOOR = 0.90
"""`I(mu)` on the constrained fit, as a fraction of the baseline's, below which the fit is
UNINFORMATIVE. Pre-registered; it is a check that the change did not break what already worked,
not a criterion on the effect."""


def theta_diagnostic(
    model, data, cfg, *, n_cells: int = 4096, chunk: int = 128, seed: int = 0
) -> dict:
    """`theta_learned` vs `theta_moment_matched`, per gene. Reported, never thresholded.

    The learned dispersion is read off the **reconstruction** path on real training cells — the
    path the optimiser actually shaped — over a deterministic subsample of at most ``n_cells``.
    Three numbers matter and each answers a different question:

    * ``within_gene_sd_log`` — median over genes of `sd_cells(log theta)`. **How much per-cell
      freedom the constraint removes.** Zero means the learned head had already settled on one
      dispersion per gene and `moment_matched` is a no-op.
    * ``log_ratio_median`` — median over genes of `log(median_cells(theta) / theta_moment)`.
      **How far the two values sit apart.** Near zero means the fixed value is where the head
      had gone anyway.
    * ``log_ratio_spearman`` — do the two orderings of genes agree at all.

    ``theta_marginal_median`` and ``log_ratio_marginal_median`` report the estimator the field
    **does not** use — the marginal moment, which the fixture measured as biased the wrong way —
    so the size of that choice is visible next to the result it feeds.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    n = int(data.n_cells)
    rows = np.sort(rng.choice(n, size=min(n_cells, n), replace=False))
    genes = np.arange(int(data.stats.n_genes), dtype=np.int64)
    # Chunked over cells, and that is not a nicety: the decoder trunk materialises
    # (N, G, decoder_hidden), so 4096 cells x a 1017-gene panel x 256 is ~4 GB in one call.
    gene_rows = torch.from_numpy(genes)
    blocks = []
    with torch.no_grad():
        gene_emb = model.embeddings.gene(gene_rows)
        for start in range(0, len(rows), int(chunk)):
            block = rows[start : start + int(chunk)]
            counts = torch.from_numpy(np.asarray(data.counts[block].todense(), dtype=np.float32))
            totals = torch.from_numpy(np.asarray(data.total_counts[block], dtype=np.float32))
            size = totals / float(data.stats.median_total)
            h1 = model.encoder(counts, gene_emb, size)
            _, theta, _ = model.decoder(h1, gene_emb, size, gene_rows)
            blocks.append(theta.numpy().astype(np.float64))
    learned = np.concatenate(blocks, axis=0)
    groups = np.concatenate(
        [np.asarray(sec.cell_type, dtype=np.int64) for sec in data.vol.sections]
    )
    moment = gene_theta_moments(
        data.counts,
        data.total_counts,
        cfg,
        reference_total=data.stats.median_total,
        groups=groups,
    )
    # Reported alongside, never used: the estimator the field does NOT use, so the size of that
    # choice is visible in the same table as the result it feeds.
    marginal = gene_theta_moments(
        data.counts, data.total_counts, cfg, reference_total=data.stats.median_total
    )
    log_learned = np.log(np.maximum(learned, 1e-12))
    per_gene = np.median(learned, axis=0)
    ratio = np.log(np.maximum(per_gene, 1e-12)) - np.log(np.maximum(moment, 1e-12))
    return {
        "theta_marginal_median": float(np.median(marginal)),
        "log_ratio_marginal_median": float(
            np.median(np.log(np.maximum(per_gene, 1e-12)) - np.log(np.maximum(marginal, 1e-12)))
        ),
        "n_cells_read": len(rows),
        "n_genes": len(genes),
        "within_gene_sd_log": float(np.median(log_learned.std(axis=0))),
        "within_gene_sd_log_p90": float(np.quantile(log_learned.std(axis=0), 0.90)),
        "theta_learned_median": float(np.median(per_gene)),
        "theta_moment_median": float(np.median(moment)),
        "log_ratio_median": float(np.median(ratio)),
        "log_ratio_iqr": float(np.subtract(*np.quantile(ratio, [0.75, 0.25]))),
        "log_ratio_spearman": float(spearmanr(per_gene, moment).statistic),
        "theta_moment_at_max": float(np.mean(moment >= float(cfg.zinb_theta_max) * (1 - 1e-9))),
    }


def reconstruction_nll(model, data, cfg, *, seed: int, steps: int = 8) -> float:
    """Mean reconstruction NLL over ``steps`` deterministic batches. The UNINFORMATIVE check.

    The training path's own term, read through ``forward_train`` rather than re-implemented, so
    it is the quantity the optimiser minimised and not a near-relative of it.
    """
    values = []
    with torch.no_grad():
        for step in range(steps):
            batch = data.sample_batch(cfg, seed=seed, step=step)
            values.append(float(model.forward_train(batch)["recon"]))
    return float(np.mean(values))


def summarise(paths: list[str]) -> int:
    """Apply the pre-registered criteria across the baseline and variant fits."""
    rows = [r for p in paths for r in json.loads(Path(p).read_text())]
    modes = sorted({r["theta_mode"] for r in rows})
    print(
        f"{'mode':<16}{'seed':<6}{'fold':<11}{'retention':>11}{'I(mu)':>9}"
        f"{'NLL':>10}{'s':>9}{'f_od':>8}"
    )
    for row in sorted(rows, key=lambda x: (x["theta_mode"], x["seed"], x["fold"])):
        head = f"{row['theta_mode']:<16}{row['seed']:<6}{row['fold']:<11}"
        if row.get("detection_rate_failed"):
            print(f"{head}{'generation refused (assert_detection_rate)':>49}")
            continue
        print(
            f"{head}{row['retention_top']:>11.4f}{row['cond_mean_morans_top']:>9.4f}"
            f"{row['recon_nll']:>10.4f}{row['s']:>9.4f}{row['f_overdispersion']:>8.4f}"
        )
    if len(modes) < 2:
        print(f"\n  only {modes} present — the comparison needs both modes. No verdict.")
        return 0

    def cells(mode: str, key: str) -> np.ndarray:
        return np.array(
            [
                r[key]
                for r in rows
                if r["theta_mode"] == mode and not r.get("detection_rate_failed")
            ],
            dtype=np.float64,
        )

    base, var = "learned", "moment_matched"
    # --- UNINFORMATIVE first, as pre-registered ------------------------------------------
    i_mu = float(np.mean(cells(var, "cond_mean_morans_top"))) / float(
        np.mean(cells(base, "cond_mean_morans_top"))
    )
    nll_base, nll_var = cells(base, "recon_nll"), cells(var, "recon_nll")
    nll_spread = float(nll_base.max() - nll_base.min())
    nll_gap = float(nll_var.mean() - nll_base.mean())
    print(f"\n  I(mu) variant / baseline = {i_mu:.4f}   (floor {I_MU_FLOOR})")
    print(
        f"  recon NLL {nll_base.mean():.4f} -> {nll_var.mean():.4f} "
        f"({nll_gap:+.4f}); the baseline's own across-seed spread is {nll_spread:.4f}"
    )
    print("  NLL is reported, not a criterion: a likelihood that worsens while spatial fidelity")
    print("  improves is R4's signature, not a broken fit. See this file's header.")
    broken = [r for r in rows if r.get("detection_rate_failed")]
    if broken:
        print(
            f"  {len(broken)} fit(s) tripped assert_detection_rate: "
            + ", ".join(f"{r['theta_mode']}/seed{r['seed']}" for r in broken)
        )
    if i_mu < I_MU_FLOOR or broken:
        print("\n  -> **UNINFORMATIVE**")
        print("     The constraint broke what was already working rather than testing what")
        print("     survives it. No retention number from this run may be read.")
        return 0

    # --- the effect, under BOTH constructions of §4.2d --------------------------------------
    ret_base, ret_var = cells(base, "retention_top"), cells(var, "retention_top")
    delta = float(ret_var.mean() - ret_base.mean())
    by_fold: dict[str, list[float]] = {}
    for r in rows:
        by_fold.setdefault(f"{r['theta_mode']}/{r['fold']}", []).append(r["retention_top"])
    fold_mean_env = float(
        max(np.ptp(np.array(v)) for v in by_fold.values() if len(v) > 1) if len(rows) > 2 else 0.0
    )
    pooled_env = float(np.ptp(ret_base))
    print(f"\n  retention {ret_base.mean():.4f} -> {ret_var.mean():.4f}  (delta {delta:+.4f})")
    print(f"  §4.2d per-fold across-seed envelope: {fold_mean_env:.4f}")
    print(f"  §4.2d pooled across-seed envelope:   {pooled_env:.4f}")
    signs = _signs_agree(rows, base, var)
    print(f"  sign agrees on every seed and fold: {signs}")
    stands = delta > max(fold_mean_env, pooled_env) and signs
    print(f"\n  -> **{'ANSWERED' if stands else 'NOT ANSWERED'}**")
    if stands:
        print("     Dispersion governs retention, and R4 has its first measured instance in")
        print("     the emission model.")
    else:
        print("     Dispersion is not the lever. R12's expression half stays open with one")
        print("     more candidate eliminated. Read the first fit's theta diagnostic before")
        print("     calling this a null effect rather than a null change.")
    return 0


def _signs_agree(rows: list[dict], base: str, var: str) -> bool:
    """True when the variant beats the baseline in every (seed, fold) cell that has both."""
    keyed = {(r["theta_mode"], r["seed"], r["fold"]): r["retention_top"] for r in rows}
    pairs = [
        (v, keyed[(var, s, f)])
        for (m, s, f), v in keyed.items()
        if m == base and (var, s, f) in keyed
    ]
    return bool(pairs) and all(b > a for a, b in pairs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--summarise", nargs="*", default=None)
    ap.add_argument("--theta-mode", choices=["learned", "moment_matched"])
    ap.add_argument("--arm", default="lookup", choices=["medcpt", "lookup"])
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
    missing = [f for f in ("theta_mode", "seed", "workdir", "out") if getattr(args, f) is None]
    if missing:
        raise SystemExit(f"--{', --'.join(missing)} required unless --summarise is given")

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")
    cfg = arm_config(args.arm, args.seed, paths.input, train_steps=args.train_steps)
    cfg = cfg.replace(decoder_theta_mode=args.theta_mode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    # Full panel: no gene is held out, so `kept` is every column and the pool is a no-op. It is
    # passed anyway so this driver goes down the same `build_and_fit` path the campaign used.
    kept = np.arange(volume.n_genes, dtype=np.int64)
    held = np.zeros(0, dtype=np.int64)
    print(f"  {cfg.content_hash()}  decoder_theta_mode = {cfg.decoder_theta_mode!r}")
    print(f"  decoder_mu_link = {cfg.decoder_mu_link!r}  panel = {len(kept)} genes")

    checkpoint = Path(args.workdir) / f"fit_theta_{args.theta_mode}_seed{args.seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = build_and_fit(
        cfg, volume, embeddings_factory(volume), kept, held, seed=args.seed, checkpoint=checkpoint
    )
    fit_seconds = time.monotonic() - started
    print(f"\n  fit finished in {fit_seconds / 3600.0:.2f} h ({fit_seconds:.0f} s)")

    data = TrainingData.build(volume, cfg, gene_pool=kept)
    # Only the baseline can carry it: under `moment_matched` the decoder returns the fixed
    # vector, so `within_gene_sd_log` would be 0 and `log_ratio_median` 0 **by construction**,
    # and reading `head_theta` instead would report a head that received no gradient. Run the
    # `learned` fit first and the diagnostic comes off it.
    diagnostic: dict = {}
    if cfg.decoder_theta_mode == "learned":
        diagnostic = theta_diagnostic(model, data, cfg, seed=args.seed)
        print("\n  theta_learned vs theta_moment_matched (reported, not thresholded):")
        for name, value in diagnostic.items():
            print(f"    {name:<26}{value}")
    else:
        print("\n  theta diagnostic: skipped — it is only meaningful on the learned baseline.")
    nll = reconstruction_nll(model, data, cfg, seed=args.seed)
    print(f"    {'recon_nll':<26}{nll:.4f}")

    rows: list[dict] = []
    for index, hidden in enumerate(selection_folds(volume, cfg)):
        try:
            adata, h = generate_capturing_latent(model, hidden, volume, cfg, args.seed + index)
        except ExpressionError as failure:
            # `assert_detection_rate` is the UNINFORMATIVE trigger. Record it as an outcome
            # rather than dying, so the summariser reports why no retention number exists.
            print(f"  {hidden.section_id}: generation refused - {failure}")
            rows.append(
                {
                    "fold": hidden.section_id,
                    "theta_mode": args.theta_mode,
                    "arm": args.arm,
                    "seed": int(args.seed),
                    "config_hash": cfg.content_hash(),
                    "fit_seconds": fit_seconds,
                    "detection_rate_failed": str(failure),
                    "recon_nll": nll,
                    "theta_diagnostic": diagnostic,
                }
            )
            continue
        counts = emitted_counts(adata)
        xy = np.asarray(adata.obsm["xyz"], dtype=np.float64)[:, :2]
        real = np.asarray(hidden.counts.todense(), dtype=np.float64)
        k = int(cfg.metric_knn_k)
        real_i = morans_i(np.asarray(hidden.coords, dtype=np.float64), rank_normalize(real), k)
        top = np.argsort(real_i)[::-1][: int(cfg.metric_marker_genes)]
        counts_top = float(np.median(morans_i(xy, rank_normalize(counts)[:, top], k)))
        real_top = float(np.median(real_i[top]))
        shares, conditional_mean, fractions = structured_share(model, h, cfg, kept[top])
        mean_top = float(np.median(morans_i(xy, rank_normalize(conditional_mean), k)))
        row = {
            "fold": hidden.section_id,
            "theta_mode": args.theta_mode,
            "arm": args.arm,
            "seed": int(args.seed),
            "config_hash": cfg.content_hash(),
            "fit_seconds": fit_seconds,
            "n_top_genes": len(top),
            "s": float(np.median(shares)),
            "s_min": float(shares.min()),
            "s_max": float(shares.max()),
            "counts_morans_top": counts_top,
            "real_morans_top": real_top,
            "cond_mean_morans_top": mean_top,
            "retention_top": counts_top / real_top if real_top else float("nan"),
            "mean_vs_real": mean_top / real_top if real_top else float("nan"),
            "draw_retention": counts_top / mean_top if mean_top else float("nan"),
            "recon_nll": nll,
            "theta_diagnostic": diagnostic,
            **{name: float(np.median(v)) for name, v in fractions.items()},
        }
        rows.append(row)
        print(
            f"  {hidden.section_id}: retention {row['retention_top']:.4f} = "
            f"mean/real {row['mean_vs_real']:.4f} x draw {row['draw_retention']:.4f}   "
            f"s {row['s']:.4f}  f_od {row['f_overdispersion']:.4f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
