"""Does `t09_depth_mechanism`'s test do what it claims — and how strong an effect can it see?

The diagnostic makes two claims that a reader should not have to take on trust:

1. the **partial** correlation strips the conditioning confound, so a world where text space
   happens to encode the depth gradient but the per-gene gain depends only on each gene's *own*
   gradient does **not** produce a significant result;
2. the permutation p is calibrated — a world with no structure at all rejects at about the
   nominal rate.

Both are checked here on planted data at the diagnostic's own scale
(``Config.metric_marker_genes`` genes, ``Config.text_diag_knn_k`` neighbours), with the same
functions the diagnostic uses. The third world ("borrowing") is what the hypothesis predicts,
and its rejection rate is the test's **power** — which turns out to be the number that decides
how a null result may be read.

Three worlds, all sharing the same confound structure (``contrast`` correlated with ``trend``):

======================  ==========================================  ====================
world                   how the per-gene gain is generated          what should happen
======================  ==========================================  ====================
``null``                pure noise                                  reject at ~5%
``text-carries-trend``  the gene's **own** trend only, while the     reject at ~5% — the
                        text vectors encode the trend               partial must strip it
``borrowing``           the **neighbours'** trend only              reject as often as
                                                                    the test has power
======================  ==========================================  ====================

Usage::

    python scripts/t09_depth_mechanism_calibration.py --replicates 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from spatialcpav25_gen.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t09_depth_mechanism import neighbour_gradient, partial_spearman

WORLDS = ("null", "text-carries-trend", "borrowing")


def draw(world: str, rng, n_genes: int, k: int, n_folds: int):
    """One planted dataset: ``(gain, trend, contrast, text_vecs)`` at the diagnostic's scale."""
    trend = rng.uniform(0.0, 1.0, n_genes)
    # `contrast` is correlated with `trend` on real profiles, and both condition the per-gene
    # Pearson r. Planting that correlation is what makes the confound world a real test.
    contrast = 0.3 * trend + rng.normal(0.0, 0.1, n_genes)
    text_vecs = (
        np.column_stack([4.0 * trend, rng.normal(0.0, 1.0, (n_genes, 16))])
        if world == "text-carries-trend"
        else rng.normal(0.0, 1.0, (n_genes, 17))
    )
    nbr = neighbour_gradient(text_vecs, trend, k)
    signal = {"null": np.zeros(n_genes), "text-carries-trend": 3.0 * trend, "borrowing": 3.0 * nbr}
    gain = np.mean([signal[world] + rng.normal(0.0, 1.0, n_genes) for _ in range(n_folds)], axis=0)
    return gain, trend, contrast, text_vecs


def reject(gain, trend, contrast, text_vecs, k: int, *, n_perm: int, rng, one_sided: bool) -> float:
    """The permutation p for one planted dataset, by the diagnostic's own statistic."""
    nbr = neighbour_gradient(text_vecs, trend, k)
    controls = np.column_stack([trend, contrast]).T
    rho = partial_spearman(gain, nbr, controls)
    order = np.arange(len(gain))
    null = np.array(
        [
            partial_spearman(
                gain, neighbour_gradient(text_vecs[rng.permutation(order)], trend, k), controls
            )
            for _ in range(n_perm)
        ]
    )
    null = null[np.isfinite(null)]
    if not null.size or not np.isfinite(rho):
        return float("nan")
    return float((null >= rho).mean()) if one_sided else float((np.abs(null) >= abs(rho)).mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--replicates", type=int, default=150)
    ap.add_argument("--permutations", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None, help="write a markdown table here as well as stdout")
    args = ap.parse_args(argv)

    cfg = Config()
    n_genes, k = int(cfg.metric_marker_genes), int(cfg.text_diag_knn_k)
    print(
        f"{n_genes} genes, text kNN {k}, {args.replicates} replicates x {args.permutations} "
        f"permutations, reject at p < {args.alpha}"
    )
    rows = []
    for one_sided in (False, True):
        for n_folds in (1, 2):
            rates = {}
            for world in WORLDS:
                rng = np.random.default_rng(args.seed)
                ps = [
                    reject(
                        *draw(world, rng, n_genes, k, n_folds),
                        k,
                        n_perm=int(args.permutations),
                        rng=rng,
                        one_sided=one_sided,
                    )
                    for _ in range(int(args.replicates))
                ]
                rates[world] = float(np.mean(np.asarray(ps) < args.alpha))
            tag = "one-sided" if one_sided else "two-sided"
            label = "pooled over 2 folds" if n_folds == 2 else "single fold"
            rows.append({"test": tag, "folds": label, **rates})
            print(
                f"  {tag}, {label:<20} " + "  ".join(f"{w} {100 * rates[w]:>3.0f}%" for w in WORLDS)
            )

    # Monte-Carlo standard error on each rate, so a reader can tell 9% from 5% honestly.
    se = 100.0 * np.sqrt(0.05 * 0.95 / args.replicates)
    print(f"\n  Monte-Carlo s.e. on a 5% rate at {args.replicates} replicates: +-{se:.1f} points")
    print(
        "\n  Read: `null` and `text-carries-trend` are the false-positive rates — the second is "
        "the confound the partial exists to strip. `borrowing` is the power."
    )

    if args.out:
        lines = [
            "# `t09_depth_mechanism` — does the test do what it claims?",
            "",
            f"Planted data at the diagnostic's own scale: {n_genes} marker genes, text kNN {k}, "
            f"{args.replicates} replicates x {args.permutations} permutations, seed {args.seed}, "
            f"rejecting at p < {args.alpha}. Monte-Carlo s.e. on a 5% rate: "
            f"+-{100.0 * np.sqrt(0.05 * 0.95 / args.replicates):.1f} points.",
            "",
            "`null` and `text-carries-trend` are **false-positive** rates and should sit near "
            "5%. The second is the confound the partial correlation exists to strip: text space "
            "encodes the depth gradient, but the per-gene gain depends only on each gene's "
            "**own** gradient. `borrowing` is the **power** — the rate at which a real "
            "neighbour effect is detected.",
            "",
            "| test | folds | " + " | ".join(f"`{w}`" for w in WORLDS) + " |",
            "|---|---|" + "---|" * len(WORLDS),
        ]
        lines += [
            f"| {r['test']} | {r['folds']} | "
            + " | ".join(f"{100 * r[w]:.0f}%" for w in WORLDS)
            + " |"
            for r in rows
        ]
        lines += [
            "",
            "**What this licenses.** A significant partial on the real run is informative — the "
            "confound cannot manufacture one. A null is **not** evidence of absence at this "
            "power, and does not replace the repeated-seed run `specs/09` §3 requires.",
        ]
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
