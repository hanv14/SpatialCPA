"""T08's two deliverables as one measurement: ablation A2 and the R4 verdict.

Trains four arms on the synthetic fixture — the metric-aware terms off and on, at T06's
1200-step budget and at the 2400 steps where T06 measured the likelihood/fidelity divergence
(open risk R4) — and reports, for each, the statistics both questions are stated on.

The 2400-step pair is the part that 1200 steps cannot answer. R4 is not "is the model good at
1200 steps"; it is "does training longer make the *generated section* worse while the
likelihood improves", and the terms exist to stop that. A comparison at one budget cannot see
it.

    python scripts/t08_metric_report.py [--steps 1200 2400] [--holdout alternating]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import to_xyz
from spatialcpav25_gen.model.expression import detection_rate
from tests.fixtures.synthetic import make_synthetic_volume
from tests.test_expression import (
    donor_baseline,
    expression_cfg,
    frobenius_offdiag,
    gene_correlation,
    generate_counts,
    informative_genes,
    mean_variance_slope,
    train_model,
)
from tests.test_metric_aware import METRIC_OFF, METRIC_ON, reconstruction_scores

SEED = 20260818


def measure(trained, gt, *, seed: int) -> dict[str, float]:
    """Every number the A2 table and the R4 verdict are stated on, for one arm."""
    counts, xyz = generate_counts(trained, seed=seed)
    section = trained.held_out
    real = np.asarray(section.counts.todense(), dtype=np.float64)
    keep = informative_genes(real)
    target = gene_correlation(real, keep)
    baseline = donor_baseline(trained, xyz, seed=seed + 1)
    true_xyz = to_xyz(section).astype(np.float64)
    mu = gt.expression_mu(gt.latent(true_xyz), true_xyz, section.cell_type)
    draws = [gene_correlation(gt.sample_counts(mu, seed=600 + s), keep) for s in range(3)]
    ceiling = float(np.mean([frobenius_offdiag(draw, target) for draw in draws]))
    rate_real = detection_rate(real)
    rate_gen = detection_rate(counts)
    held_in = trained.training.sections[trained.training.n_sections // 2]
    scores = reconstruction_scores(trained.model, held_in, trained.cfg, seed=seed + 9)
    return {
        "recon": float(trained.history.terms["recon"][-1]),
        "frobenius": frobenius_offdiag(gene_correlation(counts, keep), target),
        "frobenius_baseline": frobenius_offdiag(gene_correlation(baseline, keep), target),
        "frobenius_ceiling": ceiling,
        "detection_r": float(np.corrcoef(rate_gen, rate_real)[0, 1]),
        "detection_mad": float(np.abs(rate_gen - rate_real).mean()),
        "mv_slope": mean_variance_slope(counts),
        "mv_slope_real": mean_variance_slope(real),
        **{key: float(value) for key, value in scores.items()},
    }


def main() -> None:
    """Train every arm and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=[1200, 2400])
    parser.add_argument("--holdout", default="alternating")
    args = parser.parse_args()

    _, gt = make_synthetic_volume(seed=0)
    base: Config = expression_cfg()
    rows: dict[str, dict[str, float]] = {}
    for steps in args.steps:
        for arm, override in (("off", METRIC_OFF), ("on", METRIC_ON)):
            name = f"{arm}@{steps}"
            started = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trained = train_model(
                    holdout_mode=args.holdout, steps=steps, cfg=base.replace(**override)
                )
                rows[name] = measure(trained, gt, seed=SEED + 5)
            rows[name]["seconds"] = time.time() - started
            summary = json.dumps({k: round(v, 4) for k, v in rows[name].items()})
            print(f"{name}: {summary}", flush=True)

    header = (
        "| arm | recon | Frobenius | baseline | detection MAD | m-v slope | Moran MAE | depth r |"
    )
    print("\n" + header)
    print("|---|---|---|---|---|---|---|---|")
    for name, row in rows.items():
        print(
            f"| {name} | {row['recon']:.4f} | {row['frobenius']:.3f} | "
            f"{row['frobenius_baseline']:.3f} | {row['detection_mad']:.4f} | "
            f"{row['mv_slope']:.4f} | {row['morans_mae']:.4f} | {row['marker_depth_r']:.4f} |"
        )


if __name__ == "__main__":
    main()
