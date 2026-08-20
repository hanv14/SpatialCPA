"""R10: measure the reproducibility envelope that gate decisions are judged against.

Three representative cells of the merged full-budget gate, each fitted at three seeds:

* the **winner** (``resample`` + ``correlated`` + ``zinb-flow``);
* the **closest rival** (``hybrid`` + ``correlated`` + ``zinb-flow``, 0.0344 from the winner);
* a **far-from-winner** arm (``resample`` + ``correlated`` + ``cross-mix``, 0.2900 away), to
  check whether the envelope is score-dependent rather than constant.

All nine fits run in one process, so the spread is across **seeds** and is not confounded with
the cross-process nondeterminism R10 also observed (0.0120 on one config at a fixed seed).
Both numbers matter and they are different things: the seed spread is what a decision
threshold must clear, the cross-process drift is a separate defect.

Checkpointed per fit to a CSV keyed by (cell, seed); re-running the identical command resumes.

    python scripts/t09_envelope.py [--out reports/envelope_synthetic.csv]
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.train.select import METRIC_NAMES, FitScorer
from tests.fixtures.synthetic import make_synthetic_volume
from tests.test_expression import build_embeddings, expression_cfg

SEEDS = (20260819, 20260820, 20260821)
"""Three seeds, per ``specs/09`` §3's repeated-seed rule. The first is the selection run's."""

CELLS: dict[str, dict[str, str]] = {
    "winner": {"layout_mode": "resample", "prior_mode": "correlated", "expr_mode": "zinb-flow"},
    "closest_rival": {
        "layout_mode": "hybrid",
        "prior_mode": "correlated",
        "expr_mode": "zinb-flow",
    },
    "far_arm": {"layout_mode": "resample", "prior_mode": "correlated", "expr_mode": "cross-mix"},
}


def main() -> None:
    """Fit each cell at each seed, appending every result before the next fit starts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/envelope_synthetic.csv")
    args = parser.parse_args()

    path = Path(args.out)
    done: set[tuple[str, int]] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            done = {(row["cell"], int(row["seed"])) for row in csv.DictReader(handle)}
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["cell", "seed", *METRIC_NAMES])
    print(f"{len(done)} of {len(CELLS) * len(SEEDS)} fits already recorded", flush=True)

    volume, _ = make_synthetic_volume(seed=0)
    base = expression_cfg().replace(
        train_steps=2400,
        selection_n_folds=3,
        selection_passes=2,
        w_autocorr=0.5,
        w_profile=0.5,
        w_distribution=0.5,
    )
    training, _ = split_holdout(volume, "alternating", 0, base)
    scorer = FitScorer(training, lambda cfg: build_embeddings(cfg, volume))

    for name, overrides in CELLS.items():
        for seed in SEEDS:
            if (name, seed) in done:
                continue
            started = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = scorer(base.replace(**overrides), steps=2400, seed=seed)
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [name, seed, *[f"{scores[m]:.6f}" for m in METRIC_NAMES]]
                )
                handle.flush()
            print(
                f"{name},{seed},"
                + ",".join(f"{scores[m]:.4f}" for m in METRIC_NAMES)
                + f"   # {time.time() - started:.0f}s",
                flush=True,
            )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print("\nacross-seed spread (max - min) per cell:")
    for name in CELLS:
        vals = np.array([[float(r[m]) for m in METRIC_NAMES] for r in rows if r["cell"] == name])
        if len(vals) < 2:
            continue
        spread = vals.max(axis=0) - vals.min(axis=0)
        per_metric = "  ".join(f"{m}={v:.4f}" for m, v in zip(METRIC_NAMES, spread, strict=True))
        print(f"  {name:<14} {per_metric}")
        print(f"  {'':<14} max {spread.max():.4f}")


if __name__ == "__main__":
    main()
