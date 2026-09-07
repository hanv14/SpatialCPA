"""How many cells carry no counts at all on a scored gene pool, and what that does to `n_eff`.

``select._normalised`` library-size normalises **within the scored pool** — the size factor is a
sum over the pool's columns, not the panel's, and deliberately so: a gene outside the pool would
otherwise rescale every gene inside it (the defect closed in ``model/retrieval.py`` and then again
in ``train/select.py``). The consequence nothing reports is that a cell whose counts across the
pool sum to zero gets an all-zero normalised row. It is ``eps``-guarded, so nothing raises; the
row simply carries no information, and every metric scored on that pool is computed over fewer
**effective** cells than the fold contains.

This measures that fraction. It is the instrument for the guess pre-registered in
``progress/t09_inference_and_calibration.md`` (2026-09-07) — that a sparse held-out pool on
``cosmx`` is a candidate for both the 5.6x shrinkage of the held-out A2-A3 effect and A4's 0.2015
across-seed swing, the spread that set the shared envelope and decided both verdicts. **The
thresholds were fixed before this file was written**, and are restated in
:data:`PREREGISTERED_THRESHOLDS` so a reader does not have to take that on trust:

* **SUPPORTS** — ``cosmx`` held-out zero-total fraction >= 0.20 **and** >= 3x ``deep_starmap``'s;
* **RULES OUT** — < 0.05, **or** within 1.5x of ``deep_starmap``'s;
* **INCONCLUSIVE** — anything between.

The comparison is across **datasets**, so this must be run on both and the two reports read
together; the kept pool is the within-dataset control and is measured in the same pass.

⚠️ **No verdict moves on this, whatever it returns.** Part 1 is PARTIAL and Part 2 DOES NOT
REPLICATE under criteria fixed before the fits. An explanation for why an effect was undetectable
is not a licence to re-read it as detected.

**Model-free.** No model, no checkpoint, no embeddings — so ``Config.gene_meta_path`` is never
read and there is no ``--gene-meta`` flag. It reads the real sections' counts and the split.

Usage::

    python scripts/t09_zeroshot_pool_sparsity.py --dataset cosmx --holdout paper_2_4 \\
        --split reports/t09_gene_split_cosmx.json \\
        --out reports/t09_pool_sparsity_cosmx.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import load_training_volume
from t09_zeroshot_run import arm_config, load_split

PREREGISTERED_THRESHOLDS = {
    "supports_min_fraction": 0.20,
    "supports_min_ratio_to_reference": 3.0,
    "rules_out_max_fraction": 0.05,
    "rules_out_max_ratio_to_reference": 1.5,
}
"""Fixed in ``progress/t09_inference_and_calibration.md`` before this script was written."""


def pool_statistics(counts: np.ndarray, pool: np.ndarray) -> dict[str, Any]:
    """Sparsity of one ``(N, G)`` count matrix restricted to one gene pool.

    ``zero_total_fraction`` is the quantity the guess is about: cells whose counts over the pool
    sum to zero, which ``_normalised`` turns into an all-zero row. ``detection_rate_median`` is
    the corroborating statistic — a pool of near-undetected genes attenuates a per-gene
    correlation through a different mechanism, and the two are worth telling apart.
    """
    scored = counts[:, np.asarray(pool, dtype=np.int64)]
    totals = scored.sum(axis=1)
    zero = totals <= 0.0
    detection = (scored > 0.0).mean(axis=0)
    return {
        "n_cells": int(scored.shape[0]),
        "n_genes": int(scored.shape[1]),
        "n_zero_total": int(zero.sum()),
        "zero_total_fraction": float(zero.mean()),
        "n_eff": int((~zero).sum()),
        "detection_rate_median": float(np.median(detection)),
        "detection_rate_min": float(detection.min()),
        "median_pool_total": float(np.median(totals)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True, help="the .json written by t09_zeroshot_ceiling.py")
    ap.add_argument("--out", default=None, help="destination .json")
    ap.add_argument("--seed", type=int, default=2, help="only selects the config; nothing is fit")
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")
    kept, held, _ = load_split(args.split)
    cfg = arm_config(
        "medcpt",
        args.seed,
        paths.input,
        train_steps=args.train_steps,
        expr_pca_dim=args.expr_pca_dim,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    if len(kept) + len(held) != volume.n_genes:
        raise SystemExit(
            f"{args.split} splits {len(kept) + len(held)} genes but the volume has "
            f"{volume.n_genes}. The split was written for a different panel."
        )

    rows: list[dict[str, Any]] = []
    for section in volume.sections:
        counts = np.asarray(section.counts.todense(), dtype=np.float64)
        for name, pool in (("held_out", held), ("kept", kept)):
            row = {
                "section": str(section.section_id),
                "side": name,
                **pool_statistics(counts, pool),
            }
            rows.append(row)
            print(
                f"  {row['section']:12s} {name:9s} {row['n_genes']:4d} genes  "
                f"zero-total {row['n_zero_total']:6d}/{row['n_cells']:6d} = "
                f"{row['zero_total_fraction']:.4f}  n_eff {row['n_eff']:6d}  "
                f"detection median {row['detection_rate_median']:.4f}"
            )

    summary = {}
    for name in ("held_out", "kept"):
        side = [r for r in rows if r["side"] == name]
        cells = sum(r["n_cells"] for r in side)
        zeros = sum(r["n_zero_total"] for r in side)
        summary[name] = {
            "n_cells": cells,
            "n_zero_total": zeros,
            "zero_total_fraction": zeros / cells if cells else 0.0,
            "detection_rate_median": float(np.median([r["detection_rate_median"] for r in side])),
        }

    held_frac = summary["held_out"]["zero_total_fraction"]
    kept_frac = summary["kept"]["zero_total_fraction"]
    ratio = f" ({held_frac / kept_frac:.2f}x)" if kept_frac > 0 else ""
    print(f"\n  pooled: held-out {held_frac:.4f}, kept {kept_frac:.4f}{ratio}")
    print(
        "\n  Pre-registered thresholds are on the CROSS-DATASET comparison and cannot be read "
        "from one run: SUPPORTS needs held-out >= "
        f"{PREREGISTERED_THRESHOLDS['supports_min_fraction']:.2f} AND >= "
        f"{PREREGISTERED_THRESHOLDS['supports_min_ratio_to_reference']:.0f}x deep_starmap's; "
        f"RULES OUT is < {PREREGISTERED_THRESHOLDS['rules_out_max_fraction']:.2f} OR within "
        f"{PREREGISTERED_THRESHOLDS['rules_out_max_ratio_to_reference']:.1f}x of it. Run this on "
        "both datasets and read the two reports together."
    )
    print(
        "  Whatever it shows, no verdict moves: Part 1 is PARTIAL and Part 2 DOES NOT REPLICATE "
        "under criteria fixed before the fits."
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "thresholds": PREREGISTERED_THRESHOLDS,
                    "summary": summary,
                    "sections": rows,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
