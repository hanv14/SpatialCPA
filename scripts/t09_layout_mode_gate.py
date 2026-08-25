"""T09's ``layout_mode`` gate, re-run on the corrected sampler.

The last outstanding invalidation from ``reports/r11_fix_options.md``. T09 decided
``layout_mode`` inside the merged 18-cell full-budget gate, on the **rejection** sampler that
``reports/r11_envelope.md`` later measured as biased — it drew from ``min(lambda, envelope)``
rather than from ``lambda`` — and the margin it decided on (0.0344) sat *inside* R10's
across-seed envelope (0.0335), so ``hybrid`` shipped on a capability tie-break rather than on a
measurement. This re-runs that leg on the grid sampler.

**One fit per seed, not one per arm.** ``layout_mode`` is read only at generation time:
``sample_layout`` is never called during training, and ``_layout_term`` computes its intensity
at the *real* cells' positions. Measured rather than assumed — fitting the fixture at three
modes with one seed gives **bitwise identical** weights across all 96 parameter and buffer
tensors — so every arm here shares one set of weights and the comparison carries no fit-to-fit
noise at all. T09's own gate refits per cell, which for this gate is 3x the compute for a
strictly noisier contrast; that is worth fixing in ``specs/09`` separately.

**Three seeds**, because one is what made the original call undecidable: ``specs/10`` §3's
repeated-seed rule wants ``Config.claim_min_seeds`` before a claim rests on a number, and the
whole point of this re-run is to say whether the fixture can separate the arms at all.

Scored exactly as the gate scores: :func:`selection_scores` over internal LOSO on *training*
sections (leakage-free by type), six metrics, ranked per metric, median rank per arm.

    python scripts/t09_layout_mode_gate.py --seeds 1 2 3 --steps 2400
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import TrainingVolume, Volume
from spatialcpav25_gen.model.embeddings import EntityEmbeddings
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import fit_repulsion
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow
from spatialcpav25_gen.train.select import METRIC_NAMES, selection_scores
from tests.fixtures.synthetic import make_synthetic_volume
from tests.fixtures.text import fake_text_vecs

MODES = ("field", "hybrid", "resample")


def build_embeddings(cfg: Config, vol: Volume) -> EntityEmbeddings:
    """Deterministic stand-in text vectors — the encoder needs a network (Convention 7).

    Constant across arms, so it cannot affect the contrast; it does mean every number here is
    ablation A3 (``text_emb_mode="lookup"``) rather than the shipped ``medcpt``, exactly as the
    STARmap measurement was.
    """
    return EntityEmbeddings(
        cfg,
        torch.from_numpy(fake_text_vecs(vol.n_genes, cfg.text_dim_in, 1)),
        torch.from_numpy(fake_text_vecs(len(vol.celltype_names), cfg.text_dim_in, 2)),
        None
        if vol.region_names is None
        else torch.from_numpy(fake_text_vecs(len(vol.region_names), cfg.text_dim_in, 3)),
    )


def fit_once(train: TrainingVolume, vol: Volume, cfg: Config, seed: int) -> CTFFlow:
    """Fit the shipped configuration once. The arms differ only after this returns."""
    model = CTFFlow(cfg, TrainingData.build(train, cfg), build_embeddings(cfg, vol), grf_seed=seed)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        train_ctfflow(model, cfg, steps=int(cfg.train_steps), seed=seed)
        if cfg.repulsion:
            model.repulsion = fit_repulsion(train, cfg, seed=seed + 1)
    model.eval()
    print(f"  fit: {cfg.train_steps} steps in {time.time() - t0:.1f}s", flush=True)
    return model


def median_ranks(by_arm: dict[str, dict[str, float]]) -> dict[str, float]:
    """Median rank over the six metrics, 1 = best. The gate's own summary statistic."""
    arms = list(by_arm)
    ranks: dict[str, list[float]] = {a: [] for a in arms}
    for metric in METRIC_NAMES:
        order = sorted(arms, key=lambda a: -by_arm[a][metric])
        for position, arm in enumerate(order, start=1):
            ranks[arm].append(float(position))
    return {a: float(np.median(v)) for a, v in ranks.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--out", default="reports/t09_layout_mode_gate_grid.md")
    args = ap.parse_args(argv)

    base = Config().replace(
        train_steps=args.steps,
        text_emb_mode="lookup",
        layout_sampler="grid",
    )
    print(
        f"layout_mode gate on the grid sampler: seeds {args.seeds}, {base.train_steps} steps, "
        f"prior_mode={base.prior_mode}, expr_mode={base.expr_mode}",
        flush=True,
    )
    vol, _ = make_synthetic_volume(seed=0)
    train, _ = split_holdout(vol, "alternating", 0, base)

    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        model = fit_once(train, vol, base, seed)
        by_arm: dict[str, dict[str, float]] = {}
        for mode in MODES:
            cfg = base.replace(layout_mode=mode)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                by_arm[mode] = selection_scores(model, train, cfg, seed=seed)
            summary = "  ".join(f"{m.split('_')[0]}={by_arm[mode][m]:+.4f}" for m in METRIC_NAMES)
            print(f"    {mode:9s} {summary}", flush=True)
        per_seed[seed] = by_arm

    pooled = {
        mode: {
            m: float(np.median([per_seed[s][mode][m] for s in args.seeds])) for m in METRIC_NAMES
        }
        for mode in MODES
    }
    spread = {
        mode: {m: float(np.ptp([per_seed[s][mode][m] for s in args.seeds])) for m in METRIC_NAMES}
        for mode in MODES
    }
    ranks_pooled = median_ranks(pooled)
    ranks_per_seed = {s: median_ranks(per_seed[s]) for s in args.seeds}

    envelope = 0.0335
    lines = [
        "# T09's `layout_mode` gate, re-run on the grid sampler",
        "",
        f"Synthetic fixture, `alternating` holdout, internal LOSO over training sections, "
        f"{base.train_steps} steps, seeds {args.seeds}, `layout_sampler=grid`.",
        "",
        "**One fit per seed.** `layout_mode` is read only at generation time, and fitting the",
        "fixture at all three modes with one seed gives bitwise identical weights over all 96",
        "parameter and buffer tensors — so the three arms below share one model and the contrast",
        "carries no fit-to-fit noise. `text_emb_mode=lookup` (ablation A3): the encoder needs a",
        "network. It is constant across arms and cannot affect the contrast.",
        "",
        "## Median over seeds, per metric (higher is better)",
        "",
        "| arm | " + " | ".join(f"`{m}`" for m in METRIC_NAMES) + " | median rank |",
        "|---" * (len(METRIC_NAMES) + 2) + "|",
    ]
    for mode in MODES:
        lines.append(
            f"| `{mode}` | "
            + " | ".join(f"{pooled[mode][m]:+.4f}" for m in METRIC_NAMES)
            + f" | **{ranks_pooled[mode]:.1f}** |"
        )
    lines += [
        "",
        "## Across-seed spread (max minus min over the seeds)",
        "",
        "| arm | " + " | ".join(f"`{m}`" for m in METRIC_NAMES) + " | max |",
        "|---" * (len(METRIC_NAMES) + 2) + "|",
    ]
    for mode in MODES:
        worst = max(spread[mode].values())
        lines.append(
            f"| `{mode}` | "
            + " | ".join(f"{spread[mode][m]:.4f}" for m in METRIC_NAMES)
            + f" | **{worst:.4f}** |"
        )
    lines += [
        "",
        f"R10's measured across-seed envelope is **{envelope}**. A gap below it is not a gap.",
        "",
        "## Median rank per seed",
        "",
        "| seed | " + " | ".join(f"`{m}`" for m in MODES) + " |",
        "|---" * (len(MODES) + 1) + "|",
    ]
    for s in args.seeds:
        lines.append(f"| {s} | " + " | ".join(f"{ranks_per_seed[s][m]:.1f}" for m in MODES) + " |")
    lines += [
        "",
        "## Pairwise gaps against `resample`, per metric (median over seeds)",
        "",
        "| metric | `field` less `resample` | `hybrid` less `resample` | vs envelope |",
        "|---|---|---|---|",
    ]
    for m in METRIC_NAMES:
        f_gap = pooled["field"][m] - pooled["resample"][m]
        h_gap = pooled["hybrid"][m] - pooled["resample"][m]
        big = max(abs(f_gap), abs(h_gap))
        lines.append(f"| `{m}` | {f_gap:+.4f} | {h_gap:+.4f} | {big / envelope:.1f}x |")

    text = "\n".join(lines)
    print()
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "steps": int(base.train_steps),
                "seeds": list(args.seeds),
                "layout_sampler": base.layout_sampler,
                "prior_mode": base.prior_mode,
                "expr_mode": base.expr_mode,
                "text_emb_mode": base.text_emb_mode,
                "per_seed": {str(s): per_seed[s] for s in args.seeds},
                "pooled_median": pooled,
                "across_seed_spread": spread,
                "median_rank_pooled": ranks_pooled,
                "median_rank_per_seed": {str(s): ranks_per_seed[s] for s in args.seeds},
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
