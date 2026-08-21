#!/usr/bin/env python
"""Split R11's count error into its two causes: a wrong integral and a starved sampler.

``reports/r11_coupling.md`` reports ``n_expected`` beside the drawn count, and the two disagree
by two orders of magnitude on ``section_4`` (6203.2 expected, 146 drawn). A Poisson draw cannot
do that, so the shortfall has to be ``_propose_points`` exhausting
``Config.layout_max_proposal_factor``. This measures it rather than inferring it: it re-runs the
layout on the two saved coupling models, captures ``ProposalBudgetWarning`` instead of ignoring
it, and reports the acceptance rate the rejection sampler actually achieved.

No fit — both models are on disk::

    python scripts/t10_r11_budget.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark-pbya-v3" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.infer.generate import _layout_on, plane_at_z
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import (
    ProposalBudgetWarning,
    fit_repulsion,
    intensity_fn_from_head,
    uniform_plane_points,
)
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData
from t10_chain_diagnostic import build_embeddings, load_training_volume
from t10_r11_coupling import SEED, TARGETS, gt_counts


def main() -> int:
    truth = gt_counts()
    rows = []
    for link in ("softplus", "exp"):
        path = f"runs/pilot/model_r11_{link}_2400.pt"
        ckpt = torch.load(path, map_location="cpu")
        cfg = Config(**ckpt["config"])
        vol = load_training_volume(cfg)
        model = CTFFlow(
            cfg, TrainingData.build(vol, cfg), build_embeddings(cfg, vol), grf_seed=SEED
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.repulsion = fit_repulsion(vol, cfg, seed=SEED + 1)
        for name, z in TARGETS:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warnings.simplefilter("ignore", BBoxClampWarning)
                layout = _layout_on(model, plane_at_z(vol, z, cfg), vol, cfg, SEED)
            budget = [w for w in caught if issubclass(w.category, ProposalBudgetWarning)]
            n_drawn = int(layout.coords_xyz.shape[0])
            # The rejection sampler accepts with probability lam(x) / (max(lam) * slack) on the
            # mid-plane, so its acceptance rate is bounded by the field's own dynamic range there.
            # Measure it: this is what decides whether a starved sampler is the envelope's fault
            # or the repulsion's, and the warning blames the repulsion.
            plane = plane_at_z(vol, z, cfg)
            gen_mc = np.random.default_rng(SEED)
            fn = intensity_fn_from_head(model.intensity, model.field)
            uv = uniform_plane_points(plane, int(cfg.layout_n_mc), gen_mc)
            lam_tot = np.asarray(fn(plane.to_xyz(uv))).sum(axis=1)
            dyn = float(lam_tot.max() / max(lam_tot.mean(), 1e-12))
            rows.append(
                {
                    "link": link,
                    "section": name,
                    "n_expected": float(layout.n_expected),
                    "n_drawn": n_drawn,
                    "n_gt": truth[name],
                    "starved": bool(budget),
                    "dynamic_range": dyn,
                    "bound": 1.0 / (dyn * float(cfg.layout_envelope_slack)),
                    "msg": str(budget[0].message).split("\n")[0] if budget else "",
                }
            )
            r = rows[-1]
            print(
                f"{link:>8} {name}: expected={r['n_expected']:9.1f} "
                f"drawn={r['n_drawn']:6d} gt={r['n_gt']:5d} "
                f"integral={r['n_expected'] / r['n_gt']:6.2f}x "
                f"placed={r['n_drawn'] / max(r['n_expected'], 1e-9):6.1%} "
                f"budget_starved={r['starved']} "
                f"max/mean={r['dynamic_range']:8.1f} accept_bound={r['bound']:.4%}",
                flush=True,
            )
            if budget:
                print(f"           {r['msg']}", flush=True)

    out = Path("reports/r11_budget.md")
    lines = [
        "# R11, split in two: a wrong integral and a starved sampler",
        "",
        "No fit — the two saved coupling models re-run their layouts with",
        "`ProposalBudgetWarning` captured rather than ignored.",
        "",
        "`integral` is `n_expected / ground truth`: how wrong the learned intensity's *scale* is.",
        "`placed` is `drawn / n_expected`: how much of what it asked for the rejection sampler",
        "managed to place. They are independent failures and `cell_count_ratio` conflates them.",
        "",
        "| link | section | `n_expected` | drawn | GT | integral | placed | starved "
        "| mid-plane max/mean | acceptance bound |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['link']}` | {r['section']} | {r['n_expected']:.1f} | {r['n_drawn']} | "
            f"{r['n_gt']} | **{r['n_expected'] / r['n_gt']:.2f}x** | "
            f"**{r['n_drawn'] / max(r['n_expected'], 1e-9):.1%}** | "
            f"{'**yes**' if r['starved'] else 'no'} | {r['dynamic_range']:.1f} | "
            f"**{r['bound']:.3%}** |"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
