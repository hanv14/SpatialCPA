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
        rep = model.repulsion
        for name, z in TARGETS:
            # Ablation A4: the same layout with the Strauss interaction switched off. The
            # envelope is unchanged by it, so whatever the two arms differ by is the
            # interaction's thinning and nothing else. This is the experiment that separates
            # the two candidate causes of a starved sampler.
            cfg_off = cfg.replace(repulsion=False)
            with warnings.catch_warnings(record=True) as caught_off:
                warnings.simplefilter("always")
                warnings.simplefilter("ignore", BBoxClampWarning)
                layout_off = _layout_on(model, plane_at_z(vol, z, cfg_off), vol, cfg_off, SEED)
            n_off = int(layout_off.coords_xyz.shape[0])
            starved_off = any(issubclass(w.category, ProposalBudgetWarning) for w in caught_off)
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
                    "n_drawn_norep": n_off,
                    "starved_norep": starved_off,
                    "r0": float(rep.r0),
                    "R": float(rep.R),
                    "gamma": float(rep.gamma),
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
                f"max/mean={r['dynamic_range']:8.1f} accept_bound={r['bound']:.4%} "
                f"| no_repulsion drawn={r['n_drawn_norep']:6d} "
                f"placed={r['n_drawn_norep'] / max(r['n_expected'], 1e-9):6.1%} "
                f"starved={r['starved_norep']}",
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
        "| link | section | `n_expected` | GT | integral | max/mean | bound | drawn | placed "
        "| drawn, no repulsion | placed | starved on/off |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['link']}` | {r['section']} | {r['n_expected']:.1f} | {r['n_gt']} | "
            f"**{r['n_expected'] / r['n_gt']:.2f}x** | {r['dynamic_range']:.1f} | "
            f"{r['bound']:.3%} | {r['n_drawn']} | "
            f"**{r['n_drawn'] / max(r['n_expected'], 1e-9):.1%}** | {r['n_drawn_norep']} | "
            f"**{r['n_drawn_norep'] / max(r['n_expected'], 1e-9):.1%}** | "
            f"{'yes' if r['starved'] else 'no'} / "
            f"{'yes' if r['starved_norep'] else 'no'} |"
        )
    rep0 = rows[0]
    lines += [
        "",
        "## The fitted Strauss interaction, which the warning does not print",
        "",
        f"`r0` **{rep0['r0']:.3f} um**, `R` **{rep0['R']:.3f} um**, "
        f"`gamma` **{rep0['gamma']:.4f}**. `gamma = 1` is no soft repulsion at all.",
        "",
        "`ProposalBudgetWarning` names `r0` alone. `r0` is the hard core; `R` and `gamma` are the",
        "soft part, and they are what a rejection sampler actually spends its budget on. The",
        "`no repulsion` columns above are ablation A4 on the same weights and the same seed: the",
        "envelope is identical between the two arms, so the difference between them is the",
        "interaction's thinning and nothing else.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
