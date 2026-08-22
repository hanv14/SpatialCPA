#!/usr/bin/env python
"""Why the starved sampler places 24x fewer points than its own acceptance bound predicts.

``reports/r11_budget.md`` measured that the interaction is not the cause -- with repulsion
disabled ``section_4`` places 152 instead of 146, six cells out of a six-thousand shortfall --
and that the mid-plane acceptance bound predicts ~56% placement where 2.4% happens. That leaves
the envelope itself. ``sample_layout`` sets it as ``max(lam) * slack`` over a single MC draw of
``Config.layout_n_mc`` points, and a *sampled maximum* of a field whose max/mean is in the
hundreds is not a stable quantity: whether the draw lands on the spike decides the envelope, and
the envelope divides the acceptance rate.

Measure that directly: draw the envelope MC sample many times and report the spread of
``max(lam)``, the implied acceptance, and how far each lands from the acceptance the run actually
achieved. No fit -- the saved coupling models are on disk::

    python scripts/t10_r11_envelope.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark-pbya-v3" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.infer.generate import plane_at_z
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import intensity_fn_from_head, uniform_plane_points
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData
from t10_chain_diagnostic import build_embeddings, load_training_volume
from t10_r11_coupling import SEED, TARGETS

N_DRAWS = 24


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["runs/pilot/model_r11_softplus_2400.pt", "runs/pilot/model_r11_exp_2400.pt"],
        help="Checkpoints to measure. Each supplies its own decoder_mu_link, so the arm label "
        "is read from the config rather than assumed. Default: the two R11 coupling fits.",
    )
    args = ap.parse_args(argv)
    rows = []
    for path in args.models:
        ckpt = torch.load(path, map_location="cpu")
        cfg = Config(**ckpt["config"])
        link = str(cfg.decoder_mu_link)
        vol = load_training_volume(cfg)
        model = CTFFlow(
            cfg, TrainingData.build(vol, cfg), build_embeddings(cfg, vol), grf_seed=SEED
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        fn = intensity_fn_from_head(model.intensity, model.field)
        slack = float(cfg.layout_envelope_slack)
        for name, z in TARGETS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BBoxClampWarning)
                plane = plane_at_z(vol, z, cfg)
                # A reference mean from a sample far larger than layout_n_mc: the numerator of
                # the acceptance rate is a mean and converges, unlike the maximum.
                big = uniform_plane_points(
                    plane, 16 * int(cfg.layout_n_mc), np.random.default_rng(0)
                )
                lam_big = np.asarray(fn(plane.to_xyz(big))).sum(axis=1)
                mean_ref = float(lam_big.mean())
                maxima = []
                for s in range(N_DRAWS):
                    uv = uniform_plane_points(
                        plane, int(cfg.layout_n_mc), np.random.default_rng(1000 + s)
                    )
                    maxima.append(float(np.asarray(fn(plane.to_xyz(uv))).sum(axis=1).max()))
            m = np.asarray(maxima)
            accept = mean_ref / (m * slack)
            rows.append(
                {
                    "link": link,
                    "section": name,
                    "mean": mean_ref,
                    "max_ref": float(lam_big.max()),
                    "max_med": float(np.median(m)),
                    "max_min": float(m.min()),
                    "max_max": float(m.max()),
                    "spread": float(m.max() / max(m.min(), 1e-12)),
                    "acc_med": float(np.median(accept)),
                    "acc_min": float(accept.min()),
                    "acc_max": float(accept.max()),
                }
            )
            r = rows[-1]
            print(
                f"{link:>8} {name}: mean={r['mean']:.4g} "
                f"max over {N_DRAWS} draws of {cfg.layout_n_mc}: "
                f"min={r['max_min']:.4g} med={r['max_med']:.4g} max={r['max_max']:.4g} "
                f"(spread {r['spread']:.1f}x, 16x sample sees {r['max_ref']:.4g}) "
                f"=> acceptance {r['acc_min']:.3%}..{r['acc_max']:.3%}",
                flush=True,
            )

    out = Path("reports/r11_envelope.md")
    lines = [
        "# The envelope is a sampled maximum, and that is the starved sampler's real cause",
        "",
        f"`sample_layout` sets `envelope = max(lam) * slack` over **one** draw of "
        f"`Config.layout_n_mc` points. Below: {N_DRAWS} independent draws of that same size, the "
        "spread of the maximum they find, and the acceptance rate each would imply against a "
        "reference mean taken from a 16x larger sample (a mean converges; a maximum does not).",
        "",
        "| link | section | mean lam | max: min / median / max | spread | implied acceptance |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['link']}` | {r['section']} | {r['mean']:.4g} | "
            f"{r['max_min']:.4g} / {r['max_med']:.4g} / {r['max_max']:.4g} | "
            f"**{r['spread']:.1f}x** | {r['acc_min']:.3%} .. {r['acc_max']:.3%} |"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
