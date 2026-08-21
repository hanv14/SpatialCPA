"""R11: is the layout's cell-count error coupled to the decoder's ``mu`` link?

T10's pilot found the ``exp`` link recovering the emission collapse and, in the same arm, the
layout emitting **48 343 cells for a section whose ground truth has 4 187** — 11.5x, larger than
anything the layout head had been measured doing. The question this answers is whether the two are
connected.

**Structurally they are not directly connected.** ``IntensityHead.forward(xyz, field_feat, region)``
takes coordinates and field features only: no gene embedding, no size factor, no ``mu``, no decoder
output. The link function has no direct path to the intensity integral.

**But they share the anatomical field and one objective.** ``CTFFlow._layout_term`` computes
``lam = self.intensity(cells_data, self.field(cells_model))`` from the *same* triplane the
decoder's conditioning reads, and ``forward_train`` sums ``recon`` and ``layout`` and
backpropagates them together — so the ZINB NLL's gradients reach the shared field, and the
intensity head reads whatever field results. That is an indirect path, and its size is what this
script measures.

Two fits, identical in **every** respect except ``decoder_mu_link``: same seed, same volume, same
budget, same ``ell`` — and therefore the same derived intensity basis, since
``fourier_bands_for_lengthscale`` returns 4 at both 100 and 116.3 µm, so B10's basis fix is not a
confound. Both models are saved.

Reported per held-out section: ``n_expected`` (the intensity integral over the slab), the drawn
count, the ground-truth count, and the ratio — plus the field's own scale, so a change in ``lam``
can be traced to the field rather than to the head.

Usage::

    python scripts/t10_r11_coupling.py --steps 2400
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
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.infer.generate import _layout_on, plane_at_z
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import fit_repulsion
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t10_chain_diagnostic import build_embeddings, load_training_volume

SEED = 1
GROUND_TRUTH = "benchmark-pbya-v3/data/processed/starmap_visual_cortex/data.h5ad"
TARGETS = (("section_2", 30.0), ("section_4", 52.0), ("section_6", 74.0))


def gt_counts() -> dict[str, int]:
    """Ground-truth cell count per held-out section."""
    import anndata as ad

    gt = ad.read_h5ad(GROUND_TRUTH, backed="r")
    try:
        sections = gt.obs["section"].values.astype(str)
        return {s: int((sections == s).sum()) for s, _z in TARGETS}
    finally:
        gt.file.close()


def field_scale(model: CTFFlow, vol) -> dict[str, float]:
    """Summary statistics of the shared triplane field at the training cells.

    The coupling, if any, runs through this field: the decoder's gradients change it and the
    intensity head reads it. Reporting its scale lets a change in ``lam`` be attributed to the
    field rather than to the intensity head's own parameters.
    """
    from spatialcpav25_gen.data.schema import to_xyz

    xyz = np.concatenate([to_xyz(s)[:400] for s in vol.sections], axis=0).astype(np.float32)
    with torch.no_grad():
        feat = model.field(torch.from_numpy(xyz))
    values = feat.numpy()
    return {
        "field_abs_mean": float(np.abs(values).mean()),
        "field_sd": float(values.std()),
        "field_p99_abs": float(np.percentile(np.abs(values), 99)),
    }


def run_arm(link: str, steps: int, save_to: str | None) -> dict:
    """Fit one arm and measure the layout integral on every held-out section."""
    cfg = Config(
        seed=SEED,
        text_emb_mode="lookup",
        train_steps=steps,
        expr_pca_dim=16,
        ell_xy=116.3,
        ell_z=132.0,
        decoder_mu_link=link,
    ).replace(section_key="section", coord_key="spatial", celltype_key="cell_type", region_key=None)

    vol = load_training_volume(cfg)
    model = CTFFlow(cfg, TrainingData.build(vol, cfg), build_embeddings(cfg, vol), grf_seed=SEED)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        train_ctfflow(model, cfg, steps=steps, seed=SEED)
        model.repulsion = fit_repulsion(vol, cfg, seed=SEED + 1)
    wall = time.time() - t0

    truth = gt_counts()
    sections = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, z in TARGETS:
            layout = _layout_on(model, plane_at_z(vol, z, cfg), vol, cfg, SEED)
            n_drawn = int(layout.coords_xyz.shape[0])
            sections.append(
                {
                    "section": name,
                    "n_expected": float(layout.n_expected),
                    "n_drawn": n_drawn,
                    "n_gt": truth[name],
                    "ratio_expected": float(layout.n_expected) / truth[name],
                    "ratio_drawn": n_drawn / truth[name],
                }
            )
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, save_to)
    return {
        "link": link,
        "steps": steps,
        "fit_seconds": wall,
        "intensity_bands": int(model.intensity_bands),
        "field": field_scale(model, vol),
        "sections": sections,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--out", default="reports/r11_coupling.md")
    args = ap.parse_args(argv)

    arms = []
    for link in ("softplus", "exp"):
        print(f"=== arm: decoder_mu_link={link} ===", flush=True)
        arm = run_arm(link, args.steps, f"runs/pilot/model_r11_{link}_{args.steps}.pt")
        print(
            f"  fit {arm['fit_seconds']:.1f}s, intensity_bands={arm['intensity_bands']}", flush=True
        )
        for row in arm["sections"]:
            print(
                f"  {row['section']}: n_expected={row['n_expected']:9.1f} "
                f"drawn={row['n_drawn']:6d} gt={row['n_gt']:5d} "
                f"ratio={row['ratio_drawn']:6.2f}x",
                flush=True,
            )
        arms.append(arm)

    lines = [
        f"# R11 — is the layout's count error coupled to the `mu` link? ({args.steps} steps)",
        "",
        "Two fits identical in every respect except `decoder_mu_link`. `IntensityHead` takes only",
        "coordinates and field features, so there is no direct path; the shared triplane field and",
        "the joint objective are the indirect one.",
        "",
        "| link | section | `n_expected` | drawn | ground truth | ratio |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        for row in arm["sections"]:
            lines.append(
                f"| `{arm['link']}` | {row['section']} | {row['n_expected']:.1f} | "
                f"{row['n_drawn']} | {row['n_gt']} | **{row['ratio_drawn']:.2f}x** |"
            )
    lines += [
        "",
        "| link | intensity bands | field abs-mean | field sd | field p99 abs | fit s |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        f = arm["field"]
        lines.append(
            f"| `{arm['link']}` | {arm['intensity_bands']} | {f['field_abs_mean']:.4f} | "
            f"{f['field_sd']:.4f} | {f['field_p99_abs']:.4f} | {arm['fit_seconds']:.0f} |"
        )
    text = "\n".join(lines)
    print()
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(arms, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
