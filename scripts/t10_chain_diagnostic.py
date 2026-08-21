"""Where does the spatial structure go? — the generation chain, stage by stage.

T10's pilot found the generated field carrying 15-22% of STARmap's per-gene Moran's I, with
calibration and the layout head both ruled out by direct experiment (``reports/pilot.md`` §6).
That leaves a whole-pipeline number and no location. This script measures autocorrelation at
each link of the chain ``generate_section`` actually runs, so the loss can be attributed to the
prior, the flow, or the decoder rather than to "the expression path".

The chain, mirroring ``infer/generate.py::_expression`` under ``expr_mode="zinb-flow"``::

    h0 = model.prior_latent(xyz, seed)          # the 3-D GRF, at the generated positions
    h  = model.flow.sample(h0, cond, ode_steps) # the latent after the flow
    counts = decode(h) then sample              # the emitted counts

with two references measured on the *real* held-out section:

    real counts                                  # what the tissue has
    h1 = model.encoder(real counts, ...)         # what the encoder makes of it

Reading it:

* ``h0`` low        -> the prior is not delivering structure at these positions at all.
* ``h0`` high, ``h`` low  -> the flow destroys it.
* ``h`` matches ``h1``, counts low -> the decoder destroys it.

Every stage uses the SAME estimator — bench3's row-standardised kNN Moran's I, at
``Config.metric_knn_k`` — so the numbers are comparable down the chain. Latents are compared
per dimension and counts per gene; both are summarised by the median over channels, because a
mean over channels is dominated by whichever few carry the most variance.

Usage::

    python scripts/t10_chain_diagnostic.py --steps 1200
    python scripts/t10_chain_diagnostic.py --steps 2400 --out reports/chain_2400.md
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
from scipy.spatial import cKDTree
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import load_volume
from spatialcpav25_gen.data.schema import TrainingVolume
from spatialcpav25_gen.infer.generate import plane_at_z
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import fit_repulsion
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

SEED = 1
INPUT = "benchmark-pbya-v3/results/_inputs/starmap_visual_cortex/paper_2_4_6/train_registered.h5ad"
GROUND_TRUTH = "benchmark-pbya-v3/data/processed/starmap_visual_cortex/data.h5ad"


def morans_i(xy: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Per-column Moran's I on a row-standardised kNN graph. ``(N, 2)``, ``(N, C)`` -> ``(C,)``.

    Reimplemented here rather than imported from bench3 because this runs on latents, which
    never leave the package — but it is the same estimator: row-standardised kNN weights,
    self excluded. ``tests/conftest.py::morans_i`` is the reference it matches.
    """
    n = xy.shape[0]
    k = min(int(k), n - 1)
    idx = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]
    x = np.asarray(values, dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    denom = (xc**2).sum(axis=0)
    lagged = xc[idx].mean(axis=1)
    numer = (xc * lagged).sum(axis=0)
    out = np.full(x.shape[1], np.nan)
    ok = denom > 0
    out[ok] = numer[ok] / denom[ok]
    return out


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Per-column average ranks, as the scoreboard does before every spatial metric."""
    from scipy.stats import rankdata

    return np.column_stack([rankdata(col, method="average") for col in np.asarray(x).T])


def summarise(name: str, xy: np.ndarray, values: np.ndarray, k: int) -> dict[str, float]:
    """Median / IQR of per-channel Moran's I, plus the channel count."""
    i = morans_i(xy, values, k)
    finite = i[np.isfinite(i)]
    return {
        "stage": name,
        "median_I": float(np.median(finite)) if finite.size else float("nan"),
        "p25": float(np.percentile(finite, 25)) if finite.size else float("nan"),
        "p75": float(np.percentile(finite, 75)) if finite.size else float("nan"),
        "n_channels": int(finite.size),
    }


def load_training_volume(cfg: Config) -> TrainingVolume:
    """bench3's training-only input as a ``TrainingVolume`` (the wrapper's own path)."""
    import anndata as ad

    adata = ad.read_h5ad(INPUT)
    tmp = Path(INPUT).with_suffix(".chain.h5ad")
    adata.write_h5ad(tmp)
    try:
        vol = load_volume(tmp, cfg, flattened_sections=True)
    finally:
        tmp.unlink(missing_ok=True)
    return TrainingVolume(
        specimen_id=vol.specimen_id,
        sections=vol.sections,
        gene_names=vol.gene_names,
        celltype_names=vol.celltype_names,
        region_names=vol.region_names,
        flattened_sections=vol.flattened_sections,
    )


def build_embeddings(cfg: Config, vol: TrainingVolume):
    """Lookup-only embeddings: the MedCPT encoder is unreachable off the campaign machine."""
    from spatialcpav25_gen.model.embeddings import EntityEmbeddings

    zeros = torch.zeros((vol.n_genes, cfg.text_dim_in), dtype=torch.float32)
    types = torch.zeros((len(vol.celltype_names), cfg.text_dim_in), dtype=torch.float32)
    return EntityEmbeddings(cfg, zeros, types, None)


def real_section_reference(cfg: Config, model: CTFFlow, section_id: str, k: int) -> list[dict]:
    """Moran's I of the real held-out section's counts, and of the latent the encoder makes."""
    import anndata as ad
    import scipy.sparse as sp

    gt = ad.read_h5ad(GROUND_TRUTH)
    mask = gt.obs["section"].values.astype(str) == section_id
    xy = np.asarray(gt.obsm["spatial"], dtype=np.float64)[mask, :2]
    counts = gt.X[mask]
    counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    counts = np.asarray(counts, dtype=np.float32)

    rows = [summarise("REF real counts (rank-normalised)", xy, rank_normalize(counts), k)]

    gene_idx = torch.arange(counts.shape[1], dtype=torch.long)
    with torch.no_grad():
        gene_emb = model.embeddings.gene(gene_idx)
        totals = torch.from_numpy(counts.sum(axis=1))  # (N,), not (N, 1)
        size_factor = totals / max(float(model.stats.median_total), 1.0)
        h1 = model.encoder(torch.from_numpy(counts), gene_emb, size_factor)
    rows.append(summarise("REF real latent h1 = encoder(real counts)", xy, h1.numpy(), k))
    return rows


def mean_variance_slope(counts: np.ndarray) -> float:
    """Log-log slope of per-gene variance against per-gene mean. The tissue's is ~1.74."""
    m = np.asarray(counts, dtype=np.float64).mean(axis=0)
    v = np.asarray(counts, dtype=np.float64).var(axis=0)
    return float(np.polyfit(np.log(m + 1e-9), np.log(v + 1e-9), 1)[0])


def _verdict(rows: list[dict], emitted: dict, cfg: Config, args) -> list[str]:
    """The three numbers the decision turns on: retention, slope, and the counts' own Moran's I."""
    import anndata as ad
    import scipy.sparse as sp

    by = {r["stage"]: r["median_I"] for r in rows}
    gt = ad.read_h5ad(GROUND_TRUTH)
    mask = gt.obs["section"].values.astype(str) == args.section
    real = gt.X[mask]
    real = np.asarray(real.toarray() if sp.issparse(real) else real, dtype=np.float64)

    real_latent = by.get("REF real latent h1 = encoder(real counts)", float("nan"))
    real_counts = by.get("REF real counts (rank-normalised)", float("nan"))
    real_retention = real_counts / real_latent if real_latent else float("nan")
    latent = by.get("2. latent h after the flow", float("nan"))

    out = [
        "",
        "## The three numbers",
        "",
        "**Retention across the latent -> counts step** — what the emission costs, against what",
        "the tissue's own sampling noise costs.",
        "",
        "| arm | counts I | latent I | retention | slope | tissue slope |",
        "|---|---|---|---|---|---|",
    ]
    real_slope = mean_variance_slope(real)
    out.append(
        f"| **real tissue** | {real_counts:+.4f} | {real_latent:+.4f} | "
        f"**{real_retention:.1%}** | {real_slope:.3f} | — |"
    )
    for label, stage in (
        ("uncalibrated", "4. sampled counts (rank-normalised)"),
        ("calibrated", "4c. sampled counts, CALIBRATED (rank-norm)"),
    ):
        if label not in emitted:
            continue
        ci = by.get(stage, float("nan"))
        out.append(
            f"| {label} | {ci:+.4f} | {latent:+.4f} | **{ci / latent:.1%}** | "
            f"{mean_variance_slope(emitted[label]):.3f} | {real_slope:.3f} |"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--section", default="section_2")
    ap.add_argument("--target-z", type=float, default=30.0)
    ap.add_argument("--out", default="reports/chain_diagnostic.md")
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="also fit T09 §2's pi / log-theta calibration and re-measure the emission stages",
    )
    args = ap.parse_args(argv)

    cfg = Config(
        seed=SEED,
        text_emb_mode="lookup",
        train_steps=args.steps,
        expr_pca_dim=16,
        ell_xy=116.3,
        ell_z=132.0,
    ).replace(section_key="section", coord_key="spatial", celltype_key="cell_type", region_key=None)
    k = int(cfg.metric_knn_k)

    print(f"chain diagnostic: {args.steps} steps, {args.section} at z={args.target_z}")
    vol = load_training_volume(cfg)
    data = TrainingData.build(vol, cfg)
    model = CTFFlow(cfg, data, build_embeddings(cfg, vol), grf_seed=SEED)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        train_ctfflow(model, cfg, steps=int(cfg.train_steps), seed=SEED)
        if cfg.repulsion:
            model.repulsion = fit_repulsion(vol, cfg, seed=SEED + 1)
    print(f"  fit: {cfg.train_steps} steps in {time.time() - t0:.1f}s")

    # --- the generated chain, reproducing infer/generate.py::_expression step by step ---
    from spatialcpav25_gen.infer.generate import _decode, _default_exclusions, _layout_on
    from spatialcpav25_gen.model.expression import sample_counts

    plane = plane_at_z(vol, float(args.target_z), cfg)
    rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layout = _layout_on(model, plane, vol, cfg, SEED)
        xyz = layout.coords_xyz.astype(np.float64)
        xy = xyz[:, :2]
        cell_type = torch.from_numpy(layout.cell_type.astype(np.int64))
        neighbours, _w = model.data.index.query(
            xyz, _default_exclusions(vol, float(plane.origin[2])), seed=SEED
        )
        points = torch.from_numpy(xyz.astype(np.float32))
        with torch.no_grad():
            tokens, mask = model.data.index.neighbour_tokens(xyz, neighbours)
            cond, _ = model.conditioning(points, points, cell_type, None, tokens, mask)
            h0 = model.prior_latent(xyz, seed=SEED)
            h = model.flow.sample(h0, cond, int(cfg.ode_steps))
            mu, theta, pi = _decode(model, h, cfg, None)
            counts = sample_counts(mu, theta, pi, np.random.default_rng(SEED))

    rows.append(summarise("1. prior h0 = GRF at generated xyz", xy, h0.numpy(), k))
    rows.append(summarise("2. latent h after the flow", xy, h.numpy(), k))
    rows.append(summarise("3. decoded mu (before sampling)", xy, mu.numpy(), k))
    rows.append(
        summarise("4. sampled counts (rank-normalised)", xy, rank_normalize(counts.numpy()), k)
    )
    emitted = {"uncalibrated": counts.numpy()}

    if args.calibrate:
        # T09 §2's calibrator solves log theta per gene against the mean-variance relation at the
        # model's own mean — the quantity the pilot measured wrong on real data (slope 2.120
        # against the tissue's 1.738). It ships unapplied because the fixture gave it no headroom.
        from spatialcpav25_gen.infer.calibrate import calibrate_detection

        t1 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibration = calibrate_detection(model, vol, cfg, seed=SEED)
            with torch.no_grad():
                mu_c, theta_c, pi_c = _decode(model, h, cfg, calibration)
                counts_c = sample_counts(mu_c, theta_c, pi_c, np.random.default_rng(SEED))
        print(f"  calibration fitted in {time.time() - t1:.1f}s on {list(calibration.section_ids)}")
        rows.append(summarise("3c. decoded mu, CALIBRATED", xy, mu_c.numpy(), k))
        rows.append(
            summarise(
                "4c. sampled counts, CALIBRATED (rank-norm)",
                xy,
                rank_normalize(counts_c.numpy()),
                k,
            )
        )
        emitted["calibrated"] = counts_c.numpy()

    for r in rows:  # print the generated chain before anything else can fail
        print(f"  {r['stage']:<48s} median I = {r['median_I']:+.4f}  (n={r['n_channels']})")
    try:
        rows.extend(real_section_reference(cfg, model, args.section, k))
    except Exception as exc:  # a reference failure must not discard the chain above
        print(
            f"  !! reference stage failed ({type(exc).__name__}: {exc}); "
            f"the generated chain above still stands",
            file=sys.stderr,
        )

    width = max(len(r["stage"]) for r in rows)
    lines = [
        f"# Chain diagnostic — where the spatial structure is lost ({args.steps} steps)",
        "",
        f"STARmap tier 1, `{args.section}` at z={args.target_z}, {xy.shape[0]} generated cells.",
        "Median per-channel Moran's I on a row-standardised kNN graph "
        f"(k={k}), the same estimator at every stage.",
        "",
        f"| {'stage':<{width}} | median I | p25 | p75 | channels |",
        f"|{'-' * (width + 2)}|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['stage']:<{width}} | **{r['median_I']:+.4f}** | {r['p25']:+.4f} "
            f"| {r['p75']:+.4f} | {r['n_channels']} |"
        )
    lines.extend(_verdict(rows, emitted, cfg, args))
    text = "\n".join(lines)
    print()
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
