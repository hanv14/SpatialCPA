"""Synthetic stack with smoothly **drifting** niches — the ordinary regime.

Composition and geometry change monotonically with depth: radial niche bands
migrate, the tissue centre drifts, and within-type expression carries a spatial
gradient.  Nothing here is non-monotone, so linear interpolation of the field is a
strong baseline and the Phase 2.5 gate is expected to be *close* — that is the
point of having both this and ``make_synth_events.py``.

Usage:  python make_synth_drift.py out.h5ad
"""

import sys

import anndata as ad
import numpy as np
import pandas as pd


def make(seed: int = 0, n_sections: int = 9, n_per: int = 420, n_genes: int = 40,
         n_types: int = 5, drift: float = 7.0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    type_means = rng.gamma(2.0, 1.5, size=(n_types, n_genes)).astype(np.float32)
    for t in range(n_types):
        type_means[t, t * 5:(t + 1) * 5] *= 5.0

    xs, ys, zs, secs, cts, X = [], [], [], [], [], []
    for si in range(n_sections):
        z = float(si)
        r = np.sqrt(rng.uniform(0, 1, n_per)) * 40.0
        th = rng.uniform(0, 2 * np.pi, n_per)
        cx, cy = 0.6 * z, -0.4 * z                  # tissue centre drifts with z
        x = cx + r * np.cos(th)
        y = cy + r * np.sin(th)
        edges = np.linspace(0, 40.0, n_types + 1) + drift * (z / max(n_sections - 1, 1))
        t = np.clip(np.digitize(r, edges) - 1, 0, n_types - 1)
        grad = 1.0 + 0.45 * np.cos(0.09 * x + 0.2 * z)
        expr = rng.poisson(np.clip(type_means[t] * grad[:, None] * 40.0, 0, None))
        xs.append(x)
        ys.append(y)
        zs.append(np.full(n_per, z))
        secs.append(np.array([f"S{si}"] * n_per))
        cts.append(t)
        X.append(expr.astype(np.float32))

    obs = pd.DataFrame({
        "section": np.concatenate(secs),
        "cell_type": np.array([f"type_{i}" for i in np.concatenate(cts)]),
    })
    A = ad.AnnData(X=np.vstack(X), obs=obs,
                   var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    A.obsm["spatial"] = np.column_stack(
        [np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)]).astype(np.float64)
    A.uns["expression_type"] = "raw_counts"
    return A


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_drift.h5ad"
    A = make()
    A.write_h5ad(out)
    print("wrote", out, A.shape, "sections", sorted(A.obs["section"].unique()))
