"""Synthetic stack with **non-monotone z events** — the regime Phase 2.3 targets.

The proposal's argument for a flow-free, directly generative interpolator is that
warping assumes conserved, translating content and therefore cannot represent
*branching, merging or termination* — the events that actually matter.  Linear
interpolation of the field has the same blind spot for a different reason: it can
only produce a monotone ramp between the brackets.

This generator builds exactly those events into the stack:

* a **transient** population that appears at one depth, peaks mid-gap and is gone
  two sections later (invisible to any monotone interpolation of its brackets);
* a **branching** population whose single lobe splits into two with depth;
* a **stable** background population, so the easy part of the field is still easy.

Held out at the peak of the transient, a method that only ramps between brackets
must under-predict it.  That is what the Phase 2.5 gate is there to detect.

Usage:  python make_synth_events.py out.h5ad
"""

import sys

import anndata as ad
import numpy as np
import pandas as pd


def make(seed: int = 0, n_sections: int = 9, n_genes: int = 40,
         n_background: int = 320) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n_types = 4
    type_means = rng.gamma(2.0, 1.5, size=(n_types, n_genes)).astype(np.float32)
    # give each type a few private markers so the Phase 1.2 signatures are real
    for t in range(n_types):
        type_means[t, t * 6:(t + 1) * 6] *= 6.0

    xs, ys, zs, secs, cts, X = [], [], [], [], [], []
    for si in range(n_sections):
        z = float(si)
        px, py, pt = [], [], []

        # background: a stable disc (type 0/1 split by radius)
        r = np.sqrt(rng.uniform(0, 1, n_background)) * 40.0
        th = rng.uniform(0, 2 * np.pi, n_background)
        px.append(r * np.cos(th))
        py.append(r * np.sin(th))
        pt.append(np.where(r < 22.0, 0, 1))

        # transient: present only for z in [3, 5], peaking hard at z = 4
        amp = max(0.0, 1.0 - abs(z - 4.0) / 1.2)
        n_tr = int(round(220 * amp ** 1.5))
        if n_tr:
            a = rng.normal(0, 5.0, n_tr)
            b = rng.normal(0, 5.0, n_tr)
            px.append(a - 16.0)
            py.append(b + 14.0)
            pt.append(np.full(n_tr, 2))

        # branching: one lobe until z = 4, two lobes after
        n_br = 150
        if z <= 4.0:
            cx = np.full(n_br, 18.0)
            cy = np.full(n_br, -12.0)
        else:
            split = 4.0 * (z - 4.0)
            side = rng.integers(0, 2, n_br) * 2 - 1
            cx = 18.0 + side * split
            cy = np.full(n_br, -12.0) - side * split * 0.5
        px.append(cx + rng.normal(0, 4.0, n_br))
        py.append(cy + rng.normal(0, 4.0, n_br))
        pt.append(np.full(n_br, 3))

        x = np.concatenate(px)
        y = np.concatenate(py)
        t = np.concatenate(pt)
        # a smooth within-type spatial gradient, so type-mean assignment is wrong
        grad = 1.0 + 0.45 * np.cos(0.09 * x + 0.25 * z)
        expr = rng.poisson(np.clip(type_means[t] * grad[:, None] * 40.0, 0, None))
        xs.append(x)
        ys.append(y)
        zs.append(np.full(x.size, z))
        secs.append(np.array([f"S{si}"] * x.size))
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
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_events.h5ad"
    A = make()
    A.write_h5ad(out)
    print("wrote", out, A.shape, "sections", sorted(A.obs["section"].unique()))
