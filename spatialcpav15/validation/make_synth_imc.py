"""Synthetic stack in the **continuous-intensity** regime (IMC-like).

Most spatial transcriptomics ships counts; imaging mass cytometry and similar
protein panels ship *continuous intensities*, which take a different path through
Phase 3.1 (a Gaussian likelihood rather than NB) and a different registration
policy (``rigid`` — IMC sections are not cross-registered by the provider).  That
path is easy to leave untested, so it gets its own stack:

* ~35 protein channels of strictly positive, log-normally distributed intensity —
  no integers anywhere, so ``expression_type = "mean_intensity"``;
* each section in **its own arbitrary rigid frame**, as IMC ROIs are;
* a handful of sections at coarse z spacing.

Usage:  python make_synth_imc.py out.h5ad
"""

import sys

import anndata as ad
import numpy as np
import pandas as pd


def make(seed: int = 0, n_sections: int = 6, n_per: int = 700,
         n_channels: int = 35, n_types: int = 6, with_celltype: bool = True
         ) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    proto = rng.gamma(2.0, 1.0, size=(n_types, n_channels)) * 30.0
    for t in range(n_types):
        proto[t, t * 5:(t + 1) * 5] *= 8.0

    X, xyz, secs, ct = [], [], [], []
    for si in range(n_sections):
        z = float(si) * 4.0
        r = np.sqrt(rng.uniform(0, 1, n_per)) * 300.0
        th = rng.uniform(0, 2 * np.pi, n_per)
        x, y = r * np.cos(th), r * np.sin(th)
        t = np.clip(np.digitize(r, np.linspace(0, 300, n_types + 1)) - 1, 0, n_types - 1)
        grad = 1.0 + 0.4 * np.cos(0.01 * x + 0.15 * si)
        inten = proto[t] * grad[:, None] * rng.lognormal(0.0, 0.35, (n_per, n_channels))
        # Each ROI sits in its own frame: IMC sections are not cross-registered.
        ang = rng.uniform(-0.25, 0.25)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        p = np.column_stack([x, y]) @ R.T + rng.uniform(-40, 40, 2)
        X.append(inten.astype(np.float32))
        xyz.append(np.column_stack([p, np.full(n_per, z)]))
        secs += [f"ROI{si}"] * n_per
        ct.append(t)

    obs = {"section": np.array(secs)}
    if with_celltype:
        obs["cell_type"] = np.array([f"ct_{i}" for i in np.concatenate(ct)])
    A = ad.AnnData(X=np.vstack(X), obs=pd.DataFrame(obs),
                   var=pd.DataFrame(index=[f"prot{i}" for i in range(n_channels)]))
    A.obsm["spatial"] = np.vstack(xyz).astype(np.float64)
    A.uns["expression_type"] = "mean_intensity"
    return A


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_imc.h5ad"
    A = make()
    A.write_h5ad(out)
    print("wrote", out, A.shape, "sections", sorted(A.obs["section"].unique()))
