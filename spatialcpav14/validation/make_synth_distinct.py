"""Build a small synthetic multi-slice spatial dataset with real spatial
structure, cell types, and z-varying composition/geometry, for smoke-testing
the virtual-slice generators against the benchmark evaluators."""
import numpy as np
import anndata as ad
import pandas as pd

def make(seed=0, n_sections=7, n_per=380, n_genes=60, n_types=5, drift=6.0):
    rng = np.random.default_rng(seed)
    # Gene "programs": each cell type has a mean expression profile.
    type_means = rng.gamma(2.0, 1.0, size=(n_types, n_genes)).astype(np.float32)
    # Spatial layout: types organized in radial bands (a niche), band radius
    # drifts slightly with z so composition/geometry change smoothly.
    xs, ys, zs, secs, cts, X = [], [], [], [], [], []
    for si in range(n_sections):
        z = float(si)
        # tissue center drifts with z (so training re-registration matters)
        cx, cy = 0.0, 0.0
        # sample positions in a disc
        r = np.sqrt(rng.uniform(0, 1, n_per)) * 40.0
        th = rng.uniform(0, 2*np.pi, n_per)
        x = cx + r*np.cos(th)
        y = cy + r*np.sin(th)
        # cell type by radial band, band edges drift with z
        edges = np.linspace(0, 40.0, n_types+1) + drift*np.sin(0.4*z)
        t = np.clip(np.digitize(r, edges) - 1, 0, n_types-1)
        # composition shift with z: upweight higher types deeper
        # expression = type mean * spatial gradient + noise
        grad = 1.0 + 0.5*np.cos(0.1*(x) + 0.2*z)
        base = type_means[t] * grad[:, None]
        expr = rng.poisson(np.clip(base, 0, None)).astype(np.float32)
        xs.append(x); ys.append(y); zs.append(np.full(n_per, z))
        secs.append(np.array([f"S{si}"]*n_per)); cts.append(t)
        X.append(expr)
    X = np.vstack(X)
    obs = pd.DataFrame({
        "section": np.concatenate(secs),
        "cell_type": np.array([f"type_{i}" for i in np.concatenate(cts)]),
    })
    spatial = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    A = ad.AnnData(X=X, obs=obs, var=var)
    A.obsm["spatial"] = spatial.astype(np.float64)
    A.uns["expression_type"] = "raw_counts"
    return A

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synth.h5ad"
    A = make()
    A.write_h5ad(out)
    print("wrote", out, A.shape, "sections", sorted(A.obs['section'].unique()))
