import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings("ignore")

"""
SpatialCPA v3 — Virtual Slice GENERATION evaluation on STARmap.

Difference from run_spatialcpa_eval.py (v2)
-------------------------------------------
v2 "reconstructs" a held-out slice by predicting expression at the slice's
TRUE cell (x, y) positions, conditioned on its TRUE cell types. It therefore
needs the very slice it is meant to produce, so it is not virtual-slice
generation at all.

v3 GENERATES the held-out slice from scratch using only:
    * the target z, and
    * the two neighboring training slices (already registered / aligned).

It never sees the held-out positions, counts, or cell types. Because the
generated cells do not correspond 1:1 with the real cells, we evaluate with
correspondence-free, distribution-level metrics:

    * gene-wise spatial autocorrelation (Moran's I / Geary's C) correlation
    * cell-type composition correlation
    * pseudobulk mean-expression correlation
    * nearest-neighbor-matched gene-wise Pearson r (spatially-aware fidelity)

A naive baseline (copy the nearest real neighbor slice — i.e. pure "linear"
interpolation) is reported alongside so the generative gain is visible.

Protocol (matches v2): split STARmap into 7 slices, train on 1,3,5,7,
generate held-out 2,4,6 from their flanking slices.

Usage:
    python run_spatialcpa_v3_eval.py [--epochs 50]
"""

import argparse
import time

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from scipy.spatial import cKDTree

from spatialcpa.model import SpatialCPA
from spatialcpa.data import SpatialSection
from spatialcpa.trainer import SpatialCPATrainer
from spatialcpa.virtual_slice import VirtualSliceGeneratorV3
from spatialcpa.fourier import FourierFeatureEncoder


# ===================================================================
# Metrics (correspondence-free)
# ===================================================================

def spatial_weights_knn(positions, k=6):
    tree = cKDTree(positions)
    _, indices = tree.query(positions, k=k + 1)
    N = len(positions)
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        W[i, indices[i, 1:]] = 1.0
    row = W.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return W / row


def morans_I(x, W):
    z = x - x.mean()
    zz = z @ z
    if zz == 0:
        return 0.0
    N = len(x)
    return float((N / W.sum()) * (z @ W @ z) / zz)


def gearys_C(x, W):
    z = x - x.mean()
    zz = z @ z
    if zz == 0:
        return 1.0
    N = len(x)
    diff = x[:, None] - x[None, :]
    return float(((N - 1) / (2 * W.sum())) * (W * diff ** 2).sum() / zz)


def spatial_autocorr(X, positions, k=6):
    W = spatial_weights_knn(positions, k=k)
    n_genes = X.shape[1]
    m = np.zeros(n_genes)
    g = np.zeros(n_genes)
    for j in range(n_genes):
        m[j] = morans_I(X[:, j], W)
        g[j] = gearys_C(X[:, j], W)
    return m, g


def pearson_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def celltype_composition(labels, names):
    counts = np.array([np.sum(labels == n) for n in names], dtype=np.float64)
    return counts / max(counts.sum(), 1)


def nn_matched_genewise_r(real_X, real_xy, gen_X, gen_xy):
    """For each real cell, borrow the nearest generated cell's expression, then
    correlate per gene. Spatially-aware fidelity without exact correspondence."""
    tree = cKDTree(gen_xy)
    _, idx = tree.query(real_xy, k=1)
    matched = gen_X[idx]
    n_genes = real_X.shape[1]
    rs = np.zeros(n_genes)
    for j in range(n_genes):
        rs[j] = pearson_r(real_X[:, j], matched[:, j])
    return rs


def pseudobulk_r(real_X, gen_X):
    return pearson_r(real_X.mean(axis=0), gen_X.mean(axis=0))


# ===================================================================
# Data prep (mirrors v2 eval)
# ===================================================================

def load_starmap():
    adata = sc.read_h5ad('./data/starmap/STARmap_Wang2018three_data_3D_data.h5ad')
    adata.obs['cell_class'] = adata.obs['leiden']
    if 'X_umap' in adata.obsm:
        del adata.obsm['X_umap']
    adata.uns = {}
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]

    exclude_z = [6, 7, 8, 9, 10, 11, 12, 13, 91, 92, 93, 94]
    adata = adata[~adata.obs['z'].isin(exclude_z)].copy()

    if issparse(adata.X):
        adata.X = np.asarray(adata.X.todense())
    adata.X = np.asarray(adata.X, dtype=np.float32)
    adata.X = np.log1p(adata.X)

    gene_means = adata.X.mean(axis=0)
    gene_stds = adata.X.std(axis=0)
    gene_stds[gene_stds < 1e-6] = 1.0
    adata.X = (adata.X - gene_means) / gene_stds
    return adata, gene_means, gene_stds


def split_into_slices(adata, num_slices=7):
    unique_z = np.sort(np.unique(adata.obs['z']))
    base = len(unique_z) // num_slices
    rem = len(unique_z) % num_slices
    idx = [0]
    for i in range(num_slices):
        idx.append(idx[-1] + base + (1 if i < rem else 0))
    sv = [unique_z[idx[i]:idx[i + 1]] for i in range(num_slices)]

    def assign(z):
        for i, s in enumerate(sv):
            if z in s:
                return f'slice_{i + 1}'
        return None
    adata.obs['slice_id'] = adata.obs['z'].apply(assign)
    return {i: adata[adata.obs['slice_id'] == f'slice_{i}'].copy()
            for i in range(1, num_slices + 1)}


def slice_to_section(adata_slice, ct_to_idx, section_id):
    """Pool an entire slice into ONE SpatialSection (per-cell z retained)."""
    X = np.asarray(adata_slice.X, dtype=np.float32)
    xy = np.asarray(adata_slice.obsm['spatial'])[:, :2].astype(np.float32)
    ct = np.array([ct_to_idx[c] for c in adata_slice.obs['cell_class'].values],
                  dtype=np.int64)
    z = adata_slice.obs['z'].values.astype(np.float32)
    return SpatialSection(X, xy, z, ct, section_id=section_id)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 64)
    print("SpatialCPA v3 — Virtual Slice GENERATION (neighbors only)")
    print("=" * 64)

    adata, gene_means, gene_stds = load_starmap()
    slices = split_into_slices(adata, 7)
    gene_names = list(adata.var_names)
    n_genes = len(gene_names)

    all_ct = sorted(set().union(*[set(slices[i].obs['cell_class'].unique())
                                  for i in range(1, 8)]))
    ct_to_idx = {n: i for i, n in enumerate(all_ct)}
    n_cell_types = len(all_ct)
    print(f"  Cells total: {adata.n_obs}, genes: {n_genes}, "
          f"cell types: {n_cell_types}, device: {device}")

    # One SpatialSection per slice.
    slice_sections = {i: slice_to_section(slices[i], ct_to_idx, f'slice_{i}')
                      for i in range(1, 8)}

    train_ids = [1, 3, 5, 7]
    held_out_ids = [2, 4, 6]
    train_sections = [slice_sections[i] for i in train_ids]

    all_coords = np.vstack([s.get_3d_coords() for s in train_sections])
    xy_scale, z_scale = FourierFeatureEncoder.estimate_scales(all_coords)
    print(f"  Scales: xy={xy_scale:.2f}, z={z_scale:.2f}")

    # ── Train a GENERATIVE model (Gaussian head) ──────────────────────────
    print("\n" + "=" * 64)
    print("Training SpatialCPA v3 (generative Gaussian expression head)")
    print("=" * 64)
    model = SpatialCPA(
        n_genes=n_genes, n_cell_types=n_cell_types, n_regions=None,
        n_freq_xy=48, n_freq_z=32, xy_scale=xy_scale, z_scale=z_scale,
        backbone_hidden=512, backbone_output=256, backbone_layers=8,
        dropout=0.05, expression_mode='gaussian',
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainer = SpatialCPATrainer(
        model=model, sections=train_sections, device=device,
        lr=5e-4, batch_size=1024, n_z_samples=3, z_jitter=0.3,
        loo_weight=0.3, expression_weight=1.0, corr_weight=0.5,
    )
    t0 = time.time()
    history = trainer.train(n_epochs=args.epochs, verbose=True)
    print(f"  Trained in {time.time() - t0:.1f}s, final loss "
          f"{history[-1]['total']:.4f}")

    # ── Generate held-out slices from their neighbors ─────────────────────
    print("\n" + "=" * 64)
    print("Generating held-out slices from flanking sections")
    print("=" * 64)
    generator = VirtualSliceGeneratorV3(model, all_ct, gene_names, device=device)

    rows = []
    for held in held_out_ids:
        below = slice_sections[held - 1]   # e.g. slice_1 for held-out 2
        above = slice_sections[held + 1]   # e.g. slice_3
        real = slice_sections[held]
        target_z = float(np.median(real.z_values))
        print(f"\n--- held-out slice_{held} (z≈{target_z:.1f}) "
              f"from slice_{held - 1} & slice_{held + 1} ---")

        t0 = time.time()
        virt = generator.generate(
            below, above, target_z=target_z,
            ct_model_weight=0.5, ct_smooth_k=8, ct_smooth_iters=1,
            ct_temperature=1.0, expr_temperature=1.0,
            relax_iters=2, seed=args.seed,
        )
        print(f"    generated {virt.n_obs} cells (real has {real.n_cells}) "
              f"in {time.time() - t0:.1f}s")

        real_X = real.expression
        real_xy = real.coords_xy
        gen_X = np.asarray(virt.X)
        gen_xy = np.asarray(virt.obsm['spatial'])

        # metrics
        m_real, g_real = spatial_autocorr(real_X, real_xy)
        m_gen, g_gen = spatial_autocorr(gen_X, gen_xy)
        moran_r = pearson_r(m_real, m_gen)
        geary_r = pearson_r(g_real, g_gen)

        comp_real = celltype_composition(real.cell_type_indices,
                                         np.arange(n_cell_types))
        gen_ct = virt.obs['cell_type_idx'].values
        comp_gen = celltype_composition(gen_ct, np.arange(n_cell_types))
        comp_r = pearson_r(comp_real, comp_gen)

        pb_r = pseudobulk_r(real_X, gen_X)
        nn_r = np.nanmean(nn_matched_genewise_r(real_X, real_xy, gen_X, gen_xy))

        # baseline: nearest real neighbor slice (pure "linear" copy)
        base = below if abs(below.z_center - target_z) <= abs(
            above.z_center - target_z) else above
        b_m, b_g = spatial_autocorr(base.expression, base.coords_xy)
        base_moran_r = pearson_r(m_real, b_m)
        base_comp_r = pearson_r(comp_real, celltype_composition(
            base.cell_type_indices, np.arange(n_cell_types)))
        base_pb_r = pseudobulk_r(real_X, base.expression)
        base_nn_r = np.nanmean(nn_matched_genewise_r(
            real_X, real_xy, base.expression, base.coords_xy))

        rows.append(dict(
            slice=held, moran_r=moran_r, geary_r=geary_r, comp_r=comp_r,
            pb_r=pb_r, nn_r=nn_r,
            base_moran_r=base_moran_r, base_comp_r=base_comp_r,
            base_pb_r=base_pb_r, base_nn_r=base_nn_r,
        ))
        print(f"    Moran r={moran_r:.3f} Geary r={geary_r:.3f} "
              f"comp r={comp_r:.3f} pseudobulk r={pb_r:.3f} NN-gene r={nn_r:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Summary — SpatialCPA v3 (generation) vs nearest-slice baseline")
    print("=" * 64)
    hdr = (f"{'Slice':<8}{'Moran r':>9}{'Geary r':>9}{'Comp r':>9}"
           f"{'PBulk r':>9}{'NNgene r':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"slice_{r['slice']:<2}{r['moran_r']:>9.3f}{r['geary_r']:>9.3f}"
              f"{r['comp_r']:>9.3f}{r['pb_r']:>9.3f}{r['nn_r']:>10.3f}")
    print("-" * len(hdr))
    avg = lambda k: np.nanmean([r[k] for r in rows])
    print(f"{'v3 avg':<8}{avg('moran_r'):>9.3f}{'':>9}"
          f"{avg('comp_r'):>9.3f}{avg('pb_r'):>9.3f}{avg('nn_r'):>10.3f}")
    print(f"{'base avg':<8}{avg('base_moran_r'):>9.3f}{'':>9}"
          f"{avg('base_comp_r'):>9.3f}{avg('base_pb_r'):>9.3f}"
          f"{avg('base_nn_r'):>10.3f}")
    print("\nDone. v3 generates de-novo positions, cell types, and expression")
    print("from neighboring sections only — no held-out leakage.")


if __name__ == '__main__':
    main()
