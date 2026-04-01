import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings("ignore")

"""
SpatialCPA — Spatial Autocorrelation Evaluation (Moran's I & Geary's C).

This script trains SpatialCPA on the STARmap dataset using a leave-one-out
protocol (train on slices 1,3,5,7; reconstruct slices 2,4,6) and evaluates
spatial metrics — the same protocol as run_axialst_eval.py for fair comparison.

Metrics:
  - Gene-wise Pearson r (predicted vs real expression per gene)
  - Moran's I correlation (spatial autocorrelation preservation)
  - Geary's C correlation (local spatial variation preservation)

Usage:
    python run_spatialcpa_eval.py

Requirements:
    - STARmap data at: ./data/starmap/STARmap_Wang2018three_data_3D_data.h5ad
    - pip install anndata scanpy numpy scipy scikit-learn matplotlib tqdm torch
"""

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import time
import torch
from scipy.sparse import issparse
from scipy.spatial import cKDTree

from spatialcpa.model import SpatialCPA
from spatialcpa.data import SpatialSection, adata_to_sections, compute_gap_weights
from spatialcpa.trainer import SpatialCPATrainer
from spatialcpa.inference import VirtualSliceGenerator
from spatialcpa.fourier import FourierFeatureEncoder


# ===================================================================
# Moran's I and Geary's C — lightweight implementations
# ===================================================================

def spatial_weights_knn(positions, k=6):
    """
    Build a binary k-NN spatial weight matrix (row-normalised).

    Returns
    -------
    W : (N, N) array, row-stochastic
    """
    tree = cKDTree(positions)
    dists, indices = tree.query(positions, k=k + 1)  # +1 for self
    N = len(positions)
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        nbrs = indices[i, 1:]       # exclude self
        W[i, nbrs] = 1.0
    # Row-normalise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums
    return W


def morans_I(x, W):
    """Moran's I for a single variable x given weight matrix W."""
    z = x - x.mean()
    zz = z @ z
    if zz == 0:
        return 0.0
    N = len(x)
    S0 = W.sum()
    I = (N / S0) * (z @ W @ z) / zz
    return float(I)


def gearys_C(x, W):
    """Geary's C for a single variable x given weight matrix W."""
    z = x - x.mean()
    zz = z @ z
    if zz == 0:
        return 1.0
    N = len(x)
    S0 = W.sum()
    diff = x[:, None] - x[None, :]
    numerator = (W * (diff ** 2)).sum()
    C = ((N - 1) / (2 * S0)) * numerator / zz
    return float(C)


def compute_spatial_autocorr(adata, k_neighbors=6, verbose=True):
    """Compute Moran's I and Geary's C for every gene in an AnnData."""
    positions = np.asarray(adata.obsm['spatial'])
    X = np.asarray(adata.X)
    n_genes = X.shape[1]

    if verbose:
        print(f"    Building {k_neighbors}-NN weight matrix "
              f"({adata.n_obs} cells)...")
    W = spatial_weights_knn(positions, k=k_neighbors)

    morans = np.zeros(n_genes)
    gearys = np.zeros(n_genes)

    for g in range(n_genes):
        morans[g] = morans_I(X[:, g], W)
        gearys[g] = gearys_C(X[:, g], W)

    if verbose:
        print(f"    Moran's I: mean={morans.mean():.4f}  "
              f"Geary's C: mean={gearys.mean():.4f}")

    return morans, gearys


def pearson_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def genewise_pearson(real_X, sim_X):
    """Compute per-gene Pearson r between real and simulated expression."""
    n_genes = real_X.shape[1]
    rs = np.zeros(n_genes)
    for g in range(n_genes):
        r_gene = real_X[:, g]
        s_gene = sim_X[:, g]
        if r_gene.std() < 1e-8 or s_gene.std() < 1e-8:
            rs[g] = 0.0
        else:
            rs[g] = float(np.corrcoef(r_gene, s_gene)[0, 1])
    return rs


# ===================================================================
# 1. Data preprocessing (same as AxialST eval)
# ===================================================================

print("=" * 60)
print("Loading and preprocessing STARmap data")
print("=" * 60)

adata_raw = sc.read_h5ad('./data/starmap/STARmap_Wang2018three_data_3D_data.h5ad')
adata_raw.obs['cell_class'] = adata_raw.obs['leiden']
adata = adata_raw.copy()

if 'X_umap' in adata.obsm:
    del adata.obsm['X_umap']
if adata.uns:
    adata.uns = {}
if adata.obsp is not None:
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]

exclude_z = [6, 7, 8, 9, 10, 11, 12, 13, 91, 92, 93, 94]
adata = adata[~adata.obs['z'].isin(exclude_z)].copy()


def split_data_into_slices(adata, num_slices):
    unique_z = np.sort(np.unique(adata.obs['z']))
    base = len(unique_z) // num_slices
    rem = len(unique_z) % num_slices
    indices = [0]
    for i in range(num_slices):
        indices.append(indices[-1] + base + (1 if i < rem else 0))
    sv_list = [unique_z[indices[i]:indices[i + 1]] for i in range(num_slices)]
    def assign(z):
        for i, sv in enumerate(sv_list):
            if z in sv:
                return f'slice_{i + 1}'
        return None
    adata.obs['slice_id'] = adata.obs['z'].apply(assign)

split_data_into_slices(adata, 7)

slices = {}
for i in range(1, 8):
    slices[i] = adata[adata.obs['slice_id'] == f'slice_{i}'].copy()
    print(f"  slice_{i}: {slices[i].n_obs} cells")


# ===================================================================
# 2. Prepare training sections from slices 1, 3, 5, 7
# ===================================================================

print("\n" + "=" * 60)
print("Preparing SpatialCPA training data")
print("=" * 60)

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

# Build cell type mapping from ALL slices (for consistent encoding)
all_cell_types = set()
for i in range(1, 8):
    all_cell_types.update(slices[i].obs['cell_class'].unique())
cell_type_names = sorted(all_cell_types)
ct_to_idx = {name: i for i, name in enumerate(cell_type_names)}
n_cell_types = len(cell_type_names)
print(f"  Cell types: {n_cell_types} — {cell_type_names}")

# Get gene info
gene_names = list(slices[1].var_names)
n_genes = len(gene_names)
print(f"  Genes: {n_genes}")


def slice_to_section(adata_slice, z_position, section_id=''):
    """Convert an AnnData slice to a SpatialSection."""
    X = adata_slice.X
    if issparse(X):
        X = np.asarray(X.todense())
    X = np.asarray(X, dtype=np.float32)

    # Get 2D spatial coordinates
    if 'spatial' in adata_slice.obsm:
        coords_xy = np.asarray(adata_slice.obsm['spatial'])[:, :2]
    elif 'x' in adata_slice.obs.columns and 'y' in adata_slice.obs.columns:
        coords_xy = np.column_stack([
            adata_slice.obs['x'].values.astype(np.float32),
            adata_slice.obs['y'].values.astype(np.float32),
        ])
    else:
        # Use z values as a proxy — generate random xy if missing
        # For STARmap, coordinates should be in obs
        raise ValueError(f"No spatial coordinates found for {section_id}")

    ct_indices = np.array(
        [ct_to_idx[c] for c in adata_slice.obs['cell_class'].values],
        dtype=np.int64,
    )

    return SpatialSection(
        expression=X,
        coords_xy=coords_xy,
        z_position=z_position,
        thickness=10.0,  # approximate section thickness in µm
        cell_type_indices=ct_indices,
        section_id=section_id,
    )


# Build training sections from slices 1, 3, 5, 7
train_slice_ids = [1, 3, 5, 7]
held_out_ids = [2, 4, 6]

# Compute z-positions for each slice (use median z value of cells)
slice_z_positions = {}
for i in range(1, 8):
    z_vals = slices[i].obs['z'].values.astype(float)
    slice_z_positions[i] = float(np.median(z_vals))
    print(f"  Slice {i}: z_center = {slice_z_positions[i]:.1f}")

train_sections = []
for sid in train_slice_ids:
    sec = slice_to_section(slices[sid], z_position=slice_z_positions[sid],
                           section_id=f'slice_{sid}')
    train_sections.append(sec)
    print(f"  Training section slice_{sid}: {sec.n_cells} cells, "
          f"z={sec.z_position:.1f}")

# Estimate spatial scales from training data
all_coords = np.vstack([
    sec.get_3d_coords() for sec in train_sections
])
xy_scale, z_scale = FourierFeatureEncoder.estimate_scales(all_coords)
print(f"  Estimated scales: xy={xy_scale:.2f}, z={z_scale:.2f}")


# ===================================================================
# 3. Build and train SpatialCPA model
# ===================================================================

print("\n" + "=" * 60)
print("Training SpatialCPA")
print("=" * 60)

model = SpatialCPA(
    n_genes=n_genes,
    n_cell_types=n_cell_types,
    n_regions=None,  # STARmap dataset doesn't have region labels
    n_freq_xy=48,
    n_freq_z=32,
    xy_scale=xy_scale,
    z_scale=z_scale,
    backbone_hidden=512,
    backbone_output=256,
    backbone_layers=8,
    dropout=0.1,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params:,}")

trainer = SpatialCPATrainer(
    model=model,
    sections=train_sections,
    device=device,
    lr=1e-3,
    batch_size=512,
    n_z_samples=5,
    loo_weight=0.5,
    expression_weight=1.0,
)

t0 = time.time()
history = trainer.train(n_epochs=80, verbose=True)
train_time = time.time() - t0
print(f"\n  Training completed in {train_time:.1f}s")
print(f"  Final loss: {history[-1]['total']:.4f}")


# ===================================================================
# 4. Reconstruct held-out slices
# ===================================================================

print("\n" + "=" * 60)
print("Reconstructing held-out slices with SpatialCPA")
print("=" * 60)

generator = VirtualSliceGenerator(
    model=model,
    cell_type_names=cell_type_names,
    gene_names=gene_names,
    region_names=None,
    device=device,
)

reconstructions = {}
for held_out in held_out_ids:
    print(f"\n--- slice_{held_out} (z={slice_z_positions[held_out]:.1f}) ---")
    t0 = time.time()

    # Use generate_matching to predict at exact positions of real cells
    sim = generator.generate_matching(
        z=slice_z_positions[held_out],
        reference_adata=slices[held_out],
        cell_type_key='cell_class',
        sample_expression=False,  # Use mean for more stable evaluation
        batch_size=2048,
    )

    print(f"  -> {sim.n_obs} cells in {time.time() - t0:.1f}s")
    reconstructions[held_out] = sim


# ===================================================================
# 5. Normalize and evaluate
# ===================================================================

print("\n" + "=" * 60)
print("Normalizing and computing metrics")
print("=" * 60)

pairs = []   # (name, real_adata, sim_adata)
for held_out in held_out_ids:
    real = slices[held_out].copy()
    sim  = reconstructions[held_out].copy()

    if issparse(real.X):
        real.X = np.asarray(real.X.todense())
    if issparse(sim.X):
        sim.X = np.asarray(sim.X.todense())

    sc.pp.normalize_total(real, target_sum=1e4)
    sc.pp.log1p(real)
    sc.pp.normalize_total(sim, target_sum=1e4)
    sc.pp.log1p(sim)

    pairs.append((f"Slice {held_out}", real, sim))


# ===================================================================
# 6. Compute gene-wise Pearson r
# ===================================================================

print("\n" + "=" * 60)
print("Gene-wise Pearson correlation")
print("=" * 60)

gene_r_results = {}
for name, real, sim in pairs:
    real_X = np.asarray(real.X)
    sim_X = np.asarray(sim.X)

    # Ensure same number of cells (generate_matching guarantees this)
    n_common = min(real_X.shape[0], sim_X.shape[0])
    gene_rs = genewise_pearson(real_X[:n_common], sim_X[:n_common])

    gene_r_results[name] = gene_rs
    print(f"  {name}: mean gene-wise r = {np.nanmean(gene_rs):.4f}, "
          f"median = {np.nanmedian(gene_rs):.4f}")


# ===================================================================
# 7. Compute Moran's I and Geary's C
# ===================================================================

k_nn = 6   # spatial neighbours

results = {}
for name, real, sim in pairs:
    print(f"\n  {name} -- real:")
    m_real, g_real = compute_spatial_autocorr(real, k_neighbors=k_nn)
    print(f"  {name} -- reconstructed:")
    m_sim,  g_sim  = compute_spatial_autocorr(sim,  k_neighbors=k_nn)

    results[name] = {
        'moran_real': m_real,  'moran_sim': m_sim,
        'geary_real': g_real,  'geary_sim': g_sim,
    }


# ===================================================================
# 8. Print quantitative comparison
# ===================================================================

print("\n" + "=" * 60)
print("Spatial Autocorrelation: Real vs SpatialCPA")
print("=" * 60)

for name in results:
    r = results[name]
    r_moran = pearson_r(r['moran_real'], r['moran_sim'])
    r_geary = pearson_r(r['geary_real'], r['geary_sim'])
    print(f"\n  {name}:")
    print(f"    Moran's I   Pearson r = {r_moran:.4f}   "
          f"(real mean={r['moran_real'].mean():.4f}, "
          f"recon mean={r['moran_sim'].mean():.4f})")
    print(f"    Geary's C   Pearson r = {r_geary:.4f}   "
          f"(real mean={r['geary_real'].mean():.4f}, "
          f"recon mean={r['geary_sim'].mean():.4f})")


# ===================================================================
# 9. Scatter plots: per-gene Moran's I and Geary's C
# ===================================================================

print("\n" + "=" * 60)
print("Generating scatter plots")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

colors = {'Slice 2': '#D54151', 'Slice 4': '#549745', 'Slice 6': '#B185DC'}

for col, name in enumerate(results):
    r = results[name]
    c = colors[name]

    # --- Moran's I ---
    ax = axes[0, col]
    ax.scatter(r['moran_real'], r['moran_sim'],
               s=30, alpha=0.7, c=c, edgecolors='white', linewidths=0.3)
    lim = [min(r['moran_real'].min(), r['moran_sim'].min()) - 0.02,
           max(r['moran_real'].max(), r['moran_sim'].max()) + 0.02]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Real (Moran's I)", fontsize=10)
    ax.set_ylabel("SpatialCPA (Moran's I)", fontsize=10)
    r_val = pearson_r(r['moran_real'], r['moran_sim'])
    ax.set_title(f"{name} -- Moran's I\nr = {r_val:.4f}", fontsize=11)

    # --- Geary's C ---
    ax = axes[1, col]
    ax.scatter(r['geary_real'], r['geary_sim'],
               s=30, alpha=0.7, c=c, edgecolors='white', linewidths=0.3)
    lim = [min(r['geary_real'].min(), r['geary_sim'].min()) - 0.02,
           max(r['geary_real'].max(), r['geary_sim'].max()) + 0.02]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Real (Geary's C)", fontsize=10)
    ax.set_ylabel("SpatialCPA (Geary's C)", fontsize=10)
    r_val = pearson_r(r['geary_real'], r['geary_sim'])
    ax.set_title(f"{name} -- Geary's C\nr = {r_val:.4f}", fontsize=11)

plt.tight_layout()
plt.savefig('spatialcpa_spatial_autocorrelation.png', dpi=300, bbox_inches='tight')
print("  Saved: spatialcpa_spatial_autocorrelation.png")
plt.close()


# ===================================================================
# 10. Summary table (copy-pasteable for paper)
# ===================================================================

print("\n" + "=" * 60)
print("Summary table (for paper)")
print("=" * 60)

header = f"{'Slice':<10} {'Gene r':>10} {'Moran r':>10} {'Geary r':>10}"
print(header)
print("-" * len(header))

for name in results:
    r = results[name]
    r_m = pearson_r(r['moran_real'], r['moran_sim'])
    r_g = pearson_r(r['geary_real'], r['geary_sim'])
    r_gene = np.nanmean(gene_r_results[name])
    print(f"{name:<10} {r_gene:>10.4f} {r_m:>10.4f} {r_g:>10.4f}")

# Overall averages
avg_gene = np.nanmean([np.nanmean(gene_r_results[n]) for n in results])
avg_moran = np.mean([pearson_r(results[n]['moran_real'], results[n]['moran_sim'])
                      for n in results])
avg_geary = np.mean([pearson_r(results[n]['geary_real'], results[n]['geary_sim'])
                      for n in results])
print("-" * len(header))
print(f"{'Average':<10} {avg_gene:>10.4f} {avg_moran:>10.4f} {avg_geary:>10.4f}")

print(f"\nTraining time: {train_time:.1f}s")
print(f"Model parameters: {total_params:,}")
print("\nDone!")
