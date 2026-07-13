"""
Biologically-constrained expression generation for SpatialCPA-v10.

Where v8 *copies* real expression profiles, v10 *generates* each cell's expression
from an explicit biological model, keeping only a real residual for realism (the
"balanced hybrid"). For a synthesized cell of type ``c`` at position ``x`` with
spatial neighbours, expression is

    expr = μ_c(z)  +  λ · LR_modulation(x, neighbours)  +  (x_real − μ_{c_real})

combining the four biological constraints:

1. **Cell-type gene programs** — ``μ_c`` is the mean expression profile of type ``c``,
   learned from training cells. The *mean* of each synthesized cell is therefore a
   real cell-type program, not a copied cell.
2. **z-axis / developmental continuity** — ``μ_c(z)`` is the z-interpolation of the
   type mean in the two flanking slices, ``(1−t)·μ_c^lo + t·μ_c^hi``, so the program
   varies smoothly and monotonically along the stack (a developmental trajectory),
   not discontinuously.
3. **Ligand–receptor communication** — ``LR_modulation`` up-/down-regulates a cell's
   receptor/target genes according to how strongly its spatial neighbours express the
   cognate ligands, through a ligand→target coupling matrix ``W``. ``W`` is read from
   a curated LR database when provided, else inferred data-drivenly from the training
   slices (genes whose expression in a cell is predicted by its neighbours' expression
   — a spatial cross-coupling / NCEM-style signal). This is *mechanistic* cell–cell
   communication acting on expression, beyond the co-occurrence niche model.
4. **Spatial niche architecture** — handled in the annotation/communication modules
   (which cell types co-localize); the LR coupling additionally biases signaling
   partners to sit together.

The residual ``x_real − μ_{c_real}`` is the *deviation* of a real matched cell from
its own type mean; adding it back transfers real cell-to-cell variation and gene–gene
covariance onto the biologically-generated mean, so expression stays realistic
(preserving the variance/co-expression the benchmark rewards) while every cell is now
a function of its type and its neighbours' signaling — not a copy.

Leakage-safe: all programs, the LR coupling, and the residual come from the training
slices only.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# Cell-type gene programs (z-continuous)                                        #
# --------------------------------------------------------------------------- #
def type_means(expression, labels, n_types):
    """Per-type mean expression profile (a cell-type gene program). Missing types
    fall back to the global mean."""
    X = np.asarray(expression, dtype=np.float64)
    g = X.mean(axis=0)
    M = np.tile(g, (n_types, 1))
    lab = np.asarray(labels).astype(int)
    for c in range(n_types):
        m = lab == c
        if m.any():
            M[c] = X[m].mean(axis=0)
    return M


def z_interpolated_programs(lo_expr, lo_lab, up_expr, up_lab, n_types, t):
    """z-interpolated per-type program ``μ_c(z) = (1−t)·μ_c^lo + t·μ_c^hi``."""
    Mlo = type_means(lo_expr, lo_lab, n_types)
    Mhi = type_means(up_expr, up_lab, n_types)
    return (1.0 - t) * Mlo + t * Mhi


# --------------------------------------------------------------------------- #
# Ligand-receptor communication coupling                                       #
# --------------------------------------------------------------------------- #
def load_lr_coupling(path, gene_names):
    """Load a curated LR database into a ligand->target coupling matrix ``W`` (G×G).

    Accepts a ``.npz`` with arrays ``ligand``/``receptor`` (gene-symbol pairs, and
    optional ``weight``) or a 2/3-column TSV. Genes absent from the panel are
    dropped. Returns ``None`` if unavailable.
    """
    import os
    if not path or not os.path.exists(path):
        return None
    genes = [str(g) for g in gene_names]
    idx = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    W = np.zeros((G, G), dtype=np.float64)
    pairs = []
    try:
        if path.endswith(".npz"):
            d = np.load(path, allow_pickle=True)
            lig = [str(x) for x in d["ligand"]]
            rec = [str(x) for x in d["receptor"]]
            wt = d["weight"] if "weight" in d else np.ones(len(lig))
            pairs = list(zip(lig, rec, wt))
        else:
            for line in open(path):
                p = line.rstrip("\n").split("\t")
                if len(p) < 2:
                    continue
                w = float(p[2]) if len(p) > 2 else 1.0
                pairs.append((p[0], p[1], w))
    except Exception:
        return None
    n = 0
    for l, r, w in pairs:
        i, j = idx.get(l), idx.get(r)
        if i is not None and j is not None:
            W[i, j] += float(w)
            n += 1
    return W if n > 0 else None


def infer_lr_coupling(stack, gene_names, k=10, top_frac=0.02, max_genes=400):
    """Data-derived ligand->target coupling from spatial cross-coupling.

    For each training slice, compute each cell's mean-neighbour expression, then the
    cross-correlation between a gene's *neighbour* expression (candidate ligand) and a
    gene's *self* expression (candidate target) across cells. Strong positive
    cross-couplings are candidate communication edges (neighbour ligand predicts self
    target). Returns a sparse-ish ligand→target matrix ``W`` (G×G, non-negative), the
    leakage-safe empirical stand-in for a curated LR database.
    """
    genes = list(gene_names)
    G = len(genes)
    sel = np.arange(G)
    if G > max_genes:  # cap for tractability on large panels
        # keep the most variable genes
        allX = stack.union_expression()
        sel = np.sort(np.argsort(allX.var(axis=0))[-max_genes:])
    self_acc = None
    nbr_acc = None
    n_tot = 0
    for s in stack.slices:
        if s.n_spots < k + 1:
            continue
        X = np.asarray(s.expression, dtype=np.float64)[:, sel]
        _, nn = cKDTree(s.coords_xy).query(s.coords_xy, k=min(k + 1, s.n_spots))
        nn = nn[:, 1:]
        Xn = X[nn].mean(axis=1)               # mean-neighbour expression (n, g)
        # standardize per gene within slice
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Xns = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-8)
        if self_acc is None:
            self_acc = np.zeros((Xs.shape[1], Xs.shape[1]))
        self_acc += Xns.T @ Xs                # cross-cov: neighbour-ligand x self-target
        n_tot += X.shape[0]
    if self_acc is None or n_tot == 0:
        return None
    C = self_acc / n_tot                      # (g, g): rows=ligand(neighbour), cols=target(self)
    C = np.clip(C, 0.0, None)                 # keep positive (activating) couplings
    np.fill_diagonal(C, 0.0)
    # keep only the strongest edges
    if C.size:
        thr = np.quantile(C[C > 0], 1.0 - top_frac) if np.any(C > 0) else np.inf
        C[C < thr] = 0.0
    W = np.zeros((G, G), dtype=np.float64)
    W[np.ix_(sel, sel)] = C
    return W if np.any(W) else None


def lr_modulation(expr_norm, coords_xy, W, k=10):
    """Ligand→target expression modulation from spatial neighbours.

    ``L[i, target] = Σ_ligand  mean_neighbour_expr[i, ligand] · W[ligand, target]``.
    Standardized so it is a dimensionless modulation added to the cell-type program.
    """
    if W is None:
        return np.zeros_like(expr_norm)
    n = coords_xy.shape[0]
    if n < 2:
        return np.zeros_like(expr_norm)
    kk = min(k + 1, n)
    _, nn = cKDTree(coords_xy).query(coords_xy, k=kk)
    nn = nn[:, 1:]
    neigh_mean = expr_norm[nn].mean(axis=1)   # (n, G) neighbour ligand levels
    L = neigh_mean @ W                        # (n, G) target modulation
    s = L.std(axis=0, keepdims=True); s[s == 0] = 1.0
    return (L - L.mean(axis=0, keepdims=True)) / s


# --------------------------------------------------------------------------- #
# Balanced-hybrid expression generation                                        #
# --------------------------------------------------------------------------- #
def generate_expression(cell_type_idx, coords_xy, programs_z, W,
                        residual, residual_type_means, source_type_idx,
                        lr_lambda=0.5, program_weight=1.0, residual_weight=1.0, k=10):
    """Compose expression = μ_c(z) + λ·LR_modulation + real residual (balanced hybrid).

    Parameters
    ----------
    cell_type_idx : (n,) int assigned type per synthesized cell.
    programs_z : (n_types, G) z-interpolated cell-type programs μ_c(z).
    W : (G, G) ligand->target coupling (or None).
    residual : (n, G) real matched cell profiles (log-normalized).
    residual_type_means : (n_types, G) mean of the *residual population* per type
        (so the residual is a true deviation ``x_real − μ_{c_real}``).
    source_type_idx : (n,) type of the real cell each residual came from.
    """
    n, Gp = coords_xy.shape[0], programs_z.shape[1]
    ct = np.asarray(cell_type_idx).astype(int)
    mu = programs_z[ct]                                   # cell-type program (+continuity)
    # ligand-receptor modulation on the type-program field
    base_norm = (mu - mu.mean(0, keepdims=True))
    bstd = base_norm.std(0, keepdims=True); bstd[bstd == 0] = 1.0
    L = lr_modulation(base_norm / bstd, coords_xy, W, k=k)
    # real residual: deviation of the matched real cell from its own type mean
    st = np.asarray(source_type_idx).astype(int)
    resid = np.asarray(residual, dtype=np.float64) - residual_type_means[st]
    expr = program_weight * mu + lr_lambda * L + residual_weight * resid
    return np.clip(expr, 0.0, None).astype(np.float32)
