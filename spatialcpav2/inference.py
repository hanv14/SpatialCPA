"""
Virtual-slice generation and inference for SpatialCPA v2.

Two ideas make v2's reconstructions dominate v1's on the benchmark metrics:

1. **Hybrid cell-type prediction.** A coordinate MLP must commit to a single
   most-likely type at each location, which caps its accuracy on fine-grained,
   spatially-interleaved cell types (v1 sat near chance). v2 fuses the neural
   posterior P(c | x, y, z) with a 3D k-NN label vote over the training cells.
   The k-NN vote exploits the strong spatial autocorrelation of cell identity;
   the neural posterior supplies a smooth continuous prior and fills gaps where
   training cells are far away (large z gaps). The fusion is a
   log-linear (geometric) blend, so both sources must agree for high confidence.
   Because expression is looked up per predicted type, better types also lift
   every expression metric.

2. **Neural / k-NN expression fusion + moment calibration.** Expression is a
   blend of the FiLM neural field and the cell-type-conditioned k-NN, followed
   by an optional per-gene affine calibration to the training moments. The
   affine step is monotonic per gene (it leaves per-gene Pearson/Spearman
   unchanged) but repairs the variance shrinkage that pure k-NN averaging
   causes — lifting the gene-variance metric and reducing RMSE.
"""

import numpy as np
import torch
import anndata as ad
import pandas as pd
from scipy.spatial import cKDTree


# ── k-NN helpers ──────────────────────────────────────────────────────────────

def _weighted_3d(xy, z, z_weight):
    return np.hstack([np.asarray(xy, np.float64),
                      (np.asarray(z, np.float64) * z_weight).reshape(-1, 1)])


def knn_celltype_vote(pred_xy, pred_z, train_xy, train_z, train_ct, n_types,
                      k=15, z_weight=3.0, temperature=1.0):
    """Return (N, n_types) soft label distribution from a 3D k-NN vote."""
    pred_3d = _weighted_3d(pred_xy, pred_z, z_weight)
    train_3d = _weighted_3d(train_xy, train_z, z_weight)
    k = min(k, len(train_3d))
    tree = cKDTree(train_3d)
    dists, idx = tree.query(pred_3d, k=k)
    if k == 1:
        dists, idx = dists.reshape(-1, 1), idx.reshape(-1, 1)
    # distance-weighted soft vote
    scale = np.median(dists[:, -1]) + 1e-8
    w = np.exp(-(dists / (temperature * scale)) ** 2)
    w /= w.sum(axis=1, keepdims=True) + 1e-8
    probs = np.zeros((len(pred_3d), n_types), dtype=np.float64)
    nbr_ct = train_ct[idx]                       # (N, k)
    for j in range(k):
        np.add.at(probs, (np.arange(len(pred_3d)), nbr_ct[:, j]), w[:, j])
    probs /= probs.sum(axis=1, keepdims=True) + 1e-8
    return probs


def knn_expr_by_celltype(pred_xy, pred_z, pred_ct, train_expr, train_xy, train_z,
                         train_ct, k=15, z_weight=3.0):
    """Cell-type-conditioned IDW expression from same-type training cells."""
    n_pred, n_genes = len(pred_ct), train_expr.shape[1]
    out = np.zeros((n_pred, n_genes), dtype=np.float32)
    for ct in np.unique(pred_ct):
        pidx = np.where(pred_ct == ct)[0]
        tidx = np.where(train_ct == ct)[0]
        if len(tidx) == 0:
            tidx = np.arange(len(train_ct))       # fallback: all cells
        p3 = _weighted_3d(pred_xy[pidx], pred_z[pidx], z_weight)
        t3 = _weighted_3d(train_xy[tidx], train_z[tidx], z_weight)
        ka = min(k, len(t3))
        tree = cKDTree(t3)
        d, ii = tree.query(p3, k=ka)
        if ka == 1:
            d, ii = d.reshape(-1, 1), ii.reshape(-1, 1)
        w = 1.0 / (d + 1e-8)
        w /= w.sum(axis=1, keepdims=True)
        te = train_expr[tidx]
        out[pidx] = np.einsum("nk,nkg->ng", w.astype(np.float32), te[ii])
    return out


def unconstrained_knn_mean(pred_xy, pred_z, train_expr, train_xy, train_z,
                           k=8, z_weight=3.0):
    """Per-gene mean of a type-agnostic 3D spatial k-NN reconstruction.

    Ignoring cell type, each predicted location borrows expression from its
    nearest training cells regardless of identity. The *global mean* of this
    reconstruction tracks the true slice's per-gene mean well because it follows
    the true local cell density — it does not depend on getting the cell-type
    composition exactly right. It is used purely as a per-gene mean target.
    """
    pred_3d = _weighted_3d(pred_xy, pred_z, z_weight)
    train_3d = _weighted_3d(train_xy, train_z, z_weight)
    ka = min(k, len(train_3d))
    tree = cKDTree(train_3d)
    d, ii = tree.query(pred_3d, k=ka)
    if ka == 1:
        d, ii = d.reshape(-1, 1), ii.reshape(-1, 1)
    w = 1.0 / (d + 1e-8)
    w /= w.sum(axis=1, keepdims=True)
    est = np.einsum("nk,nkg->ng", w.astype(np.float32), train_expr[ii])
    return est.mean(axis=0)


def mean_anchor(pred, target_mean, strength=1.0):
    """Shift each gene's predicted mean toward ``target_mean``.

    This is a pure additive per-gene offset, so it leaves every cell-wise and
    gene-wise *correlation* unchanged while improving per-gene *mean*
    reproduction (and RMSE/MAE). Variance and spatial structure are untouched.
    """
    cur = pred.mean(axis=0)
    return (pred + strength * (target_mean - cur)).astype(np.float32)


def calibrate_moments(pred, target_mean, target_std, strength=1.0):
    """Per-gene affine map so pred matches target moments (monotone per gene)."""
    mu = pred.mean(axis=0)
    sd = pred.std(axis=0) + 1e-8
    scale = 1.0 + strength * ((target_std / sd) - 1.0)
    scale = np.clip(scale, 0.25, 4.0)
    out = (pred - mu) * scale + (mu + strength * (target_mean - mu))
    return out.astype(np.float32)


def spatial_smooth(expression, positions, k=20, sigma_mult=1.5):
    """Gaussian-weighted k-NN spatial smoothing (kept for parity with v1)."""
    tree = cKDTree(positions)
    dists, indices = tree.query(positions, k=k + 1)
    sigma = sigma_mult * np.median(dists[:, 1])
    out = np.zeros_like(expression)
    for i in range(len(positions)):
        w = np.exp(-dists[i] ** 2 / (2 * sigma ** 2))
        w /= w.sum()
        out[i] = (w[:, None] * expression[indices[i]]).sum(axis=0)
    return out


class VirtualSliceGenerator:
    """
    Generate virtual slices from a trained SpatialCPAv2 model.

    Parameters
    ----------
    model : SpatialCPAv2
    cell_type_names, gene_names : list[str]
    region_names : list[str] or None
    device : str
    train_sections : list[SpatialSection] or None
        Enables k-NN cell-type voting, k-NN expression lookup, and moment
        calibration. Strongly recommended.
    """

    def __init__(self, model, cell_type_names, gene_names, region_names=None,
                 device="cpu", train_sections=None):
        self.model = model.to(device)
        self.model.eval()
        self.cell_type_names = cell_type_names
        self.gene_names = gene_names
        self.region_names = region_names
        self.device = device
        self.n_types = len(cell_type_names)

        self.train_expr = self.train_xy = self.train_z = self.train_ct = None
        self.gene_mean = self.gene_std = None
        if train_sections:
            self.train_expr = np.concatenate([s.expression for s in train_sections], 0)
            self.train_xy = np.concatenate([s.coords_xy for s in train_sections], 0)
            self.train_z = np.concatenate([s.z_values for s in train_sections], 0)
            self.train_ct = np.concatenate([s.cell_type_indices for s in train_sections], 0)
            self.gene_mean = self.train_expr.mean(axis=0)
            self.gene_std = self.train_expr.std(axis=0)

    @torch.no_grad()
    def _neural_ct_probs(self, coords_3d, batch_size):
        out = []
        for s in range(0, len(coords_3d), batch_size):
            b = torch.tensor(coords_3d[s:s + batch_size], device=self.device)
            out.append(self.model.predict_cell_type(b).cpu().numpy())
        return np.concatenate(out, 0)

    @torch.no_grad()
    def _neural_expr(self, coords_3d, ct, batch_size):
        out = []
        for s in range(0, len(coords_3d), batch_size):
            b = torch.tensor(coords_3d[s:s + batch_size], device=self.device)
            bc = torch.tensor(ct[s:s + batch_size], dtype=torch.long, device=self.device)
            out.append(self.model.predict_expression(b, bc).cpu().numpy())
        return np.concatenate(out, 0)

    @torch.no_grad()
    def generate_matching(self, reference_adata, cell_type_key="cell_class",
                          true_cell_types=None,
                          knn_k=8, knn_z_weight=3.0,
                          ct_knn_weight=0.7, expr_knn_alpha=0.3,
                          calibrate=False, calibrate_strength=1.0,
                          anchor_mean=True, anchor_strength=1.0, anchor_k=5,
                          smooth_k=0, smooth_sigma=1.5, batch_size=4096):
        """
        Predict cell type + expression at the reference slice's cell positions.

        Parameters
        ----------
        ct_knn_weight : float in [0, 1]
            Weight of the k-NN vote in the geometric cell-type blend
            (0 = pure neural posterior, 1 = pure k-NN vote).
        expr_knn_alpha : float in [0, 1]
            Blend for expression: alpha * neural + (1 - alpha) * kNN.
        calibrate : bool
            Apply per-gene moment calibration to the training statistics.
        """
        if "spatial" in reference_adata.obsm:
            positions = np.asarray(reference_adata.obsm["spatial"])[:, :2].astype(np.float32)
        elif "x" in reference_adata.obs.columns:
            positions = np.column_stack([reference_adata.obs["x"].values,
                                         reference_adata.obs["y"].values]).astype(np.float32)
        else:
            raise ValueError("Cannot find spatial coordinates in reference")
        z_values = reference_adata.obs["z"].values.astype(np.float32)
        coords_3d = np.hstack([positions, z_values.reshape(-1, 1)]).astype(np.float32)
        n = len(positions)

        # ── Cell type ─────────────────────────────────────────────────────────
        if true_cell_types is not None:
            cell_types = np.asarray(true_cell_types, dtype=np.int64)
            ct_probs = None
        else:
            p_nn = self._neural_ct_probs(coords_3d, batch_size)
            if self.train_ct is not None and ct_knn_weight > 0:
                p_knn = knn_celltype_vote(
                    positions, z_values, self.train_xy, self.train_z, self.train_ct,
                    self.n_types, k=knn_k, z_weight=knn_z_weight)
                # geometric (log-linear) blend
                eps = 1e-8
                logp = ((1 - ct_knn_weight) * np.log(p_nn + eps)
                        + ct_knn_weight * np.log(p_knn + eps))
                ct_probs = np.exp(logp)
                ct_probs /= ct_probs.sum(axis=1, keepdims=True) + eps
            else:
                ct_probs = p_nn
            cell_types = ct_probs.argmax(axis=1)
        cell_type_labels = [self.cell_type_names[i] for i in cell_types]

        # ── Expression ────────────────────────────────────────────────────────
        expr_knn = None
        if self.train_expr is not None:
            expr_knn = knn_expr_by_celltype(
                positions, z_values, cell_types, self.train_expr,
                self.train_xy, self.train_z, self.train_ct,
                k=knn_k, z_weight=knn_z_weight)

        expr_nn = None
        if expr_knn_alpha > 0:
            if ct_probs is not None:
                # posterior-marginalised neural prediction (coherent with head)
                expr_nn = self._neural_expr_marginal(coords_3d, ct_probs, batch_size)
            else:
                expr_nn = self._neural_expr(coords_3d, cell_types, batch_size)

        if expr_knn is not None and expr_nn is not None:
            expr_all = expr_knn_alpha * expr_nn + (1 - expr_knn_alpha) * expr_knn
        elif expr_knn is not None:
            expr_all = expr_knn
        elif expr_nn is not None:
            expr_all = expr_nn
        else:
            expr_all = self._neural_expr(coords_3d, cell_types, batch_size)

        if calibrate and self.gene_mean is not None:
            expr_all = calibrate_moments(expr_all, self.gene_mean, self.gene_std,
                                         strength=calibrate_strength)
        elif anchor_mean and self.train_expr is not None:
            target_mean = unconstrained_knn_mean(
                positions, z_values, self.train_expr, self.train_xy, self.train_z,
                k=anchor_k, z_weight=knn_z_weight)
            expr_all = mean_anchor(expr_all, target_mean, strength=anchor_strength)

        if smooth_k > 0:
            expr_all = spatial_smooth(expr_all, positions, k=smooth_k, sigma_mult=smooth_sigma)

        obs = pd.DataFrame({"cell_class": cell_type_labels, "cell_type_idx": cell_types})
        obs.index = reference_adata.obs.index.copy()
        virtual = ad.AnnData(X=expr_all.astype(np.float32), obs=obs,
                             var=reference_adata.var.copy())
        virtual.obsm["spatial"] = positions.astype(np.float32)
        return virtual

    @torch.no_grad()
    def _neural_expr_marginal(self, coords_3d, ct_probs, batch_size, top_k=3):
        out = []
        for s in range(0, len(coords_3d), batch_size):
            b = torch.tensor(coords_3d[s:s + batch_size], device=self.device)
            p = torch.tensor(ct_probs[s:s + batch_size], dtype=torch.float32,
                             device=self.device)
            out.append(self.model.predict_expression_marginal(b, p, top_k=top_k).cpu().numpy())
        return np.concatenate(out, 0)
