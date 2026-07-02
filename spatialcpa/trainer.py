"""
SpatialCPA Trainer with MSE + Pearson correlation loss,
z-marginalization, and gap-aware leave-one-out.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from spatialcpa.model import SpatialCPA
from spatialcpa.data import SectionDataset, compute_gap_weights
from spatialcpa.heads import zinb_log_prob


def pearson_corr_loss(predicted, target):
    """
    Compute 1 - mean(per-gene Pearson correlation).

    Directly optimises the gene-wise Pearson r evaluation metric.
    """
    # predicted, target: (N, G)
    N = predicted.shape[0]
    if N < 4:
        return torch.tensor(0.0, device=predicted.device)

    pred_mean = predicted.mean(dim=0, keepdim=True)
    tgt_mean = target.mean(dim=0, keepdim=True)
    pred_centered = predicted - pred_mean
    tgt_centered = target - tgt_mean

    # Per-gene correlation
    cov = (pred_centered * tgt_centered).sum(dim=0)
    pred_std = (pred_centered ** 2).sum(dim=0).sqrt().clamp(min=1e-8)
    tgt_std = (tgt_centered ** 2).sum(dim=0).sqrt().clamp(min=1e-8)

    corr = cov / (pred_std * tgt_std)
    return 1.0 - corr.mean()


class SpatialCPATrainer:
    """
    End-to-end trainer for SpatialCPA.

    Supports both MSE (for pre-normalized data) and ZINB (for count data) modes.

    Parameters
    ----------
    model : SpatialCPA
    sections : list of SpatialSection
    device : str
    lr : float
    batch_size : int
    n_z_samples : int
        Number of z-samples for slab marginalization.
    z_jitter : float
        Max z-offset for z-marginalization (half-thickness of slab).
    loo_weight : float
        Weight for leave-one-out loss.
    expression_weight : float
        Weight for expression loss.
    corr_weight : float
        Weight for Pearson correlation loss (MSE mode only).
    """

    def __init__(
        self,
        model,
        sections,
        device='cpu',
        lr=1e-3,
        batch_size=512,
        n_z_samples=5,
        z_jitter=0.5,
        loo_weight=0.5,
        expression_weight=1.0,
        corr_weight=0.5,
    ):
        self.model = model.to(device)
        self.sections = sections
        self.device = device
        self.batch_size = batch_size
        self.n_z_samples = n_z_samples
        self.z_jitter = z_jitter
        self.loo_weight = loo_weight
        self.expression_weight = expression_weight
        self.corr_weight = corr_weight

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                            weight_decay=1e-4)

        self.dataset = SectionDataset(sections)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device != 'cpu'), drop_last=True,
        )

        self.gap_sizes, self.section_loo_weights = compute_gap_weights(sections)

    def _build_coords_with_z_margin(self, batch):
        """
        Construct 3D coordinates with z-marginalization.

        Uses each cell's actual z with small jitter offsets to model
        section thickness (slab model).
        """
        xy = batch['xy'].to(self.device)       # (N, 2)
        z = batch['z'].to(self.device)          # (N,) — actual per-cell z
        N = xy.shape[0]
        K = self.n_z_samples

        # Sample z offsets within [-z_jitter, +z_jitter]
        z_offsets = (torch.rand(N, K, device=self.device) - 0.5) * 2 * self.z_jitter
        z_samples = z.unsqueeze(1) + z_offsets  # (N, K)

        # Repeat xy for each z-sample
        xy_rep = xy.unsqueeze(1).expand(N, K, 2).reshape(N * K, 2)
        z_flat = z_samples.reshape(N * K, 1)
        coords = torch.cat([xy_rep, z_flat], dim=1)  # (N*K, 3)

        return coords, N

    def _supervised_step(self, batch):
        """Standard supervised training step with z-marginalization."""
        coords, N = self._build_coords_with_z_margin(batch)
        K = self.n_z_samples

        cell_type = batch['cell_type'].to(self.device)
        expression = batch['expression'].to(self.device)
        region = batch['region'].to(self.device)

        ct_rep = cell_type.unsqueeze(1).expand(N, K).reshape(N * K)

        out = self.model(coords, ct_rep)

        # Cell type loss: average logits over z-samples then cross-entropy
        ct_logits = out['cell_type_logits'].reshape(N, K, -1).mean(dim=1)
        ct_loss = F.cross_entropy(ct_logits, cell_type)

        # Region loss
        region_loss = torch.tensor(0.0, device=self.device)
        if out['region_logits'] is not None and (region >= 0).any():
            valid = region >= 0
            if valid.any():
                reg_logits = out['region_logits'].reshape(N, K, -1).mean(dim=1)
                region_loss = F.cross_entropy(reg_logits[valid], region[valid])

        # Expression loss
        if self.model.use_zinb:
            expr_rep = expression.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)
            log_prob = zinb_log_prob(expr_rep, out['mu'], out['theta'],
                                     out['pi_logits'])
            log_prob_avg = log_prob.reshape(N, K, -1).mean(dim=1)
            expr_loss = -log_prob_avg.mean()
            corr_loss = torch.tensor(0.0, device=self.device)
        else:
            predicted = out['predicted_expr'].reshape(N, K, -1).mean(dim=1)
            expr_loss = F.mse_loss(predicted, expression)
            corr_loss = pearson_corr_loss(predicted, expression)

        return {
            'ct_loss': ct_loss,
            'region_loss': region_loss,
            'expr_loss': expr_loss,
            'corr_loss': corr_loss,
        }

    def _loo_step(self):
        """Gap-aware leave-one-out self-supervision step."""
        n_sections = len(self.sections)
        if n_sections < 3:
            return torch.tensor(0.0, device=self.device)

        loo_idx = np.random.choice(n_sections, p=self.section_loo_weights)
        held_out = self.sections[loo_idx]

        n_sample = min(held_out.n_cells, self.batch_size)
        sample_idx = np.random.choice(held_out.n_cells, n_sample, replace=False)

        # Use actual per-cell z (no marginalization for LOO)
        xy = torch.tensor(held_out.coords_xy[sample_idx], dtype=torch.float32,
                          device=self.device)
        z = torch.tensor(held_out.z_values[sample_idx], dtype=torch.float32,
                         device=self.device).unsqueeze(1)
        coords = torch.cat([xy, z], dim=1)

        ct = torch.tensor(held_out.cell_type_indices[sample_idx],
                          dtype=torch.long, device=self.device)
        expr = torch.tensor(held_out.expression[sample_idx],
                            dtype=torch.float32, device=self.device)

        out = self.model(coords, ct)

        ct_loss = F.cross_entropy(out['cell_type_logits'], ct)

        if self.model.use_zinb:
            log_prob = zinb_log_prob(expr, out['mu'], out['theta'],
                                     out['pi_logits'])
            expr_loss = -log_prob.mean()
        else:
            expr_loss = F.mse_loss(out['predicted_expr'], expr)

        # Weight by gap size
        gap_weight = 1.0
        if loo_idx > 0 and loo_idx < n_sections - 1:
            left_gap = held_out.z_center - self.sections[loo_idx - 1].z_center
            right_gap = self.sections[loo_idx + 1].z_center - held_out.z_center
            avg_gap = (left_gap + right_gap) / 2.0
            median_gap = np.median(self.gap_sizes) if len(self.gap_sizes) > 0 else 1.0
            gap_weight = avg_gap / max(median_gap, 1e-6)

        return gap_weight * (ct_loss + self.expression_weight * expr_loss)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        losses = {'ct': [], 'region': [], 'expr': [], 'corr': [],
                  'loo': [], 'total': []}

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            sup = self._supervised_step(batch)
            loss = sup['ct_loss'] + sup['region_loss'] + \
                   self.expression_weight * sup['expr_loss'] + \
                   self.corr_weight * sup['corr_loss']

            # LOO step
            loo_loss = torch.tensor(0.0, device=self.device)
            if np.random.rand() < 0.3 and len(self.sections) >= 3:
                loo_loss = self._loo_step()
                loss = loss + self.loo_weight * loo_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses['ct'].append(sup['ct_loss'].item())
            losses['region'].append(sup['region_loss'].item())
            losses['expr'].append(sup['expr_loss'].item())
            losses['corr'].append(sup['corr_loss'].item())
            losses['loo'].append(loo_loss.item())
            losses['total'].append(loss.item())

        return {k: np.mean(v) for k, v in losses.items()}

    def train(self, n_epochs=100, verbose=True):
        """Full training loop with cosine annealing."""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs,
            eta_min=self.optimizer.param_groups[0]['lr'] * 0.01
        )

        history = []
        pbar = tqdm(range(n_epochs), desc='Training SpatialCPA',
                    disable=not verbose)

        for epoch in pbar:
            epoch_losses = self.train_epoch()
            scheduler.step()
            history.append(epoch_losses)

            if verbose:
                pbar.set_postfix({
                    'total': f"{epoch_losses['total']:.4f}",
                    'ct': f"{epoch_losses['ct']:.4f}",
                    'expr': f"{epoch_losses['expr']:.2f}",
                    'corr': f"{epoch_losses['corr']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

        return history
