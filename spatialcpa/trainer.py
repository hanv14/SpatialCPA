"""
SpatialCPA Trainer with z-marginalization and gap-aware leave-one-out.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from spatialcpa.model import SpatialCPA
from spatialcpa.data import SectionDataset, compute_gap_weights, SpatialSection
from spatialcpa.heads import zinb_log_prob
from spatialcpa.fourier import FourierFeatureEncoder


class SpatialCPATrainer:
    """
    End-to-end trainer for SpatialCPA.

    Handles:
    - Standard supervised training with z-marginalization (slab model)
    - Gap-aware leave-one-out self-supervision
    - Multi-head loss balancing

    Parameters
    ----------
    model : SpatialCPA
        The model to train.
    sections : list of SpatialSection
        Training sections.
    device : str
        'cuda' or 'cpu'.
    lr : float
        Learning rate.
    batch_size : int
        Batch size for training.
    n_z_samples : int
        Number of z-samples for slab marginalization.
    loo_weight : float
        Weight for leave-one-out loss relative to supervised loss.
    expression_weight : float
        Weight for expression loss relative to classification losses.
    """

    def __init__(
        self,
        model,
        sections,
        device='cpu',
        lr=1e-3,
        batch_size=512,
        n_z_samples=5,
        loo_weight=0.5,
        expression_weight=1.0,
    ):
        self.model = model.to(device)
        self.sections = sections
        self.device = device
        self.batch_size = batch_size
        self.n_z_samples = n_z_samples
        self.loo_weight = loo_weight
        self.expression_weight = expression_weight

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                            weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.01
        )

        # Build dataset
        self.dataset = SectionDataset(sections)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device != 'cpu'), drop_last=True,
        )

        # Gap-aware weights
        self.gap_sizes, self.section_loo_weights = compute_gap_weights(sections)

    def _build_coords_with_z_margin(self, batch):
        """
        Construct 3D coordinates with z-marginalization.

        For each cell, sample n_z_samples z-values uniformly within the
        section thickness, creating augmented batches.

        Returns
        -------
        coords : (N * n_z_samples, 3) tensor
        Repeat indices for averaging later.
        """
        xy = batch['xy'].to(self.device)                  # (N, 2)
        z_center = batch['z_center'].to(self.device)       # (N,)
        z_thick = batch['z_thickness'].to(self.device)     # (N,)
        N = xy.shape[0]

        # Sample z offsets within [-thickness/2, +thickness/2]
        z_offsets = (torch.rand(N, self.n_z_samples, device=self.device) - 0.5) \
                    * z_thick.unsqueeze(1)  # (N, n_z_samples)
        z_samples = z_center.unsqueeze(1) + z_offsets      # (N, n_z_samples)

        # Repeat xy for each z-sample
        xy_rep = xy.unsqueeze(1).expand(N, self.n_z_samples, 2) \
                    .reshape(N * self.n_z_samples, 2)
        z_flat = z_samples.reshape(N * self.n_z_samples, 1)
        coords = torch.cat([xy_rep, z_flat], dim=1)        # (N*K, 3)

        return coords, N

    def _supervised_step(self, batch):
        """
        Standard supervised training step with z-marginalization.

        Returns dict of losses.
        """
        coords, N = self._build_coords_with_z_margin(batch)
        K = self.n_z_samples

        cell_type = batch['cell_type'].to(self.device)     # (N,)
        expression = batch['expression'].to(self.device)   # (N, G)
        region = batch['region'].to(self.device)           # (N,)

        # Repeat targets for z-samples
        ct_rep = cell_type.unsqueeze(1).expand(N, K).reshape(N * K)
        expr_rep = expression.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)

        # Forward pass
        out = self.model(coords, ct_rep)

        # Cell type loss: average over z-samples then cross-entropy
        ct_logits = out['cell_type_logits'].reshape(N, K, -1).mean(dim=1)  # (N, C)
        ct_loss = F.cross_entropy(ct_logits, cell_type)

        # Region loss (if available)
        region_loss = torch.tensor(0.0, device=self.device)
        if out['region_logits'] is not None and (region >= 0).any():
            valid = region >= 0
            if valid.any():
                reg_logits = out['region_logits'].reshape(N, K, -1).mean(dim=1)
                region_loss = F.cross_entropy(reg_logits[valid], region[valid])

        # Expression loss (ZINB negative log-likelihood)
        log_prob = zinb_log_prob(expr_rep, out['mu'], out['theta'], out['pi_logits'])
        # Average over z-samples per cell, then over cells and genes
        log_prob_avg = log_prob.reshape(N, K, -1).mean(dim=1)  # (N, G)
        expr_loss = -log_prob_avg.mean()

        return {
            'ct_loss': ct_loss,
            'region_loss': region_loss,
            'expr_loss': expr_loss,
        }

    def _loo_step(self):
        """
        Gap-aware leave-one-out self-supervision step.

        Randomly hold out one section (weighted by gap size), predict its
        cells using the model, and compute loss.

        Returns expression + cell type loss on held-out section.
        """
        # Sample section to hold out
        n_sections = len(self.sections)
        if n_sections < 3:
            return torch.tensor(0.0, device=self.device)

        loo_idx = np.random.choice(n_sections, p=self.section_loo_weights)
        held_out = self.sections[loo_idx]

        # Sample a minibatch from held-out section
        n_sample = min(held_out.n_cells, self.batch_size)
        sample_idx = np.random.choice(held_out.n_cells, n_sample, replace=False)

        # Get exact z position (no marginalization for LOO)
        xy = torch.tensor(held_out.coords_xy[sample_idx], dtype=torch.float32,
                          device=self.device)
        z = torch.full((n_sample, 1), held_out.z_position, dtype=torch.float32,
                       device=self.device)
        coords = torch.cat([xy, z], dim=1)

        ct = torch.tensor(held_out.cell_type_indices[sample_idx],
                          dtype=torch.long, device=self.device)
        expr = torch.tensor(held_out.expression[sample_idx],
                            dtype=torch.float32, device=self.device)

        # Forward
        out = self.model(coords, ct)

        # Cell type loss
        ct_loss = F.cross_entropy(out['cell_type_logits'], ct)

        # Expression loss
        log_prob = zinb_log_prob(expr, out['mu'], out['theta'], out['pi_logits'])
        expr_loss = -log_prob.mean()

        # Weight by gap size
        gap_weight = 1.0
        if loo_idx > 0 and loo_idx < n_sections - 1:
            left_gap = self.sections[loo_idx].z_position - \
                       self.sections[loo_idx - 1].z_position
            right_gap = self.sections[loo_idx + 1].z_position - \
                        self.sections[loo_idx].z_position
            avg_gap = (left_gap + right_gap) / 2.0
            median_gap = np.median(self.gap_sizes) if len(self.gap_sizes) > 0 else 1.0
            gap_weight = avg_gap / max(median_gap, 1e-6)

        return gap_weight * (ct_loss + self.expression_weight * expr_loss)

    def train_epoch(self):
        """Train for one epoch. Returns dict of mean losses."""
        self.model.train()
        losses = {'ct': [], 'region': [], 'expr': [], 'loo': [], 'total': []}

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            # Supervised step
            sup = self._supervised_step(batch)
            loss = sup['ct_loss'] + sup['region_loss'] + \
                   self.expression_weight * sup['expr_loss']

            # LOO step (with some probability to save compute)
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
            losses['loo'].append(loo_loss.item())
            losses['total'].append(loss.item())

        self.scheduler.step()

        return {k: np.mean(v) for k, v in losses.items()}

    def train(self, n_epochs=100, verbose=True):
        """
        Full training loop.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        verbose : bool
            Print progress.

        Returns
        -------
        history : list of dicts with per-epoch losses.
        """
        history = []
        pbar = tqdm(range(n_epochs), desc='Training SpatialCPA',
                    disable=not verbose)

        for epoch in pbar:
            epoch_losses = self.train_epoch()
            history.append(epoch_losses)

            if verbose:
                pbar.set_postfix({
                    'total': f"{epoch_losses['total']:.4f}",
                    'ct': f"{epoch_losses['ct']:.4f}",
                    'expr': f"{epoch_losses['expr']:.4f}",
                    'loo': f"{epoch_losses['loo']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

        return history
