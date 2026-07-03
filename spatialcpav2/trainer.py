"""
Trainer for SpatialCPA v2.

Improvements over v1
--------------------
* **Class-balanced cell-type loss.** v1 used plain cross-entropy, which on the
  imbalanced cell-type distributions of real tissue collapses onto the majority
  classes (v1 reached ~13% accuracy / ~0.06 macro-F1 on STARmap). v2 uses
  inverse-frequency class weights plus optional label smoothing, directly
  lifting macro-F1.
* **Metric-aligned expression losses.** In addition to MSE + per-gene Pearson,
  v2 adds per-gene *mean* and *variance* matching terms (the FEAST metrics) and
  a soft, posterior-marginalised consistency term that ties the cell-type head
  and expression head together.
* **Cleaner z-marginalisation and gap-aware LOO**, retained from v1.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from spatialcpav2.data import SectionDataset, compute_gap_weights
from spatialcpav2.heads import zinb_log_prob


def pearson_corr_loss(pred, target):
    """1 - mean per-gene Pearson r (directly optimises the eval metric)."""
    if pred.shape[0] < 4:
        return torch.tensor(0.0, device=pred.device)
    pc = pred - pred.mean(0, keepdim=True)
    tc = target - target.mean(0, keepdim=True)
    cov = (pc * tc).sum(0)
    ps = (pc ** 2).sum(0).sqrt().clamp(min=1e-8)
    ts = (tc ** 2).sum(0).sqrt().clamp(min=1e-8)
    return 1.0 - (cov / (ps * ts)).mean()


def moment_loss(pred, target):
    """Per-gene mean and variance matching (FEAST's gene_mean/var metrics)."""
    if pred.shape[0] < 4:
        return torch.tensor(0.0, device=pred.device)
    mean_l = F.mse_loss(pred.mean(0), target.mean(0))
    var_l = F.mse_loss(pred.var(0, unbiased=False), target.var(0, unbiased=False))
    return mean_l + var_l


class SpatialCPAv2Trainer:
    def __init__(
        self,
        model,
        sections,
        device="cpu",
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=512,
        n_z_samples=5,
        z_jitter=0.5,
        loo_weight=0.5,
        expression_weight=1.0,
        corr_weight=0.5,
        moment_weight=0.5,
        marginal_weight=0.3,
        ct_label_smoothing=0.05,
        class_balanced=True,
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
        self.moment_weight = moment_weight
        self.marginal_weight = marginal_weight
        self.ct_label_smoothing = ct_label_smoothing

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                            weight_decay=weight_decay)
        self.dataset = SectionDataset(sections)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device != "cpu"), drop_last=True)
        self.gap_sizes, self.section_loo_weights = compute_gap_weights(sections)

        # ── Class-balanced weights from training-cell frequencies ────────────
        self.ct_weight = None
        if class_balanced:
            counts = np.zeros(model.n_cell_types, dtype=np.float64)
            for s in sections:
                for c in s.cell_type_indices:
                    counts[c] += 1
            counts = np.maximum(counts, 1.0)
            w = counts.sum() / (len(counts) * counts)   # inverse frequency
            w = np.clip(w, 0.2, 5.0)                     # avoid extreme weights
            self.ct_weight = torch.tensor(w, dtype=torch.float32, device=device)

    def _ce(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.ct_weight,
                               label_smoothing=self.ct_label_smoothing)

    def _build_coords_with_z_margin(self, batch):
        xy = batch["xy"].to(self.device)
        z = batch["z"].to(self.device)
        N, K = xy.shape[0], self.n_z_samples
        z_off = (torch.rand(N, K, device=self.device) - 0.5) * 2 * self.z_jitter
        z_samples = z.unsqueeze(1) + z_off
        xy_rep = xy.unsqueeze(1).expand(N, K, 2).reshape(N * K, 2)
        z_flat = z_samples.reshape(N * K, 1)
        return torch.cat([xy_rep, z_flat], dim=1), N

    def _supervised_step(self, batch, do_marginal=True):
        coords, N = self._build_coords_with_z_margin(batch)
        K = self.n_z_samples
        cell_type = batch["cell_type"].to(self.device)
        expression = batch["expression"].to(self.device)
        region = batch["region"].to(self.device)
        ct_rep = cell_type.unsqueeze(1).expand(N, K).reshape(N * K)

        out = self.model(coords, ct_rep)

        ct_logits = out["cell_type_logits"].reshape(N, K, -1).mean(1)
        ct_loss = self._ce(ct_logits, cell_type)

        region_loss = torch.tensor(0.0, device=self.device)
        if out["region_logits"] is not None and (region >= 0).any():
            valid = region >= 0
            reg_logits = out["region_logits"].reshape(N, K, -1).mean(1)
            region_loss = F.cross_entropy(reg_logits[valid], region[valid])

        if self.model.use_zinb:
            expr_rep = expression.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)
            lp = zinb_log_prob(expr_rep, out["mu"], out["theta"], out["pi_logits"])
            expr_loss = -lp.reshape(N, K, -1).mean(1).mean()
            corr_loss = torch.tensor(0.0, device=self.device)
            moment_l = torch.tensor(0.0, device=self.device)
            marginal_l = torch.tensor(0.0, device=self.device)
        else:
            pred = out["predicted_expr"].reshape(N, K, -1).mean(1)
            expr_loss = F.mse_loss(pred, expression)
            corr_loss = pearson_corr_loss(pred, expression)
            moment_l = moment_loss(pred, expression)
            # Posterior-marginalised consistency: expression under the *predicted*
            # type distribution should also match the target. Ties the two heads
            # together. h is detached so this term only trains the decoder (the
            # backbone is already trained by the main expression loss), which
            # keeps the extra forward/backward cheap.
            if self.marginal_weight > 0 and do_marginal:
                h = out["h"].reshape(N, K, -1).mean(1).detach()
                probs = F.softmax(ct_logits.detach(), dim=-1)
                marg = self.model.expression_decoder.forward_marginal(h, probs, top_k=2)
                marginal_l = F.mse_loss(marg, expression)
            else:
                marginal_l = torch.tensor(0.0, device=self.device)

        return {"ct_loss": ct_loss, "region_loss": region_loss,
                "expr_loss": expr_loss, "corr_loss": corr_loss,
                "moment_loss": moment_l, "marginal_loss": marginal_l}

    def _loo_step(self):
        n = len(self.sections)
        if n < 3:
            return torch.tensor(0.0, device=self.device)
        loo_idx = np.random.choice(n, p=self.section_loo_weights)
        held = self.sections[loo_idx]
        ns = min(held.n_cells, self.batch_size)
        si = np.random.choice(held.n_cells, ns, replace=False)
        xy = torch.tensor(held.coords_xy[si], dtype=torch.float32, device=self.device)
        z = torch.tensor(held.z_values[si], dtype=torch.float32,
                         device=self.device).unsqueeze(1)
        coords = torch.cat([xy, z], dim=1)
        ct = torch.tensor(held.cell_type_indices[si], dtype=torch.long, device=self.device)
        expr = torch.tensor(held.expression[si], dtype=torch.float32, device=self.device)
        out = self.model(coords, ct)
        ct_loss = self._ce(out["cell_type_logits"], ct)
        if self.model.use_zinb:
            expr_loss = -zinb_log_prob(expr, out["mu"], out["theta"], out["pi_logits"]).mean()
        else:
            expr_loss = F.mse_loss(out["predicted_expr"], expr) + \
                        self.corr_weight * pearson_corr_loss(out["predicted_expr"], expr)
        gap_weight = 1.0
        if 0 < loo_idx < n - 1:
            lg = held.z_center - self.sections[loo_idx - 1].z_center
            rg = self.sections[loo_idx + 1].z_center - held.z_center
            med = np.median(self.gap_sizes) if len(self.gap_sizes) else 1.0
            gap_weight = ((lg + rg) / 2.0) / max(med, 1e-6)
        return gap_weight * (ct_loss + self.expression_weight * expr_loss)

    def train_epoch(self):
        self.model.train()
        agg = {k: [] for k in
               ["ct", "region", "expr", "corr", "moment", "marginal", "loo", "total"]}
        for bi, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            s = self._supervised_step(batch, do_marginal=(bi % 2 == 0))
            loss = (s["ct_loss"] + s["region_loss"]
                    + self.expression_weight * s["expr_loss"]
                    + self.corr_weight * s["corr_loss"]
                    + self.moment_weight * s["moment_loss"]
                    + self.marginal_weight * s["marginal_loss"])
            loo_loss = torch.tensor(0.0, device=self.device)
            if np.random.rand() < 0.3 and len(self.sections) >= 3:
                loo_loss = self._loo_step()
                loss = loss + self.loo_weight * loo_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            _f = lambda t: t.detach().item() if torch.is_tensor(t) else float(t)
            agg["ct"].append(_f(s["ct_loss"]))
            agg["region"].append(_f(s["region_loss"]))
            agg["expr"].append(_f(s["expr_loss"]))
            agg["corr"].append(_f(s["corr_loss"]))
            agg["moment"].append(_f(s["moment_loss"]))
            agg["marginal"].append(_f(s["marginal_loss"]))
            agg["loo"].append(_f(loo_loss))
            agg["total"].append(_f(loss))
        return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}

    def train(self, n_epochs=100, verbose=True, warmup_frac=0.05):
        base_lr = self.optimizer.param_groups[0]["lr"]
        warmup = max(1, int(n_epochs * warmup_frac))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, n_epochs - warmup), eta_min=base_lr * 0.01)
        history = []
        pbar = tqdm(range(n_epochs), desc="Training SpatialCPAv2", disable=not verbose)
        for epoch in pbar:
            if epoch < warmup:  # linear warmup
                for g in self.optimizer.param_groups:
                    g["lr"] = base_lr * (epoch + 1) / warmup
            el = self.train_epoch()
            if epoch >= warmup:
                sched.step()
            history.append(el)
            if verbose:
                pbar.set_postfix({"total": f"{el['total']:.3f}", "ct": f"{el['ct']:.3f}",
                                  "expr": f"{el['expr']:.3f}", "corr": f"{el['corr']:.3f}",
                                  "lr": f"{self.optimizer.param_groups[0]['lr']:.1e}"})
        return history
