"""
Loss functions for SpatialCPA-v4.

Provides the individual terms (masked MSE, Pearson-correlation loss, masked
cross-entropy, occupancy BCE) and a :func:`compute_total_loss` helper that
combines them with the weights in :class:`~spatialcpav4.config.LossConfig`.

All expression / label terms are *masked* by ``has_target`` so background
(occupancy-only) samples contribute to the occupancy loss alone.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import LossConfig


def masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE over rows where ``mask > 0``.

    ``pred``/``target`` : (N, G).  ``mask`` : (N,) in {0,1}.
    """
    m = mask > 0
    if m.sum() == 0:
        return pred.new_tensor(0.0)
    return F.mse_loss(pred[m], target[m])


def pearson_corr_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``1 - mean(per-gene Pearson r)`` over the (masked) batch.

    Directly optimises the gene-wise correlation used at evaluation.  Needs at
    least a few samples to be meaningful; returns 0 otherwise.
    """
    if mask is not None:
        m = mask > 0
        if m.sum() < 4:
            return pred.new_tensor(0.0)
        pred = pred[m]
        target = target[m]
    if pred.shape[0] < 4:
        return pred.new_tensor(0.0)

    pred_c = pred - pred.mean(dim=0, keepdim=True)
    tgt_c = target - target.mean(dim=0, keepdim=True)
    cov = (pred_c * tgt_c).sum(dim=0)
    pred_std = pred_c.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)
    tgt_std = tgt_c.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)
    corr = cov / (pred_std * tgt_std)
    return 1.0 - corr.mean()


def masked_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, has_target: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy over samples that are supervised *and* have a valid label.

    ``target < 0`` marks an unknown/background label and is skipped, as is any
    sample with ``has_target == 0``.
    """
    valid = (has_target > 0) & (target >= 0)
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[valid], target[valid])


def occupancy_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for the occupancy head (applies to all samples)."""
    return F.binary_cross_entropy_with_logits(logits, target)


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: LossConfig,
    use_cell_type: bool,
    use_region: bool,
) -> Dict[str, torch.Tensor]:
    """Combine every active term into the total loss.

    Parameters
    ----------
    outputs
        Model forward outputs: ``expression`` (N,G), ``occupancy_logit`` (N,),
        and optionally ``cell_type_logits`` / ``region_logits``.
    batch
        Dataset batch (targets + masks).
    cfg
        Loss weights.
    use_cell_type, use_region
        Whether those heads/labels are active.

    Returns
    -------
    dict with a ``total`` key plus each individual (unweighted) term for logging.
    """
    has_target = batch["has_target"]

    # ---- expression ------------------------------------------------------- #
    mse = masked_mse_loss(outputs["expression"], batch["target_expr"], has_target)
    pear = pearson_corr_loss(outputs["expression"], batch["target_expr"], has_target)
    expr_term = cfg.mse_weight * mse + cfg.pearson_weight * pear

    # ---- labels ----------------------------------------------------------- #
    ct_loss = outputs["expression"].new_tensor(0.0)
    reg_loss = outputs["expression"].new_tensor(0.0)
    if use_cell_type and "cell_type_logits" in outputs:
        ct_loss = masked_cross_entropy(
            outputs["cell_type_logits"], batch["target_ct"], has_target
        )
    if use_region and "region_logits" in outputs:
        reg_loss = masked_cross_entropy(
            outputs["region_logits"], batch["target_reg"], has_target
        )
    label_term = cfg.cell_type_weight * ct_loss + cfg.region_weight * reg_loss

    # ---- occupancy -------------------------------------------------------- #
    occ_loss = occupancy_bce(outputs["occupancy_logit"], batch["target_occ"])

    total = (
        cfg.expression_weight * expr_term
        + cfg.label_weight * label_term
        + cfg.occupancy_weight * occ_loss
    )

    return {
        "total": total,
        "mse": mse.detach(),
        "pearson": pear.detach(),
        "cell_type": ct_loss.detach(),
        "region": reg_loss.detach(),
        "occupancy": occ_loss.detach(),
    }
