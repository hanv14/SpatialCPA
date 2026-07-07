"""
Trainer for SpatialCPA-v5.

Implements a full-featured training loop:

* mini-batch training via ``DataLoader``,
* train / validation split,
* mixed-precision (AMP) on CUDA,
* AdamW + cosine (with warmup) or ReduceLROnPlateau scheduler,
* gradient clipping,
* checkpoint saving (``best.pt`` / ``last.pt``),
* early stopping on validation loss,
* TensorBoard logging (optional),
* CUDA / CPU auto-selection and reproducible seeding.

The trainer owns nothing architecture-specific: it consumes a
:class:`~spatialcpav5.model.SpatialCPATransformer`, a
:class:`~spatialcpav5.data.SliceStack`, precomputed
:class:`~spatialcpav5.data.TripletSamples`, and a
:class:`~spatialcpav5.config.SpatialCPAv5Config`.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SpatialCPAv5Config
from .data import SliceStack, TripletSamples
from .dataset import TripletTokenDataset
from .losses import compute_total_loss
from .model import SpatialCPATransformer


def _make_grad_scaler(enabled: bool):
    """Construct a GradScaler across torch versions (``torch.amp`` vs legacy)."""
    try:  # torch >= 2.3 preferred namespace
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    """Autocast context manager across torch versions."""
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class Trainer:
    """Train a :class:`SpatialCPATransformer`.

    Parameters
    ----------
    model
        The model to train.
    stack
        The slice stack (shared feature table).
    samples
        Precomputed triplet samples.
    config
        Full experiment config.
    """

    def __init__(
        self,
        model: SpatialCPATransformer,
        stack: SliceStack,
        samples: TripletSamples,
        config: SpatialCPAv5Config,
    ) -> None:
        self.model = model
        self.stack = stack
        self.samples = samples
        self.config = config
        self.tcfg = config.train
        self.device = resolve_device(self.tcfg.device)
        self.model.to(self.device)

        set_seed(self.tcfg.seed)

        # ---- train / val split -------------------------------------------- #
        n = len(samples)
        rng = np.random.default_rng(self.tcfg.seed)
        perm = rng.permutation(n)
        n_val = int(round(config.data.val_fraction * n))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        # Guarantee a non-empty training set even for tiny inputs.
        if len(train_idx) == 0:
            train_idx, val_idx = perm, perm[:0]

        k = config.data.n_neighbors
        self.train_ds = TripletTokenDataset(stack, samples, k, train_idx)
        self.val_ds = TripletTokenDataset(stack, samples, k, val_idx) if n_val > 0 else None

        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.tcfg.batch_size,
            shuffle=True,
            num_workers=self.tcfg.num_workers,
            pin_memory=pin,
            drop_last=len(self.train_ds) > self.tcfg.batch_size,
        )
        self.val_loader = (
            DataLoader(
                self.val_ds,
                batch_size=self.tcfg.batch_size,
                shuffle=False,
                num_workers=self.tcfg.num_workers,
                pin_memory=pin,
            )
            if self.val_ds is not None and len(self.val_ds) > 0
            else None
        )

        # ---- optimiser / scheduler / AMP ---------------------------------- #
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.tcfg.lr, weight_decay=self.tcfg.weight_decay
        )
        self.scheduler = self._build_scheduler()
        self.use_amp = self.tcfg.mixed_precision and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.use_amp)

        # ---- logging / checkpoints ---------------------------------------- #
        self.ckpt_dir = Path(self.tcfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = self._build_writer()

        self.best_val = math.inf
        self.epochs_since_improve = 0
        self.global_step = 0

    # ------------------------------------------------------------------ #
    def _build_scheduler(self):
        if self.tcfg.scheduler == "cosine":
            warmup = max(self.tcfg.warmup_epochs, 0)
            total = max(self.tcfg.epochs, 1)

            def lr_lambda(epoch: int) -> float:
                if warmup > 0 and epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / max(total - warmup, 1)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        if self.tcfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        return None

    def _build_writer(self):
        if not self.tcfg.tensorboard_dir:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:  # tensorboard optional at runtime
            return None
        return SummaryWriter(self.tcfg.tensorboard_dir)

    # ------------------------------------------------------------------ #
    def _step_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.model(batch)
        return compute_total_loss(
            outputs,
            batch,
            self.config.loss,
            use_cell_type=self.model.use_cell_type,
            use_region=self.model.use_region,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        agg: Dict[str, List[float]] = {}
        for batch in self.train_loader:
            batch = _move_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with _autocast(self.use_amp):
                losses = self._step_losses(batch)
                loss = losses["total"]

            self.scaler.scale(loss).backward()
            if self.tcfg.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in losses.items():
                agg.setdefault(k, []).append(float(v.item()))

            if self.writer is not None and self.global_step % self.tcfg.log_every == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train_step/{k}", float(v.item()), self.global_step)
                self.writer.add_scalar(
                    "train_step/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )
            self.global_step += 1

        return {k: float(np.mean(v)) for k, v in agg.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        agg: Dict[str, List[float]] = {}
        for batch in self.val_loader:
            batch = _move_batch(batch, self.device)
            with _autocast(self.use_amp):
                losses = self._step_losses(batch)
            for k, v in losses.items():
                agg.setdefault(k, []).append(float(v.item()))
        return {k: float(np.mean(v)) for k, v in agg.items()}

    # ------------------------------------------------------------------ #
    def train(self, verbose: bool = True) -> List[Dict[str, float]]:
        """Run the full training loop; return per-epoch history."""
        history: List[Dict[str, float]] = []
        for epoch in range(self.tcfg.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            # Scheduler step.
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitor = val_metrics.get("total", train_metrics["total"])
                self.scheduler.step(monitor)
            elif self.scheduler is not None:
                self.scheduler.step()

            # Log epoch metrics.
            if self.writer is not None:
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

            monitor = val_metrics.get("total", train_metrics["total"])
            self.save_checkpoint("last.pt", epoch, monitor)

            improved = monitor < self.best_val - self.tcfg.early_stopping_min_delta
            if improved:
                self.best_val = monitor
                self.epochs_since_improve = 0
                self.save_checkpoint("best.pt", epoch, monitor)
            else:
                self.epochs_since_improve += 1

            record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                      **{f"val_{k}": v for k, v in val_metrics.items()},
                      "lr": self.optimizer.param_groups[0]["lr"]}
            history.append(record)

            if verbose:
                msg = (f"[epoch {epoch:03d}] train_total={train_metrics['total']:.4f}")
                if val_metrics:
                    msg += f" val_total={val_metrics['total']:.4f}"
                msg += f" lr={self.optimizer.param_groups[0]['lr']:.2e}"
                if improved:
                    msg += " *"
                print(msg, flush=True)

            # Early stopping.
            if (
                self.tcfg.early_stopping_patience is not None
                and self.val_loader is not None
                and self.epochs_since_improve >= self.tcfg.early_stopping_patience
            ):
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement for {self.epochs_since_improve} epochs)",
                          flush=True)
                break

        if self.writer is not None:
            self.writer.flush()
        return history

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, name: str, epoch: int, val_loss: float) -> Path:
        """Persist model + config + label metadata to ``checkpoint_dir/name``."""
        path = self.ckpt_dir / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.config.to_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "n_genes": self.model.n_genes,
                "n_cell_types": self.model.n_cell_types,
                "n_regions": self.model.n_regions,
                "coord_scale": float(self.model.token_embedder.coord_encoder.coord_scale.item()),
            },
            path,
        )
        return path


def load_model(checkpoint_path: str, device: Optional[str] = None) -> SpatialCPATransformer:
    """Rebuild a :class:`SpatialCPATransformer` from a checkpoint."""
    dev = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    config = SpatialCPAv5Config.from_dict(ckpt["config"])
    model = SpatialCPATransformer(
        n_genes=ckpt["n_genes"],
        n_cell_types=ckpt["n_cell_types"],
        n_regions=ckpt["n_regions"],
        cfg=config.model,
        coord_scale=ckpt.get("coord_scale", 1.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(dev)
    model.eval()
    return model
