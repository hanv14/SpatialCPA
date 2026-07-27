"""Device selection and **GPU-sharing policy** for SpatialCPA-v15.

All three trained models (the Phase 2.3 structure interpolator, the Phase 3.1
expression VAE and the Phase 3.2 latent diffusion) get their device from here, so
there is exactly one place that decides where the work runs and exactly one place
that decides how much of the GPU this process is allowed to hold.

Why this module exists at all
-----------------------------
PyTorch's caching allocator does not free GPU memory back to the driver when a
tensor is released — it keeps the block in its cache. On a shared GPU that is a
problem: a run that briefly peaks holds that peak for its whole lifetime, and
other processes see a card that is "full" even while this one is idle between
stages. The policy here is deliberately co-operative:

* **expandable segments** — the allocator is asked to back its pools with
  expandable virtual-memory segments, so a pool *grows and shrinks* with demand
  instead of reserving fixed large blocks it can never give back. This is the
  setting that lets memory expand rather than be claimed up front.
* **cache release between stages** — each training stage releases its cached
  blocks when it finishes, so the memory is available to other processes during
  the parts of the pipeline that do not need the GPU (rasterization, the point
  process, all of Phase 1) and between the three models.
* **optional hard cap** — ``memory_fraction`` caps this process at a fraction of
  the card via ``set_per_process_memory_fraction``, so it *cannot* crowd out a
  co-tenant even at peak. Off by default, because a cap that is too tight turns a
  slow run into a crashed one; set it when the GPU is genuinely shared.

The allocator configuration has to be in place *before* the CUDA caching
allocator initializes, so :func:`configure` is called once, at the start of the
fit, before any tensor is created.
"""

from __future__ import annotations

import os
from typing import Optional


_CONFIGURED = False
_LAST_REPORT = ""


def configure(cfg) -> "object":
    """Apply the GPU-sharing policy and return the torch device to use.

    ``cfg`` is a :class:`~spatialcpav15.config.RuntimeConfig`. Safe to call more
    than once; the allocator settings are applied only on the first call.
    """
    global _CONFIGURED, _LAST_REPORT
    import torch

    want_cuda = (cfg.device == "cuda"
                 or (cfg.device == "auto" and torch.cuda.is_available()))

    if want_cuda and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch reports no CUDA device")

    if not want_cuda:
        _LAST_REPORT = "cpu"
        return torch.device("cpu")

    if not _CONFIGURED:
        _apply_allocator_policy(cfg)
        _CONFIGURED = True

    index = cfg.cuda_index or 0
    device = torch.device(f"cuda:{index}")

    if cfg.memory_fraction is not None:
        # Hard ceiling: this process can never allocate more than the given
        # fraction of the card, so a co-tenant always has the rest available.
        torch.cuda.set_per_process_memory_fraction(float(cfg.memory_fraction), index)

    total = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
    cap = (f", capped at {cfg.memory_fraction:.0%}"
           if cfg.memory_fraction is not None else ", uncapped")
    _LAST_REPORT = (f"{torch.cuda.get_device_name(index)} (cuda:{index}, "
                    f"{total:.1f} GiB{cap}, "
                    f"expandable_segments={'on' if cfg.expandable_segments else 'off'})")
    return device


def _apply_allocator_policy(cfg) -> None:
    """Ask the caching allocator to use growable (returnable) segments.

    The environment variable is the documented interface and must be set before
    the allocator starts; when CUDA is already initialized (e.g. the caller
    touched torch.cuda first) the private runtime setter is tried instead, and a
    failure there is not fatal — it costs sharing friendliness, not correctness.
    """
    import torch

    if not cfg.expandable_segments:
        return

    key = "PYTORCH_CUDA_ALLOC_CONF"
    existing = os.environ.get(key, "")
    if "expandable_segments" not in existing:
        os.environ[key] = (f"{existing},expandable_segments:True"
                           if existing else "expandable_segments:True")

    if torch.cuda.is_initialized():
        setter = getattr(getattr(torch.cuda, "memory", None),
                         "_set_allocator_settings", None)
        if setter is not None:
            try:
                setter("expandable_segments:True")
            except Exception:
                pass


def release(device=None) -> None:
    """Return this process's cached GPU blocks to the driver.

    Called after each training stage.  Between stages — and for the whole of
    Phase 1, the rasterization and the point process, which are numpy — the
    memory belongs to whoever else wants it.
    """
    import torch

    if device is not None and getattr(device, "type", None) != "cuda":
        return
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def describe() -> str:
    """Human-readable description of the device chosen by the last configure()."""
    return _LAST_REPORT or "cpu"


def peak_memory_gib(device=None) -> Optional[float]:
    """Peak GPU memory this process has reserved, in GiB (None on CPU)."""
    import torch

    if device is not None and getattr(device, "type", None) != "cuda":
        return None
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return None
    return float(torch.cuda.max_memory_reserved() / 1024 ** 3)
