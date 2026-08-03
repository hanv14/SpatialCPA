"""Device selection and the GPU-sharing policy for SpatialCPA-v16.

One place decides where the work runs and one place decides how much of the card
this process is allowed to hold. v16 trains a single model (the spectral velocity
field) and does the rest — harmonics, calibration, the point process — in
numpy/scipy, so the GPU is genuinely idle for a large fraction of the run. That
makes the sharing policy worth getting right rather than incidental.

Why any of this is needed
-------------------------
PyTorch's caching allocator does not hand memory back to the driver when a tensor
is freed; it keeps the block. On a shared card that means a run which briefly
peaks holds that peak for its entire lifetime, and a co-tenant sees a full GPU
even while this process is sitting in scipy. Three settings fix it:

* **expandable segments** — back the allocator's pools with expandable
  virtual-memory segments so they *grow and shrink with demand* instead of
  reserving fixed blocks that are never returned. This is the setting that makes
  the memory expand rather than be claimed up front.
* **cache release between stages** — return cached blocks after the flow trains
  and again after each section is generated, so the memory is free during
  everything that does not need it.
* **optional hard cap** — ``memory_fraction`` ceilings this process via
  ``set_per_process_memory_fraction`` so it cannot crowd out a co-tenant even at
  peak. Off by default, because a cap set too tight turns a slow run into a
  crashed one.

The allocator configuration must be in place *before* the CUDA caching allocator
initializes, which is why :func:`configure` runs once at the start of the fit,
before any tensor exists.

This module is deliberately a standalone copy of the same policy v15 uses rather
than an import of it — the method packages do not depend on each other, and a
shared runtime would couple two independent implementations for no benefit.
"""

from __future__ import annotations

import os
from typing import Optional

_CONFIGURED = False
_LAST_REPORT = ""


def configure(cfg):
    """Apply the GPU-sharing policy and return the torch device to use.

    ``cfg`` is a :class:`~spatialcpav16.config.RuntimeConfig`. Safe to call more
    than once; the allocator settings are applied only on the first call.
    """
    global _CONFIGURED, _LAST_REPORT
    import torch

    want_cuda = (cfg.device == "cuda"
                 or (cfg.device == "auto" and torch.cuda.is_available()))
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch reports no CUDA device")
    if not want_cuda:
        _LAST_REPORT = "cpu"
        return torch.device("cpu")

    if not _CONFIGURED:
        _apply_allocator_policy(cfg)
        _CONFIGURED = True

    index = int(cfg.cuda_index or 0)
    device = torch.device(f"cuda:{index}")
    if cfg.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(cfg.memory_fraction), index)

    total = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
    cap = (f", capped at {cfg.memory_fraction:.0%}"
           if cfg.memory_fraction is not None else ", uncapped")
    _LAST_REPORT = (f"{torch.cuda.get_device_name(index)} (cuda:{index}, "
                    f"{total:.1f} GiB{cap}, expandable_segments="
                    f"{'on' if cfg.expandable_segments else 'off'})")
    return device


def _apply_allocator_policy(cfg) -> None:
    """Ask the caching allocator for growable (returnable) segments.

    The environment variable is the documented interface and has to be set before
    the allocator starts. If CUDA is already initialized — the caller touched
    ``torch.cuda`` first — the private runtime setter is tried instead, and a
    failure there is not fatal: it costs sharing friendliness, not correctness.
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
    """Return this process's cached GPU blocks to the driver."""
    import torch

    if device is not None and getattr(device, "type", None) != "cuda":
        return
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def describe() -> str:
    """Description of the device chosen by the last :func:`configure`."""
    return _LAST_REPORT or "cpu"


def peak_memory_gib(device=None) -> Optional[float]:
    """Peak GPU memory this process has reserved, in GiB (None on CPU)."""
    import torch

    if device is not None and getattr(device, "type", None) != "cuda":
        return None
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return None
    return float(torch.cuda.max_memory_reserved() / 1024 ** 3)
