"""Phase 4.3 — sample the layout from the completed field as a *marked point process*.

The proposal is specific about why this is a point process and not a raster
round-trip: rasterizing the completed field back to peaks and detecting them
throws away sub-voxel position, and peak detection on a smooth density is
ill-posed.  Instead the completed intensity ``lambda_tot`` is treated as the
intensity of an inhomogeneous point process, positions are drawn from it
continuously, and each point carries a **mark** — its cell type — drawn from the
local type posterior implied by the group-density channels and the mean
type-embedding channel.

Two departures from a textbook Poisson process are deliberate and physical:

* the emergent **count** is the field's own integral (``sum lambda_tot``), which
  is the interpolated cell count — never the held-out count;
* a **soft hardcore** relaxation is applied.  Cells occupy volume: real tissue is
  a regular point pattern, while a Poisson sample clumps and leaves holes, which
  would show up as spurious density structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from .config import LayoutConfig
from .phase2_field import FieldSpec


def _bilinear(field2d: np.ndarray, u: np.ndarray, spec: FieldSpec) -> np.ndarray:
    """Sample a (H, W) channel at continuous normalized positions."""
    yc, xc = spec.centers()
    fx = np.interp(u[:, 0], xc, np.arange(len(xc)))
    fy = np.interp(u[:, 1], yc, np.arange(len(yc)))
    x0 = np.clip(np.floor(fx).astype(int), 0, spec.grid_w - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, spec.grid_h - 1)
    x1 = np.clip(x0 + 1, 0, spec.grid_w - 1)
    y1 = np.clip(y0 + 1, 0, spec.grid_h - 1)
    ax, ay = fx - x0, fy - y0
    return ((1 - ax) * (1 - ay) * field2d[y0, x0] + ax * (1 - ay) * field2d[y0, x1]
            + (1 - ax) * ay * field2d[y1, x0] + ax * ay * field2d[y1, x1])


def _upsample(field2d: np.ndarray, factor: int) -> np.ndarray:
    """Bilinearly refine a channel before sampling from it.

    The interpolator's voxel is chosen for the *convolutions* (Phase 2.1
    anisotropy note), not for placing cells.  Sampling directly on that grid
    would make positions uniform inside a voxel several cell-diameters wide,
    which throws away the sub-voxel structure the field actually encodes.
    Refining the intensity first keeps the point process continuous in the sense
    the proposal asks for.
    """
    if factor <= 1:
        return field2d
    H, W = field2d.shape
    yi = (np.arange(H * factor) + 0.5) / factor - 0.5
    xi = (np.arange(W * factor) + 0.5) / factor - 0.5
    y0 = np.clip(np.floor(yi).astype(int), 0, H - 1)
    x0 = np.clip(np.floor(xi).astype(int), 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    ay = (yi - y0)[:, None]
    ax = (xi - x0)[None, :]
    return ((1 - ay) * (1 - ax) * field2d[np.ix_(y0, x0)]
            + (1 - ay) * ax * field2d[np.ix_(y0, x1)]
            + ay * (1 - ax) * field2d[np.ix_(y1, x0)]
            + ay * ax * field2d[np.ix_(y1, x1)])


def _sample_positions(intensity: np.ndarray, n: int, spec: FieldSpec,
                      rng: np.random.Generator, factor: int = 1) -> np.ndarray:
    """Draw ``n`` points from the intensity, jittered *within* the voxel.

    Allocation is **stratified**, not i.i.d. multinomial: each voxel gets
    ``floor(n * p)`` points deterministically and the remainders are distributed
    by a single systematic pass.  Both schemes have the same intensity, but i.i.d.
    sampling piles the full multinomial counting noise on top of it, which at
    ~1 cell per bin is larger than the density signal itself — it would show up
    as spurious structure in exactly the binned-density comparison the field is
    supposed to reproduce.  Stratification removes that noise while leaving the
    process point-wise random (positions stay continuous within the voxel).
    """
    clean = np.nan_to_num(np.asarray(intensity, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
    fine = _upsample(np.clip(clean, 0.0, None), factor)
    fh, fw = fine.shape
    p = fine.ravel()
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        p = np.ones_like(p)
        total = p.sum()
    p = p / total
    expected = p * n
    base = np.clip(np.floor(expected), 0, n).astype(int)
    remainder = expected - base
    n_left = int(n - base.sum())
    if n_left > 0:
        # Systematic (low-discrepancy) selection of the fractional part.
        order = rng.permutation(p.size)
        cum = np.cumsum(remainder[order])
        u = rng.random() * (cum[-1] / n_left) if cum[-1] > 0 else 0.0
        picks = np.searchsorted(cum, u + np.arange(n_left) * (cum[-1] / n_left))
        picks = order[np.clip(picks, 0, p.size - 1)]
        np.add.at(base, picks, 1)
    pick = np.repeat(np.arange(p.size), base)
    if pick.size > n:
        pick = pick[:n]
    elif pick.size < n:
        pick = np.concatenate([pick, rng.choice(p.size, n - pick.size, p=p)])
    n = pick.size
    yb, xb = np.divmod(pick, fw)
    xw = (spec.x_edges[-1] - spec.x_edges[0]) / fw
    yw = (spec.y_edges[-1] - spec.y_edges[0]) / fh
    x = spec.x_edges[0] + (xb + rng.random(n)) * xw
    y = spec.y_edges[0] + (yb + rng.random(n)) * yw
    return np.column_stack([x, y])


def _hardcore_relax(u: np.ndarray, min_dist: float, iters: int) -> np.ndarray:
    """Push overlapping points apart, preserving the density field."""
    if min_dist <= 0 or u.shape[0] < 3:
        return u
    for _ in range(iters):
        tree = cKDTree(u)
        pairs = tree.query_pairs(min_dist, output_type="ndarray")
        if pairs.size == 0:
            break
        d = u[pairs[:, 0]] - u[pairs[:, 1]]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n == 0] = 1e-9
        push = 0.5 * (min_dist - n) * d / n
        np.add.at(u, pairs[:, 0], push)
        np.add.at(u, pairs[:, 1], -push)
    return u


@dataclass
class Layout:
    """A sampled layout: positions, types, and the per-cell structural spread."""

    u: np.ndarray                    # (n, 2) normalized in-plane positions
    types: np.ndarray                # (n,) int type indices
    spread: np.ndarray               # (n,) structural uncertainty at the position
    posterior: np.ndarray            # (n, C) type posterior (for the repair step)
    diagnostics: Dict


def sample_layout(field: np.ndarray, spread: Optional[np.ndarray], spec: FieldSpec,
                  cfg: LayoutConfig, spacing_norm: float,
                  rng: np.random.Generator,
                  n_cells: Optional[int] = None) -> Layout:
    """Sample a marked point process from the completed field."""
    dens = np.clip(np.nan_to_num(field[0], nan=0.0, posinf=0.0, neginf=0.0),
                   0.0, None)
    n = int(n_cells) if n_cells is not None else int(round(float(dens.sum())))
    n = int(np.clip(n, 8, 10_000_000))

    u = _sample_positions(dens, n, spec, rng, factor=cfg.intensity_upsample)
    u = _hardcore_relax(u, cfg.hardcore_beta * spacing_norm, cfg.hardcore_iters)

    # ── marks: the type posterior at each point ──────────────────────────────
    n_groups = spec.n_groups
    group_of = spec.group_of_type
    C = len(group_of)
    G = np.stack([_bilinear(np.clip(field[1 + g], 0.0, None), u, spec)
                  for g in range(n_groups)], axis=1)          # (n, n_groups)
    Gs = G.sum(axis=1, keepdims=True)
    Gp = np.divide(G, Gs, out=np.full_like(G, 1.0 / max(n_groups, 1)), where=Gs > 0)

    ne = spec.type_embedding.shape[1]
    if ne:
        tot = np.maximum(_bilinear(np.clip(field[0], 1e-6, None), u, spec), 1e-6)
        E = np.stack([_bilinear(field[1 + n_groups + e], u, spec)
                      for e in range(ne)], axis=1) / tot[:, None]   # mean embedding
        d2 = ((E[:, None, :] - spec.type_embedding[None, :, :]) ** 2).sum(-1)
        logits = -d2 / max(cfg.softmax_tau ** 2, 1e-6)
    else:
        logits = np.zeros((n, C))

    # Within-group softmax over the embedding, scaled by the group density.
    post = np.zeros((n, C))
    for g in range(n_groups):
        members = np.where(group_of == g)[0]
        if members.size == 0:
            continue
        lg = logits[:, members]
        lg = lg - lg.max(axis=1, keepdims=True)
        w = np.exp(lg)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        post[:, members] = w * Gp[:, [g]]
    s = post.sum(axis=1, keepdims=True)
    post = np.divide(post, s, out=np.full_like(post, 1.0 / C), where=s > 0)
    if cfg.mark_sharpness != 1.0:
        post = post ** float(cfg.mark_sharpness)
        s = post.sum(axis=1, keepdims=True)
        post = np.divide(post, s, out=np.full_like(post, 1.0 / C), where=s > 0)

    types = np.array([rng.choice(C, p=post[i]) for i in range(n)], dtype=np.int64)

    if cfg.composition_repair:
        target = _field_composition(field, spec)
        types = _repair_composition(types, post, target, rng)

    sp = np.zeros(n)
    if spread is not None:
        sp = _bilinear(spread[0], u, spec)

    return Layout(u=u, types=types, spread=sp, posterior=post,
                  diagnostics={"n_cells": n, "field_mass": float(dens.sum())})


def _field_composition(field: np.ndarray, spec: FieldSpec) -> np.ndarray:
    """Type composition implied by the completed field (group mass x within-group
    embedding posterior at the group's own density-weighted mean embedding)."""
    C = len(spec.group_of_type)
    mass = np.array([np.clip(field[1 + g], 0.0, None).sum()
                     for g in range(spec.n_groups)])
    total = mass.sum()
    if total <= 0:
        return np.full(C, 1.0 / C)
    comp = np.zeros(C)
    for g in range(spec.n_groups):
        members = np.where(spec.group_of_type == g)[0]
        if members.size:
            comp[members] = mass[g] / total / members.size
    s = comp.sum()
    return comp / s if s > 0 else np.full(C, 1.0 / C)


def _repair_composition(types: np.ndarray, post: np.ndarray,
                        target: np.ndarray, rng: np.random.Generator,
                        max_moves_frac: float = 0.25) -> np.ndarray:
    """Nudge the sampled composition onto the field-implied one.

    A cell is only moved to an under-represented type if the *local field* was
    already reasonably compatible with that type: candidates are ranked by
    ``post[i, c_under] - post[i, current(i)]``, i.e. how little the swap costs
    under the point's own posterior.  Re-labelling by global confidence alone
    would happily drop a type into a region where it never occurs, which is
    exactly the spatial organization the neighbourhood metrics look at.  No
    transport plan and no global matching is involved — the marks stay a property
    of the local field.
    """
    n, C = post.shape
    want = np.maximum(np.round(target * n).astype(int), 0)
    types = types.copy()
    budget = int(max_moves_frac * n)
    for _ in range(3):
        have = np.bincount(types, minlength=C)
        excess = have - want
        over = np.where(excess > 0)[0]
        under = np.where(excess < 0)[0]
        if under.size == 0 or over.size == 0 or budget <= 0:
            break
        moved = 0
        for c_under in under[np.argsort(excess[under])]:
            need = int(-excess[c_under])
            cand = np.where(np.isin(types, over))[0]
            if cand.size == 0 or need <= 0:
                continue
            cost = post[cand, types[cand]] - post[cand, c_under]
            take = cand[np.argsort(cost)][:min(need, budget - moved)]
            if take.size == 0:
                break
            types[take] = c_under
            moved += take.size
            if moved >= budget:
                break
        if moved == 0:
            break
        budget -= moved
    return types
