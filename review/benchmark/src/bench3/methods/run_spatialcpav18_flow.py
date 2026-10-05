"""SpatialCPA-v18-flow — benchmark wrapper (method name ``spatialcpav18_gen_flow``).

Retrieval-based virtual-slice synthesis in which the learned flow decides ONE
thing: **which flanking section to retrieve the output section from.**

Built on SpatialCPA-v18 without changing it: ``learn_spatialcpav18.py`` and
``run_spatialcpav18.py`` are imported, never edited. The class below subclasses
``SpatialCPAv18`` and overrides four methods; everything else (training, the
store, the type vote, composition matching, gene splicing, raw output) is v18's
own code.

What differs from v18 at the benchmark configuration
----------------------------------------------------
1. **Layout = one real section, chosen by the flow.** v18's ``nearest`` layout
   copies the cells (exact positions) of the lower flank when ``t <= 0.5``, else
   the upper one. Under the alternating hold-out every target sits at
   ``t = 0.5``, so that rule always copies the section *below* — an arbitrary
   tie-break. Here the flow breaks it: the flow's prediction for depth z* is
   computed at the positions of each flank's real cells, and the flank whose real
   cells agree better with it (lower mean latent distance) is retrieved
   (``_flow_choose_flank``). ``--flank-select rule`` keeps v18's rule instead.
2. **No flow-guided donor reranking.** Re-grounding, the type-vote replacement and
   the composition replacement use each output cell's *own source cell* as the
   query (``pool_e[anchor_src]``) instead of the flow's prediction, so they keep
   the retrieved cell (re-grounding never switches) or pick the replacement most
   like it. This is the "no flow" setting of the v18 ablation, which kept
   continuity (UMAP mixing, Moran's I) that flow reranking lost.

Consequence, by construction: whenever the flow picks the same flank as v18's rule,
the output is **bitwise identical** to v18 run with ``--position-mode nearest`` and
the donor query switched off ("nearest + no flow"); ``--flank-select rule`` is that
variant exactly. The flow can change the result only through the flank it picks.
``tests/`` and REVIEW_NOTES record the check.

The flank choice uses no numpy randomness, so the numpy random stream that drives
the layout, vote, composition and gene splicing is the same as in "nearest + no
flow". The flow's own sampling uses torch's generator, which no later step reads.

``--flank-select flow-cv`` (method ``spatialcpav18_gen_flow_cv``)
-----------------------------------------------------------------
The flow still makes every flank decision, but the threshold it must clear is
calibrated by internal validation on the *training* sections, so the decision is
checked against the benchmark's metrics before it is trusted:

1. For each interior training section i (both neighbours i-1 and i+1 are training
   sections; at most ``CV_MAX_FOLDS``, evenly spaced), a fresh model is trained
   with i left out. Its flow scores the two flanks of z_i exactly as above, giving
   ``margin_i = d(rule flank) - d(other flank)`` (positive: the flow prefers the
   flank v18's rule would not take).
2. The same fold model then synthesizes section i twice, from the rule flank and
   from the other flank, and both are scored against the real section i with the
   benchmark's own per-section metric functions (``evaluate_paper``; UMAP off).
   ``gain_i`` = number of the ``CV_METRICS`` on which the other flank beats the
   rule flank, minus the number on which it loses.
3. The threshold ``delta`` is the largest value maximizing
   ``sum(gain_i for folds with margin_i > delta)``; ``delta = +inf`` (never
   switch, total gain 0) wins every tie. So the flow may switch away from v18's
   rule only where the folds showed that switching on that margin improves the
   metrics on training sections.
4. Each held-out section: the full-data flow scores its two flanks; the method
   takes the other flank iff ``margin > delta``, otherwise the rule flank.

When no switch is made the output is bitwise "nearest + no flow" (the fold models
are trained before the main one, and v18 re-seeds torch at the start of every
training run, so the main model is identical to ``--flank-select flow``'s). Every
flow-cv flank score, on a fold or a held-out section, draws the flow's noise from
its own generator seeded with ``--seed``, so each decision is a function of the
model and z* alone. Nothing here reads a held-out section: the input file
holds training sections only (``guard_no_holdout``) and every fold target is one
of them.

``--flank-select patch-cv`` (method ``spatialcpav18_gen_flow_patch``)
---------------------------------------------------------------------
The same idea one level finer: the flank is chosen per spatial PATCH, not per
section. The two flanks' cells are binned on one square grid (side
``PATCH_SIDE_SPACINGS`` median cell spacings). The flow is queried at every cell
of both flanks at z*; a patch's margin is the rule flank's mean latent distance
there minus the other flank's (positive: the flow agrees better with the other
flank in that patch). The top fraction q of patches by margin are copied, cells
at their exact positions, from the other flank; the rest from the rule flank.
Whole patches move, so neighbourhoods stay intact — the per-cell reranking of the
published configuration broke them.

The flow decides WHICH patches; the folds decide HOW MANY. Each fold synthesizes
its left-out training section at every q in ``PATCH_Q_GRID`` and scores it with
``CV_METRICS`` against the real section; q is the one with the largest total gain
over q = 0, and q = 0 wins ties. q = 0 is bitwise "nearest + no flow"; q = 1 is
the whole other flank. ``--patch-rank random`` ranks patches in a seeded random
order instead, calibrated the same way (method
``spatialcpav18_gen_flow_patch_random``): the control that says what the flow's
ranking adds over switching the same number of arbitrary patches.
"""

import argparse
import copy
import sys
import time
from collections import namedtuple
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_spatialcpav18 as _V18W    # noqa: E402  (v18's wrapper: loader, I/O helpers)

_V18 = _V18W._V18                    # the loaded learn_spatialcpav18 module
_v2_io = _V18W._v2_io
leakage_guard = _V18W.leakage_guard
torch = _V18.torch

METHOD_NAME = "spatialcpav18_gen_flow"
METHOD_NAME_CV = "spatialcpav18_gen_flow_cv"   # the same wrapper under --flank-select flow-cv
METHOD_NAME_PATCH = {"flow": "spatialcpav18_gen_flow_patch",          # --flank-select patch-cv
                     "random": "spatialcpav18_gen_flow_patch_random"}  # + --patch-rank random
# Cap on cells per flank scored by the flow when choosing a flank. Deterministic
# (evenly spaced), so the choice consumes no randomness; 4000 covers STARmap's
# sections whole and bounds the cost on million-cell volumes.
FLANK_SCORE_MAX_CELLS = 4000

# ── flow-cv (internal validation on training sections) ──
# The composite that decides whether switching flanks helped on a fold, fixed
# before any run: (evaluate_paper per-section key, +1 higher is better / -1 lower).
# These are the §7 table's metrics, with the PCA embedding mixing standing in for
# UMAP mixing (UMAP is the one stochastic, version-dependent metric).
CV_METRICS = (
    ("embedding_mixing_pca", +1),
    ("morans_pearson", +1),
    ("morans_mae", -1),
    ("marker_depth_r", +1),
    ("marker_field_r", +1),
    ("celltype_localization", +1),
    ("rare_celltype_localization", +1),
    ("gene_detection_spearman", +1),
)
# At most this many folds (one model training each), evenly spaced over the
# interior training sections.
CV_MAX_FOLDS = 8
# Metric differences at or below this count as ties: the Sinkhorn-based metrics
# vary in the last bits across CPUs and thread counts (REVIEW_NOTES §3), and that
# must not decide a fold.
CV_TIE_TOL = 1e-9

Fold = namedtuple("Fold", "section z margin gain detail")

# ── patch-cv ──
# Patch side, in median nearest-neighbour spacings of the two flanks (~8 x 8
# cells per patch), so the patch holds a neighbourhood whatever the dataset's
# units or density.
PATCH_SIDE_SPACINGS = 8.0
# A patch is ranked only when both flanks have at least this many cells in it.
PATCH_MIN_CELLS = 5
# Fractions of ranked patches a fold may switch; 0 (= v18's nearest layout) wins
# ties, 1 switches every ranked patch.
PATCH_Q_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


class SpatialCPAv18Flow(_V18.SpatialCPAv18):
    """v18 with a flow-chosen single-section layout and no flow donor reranking."""

    flank_select = "flow"            # "flow" | "flow-cv" | "patch-cv" | "rule" | "lower" | "upper"
    cv_delta = float("inf")          # flow-cv threshold (calibrate_delta); +inf = never switch
    patch_q = 0.0                    # patch-cv: fraction of patches switched (calibrate_q)
    patch_rank = "flow"              # patch-cv: "flow" ranks patches; "random" is the control

    # -- remember the target depth: _resample_layout is not given z ------------
    def _generate(self, z):
        self._z_current = float(z)
        self._query = None
        return super()._generate(z)

    # -- 1. layout: one flanking section, exact positions ----------------------
    def _resample_layout(self, lower, upper, t, n_target, rng):
        """Same as v18's ``position_mode="nearest"`` branch, except that the
        flank is chosen by ``_flow_choose_flank`` (or v18's rule), or, under
        ``patch-cv``, chosen patch by patch (``_patch_layout``)."""
        if self.flank_select == "patch-cv":
            return self._patch_layout(lower, upper, t, n_target, rng)
        rule_lower = t <= 0.5
        if self.flank_select == "flow":
            use_lower, info = self._flow_choose_flank(lower, upper, rule_lower)
        elif self.flank_select == "flow-cv":
            use_lower, info = self._flow_cv_choose_flank(lower, upper, rule_lower)
        elif self.flank_select in ("lower", "upper"):        # diagnostics only
            use_lower, info = self.flank_select == "lower", {"forced": self.flank_select}
        else:
            use_lower, info = rule_lower, {"rule": True}
        info.update(z=self._z_current, t=float(t), rule_lower=bool(rule_lower),
                    chose_lower=bool(use_lower))
        self.flank_log.append(info)

        near = lower if use_lower else upper
        near_xy = self._nxy(near.coords_xy).astype(np.float32)
        pick = (rng.choice(near_xy.shape[0], n_target, replace=False)
                if near_xy.shape[0] > n_target else np.arange(near_xy.shape[0]))
        anchor = near_xy[pick]
        anchor_src = (pick if use_lower else pick + lower.n_spots).astype(np.int64)
        return anchor, anchor_src

    def _flow_choose_flank(self, lower, upper, rule_lower):
        """Pick the flank whose real cells best match the flow's prediction at z*.

        For each flank, the flow is queried at that flank's own cell positions and
        target depth z*; its decoded prediction is compared with those cells' real
        latents (the same standardized PCA latent the store uses). The flank with
        the smaller mean distance is the better retrieval for depth z*. Ties go
        to v18's rule.
        """
        dist = self._flow_flank_distances(lower, upper)
        if dist["lower"] < dist["upper"]:
            use_lower = True
        elif dist["upper"] < dist["lower"]:
            use_lower = False
        else:
            use_lower = rule_lower
        return use_lower, {"dist_lower": dist["lower"], "dist_upper": dist["upper"]}

    def _flow_cv_choose_flank(self, lower, upper, rule_lower):
        """The flow's choice, gated by the internally validated threshold: take
        the non-rule flank iff ``margin = d(rule) - d(other) > cv_delta``."""
        dist = self._flow_flank_distances(lower, upper, generator=self._cv_generator())
        rule, other = ("lower", "upper") if rule_lower else ("upper", "lower")
        margin = dist[rule] - dist[other]
        switch = bool(margin > self.cv_delta)
        return (rule_lower != switch), {"dist_lower": dist["lower"],
                                        "dist_upper": dist["upper"],
                                        "margin": margin, "delta": _json_num(self.cv_delta)}

    def _cv_generator(self):
        """flow-cv's flow noise: a fresh generator seeded with ``cfg.seed`` for
        every decision, so a flank choice depends only on the model and z*, not on
        how much of torch's global stream earlier sections consumed (copying the
        other flank can change the cell count, and with it that consumption)."""
        g = torch.Generator(device=self.dev)
        g.manual_seed(int(self.cfg.seed))
        return g

    def _flow_flank_distances(self, lower, upper, generator=None):
        """Mean latent distance between the flow's prediction at z* and each
        flank's real cells, queried at those cells' positions:
        ``{"lower": d, "upper": d}``. ``generator=None`` (``--flank-select flow``)
        draws the flow's noise from torch's global stream."""
        li = self.stack.slices.index(lower)
        ui = self.stack.slices.index(upper)
        pool = self._flow_ctx_pool(li, ui, self._z_current)

        dist = {}
        for name, j in (("lower", li), ("upper", ui)):
            st = self.S[j]
            n = st["nxy"].shape[0]
            idx = (np.linspace(0, n - 1, FLANK_SCORE_MAX_CELLS).astype(np.int64)
                   if n > FLANK_SCORE_MAX_CELLS else np.arange(n))
            q_nxy = st["nxy"][idx].astype(np.float32)
            e_hat = self._flow_decode(pool, q_nxy, generator)
            e_real = st["e"][torch.as_tensor(idx, device=self.dev)].cpu().numpy()
            dist[name] = float(np.mean(np.linalg.norm(e_hat - e_real, axis=1)))
        return dist

    def _flow_ctx_pool(self, li, ui, z):
        """The 3-D attention context at depth z, built exactly as ``_generate``
        builds it (k nearest real sections per side, plus both flanks)."""
        cfg = self.cfg
        order = np.argsort(self.stack.z_centers())
        zc = self.stack.z_centers()
        k_side = max(cfg.context_slices_each_side, 1)
        below = [int(j) for j in order if zc[j] <= z]
        above = [int(j) for j in order if zc[j] > z]
        ctx_src = (below[-k_side:] if below else []) + (above[:k_side] if above else [])
        for extra in (li, ui):
            if extra not in ctx_src:
                ctx_src.append(extra)
        ctx_pool_nxy, ctx_pool_z, ctx_pool_owner = self._build_pool(self.S, ctx_src)
        ctx_pool_h = torch.cat([self.S[j]["h"] for j in ctx_src], 0)
        zn = self._nz(z)
        return ctx_pool_h, ctx_pool_nxy, ctx_pool_z, ctx_pool_owner, ctx_src, zn

    def _flow_decode(self, pool, q_nxy, generator=None):
        """The flow's decoded latent prediction at query positions ``q_nxy``
        (normalized xy, float32) and the pool's depth: the same Euler ODE and
        noise ensemble as ``_generate``. Returns ``(Q, d_e)`` numpy."""
        cfg = self.cfg
        ctx_pool_h, ctx_pool_nxy, ctx_pool_z, ctx_pool_owner, ctx_src, zn = pool
        with torch.no_grad():
            ctx = self._context(ctx_pool_h, ctx_pool_nxy, ctx_pool_z, ctx_pool_owner,
                                ctx_src, self.S, q_nxy, zn, self.ctxmod, self.dev)
            Q = q_nxy.shape[0]
            zt = torch.full((Q,), float(zn), device=self.dev)
            h_acc = torch.zeros((Q, cfg.joint_dim), device=self.dev)
            n_ens = max(cfg.n_ensemble, 1)
            steps = max(cfg.n_ode_steps, 1)
            for _ in range(n_ens):
                h = torch.randn((Q, cfg.joint_dim), device=self.dev,
                                generator=generator)
                for si in range(steps):
                    tt = torch.full((Q,), si / steps, device=self.dev)
                    h = h + (1.0 / steps) * self.vfield(h, tt, ctx, zt)
                h_acc += h
            return self.encoder.decode_e(h_acc / n_ens).cpu().numpy()

    # -- patch-cv: the flow ranks spatial patches; folds set how many switch ---
    def _flow_cell_distances(self, lower, upper):
        """Per-cell latent distance between the flow's prediction at z* and each
        flank's real cells, for EVERY cell (queried in chunks of
        ``FLANK_SCORE_MAX_CELLS``, one fresh seeded generator per call): returns
        ``(d_lower (n_lower,), d_upper (n_upper,))``. Cached per (flanks, z*), so
        the syntheses a fold runs at several q reuse one scoring."""
        li = self.stack.slices.index(lower)
        ui = self.stack.slices.index(upper)
        key = (li, ui, float(self._z_current))
        cache = self.__dict__.setdefault("_cell_dist_cache", {})
        if key in cache:
            return cache[key]
        pool = self._flow_ctx_pool(li, ui, self._z_current)
        gen = self._cv_generator()
        out = []
        for j in (li, ui):
            st = self.S[j]
            n = st["nxy"].shape[0]
            d = np.empty(n, dtype=np.float64)
            for a in range(0, n, FLANK_SCORE_MAX_CELLS):
                idx = np.arange(a, min(a + FLANK_SCORE_MAX_CELLS, n))
                e_hat = self._flow_decode(pool, st["nxy"][idx].astype(np.float32), gen)
                e_real = st["e"][torch.as_tensor(idx, device=self.dev)].cpu().numpy()
                d[idx] = np.linalg.norm(e_hat - e_real, axis=1)
            out.append(d)
        cache[key] = (out[0], out[1])
        return cache[key]

    def _patch_layout(self, lower, upper, t, n_target, rng):
        """``--flank-select patch-cv``: v18's ``nearest`` layout, except that the
        top ``patch_q`` fraction of spatial patches — ranked by how much more the
        flow's prediction at z* agrees with the other flank's cells than with the
        rule flank's (``--patch-rank flow``), or in a seeded random order
        (``--patch-rank random``, the control) — are copied from the other flank.
        ``patch_q = 0`` is bitwise v18's nearest layout."""
        rule_lower = t <= 0.5
        n_lo = lower.n_spots
        d_lo, d_hi = self._flow_cell_distances(lower, upper)
        side = PATCH_SIDE_SPACINGS * float(np.median([lower.median_spacing(),
                                                      upper.median_spacing()]))
        pid_lo, pid_hi = patch_ids(lower.coords_xy, upper.coords_xy, side)
        if rule_lower:
            pid_rule, d_rule, pid_other, d_other = pid_lo, d_lo, pid_hi, d_hi
        else:
            pid_rule, d_rule, pid_other, d_other = pid_hi, d_hi, pid_lo, d_lo
        patches, margins = patch_margins(pid_rule, d_rule, pid_other, d_other,
                                         PATCH_MIN_CELLS)
        switch = patches_to_switch(patches, margins, self.patch_q, self.patch_rank,
                                   int(self.cfg.seed))
        rule_keep = np.nonzero(~np.isin(pid_rule, switch))[0]
        other_take = np.nonzero(np.isin(pid_other, switch))[0]
        if rule_lower:
            cand = np.concatenate([rule_keep, n_lo + other_take])
        else:
            cand = np.concatenate([other_take, n_lo + rule_keep])
        cand = cand.astype(np.int64)

        pick = (rng.choice(cand.shape[0], n_target, replace=False)
                if cand.shape[0] > n_target else np.arange(cand.shape[0]))
        src = cand[pick].astype(np.int64)
        props = np.vstack([self._nxy(lower.coords_xy).astype(np.float32),
                           self._nxy(upper.coords_xy).astype(np.float32)])
        anchor = props[src]
        from_other = (src >= n_lo) if rule_lower else (src < n_lo)
        self.flank_log.append({
            "z": self._z_current, "t": float(t), "rule_lower": bool(rule_lower),
            "chose_lower": bool(rule_lower), "patch_q": float(self.patch_q),
            "patch_rank": self.patch_rank, "patch_side_um": side,
            "n_patches": int(len(patches)), "n_switched": int(len(switch)),
            "frac_cells_other": float(from_other.mean()) if src.size else 0.0,
            "mean_margin": float(np.mean(margins)) if len(margins) else 0.0})
        return anchor, src

    # -- 2. no flow donor reranking: query = each cell's own source latent -----
    def _ground(self, anchor, anchor_src, e_hat, pool_nxy, pool_e, pool_expr, pool_type, rng):
        self._query = pool_e[anchor_src].copy()
        return super()._ground(anchor, anchor_src, self._query, pool_nxy, pool_e,
                               pool_expr, pool_type, rng)

    def _vote_types(self, anchor, pick, ct_idx, expr, pool_nxy, pool_type,
                    pool_expr, pool_e, e_hat, rng):
        return super()._vote_types(anchor, pick, ct_idx, expr, pool_nxy, pool_type,
                                   pool_expr, pool_e, self._query, rng)

    def _match_composition(self, lower, upper, t, ct_idx, expr, anchor, pool_nxy, pool_e,
                           e_hat, pool_type, pool_expr, rng, pick=None):
        return super()._match_composition(lower, upper, t, ct_idx, expr, anchor, pool_nxy,
                                          pool_e, self._query, pool_type, pool_expr, rng, pick)


def _json_num(x):
    """A float for JSON: ``+inf`` (flow-cv's never-switch threshold) as ``"inf"``."""
    return "inf" if x == float("inf") else float(x)


def calibrate_delta(folds):
    """The flow-cv threshold from internal-validation folds.

    ``folds``: objects with ``.margin`` (the fold flow's d(rule) - d(other)) and
    ``.gain`` (metric wins minus losses of the other flank over the rule flank).
    Switching on held-out sections with ``margin > delta`` would have switched
    exactly the folds with ``margin > delta``; the returned ``delta`` is the
    largest one maximizing their total gain. ``+inf`` (switch nothing, gain 0)
    wins every tie, so a switch needs strictly positive validated gain.
    Returns ``(delta, total_gain)``.
    """
    best_gain, delta = 0, float("inf")
    for m in sorted({float(f.margin) for f in folds}, reverse=True):
        thr = float(np.nextafter(m, -np.inf))          # just below m: fold m switches
        g = sum(f.gain for f in folds if f.margin > thr)
        if g > best_gain:
            best_gain, delta = g, thr
    return delta, best_gain


def patch_ids(lo_xy, hi_xy, side):
    """Square-patch id per cell for the two flanks, on one grid of side ``side``
    (same units as the coordinates) anchored at their joint minimum corner.
    Returns ``(pid_lower (n_lower,), pid_upper (n_upper,))`` int64."""
    lo_xy = np.asarray(lo_xy, dtype=np.float64)
    hi_xy = np.asarray(hi_xy, dtype=np.float64)
    allxy = np.vstack([lo_xy, hi_xy])
    keys = np.floor((allxy - allxy.min(0)) / float(side)).astype(np.int64)
    pid = keys[:, 1] * (int(keys[:, 0].max()) + 1) + keys[:, 0]
    return pid[:lo_xy.shape[0]], pid[lo_xy.shape[0]:]


def patch_margins(pid_rule, d_rule, pid_other, d_other, min_cells):
    """Per-patch flow margin ``mean d(rule cells) - mean d(other cells)``
    (positive: the flow agrees better with the other flank there), for patches
    where both flanks have at least ``min_cells`` cells. Returns
    ``(patches (P,) sorted, margins (P,))``."""
    r_ids, r_n = np.unique(pid_rule, return_counts=True)
    o_ids, o_n = np.unique(pid_other, return_counts=True)
    ok = np.intersect1d(r_ids[r_n >= min_cells], o_ids[o_n >= min_cells])
    margins = np.array([float(np.mean(d_rule[pid_rule == p]))
                        - float(np.mean(d_other[pid_other == p])) for p in ok],
                       dtype=np.float64)
    return ok.astype(np.int64), margins


def patches_to_switch(patches, margins, q, rank, seed):
    """The ``round(q * P)`` patches to copy from the other flank: the largest
    margins (``rank="flow"``; ties by patch id) or a seeded random draw
    (``rank="random"``, the control at the same q). Returns an int64 array."""
    k = int(round(float(q) * len(patches)))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if rank == "flow":
        order = np.argsort(-np.asarray(margins), kind="stable")
    elif rank == "random":
        order = np.random.default_rng(seed).permutation(len(patches))
    else:
        raise ValueError(f"unknown patch rank {rank!r}")
    return np.asarray(patches)[order[:k]].astype(np.int64)


def calibrate_q(total_gain_by_q):
    """The patch-cv fraction: the q with the largest total fold gain over q = 0;
    the smallest q wins ties, so q = 0 (v18's nearest layout) needs no evidence.
    Returns ``(q, total_gain)``."""
    best_q, best = 0.0, 0
    for q in sorted(total_gain_by_q):
        if total_gain_by_q[q] > best:
            best_q, best = float(q), int(total_gain_by_q[q])
    return best_q, best


def _load_paper_scorer():
    """The benchmark's own scorer module (``bench3.evaluate_paper``), imported
    only for flow-cv. Its per-section functions are called exactly as
    ``evaluate_paper()`` calls them; the module itself is not modified."""
    src = str(Path(__file__).resolve().parents[2])
    if src not in sys.path:
        sys.path.insert(0, src)
    from bench3 import evaluate_paper as ep
    return ep


def _paper_section_metrics(ep, pred_X, pred_xy, pred_types, gt_X, gt_xy, gt_types,
                           gene_names, markers, layers):
    """One section's ``evaluate_paper`` metrics (UMAP off), from arrays.

    The same calls, arguments and order as the per-section loop of
    ``evaluate_paper.evaluate_paper``; ``pred_X``/``gt_X`` are raw expression with
    columns in ``gene_names`` order (sorted, as the scorer's ``np.intersect1d``).
    """
    k, grid, seed = ep.SPATIAL_K, ep.FIELD_GRID, ep.RANDOM_SEED
    pR, gR = ep._rank_normalize(pred_X), ep._rank_normalize(gt_X)
    cols, _basis = ep.scoring_columns(gene_names, markers, gt_xy, gR, spatial_k=k)
    pred_xy_al, _ainfo = ep.align_by_expression(
        pred_xy, pR[:, cols], gt_xy, gR[:, cols], grid=grid, seed=seed)
    m = {}
    m.update(ep.spatial_autocorrelation_metrics(pred_xy, pR, gt_xy, gR, k=k))
    m.update(ep.embedding_continuity(pR, gR, seed=seed, use_umap=False))
    m.update(ep.marker_metrics(pred_xy_al, pR, gt_xy, gR, gene_names,
                               markers=markers, grid=grid,
                               depth_bins=ep.DEPTH_BINS, k=k, layers=layers))
    m.update(ep.gene_detection_metrics(pred_X, gt_X))
    if gt_types is not None:
        m.update(ep.celltype_localization(pred_xy_al, pred_types, gt_xy, gt_types,
                                          seed=seed))
    return m


def fold_gain(rule_m, other_m):
    """Wins minus losses of the other flank over the rule flank on ``CV_METRICS``
    (a missing/NaN value or a difference within ``CV_TIE_TOL`` counts as a tie).
    Returns ``(gain, {metric: {"rule": v, "other": v}})``."""
    gain, detail = 0, {}
    for key, sign in CV_METRICS:
        r, o = rule_m.get(key), other_m.get(key)
        detail[key] = {"rule": None if r is None else float(r),
                       "other": None if o is None else float(o)}
        if r is None or o is None or not (np.isfinite(r) and np.isfinite(o)):
            continue
        d = sign * (float(o) - float(r))
        gain += int(d > CV_TIE_TOL) - int(d < -CV_TIE_TOL)
    return gain, detail


def _fold_models(adata, gene_names, X_log, X_raw, ct_all, cell_type_names, cfg):
    """Leave-one-training-section-out folds shared by flow-cv and patch-cv.

    Yields ``(section, z, fold_model, score)`` for each interior training section
    (at most ``CV_MAX_FOLDS``, evenly spaced); ``score(vs)`` returns that
    section's ``evaluate_paper`` metrics for a synthesized slice. Every array
    read here comes from the training-only input. Raises if a fold model fails
    to train: a fold that silently fell back would calibrate on something that
    is not this method.
    """
    ep = _load_paper_scorer()
    pp = dict(adata.uns.get("paper_protocol") or {})
    markers = [str(g) for g in pp.get("marker_genes", [])] or ep.MARKER_GENES
    layers = ([str(g) for g in pp.get("layer_superficial", [])] or ep.LAYER_SUPERFICIAL,
              [str(g) for g in pp.get("layer_deep", [])] or ep.LAYER_DEEP)
    print(f"  flow-cv: markers={markers} (uns['paper_protocol'] or the scorer's default)")

    order = np.argsort(np.asarray(gene_names, dtype=str), kind="stable")
    sorted_genes = [str(gene_names[j]) for j in order]
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    has_types = "cell_type" in adata.obs.columns

    full = _V18W.build_stack(adata, X_log, X_raw, ct_all)
    ids = [s.section_id for s in full.slices]
    zc = full.z_centers()
    interior = list(range(1, len(ids) - 1))
    if len(interior) > CV_MAX_FOLDS:
        interior = [interior[int(j)] for j in
                    np.linspace(0, len(interior) - 1, CV_MAX_FOLDS).round().astype(int)]
    print(f"  flow-cv: {len(interior)} fold(s) over training sections "
          f"{[ids[i] for i in interior]}")

    for i in interior:
        sec, z = ids[i], float(zc[i])
        keep = sections != sec
        stack = _V18W.build_stack(adata[keep], X_log[keep], X_raw[keep],
                                  None if ct_all is None else ct_all[keep])
        print(f"  flow-cv fold {sec} (z={z:.2f}): training without it ...")
        fold = SpatialCPAv18Flow(stack, gene_names=gene_names,
                                 cell_type_names=cell_type_names, cfg=copy.deepcopy(cfg))
        if not fold.trained:
            raise RuntimeError(f"flow-cv fold {sec}: the fold model did not train")
        fold.flank_log = []

        gm = sections == sec
        gt_X = X_raw[gm][:, order]
        gt_xy = coords[gm, :2]
        gt_types = adata.obs["cell_type"].values[gm].astype(str) if has_types else None

        def score(vs, gt_X=gt_X, gt_xy=gt_xy, gt_types=gt_types):
            pred_X = _V18W._to_dense_f32(vs.expression)[:, order]
            pred_types = (vs.cell_type.astype(str) if vs.cell_type is not None
                          else np.array(["NA"] * pred_X.shape[0]))
            return _paper_section_metrics(
                ep, pred_X, np.asarray(vs.coords, dtype=np.float64)[:, :2], pred_types,
                gt_X, gt_xy, gt_types, sorted_genes, markers, layers)

        yield sec, z, fold, score
        del fold


def _cv_folds(adata, gene_names, X_log, X_raw, ct_all, cell_type_names, cfg):
    """flow-cv folds (see the module docstring). Returns a list of ``Fold``."""
    folds = []
    for sec, z, fold, score in _fold_models(adata, gene_names, X_log, X_raw, ct_all,
                                            cell_type_names, cfg):
        lower, upper = fold.stack.pick_flanking_slices(z)
        t = fold._tp_frac(z, lower.z_center, upper.z_center)
        rule_lower = t <= 0.5
        fold._z_current = z
        dist = fold._flow_flank_distances(lower, upper, generator=fold._cv_generator())
        rule, other = ("lower", "upper") if rule_lower else ("upper", "lower")
        margin = dist[rule] - dist[other]

        scored = {}
        for side in (rule, other):
            fold.flank_select = side
            scored[side] = score(fold.generate_virtual_slice(z=z))
        gain, detail = fold_gain(scored[rule], scored[other])
        print(f"    rule={rule}, d_lower={dist['lower']:.4f} d_upper={dist['upper']:.4f} "
              f"margin={margin:+.4f}; other flank vs rule: gain {gain:+d} "
              f"over {len(CV_METRICS)} metrics")
        folds.append(Fold(sec, z, margin, gain,
                          {"rule": rule, "dist_lower": dist["lower"],
                           "dist_upper": dist["upper"], "metrics": detail}))
    return folds


def _patch_cv_folds(adata, gene_names, X_log, X_raw, ct_all, cell_type_names, cfg,
                    patch_rank):
    """patch-cv folds: each fold synthesizes its left-out section at every q in
    ``PATCH_Q_GRID`` and scores it against the real section; the gain of q is
    ``fold_gain(q = 0, q)``. Returns ``(total_gain_by_q, per-fold records)``."""
    total = {q: 0 for q in PATCH_Q_GRID}
    records = []
    for sec, z, fold, score in _fold_models(adata, gene_names, X_log, X_raw, ct_all,
                                            cell_type_names, cfg):
        fold.flank_select, fold.patch_rank = "patch-cv", patch_rank
        per_q = {}
        for q in PATCH_Q_GRID:
            fold.patch_q = q
            vs = fold.generate_virtual_slice(z=z)
            per_q[q] = (score(vs), fold.flank_log[-1])
        rec = {"section": sec, "z": z, "q": {}}
        for q in PATCH_Q_GRID:
            g, detail = fold_gain(per_q[0.0][0], per_q[q][0])
            total[q] += g
            lg = per_q[q][1]
            rec["q"][str(q)] = {"gain": g, "n_switched": lg["n_switched"],
                                "n_patches": lg["n_patches"],
                                "frac_cells_other": lg["frac_cells_other"],
                                "metrics": {k: v["other"] for k, v in detail.items()}}
            print(f"    q={q:<4} switched {lg['n_switched']:>3}/{lg['n_patches']} patches "
                  f"({100 * lg['frac_cells_other']:.0f}% cells from the other flank): "
                  f"gain {g:+d} vs q=0")
        records.append(rec)
    return total, records


def run_method(adata, targets, gene_names, X_log, X_raw, args):
    """v18's run_method, constructing the subclass. Returns
    ``(results, flank_log, cv)``; ``cv`` is the flow-cv record (None otherwise)."""
    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = _V18W.build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}, [], None

    cfg = _V18W._build_config(args)
    cfg.position_mode = "flanking"   # routes the layout through _resample_layout
    print(f"  epochs(A={cfg.pretrain_epochs},B={cfg.epochs}), "
          f"latent(d_e={cfg.expr_latent_dim},h={cfg.joint_dim}), "
          f"flow(steps={cfg.n_ode_steps},ens={cfg.n_ensemble}), "
          f"layout=single-section, flank_select={args.flank_select}, "
          f"ctx_slices={cfg.context_slices_each_side}, type_mode={cfg.type_mode}, "
          f"gene_mix={cfg.gene_mix_frac}, raw_output={cfg.raw_output}, "
          f"ground_sample={cfg.ground_sample}, keep_margin={cfg.ground_keep_margin}, "
          f"edit_w={cfg.edit_weight}, blend={cfg.ground_blend_flow}, k={cfg.ground_k}, "
          f"temp={cfg.ground_temp}, device={cfg.device}, donor_query=source-cell")

    # flow-cv folds run BEFORE the main model is built: v18 re-seeds torch and
    # numpy at the start of training, so the main model (and the flow's flank
    # distances) are then the same as under --flank-select flow.
    cv = None
    delta = float("inf")
    if args.flank_select == "flow-cv":
        folds = _cv_folds(adata, gene_names, X_log, X_raw, ct_all, cell_type_names, cfg)
        delta, total = calibrate_delta(folds)
        print(f"  flow-cv: delta={delta:.6g} (validated gain {total:+d}; "
              f"{'never switch: no fold showed a gain' if delta == float('inf') else 'switch when margin > delta'})")
        cv = {"delta": _json_num(delta), "validated_gain": int(total),
              "metrics": [f"{k}:{'+' if s > 0 else '-'}" for k, s in CV_METRICS],
              "max_folds": CV_MAX_FOLDS, "tie_tol": CV_TIE_TOL,
              "folds": [{"section": f.section, "z": f.z, "margin": f.margin,
                         "gain": f.gain, **f.detail} for f in folds]}

    patch_q = 0.0
    if args.flank_select == "patch-cv" and args.patch_q is not None:   # diagnostics only
        patch_q = float(args.patch_q)
        print(f"  patch-cv: q={patch_q} FORCED by --patch-q (no folds; a diagnostic)")
        cv = {"patch_q": patch_q, "forced": True, "patch_rank": args.patch_rank}
    elif args.flank_select == "patch-cv":
        total, records = _patch_cv_folds(adata, gene_names, X_log, X_raw, ct_all,
                                         cell_type_names, cfg, args.patch_rank)
        patch_q, best = calibrate_q(total)
        print(f"  patch-cv ({args.patch_rank} ranking): total gain by q "
              f"{ {q: total[q] for q in PATCH_Q_GRID} } -> q={patch_q} (gain {best:+d})")
        cv = {"patch_q": patch_q, "validated_gain": int(best),
              "patch_rank": args.patch_rank,
              "total_gain_by_q": {str(q): int(total[q]) for q in PATCH_Q_GRID},
              "metrics": [f"{k}:{'+' if s > 0 else '-'}" for k, s in CV_METRICS],
              "patch_side_spacings": PATCH_SIDE_SPACINGS,
              "patch_min_cells": PATCH_MIN_CELLS, "max_folds": CV_MAX_FOLDS,
              "tie_tol": CV_TIE_TOL, "folds": records}

    SpatialCPAv18Flow.flank_select = args.flank_select
    gen = SpatialCPAv18Flow(stack, gene_names=gene_names,
                            cell_type_names=cell_type_names, cfg=cfg)
    gen.flank_log = []
    gen.patch_q, gen.patch_rank = patch_q, args.patch_rank
    if args.flank_select == "flow-cv":
        gen.cv_delta = delta
    print(f"  flow-matching model trained: {gen.trained}")

    results = {}
    for sec, z in targets:
        print(f"  {sec}: retrieving virtual slice at z={z:.2f} ...")
        try:
            vs = gen.generate_virtual_slice(z=z)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        if gen.flank_log and "patch_q" in gen.flank_log[-1]:
            f = gen.flank_log[-1]
            print(f"    patches: switched {f['n_switched']}/{f['n_patches']} "
                  f"(q={f['patch_q']}, {f['patch_rank']} ranking; "
                  f"{100 * f['frac_cells_other']:.0f}% of cells from the other flank; "
                  f"rule flank {'lower' if f['rule_lower'] else 'upper'})")
        elif gen.flank_log:
            f = gen.flank_log[-1]
            chose = "lower" if f["chose_lower"] else "upper"
            rule = "lower" if f["rule_lower"] else "upper"
            extra = (f", dist lower={f['dist_lower']:.4f} upper={f['dist_upper']:.4f}"
                     if "dist_lower" in f else "")
            if "margin" in f:
                extra += f", margin={f['margin']:+.4f} vs delta={f['delta']}"
            print(f"    flank: {chose} (rule: {rule}{extra})")
        n = vs.coords.shape[0]
        if n == 0:
            continue
        print(f"    -> {n} cells synthesized")
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_V18W._to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64),
                        "cell_type": cell_type}
    return results, gen.flank_log, cv


def build_parser():
    """v18's flags (identical names and defaults; a test checks it) + ``--flank-select``."""
    p = argparse.ArgumentParser(description="SpatialCPA-v18-flow wrapper")
    _v2_io.add_v2_args(p)
    p.add_argument("--epochs", type=int, default=160, help="Phase B flow epochs")
    p.add_argument("--pretrain-epochs", type=int, default=60, help="Phase A epochs")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--latent-dim", type=int, default=32,
                   help="expression PCA latent dim (V18Config.expr_latent_dim)")
    p.add_argument("--joint-dim", type=int, default=48)
    p.add_argument("--ode-steps", type=int, default=12,
                   help="Euler steps when sampling the flow ODE")
    p.add_argument("--ensemble", type=int, default=4,
                   help="initial noises marginalized per query")
    p.add_argument("--context-slices", type=int, default=None,
                   help="real slices per side feeding the 3D context "
                        "(default: V18Config.context_slices_each_side)")
    p.add_argument("--position-mode", default="flanking",
                   choices=["flanking", "morph", "nearest"],
                   help="not used: this method's layout is the flow-chosen single section")
    p.add_argument("--ground-blend-flow", type=float, default=0.20,
                   help="fraction of cells offered for re-grounding")
    p.add_argument("--ground-k", type=int, default=8,
                   help="local real candidates for re-grounding, composition, gene mix")
    p.add_argument("--ground-temp", type=float, default=0.25,
                   help="softmax temperature for exemplar sampling")
    p.add_argument("--edit-weight", type=float, default=0.25,
                   help="blend toward the decoded profile (0 = pure exemplar)")
    p.add_argument("--no-coherent-source", action="store_true",
                   help="not used by this layout")
    p.add_argument("--no-output-counts", action="store_true",
                   help="emit log1p-normalized (not count-like) expression")
    p.add_argument("--no-composition-match", action="store_true",
                   help="disable cell-type composition matching")
    p.add_argument("--type-mode", default="vote", choices=["inherit", "vote"],
                   help="'vote': distance-weighted kNN type vote over both flanks")
    p.add_argument("--type-vote-k", type=int, default=12)
    p.add_argument("--gene-mix-frac", type=float, default=0.15,
                   help="fraction of each cell's genes taken from a second local "
                        "same-type real cell (0 disables)")
    p.add_argument("--no-gene-mix", action="store_true",
                   help="ablate gene-mix (equivalent to --gene-mix-frac 0)")
    p.add_argument("--no-raw-output", action="store_true",
                   help="emit expm1(log-normalized) instead of raw measurements")
    p.add_argument("--no-ground-sample", action="store_true",
                   help="ablate temperature sampling: use the argmin exemplar")
    p.add_argument("--no-dedup-ground", action="store_true",
                   help="ablate the exemplar-reuse penalty")
    p.add_argument("--ground-keep-margin", type=float, default=1.0,
                   help="keep the inherited cell unless a candidate matches the query "
                        "better by this margin")
    # ── this method's own flag ──
    p.add_argument("--flank-select", default="flow",
                   choices=["flow", "flow-cv", "patch-cv", "rule", "lower", "upper"],
                   help="'flow': the flow picks which flank to retrieve (default); "
                        "'flow-cv': the flow picks, against a threshold calibrated "
                        "by leave-one-training-section-out validation "
                        "(method spatialcpav18_gen_flow_cv); 'patch-cv': the flow "
                        "ranks spatial patches and folds set how many are copied "
                        "from the other flank (spatialcpav18_gen_flow_patch); "
                        "'rule': v18's rule (lower if t <= 0.5) — identical to v18 "
                        "'nearest + no flow'; 'lower'/'upper': force a flank "
                        "(diagnostics: is the flow's choice the better one?)")
    p.add_argument("--patch-rank", default="flow", choices=["flow", "random"],
                   help="patch-cv only: rank patches by the flow (default) or in a "
                        "seeded random order (the control that isolates the flow)")
    p.add_argument("--patch-q", type=float, default=None,
                   help="patch-cv only: force the switched fraction and skip the folds "
                        "(diagnostics: --patch-q 0 is v18 'nearest + no flow')")
    return p


def main():
    args = build_parser().parse_args()
    if args.position_mode != "flanking":
        print("ERROR: --position-mode is not used by spatialcpav18_gen_flow; its layout "
              "is the flow-chosen single section", file=sys.stderr)
        return 2
    if not _V18W.check_environment():
        return 1

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    print(f"Loading training-only input {args.input} ...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = list(adata.var_names)
    print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['section'].nunique()} sections")

    X_raw = _V18W._to_dense_f32(adata.X)
    et = _V18W._normalize_expression(adata)
    X_log = _V18W._to_dense_f32(adata.X)
    print(f"  expression_type={et}")

    print(f"Running SpatialCPA-v18-flow for targets "
          f"{[(s, round(float(z), 2)) for s, z in targets]} ...")
    t0 = time.time()
    results, flank_log, cv = run_method(adata, targets, gene_names, X_log, X_raw, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1

    method_params = {
        "seed": args.seed, "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "latent_dim": args.latent_dim, "joint_dim": args.joint_dim,
        "ode_steps": args.ode_steps, "ensemble": args.ensemble,
        "position_mode": "single_section", "flank_select": args.flank_select,
        "donor_query": "source_cell",
        "ground_blend_flow": args.ground_blend_flow, "ground_k": args.ground_k,
        "ground_temp": args.ground_temp, "edit_weight": args.edit_weight,
        "type_mode": args.type_mode, "type_vote_k": args.type_vote_k,
        "gene_mix_frac": 0.0 if args.no_gene_mix else args.gene_mix_frac,
        "raw_output": not args.no_raw_output,
        "ground_sample": not args.no_ground_sample,
        "dedup_ground": not args.no_dedup_ground,
        "ground_keep_margin": args.ground_keep_margin,
        "flank_choices": [{k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in f.items()} for f in flank_log],
        "flow_matching": True, "generation_only": True,
    }
    if cv is not None:
        method_params["patch_cv" if args.flank_select == "patch-cv" else "flow_cv"] = cv
    name = {"flow-cv": METHOD_NAME_CV,
            "patch-cv": METHOD_NAME_PATCH[args.patch_rank]}.get(args.flank_select, METHOD_NAME)
    _v2_io.write_prediction_h5(
        results, gene_names, target_sections, method_params, wall, args.output, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
