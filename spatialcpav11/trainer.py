"""
Training and inference for SpatialCPA-v11 (PyTorch).

``train_model`` builds the context encoder + the two implicit fields, precomputes the
per-slice rasters and foundation-model teacher targets, and optimizes all objectives
(layout reconstruction + distillation, expression reconstruction, cross-z consistency,
biology-informed constraints) by leave-one-slice-out self-supervision. ``infer_slice``
queries the trained fields at an arbitrary continuous z: it samples the occupancy field
(z-marginalized), reads the type field to label each position, and evaluates the
expression field conditioned on the Stage-1 layout code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import losses as L
from .nets import FourierFeatures, ContextEncoder, LayoutField, ExpressionField, sample_raster, mlp
from .teacher import build_teacher


# --------------------------------------------------------------------------- #
# Rasterization + neighbourhood enrichment                                     #
# --------------------------------------------------------------------------- #
def _rasterize(nxy, types, embed, n_types, G, sigma):
    from scipy.ndimage import gaussian_filter
    xb = np.clip(((nxy[:, 0] + 1) / 2 * G).astype(int), 0, G - 1)
    yb = np.clip(((nxy[:, 1] + 1) / 2 * G).astype(int), 0, G - 1)
    occ = np.zeros((G, G)); np.add.at(occ, (yb, xb), 1.0)
    ty = np.zeros((n_types, G, G))
    if types is not None:
        for c in range(n_types):
            m = types == c
            if m.any():
                np.add.at(ty[c], (yb[m], xb[m]), 1.0)
    emb = np.zeros((embed.shape[1], G, G))
    for d in range(embed.shape[1]):
        np.add.at(emb[d], (yb, xb), embed[:, d])
    cnt = np.maximum(occ, 1e-6)
    occ_s = gaussian_filter(occ, sigma, mode="nearest")
    occ_s = occ_s / (occ_s.max() + 1e-8)
    ty_s = np.stack([gaussian_filter(ty[c], sigma, mode="nearest") for c in range(n_types)], 0)
    ty_s = ty_s / (ty_s.sum(0, keepdims=True) + 1e-8)
    emb_s = np.stack([gaussian_filter(emb[d] / cnt, sigma, mode="nearest") for d in range(emb.shape[0])], 0)
    return np.concatenate([occ_s[None], ty_s, emb_s], axis=0).astype(np.float32)  # (1+nt+d, G, G)


def _nhood_enrichment(nxy, types, n_types, k=10):
    from scipy.spatial import cKDTree
    n = len(types)
    M = np.zeros((n_types, n_types))
    if n < k + 1:
        return M
    _, nn = cKDTree(nxy).query(nxy, k=min(k + 1, n)); nn = nn[:, 1:]
    nt = types[nn]
    for c in range(n_types):
        rows = nt[types == c]
        if rows.size:
            cnt = np.bincount(rows.ravel(), minlength=n_types).astype(float)
            if cnt.sum() > 0:
                M[c] = cnt / cnt.sum()
    return M


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #
def _device(cfg, m=None):
    if m is not None and getattr(m, "_force_cpu", False):
        return "cpu"
    d = cfg.train.device
    if d != "auto":
        return d
    if not torch.cuda.is_available():
        return "cpu"
    # Don't pick a GPU that is (nearly) full — a common cause of OOM on the first
    # allocation when another process holds the card.
    try:
        free, _ = torch.cuda.mem_get_info()
        if free < 512 * 1024 * 1024:      # < 512 MB free
            print(f"[spatialcpav11] GPU has only {free/1e6:.0f} MB free; using CPU.")
            return "cpu"
    except Exception:
        pass
    return "cuda"


def _build(m):
    cfg = m.cfg
    dev = _device(cfg, m)
    nt, d = m.n_types, m._embed_dim
    if cfg.train.verbose:
        print(f"[spatialcpav11] training device: {dev}")
    fx = FourierFeatures(cfg.fourier.xy_bands, cfg.fourier.xy_max_freq).to(dev)
    fz = FourierFeatures(cfg.fourier.z_bands, cfg.fourier.z_max_freq).to(dev)
    fx_dim = 2 * fx.out_mult
    fz_dim = 1 * fz.out_mult
    ctx = ContextEncoder(2 + d + 1, cfg.context.hidden, cfg.context.context_dim).to(dev)
    layout_in = fx_dim + fz_dim + cfg.context.context_dim + 2 * (1 + nt)
    expr_in = fx_dim + fz_dim + cfg.layout.layout_feat_dim + cfg.context.context_dim + 2 * d
    layout = LayoutField(layout_in, cfg.layout, nt).to(dev)
    expr = ExpressionField(expr_in, cfg.expression, m.n_genes).to(dev)
    domain_head = mlp([cfg.layout.layout_feat_dim, 64, cfg.teacher.n_pseudo_domains]).to(dev)
    distill_proj = nn.Linear(cfg.layout.layout_feat_dim, m._teacher_dim).to(dev)
    m._nn = dict(fx=fx, fz=fz, ctx=ctx, layout=layout, expr=expr, domain_head=domain_head,
                 distill_proj=distill_proj, dev=dev, nt=nt, d=d, fx_dim=fx_dim, fz_dim=fz_dim)
    return m._nn


def _prep_slices(m):
    """Per-slice tensors: normalized xy, expr, types, embed, teacher targets, raster."""
    cfg = m.cfg; dev = m._nn["dev"]; nt = m.n_types
    S = []
    for s in m.stack.slices:
        nxy = m._nxy(s.coords_xy).astype(np.float32)
        emb = m._embed(s.expression)
        types = (s.cell_type_indices.astype(int) if s.cell_type_indices is not None
                 else np.zeros(s.n_spots, int))
        te = m._teacher.embed(s.expression, s.coords_xy)
        dom = m._teacher.domains(s.expression, s.coords_xy)
        raster = _rasterize(nxy, types, emb, nt, cfg.context.raster_grid, cfg.context.raster_smooth)
        S.append(dict(
            nxy=torch.tensor(nxy, device=dev), nz=float(m._nz(s.z_center)),
            expr=torch.tensor(np.asarray(s.expression, np.float32), device=dev),
            types=torch.tensor(types, device=dev),
            emb=torch.tensor(emb, device=dev),
            te=torch.tensor(te, device=dev),
            dom=torch.tensor(dom, device=dev),
            raster=torch.tensor(raster, device=dev),
            M=torch.tensor(_nhood_enrichment(nxy, types, nt), dtype=torch.float32, device=dev),
        ))
    m._S = S


# --------------------------------------------------------------------------- #
# Feature assembly for a query batch                                           #
# --------------------------------------------------------------------------- #
def _context_vec(m, ctx_slices):
    feats = [torch.cat([s["nxy"], s["emb"], s["nxy"].new_full((s["nxy"].shape[0], 1), s["nz"])], dim=1)
             for s in ctx_slices]
    return m._nn["ctx"](feats)


def _layout_feats(m, xy, znorm, cvec, ctx_slices):
    nn = m._nn
    fx = nn["fx"](xy); fz = nn["fz"](xy.new_full((xy.shape[0], 1), znorm))
    ras = [sample_raster(s["raster"][:1 + nn["nt"]], xy) for s in ctx_slices]  # occ+type
    ras = torch.cat(ras, dim=1)
    ctx = cvec[None].expand(xy.shape[0], -1)
    return torch.cat([fx, fz, ctx, ras], dim=1)


def _expr_feats(m, xy, znorm, code, cvec, ctx_slices):
    nn = m._nn
    fx = nn["fx"](xy); fz = nn["fz"](xy.new_full((xy.shape[0], 1), znorm))
    ras = [sample_raster(s["raster"][1 + nn["nt"]:], xy) for s in ctx_slices]   # embed channels
    ras = torch.cat(ras, dim=1)
    ctx = cvec[None].expand(xy.shape[0], -1)
    return torch.cat([fx, fz, code, ctx, ras], dim=1)


# --------------------------------------------------------------------------- #
# Training                                                                      #
# --------------------------------------------------------------------------- #
def train_model(m):
    cfg = m.cfg
    torch.manual_seed(cfg.train.seed); np.random.seed(cfg.train.seed)
    # Build the teacher first so the network knows its embedding dim.
    m._teacher = build_teacher(cfg.teacher, m.stack)
    m._teacher_dim = int(m._teacher.embed(m.stack.slices[0].expression,
                                          m.stack.slices[0].coords_xy).shape[1])
    nn_ = _build(m)
    _prep_slices(m)
    dev = nn_["dev"]
    params = []
    for k in ("ctx", "layout", "expr", "domain_head", "distill_proj"):
        params += list(nn_[k].parameters())
    opt = torch.optim.Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    lc = cfg.loss
    order = np.argsort(m.stack.z_centers())
    interior = order[1:-1] if len(order) >= 3 else order[:1]
    rng = np.random.default_rng(cfg.train.seed)

    for ep in range(cfg.train.epochs):
        tot = 0.0; nb = 0
        for i in interior:
            i = int(i)
            pos = np.where(order == i)[0][0]
            lo, hi = int(order[pos - 1]), int(order[pos + 1])
            ctx_slices = [m._S[lo], m._S[hi]]
            tgt = m._S[i]
            znorm = tgt["nz"]
            cvec = _context_vec(m, ctx_slices)

            # positive (real) + negative (empty) queries
            npos = min(cfg.train.batch_points, tgt["nxy"].shape[0])
            pidx = torch.tensor(rng.choice(tgt["nxy"].shape[0], npos, replace=False), device=dev)
            xy_pos = tgt["nxy"][pidx]
            nneg = int(npos * cfg.train.neg_ratio)
            xy_neg = (torch.rand(nneg, 2, device=dev) * 2 - 1)

            # ---- Stage 1: layout ---- #
            occ_p, typ_p, code_p = nn_["layout"](_layout_feats(m, xy_pos, znorm, cvec, ctx_slices))
            occ_n, _, _ = nn_["layout"](_layout_feats(m, xy_neg, znorm, cvec, ctx_slices))
            loss = lc.w_layout_occ * (L.occupancy_bce(occ_p, torch.ones_like(occ_p))
                                      + L.occupancy_bce(occ_n, torch.zeros_like(occ_n)))
            if m.n_types >= 2:
                loss = loss + lc.w_layout_type * L.type_ce(typ_p, tgt["types"][pidx])
            # distillation: feature alignment + pseudo-layout (domains)
            loss = loss + lc.w_distill_embed * L.distill_embed(
                nn_["distill_proj"](code_p), tgt["te"][pidx])
            loss = loss + lc.w_distill_pseudo * L.type_ce(nn_["domain_head"](code_p), tgt["dom"][pidx])

            # ---- Stage 2: expression (conditioned on layout code) ---- #
            pred = nn_["expr"](_expr_feats(m, xy_pos, znorm, code_p, cvec, ctx_slices))
            loss = loss + lc.w_expr_recon * L.expr_mse(pred, tgt["expr"][pidx])

            # ---- cross-z consistency ---- #
            dz = lc.consistency_dz
            occ_zp, typ_zp, code_zp = nn_["layout"](_layout_feats(m, xy_pos, znorm + dz, cvec, ctx_slices))
            occ_zm, typ_zm, code_zm = nn_["layout"](_layout_feats(m, xy_pos, znorm - dz, cvec, ctx_slices))
            loss = loss + lc.w_consistency_layout * L.consistency(
                torch.sigmoid(occ_p), torch.sigmoid(occ_zp), torch.sigmoid(occ_zm))
            pred_zp = nn_["expr"](_expr_feats(m, xy_pos, znorm + dz, code_zp, cvec, ctx_slices))
            pred_zm = nn_["expr"](_expr_feats(m, xy_pos, znorm - dz, code_zm, cvec, ctx_slices))
            loss = loss + lc.w_consistency_expr * L.consistency(pred, pred_zp, pred_zm)

            # ---- biology-informed constraints ---- #
            if m.n_types >= 2:
                tp = torch.softmax(typ_p, dim=1)
                target_M = (1.0 - _t(znorm, ctx_slices)) * ctx_slices[0]["M"] + \
                           _t(znorm, ctx_slices) * ctx_slices[1]["M"]
                loss = loss + lc.w_interface * L.interface_preservation(tp, xy_pos, target_M)
                loss = loss + lc.w_domain_coherence * L.domain_coherence(tp, xy_pos)
            # within-domain expression gradient smoothness (autograd wrt xy)
            xy_g = xy_pos.detach().clone().requires_grad_(True)
            _, _, code_g = nn_["layout"](_layout_feats(m, xy_g, znorm, cvec, ctx_slices))
            eg = nn_["expr"](_expr_feats(m, xy_g, znorm, code_g, cvec, ctx_slices))
            loss = loss + lc.w_grad_smooth * L.grad_smoothness(eg, xy_g)

            opt.zero_grad(); loss.backward()
            if cfg.train.grad_clip:
                nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            tot += float(loss); nb += 1
        if cfg.train.verbose and (ep % 50 == 0 or ep == cfg.train.epochs - 1):
            print(f"    [v11] epoch {ep} loss={tot / max(nb, 1):.4f}")

    for k in ("ctx", "layout", "expr", "domain_head", "distill_proj"):
        nn_[k].eval()


def _t(znorm, ctx_slices):
    zl, zh = ctx_slices[0]["nz"], ctx_slices[1]["nz"]
    if zh == zl:
        return 0.5
    return float(np.clip((znorm - zl) / (zh - zl), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Inference (continuous z, z-marginalized)                                     #
# --------------------------------------------------------------------------- #
def infer_slice(m, z):
    from .model import VirtualSlice
    cfg = m.cfg; nn_ = m._nn; dev = nn_["dev"]
    lower, upper = m.stack.pick_flanking_slices(z)
    lo = _slice_state(m, lower); hi = _slice_state(m, upper)
    ctx_slices = [lo, hi]
    znorm = m._nz(z)
    t = tp_frac(z, lower.z_center, upper.z_center)
    n_target = max(int(round((1 - t) * lower.n_spots + t * upper.n_spots)), 1)

    # z window for marginalization / hybrid inference
    zs = [znorm]
    if cfg.inference.z_marginalize > 1:
        zs = list(np.linspace(znorm - cfg.inference.z_window, znorm + cfg.inference.z_window,
                              cfg.inference.z_marginalize))
    cvec = _context_vec(m, ctx_slices)

    with torch.no_grad():
        # occupancy grid (z-marginalized) -> sample n_target positions
        G = cfg.inference.occ_grid
        gx = torch.linspace(-1, 1, G, device=dev)
        gy = torch.linspace(-1, 1, G, device=dev)
        YY, XX = torch.meshgrid(gy, gx, indexing="ij")
        grid_xy = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)
        occ = torch.zeros(grid_xy.shape[0], device=dev)
        for zq in zs:
            o, _, _ = nn_["layout"](_layout_feats(m, grid_xy, float(zq), cvec, ctx_slices))
            occ = occ + torch.sigmoid(o)
        occ = (occ / len(zs)).cpu().numpy()
        occ = np.clip(occ, 1e-6, None); occ = occ / occ.sum()
        rng = np.random.default_rng(cfg.seed)
        sel = rng.choice(grid_xy.shape[0], size=n_target, replace=True, p=occ)
        jit = (rng.random((n_target, 2)) - 0.5) * (2.0 / G)
        xy = grid_xy.cpu().numpy()[sel] + jit
        xy_t = torch.tensor(xy, dtype=torch.float32, device=dev)

        # types + layout code + expression (z-marginalized), conditioned on layout
        typ_acc = 0.0; code_acc = 0.0; expr_acc = 0.0
        for zq in zs:
            _, typ, code = nn_["layout"](_layout_feats(m, xy_t, float(zq), cvec, ctx_slices))
            typ_acc = typ_acc + torch.softmax(typ, dim=1)
            code_acc = code_acc + code
            expr_acc = expr_acc + nn_["expr"](_expr_feats(m, xy_t, float(zq), code, cvec, ctx_slices))
        typ_prob = (typ_acc / len(zs)).cpu().numpy()
        expr = (expr_acc / len(zs)).cpu().numpy()
    ct_idx = typ_prob.argmax(1) if m.n_types >= 2 else None

    # optional real-residual decode (hybrid): layout-conditioned real profile
    if cfg.inference.expr_decode == "residual":
        expr = _add_residual(m, expr, ct_idx, xy, lower, upper)
    expr = np.clip(expr, 0.0, None).astype(np.float32)

    coords = np.column_stack([m._dxy(xy), np.full(n_target, float(z))]).astype(np.float32)
    labels = (np.array([m.cell_type_names[i] for i in ct_idx]) if (ct_idx is not None and m.cell_type_names)
              else (ct_idx.astype(str) if ct_idx is not None else None))
    return VirtualSlice(coords, expr, labels, ct_idx)


def _slice_state(m, s):
    for st, sl in zip(m._S, m.stack.slices):
        if sl is s:
            return st
    # rebuild if not found (shouldn't happen)
    return m._S[0]


def tp_frac(z, zl, zh):
    if zh == zl:
        return 0.5
    return float(np.clip((float(z) - zl) / (zh - zl), 0.0, 1.0))


def _add_residual(m, expr, ct_idx, xy_norm, lower, upper):
    """Hybrid decode (layout-conditioned): blend the field expression with a *real*
    same-type cell's profile, matched by predicted type + spatial position. Robust to
    field mean-collapse and guarantees realistic, type-consistent gene-gene structure
    (the layout still drives *which* real profile via type + location)."""
    from scipy.spatial import cKDTree
    src = np.concatenate([lower.expression, upper.expression], 0).astype(np.float32)
    src_xy = np.concatenate([m._nxy(lower.coords_xy), m._nxy(upper.coords_xy)], 0)
    src_ty = (np.concatenate([lower.cell_type_indices, upper.cell_type_indices])
              if (lower.cell_type_indices is not None and upper.cell_type_indices is not None)
              else np.zeros(src.shape[0], int))
    a = m.cfg.inference.residual_weight
    out = expr.copy()
    types = ct_idx if ct_idx is not None else np.zeros(xy_norm.shape[0], int)
    for c in np.unique(types):
        q = np.where(types == c)[0]
        s = np.where(src_ty == c)[0]
        if s.size == 0:
            s = np.arange(src.shape[0])
        _, nn = cKDTree(src_xy[s]).query(xy_norm[q], k=1)
        out[q] = (1 - a) * expr[q] + a * src[s][nn]
    return out
