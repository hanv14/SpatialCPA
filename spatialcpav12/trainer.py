"""
Training and inference for SpatialCPA-v12 (PyTorch).

``train_model`` builds the context encoder + Stage-1 layout field + Stage-2 generative
expression decoder, warm-starts the decoder's factor loadings from the real gene-gene
covariance, precomputes per-slice rasters and foundation-model teacher targets, and
optimizes all objectives (layout reconstruction + distillation, expression
mean-reconstruction + factor-analysis NLL, cross-z consistency, biology-informed
constraints) by leave-one-slice-out self-supervision.

``infer_slice`` queries the trained fields at an arbitrary continuous z and **samples**
a slice: it samples the occupancy field (z-marginalized, optionally sharpened by the
z-interpolated flanking density), reads the type field (optionally prior-corrected to
the interpolated composition), evaluates the mean expression field, and draws each
cell's expression from the factor-analysis decoder with a spatially-coherent latent —
finally pinning per-gene mean/variance to the interpolated flanking statistics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import losses as L
from .nets import (FourierFeatures, ContextEncoder, LayoutField,
                   GenerativeExpressionField, sample_raster, mlp)
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


def _density_raster(nxy, G, sigma=1.5):
    """Smoothed occupancy density on a G×G grid over [-1, 1]^2 (for calibration)."""
    from scipy.ndimage import gaussian_filter
    xb = np.clip(((nxy[:, 0] + 1) / 2 * G).astype(int), 0, G - 1)
    yb = np.clip(((nxy[:, 1] + 1) / 2 * G).astype(int), 0, G - 1)
    occ = np.zeros((G, G)); np.add.at(occ, (yb, xb), 1.0)
    occ = gaussian_filter(occ, sigma, mode="nearest")
    return occ.astype(np.float64)


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
# Data-derived covariance warm-start for the factor decoder                    #
# --------------------------------------------------------------------------- #
def _covariance_init(union_expr, r):
    """Factorize the real gene-gene covariance as L Lᵀ + Ψ (top-r PCA warm-start)."""
    X = np.asarray(union_expr, np.float64)
    G = X.shape[1]
    mean = X.mean(0, keepdims=True)
    Xc = X - mean
    n = Xc.shape[0]
    r = int(max(1, min(r, min(Xc.shape) - 1)))
    # SVD of centered data: columns of V are principal directions.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(n - 1, 1)                      # per-component variance
    comps = Vt[:r]                                      # (r, G)
    L0 = (comps.T * np.sqrt(var[:r])[None, :]).astype(np.float32)   # (G, r)
    total_var = Xc.var(0)                               # (G,)
    resid_var = np.clip(total_var - (L0 ** 2).sum(1), 1e-4, None)   # (G,)
    log_psi0 = np.log(resid_var).astype(np.float32)
    return L0, log_psi0


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
    try:
        free, _ = torch.cuda.mem_get_info()
        if free < 512 * 1024 * 1024:
            print(f"[spatialcpav12] GPU has only {free/1e6:.0f} MB free; using CPU.")
            return "cpu"
    except Exception:
        pass
    return "cuda"


def _build(m):
    cfg = m.cfg
    dev = _device(cfg, m)
    nt, d = m.n_types, m._embed_dim
    if cfg.train.verbose:
        print(f"[spatialcpav12] training device: {dev}")
    fx = FourierFeatures(cfg.fourier.xy_bands, cfg.fourier.xy_max_freq).to(dev)
    fz = FourierFeatures(cfg.fourier.z_bands, cfg.fourier.z_max_freq).to(dev)
    fx_dim = 2 * fx.out_mult
    fz_dim = 1 * fz.out_mult
    ctx = ContextEncoder(2 + d + 1, cfg.context.hidden, cfg.context.context_dim).to(dev)
    layout_in = fx_dim + fz_dim + cfg.context.context_dim + 2 * (1 + nt)
    expr_in = fx_dim + fz_dim + cfg.layout.layout_feat_dim + cfg.context.context_dim + 2 * d
    layout = LayoutField(layout_in, cfg.layout, nt).to(dev)
    expr = GenerativeExpressionField(expr_in, cfg.expression, m.n_genes).to(dev)
    # Warm-start the factor decoder from the real gene-gene covariance.
    L0, log_psi0 = _covariance_init(m.stack.union_expression(), cfg.expression.n_factors)
    expr.init_covariance(L0, log_psi0)
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
    m._teacher = build_teacher(cfg.teacher, m.stack, m.gene_names,
                               getattr(m, "gene_symbols", None))
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
            loss = loss + lc.w_distill_embed * L.distill_embed(
                nn_["distill_proj"](code_p), tgt["te"][pidx])
            loss = loss + lc.w_distill_pseudo * L.type_ce(nn_["domain_head"](code_p), tgt["dom"][pidx])

            # ---- Stage 2: generative expression (conditioned on layout code) ---- #
            mu = nn_["expr"](_expr_feats(m, xy_pos, znorm, code_p, cvec, ctx_slices))
            real = tgt["expr"][pidx]
            loss = loss + lc.w_expr_recon * L.expr_mse(mu, real)
            loss = loss + lc.w_expr_nll * L.factor_analysis_nll(
                real, mu, nn_["expr"].loadings, nn_["expr"].psi())

            # ---- cross-z consistency (mean) ---- #
            dz = lc.consistency_dz
            occ_zp, typ_zp, code_zp = nn_["layout"](_layout_feats(m, xy_pos, znorm + dz, cvec, ctx_slices))
            occ_zm, typ_zm, code_zm = nn_["layout"](_layout_feats(m, xy_pos, znorm - dz, cvec, ctx_slices))
            loss = loss + lc.w_consistency_layout * L.consistency(
                torch.sigmoid(occ_p), torch.sigmoid(occ_zp), torch.sigmoid(occ_zm))
            mu_zp = nn_["expr"](_expr_feats(m, xy_pos, znorm + dz, code_zp, cvec, ctx_slices))
            mu_zm = nn_["expr"](_expr_feats(m, xy_pos, znorm - dz, code_zm, cvec, ctx_slices))
            loss = loss + lc.w_consistency_expr * L.consistency(mu, mu_zp, mu_zm)

            # ---- biology-informed constraints ---- #
            if m.n_types >= 2:
                tp = torch.softmax(typ_p, dim=1)
                target_M = (1.0 - _t(znorm, ctx_slices)) * ctx_slices[0]["M"] + \
                           _t(znorm, ctx_slices) * ctx_slices[1]["M"]
                loss = loss + lc.w_interface * L.interface_preservation(tp, xy_pos, target_M)
                loss = loss + lc.w_domain_coherence * L.domain_coherence(tp, xy_pos)
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
            print(f"    [v12] epoch {ep} loss={tot / max(nb, 1):.4f}")

    for k in ("ctx", "layout", "expr", "domain_head", "distill_proj"):
        nn_[k].eval()


def _t(znorm, ctx_slices):
    zl, zh = ctx_slices[0]["nz"], ctx_slices[1]["nz"]
    if zh == zl:
        return 0.5
    return float(np.clip((znorm - zl) / (zh - zl), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Inference (continuous z, generative)                                         #
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

    zs = [znorm]
    if cfg.inference.z_marginalize > 1:
        zs = list(np.linspace(znorm - cfg.inference.z_window, znorm + cfg.inference.z_window,
                              cfg.inference.z_marginalize))
    cvec = _context_vec(m, ctx_slices)

    rng = np.random.default_rng(cfg.seed)
    carried_types = None
    with torch.no_grad():
        pmode = cfg.inference.position_mode
        if pmode == "auto":
            # Near-identical planes (volumetric): the flanking slices already sit on the
            # target geometry, so real flanking positions reproduce the density far better
            # than a learned grid. Distinct/drifting tissue: the tissue moves between
            # slices, so the learned occupancy field must place cells. Detect the regime by
            # the cross-slice spacing relative to the within-slice spacing.
            pmode = "flanking" if _planes_near_identical(m, lower, upper) else "field"
        if pmode in ("flanking", "hybrid"):
            # Proposal positions = the real flanking cells' (x, y), sampled with a
            # z-weight toward the nearer slice. These carry the real local density
            # (unlike a coarse occupancy grid). No transport / correspondence is used —
            # each proposal is a real position kept as-is. "hybrid" additionally
            # reweights each proposal by the learned occupancy field there, so the
            # neural layout still shapes the density.
            lo_xy = m._nxy(lower.coords_xy).astype(np.float32)
            hi_xy = m._nxy(upper.coords_xy).astype(np.float32)
            if cfg.inference.flanking_single_slice:
                # Sample from the single nearest slice — a coherent real sheet, so the
                # density and niche stay as clean as one real slice (best for near-
                # identical planes) rather than a superposition of two clouds.
                near = lower if abs(t - 0.0) <= abs(t - 1.0) else upper
                props = m._nxy(near.coords_xy).astype(np.float32)
                w = np.ones(props.shape[0], dtype=np.float64)
            else:
                props = np.vstack([lo_xy, hi_xy])
                w = np.concatenate([np.full(len(lo_xy), max(1 - t, 1e-3)),
                                    np.full(len(hi_xy), max(t, 1e-3))]).astype(np.float64)
            if pmode == "hybrid":
                pt = torch.tensor(props, dtype=torch.float32, device=dev)
                occ_at = torch.zeros(pt.shape[0], device=dev)
                for zq in zs:
                    o, _, _ = nn_["layout"](_layout_feats(m, pt, float(zq), cvec, ctx_slices))
                    occ_at = occ_at + torch.sigmoid(o)
                w = w * np.clip((occ_at / len(zs)).cpu().numpy().astype(np.float64), 1e-3, None)
            w = w / w.sum()
            sel = rng.choice(props.shape[0], size=n_target, replace=True, p=w)
            spacing = 2.0 / max(cfg.inference.occ_grid, 1)
            jit = (rng.standard_normal((n_target, 2))) * (0.5 * spacing)
            xy = props[sel] + jit
            # For near-identical planes the flanking layout *is* the target layout, so we
            # can carry each proposal cell's REAL type — preserving the niche (nhood)
            # structure that quota reassignment on a mixed cloud would scramble.
            if (cfg.inference.flanking_carry_type and m.n_types >= 2
                    and lower.cell_type_indices is not None and upper.cell_type_indices is not None):
                if cfg.inference.flanking_single_slice:
                    near = lower if abs(t - 0.0) <= abs(t - 1.0) else upper
                    src_ty = near.cell_type_indices.astype(int)
                else:
                    src_ty = np.concatenate([lower.cell_type_indices, upper.cell_type_indices]).astype(int)
                carried_types = src_ty[sel]
        else:
            # ---- occupancy grid (z-marginalized) ---- #
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
            occ = np.clip(occ, 1e-6, None)
            occ_norm = occ / occ.sum()
            if cfg.inference.calibrate_density:
                dl = _density_raster(m._nxy(lower.coords_xy), G)
                du = _density_raster(m._nxy(upper.coords_xy), G)
                dl = dl / (dl.sum() + 1e-12); du = du / (du.sum() + 1e-12)
                dens = ((1 - t) * dl + t * du).reshape(-1)
                dens = np.clip(dens, 1e-8, None); dens = dens / dens.sum()
                b = cfg.inference.density_blend
                occ_norm = (1 - b) * occ_norm + b * dens
                occ_norm = occ_norm / occ_norm.sum()
            sel = rng.choice(grid_xy.shape[0], size=n_target, replace=True, p=occ_norm)
            jit = (rng.random((n_target, 2)) - 0.5) * (2.0 / G)
            xy = grid_xy.cpu().numpy()[sel] + jit
        xy_t = torch.tensor(xy, dtype=torch.float32, device=dev)

        # ---- types + layout code + mean expression (z-marginalized) ---- #
        typ_acc = 0.0; code_acc = 0.0; mu_acc = 0.0
        for zq in zs:
            _, typ, code = nn_["layout"](_layout_feats(m, xy_t, float(zq), cvec, ctx_slices))
            typ_acc = typ_acc + torch.softmax(typ, dim=1)
            code_acc = code_acc + code
            mu_acc = mu_acc + nn_["expr"](_expr_feats(m, xy_t, float(zq), code, cvec, ctx_slices))
        typ_prob = (typ_acc / len(zs)).cpu().numpy()
        mu = (mu_acc / len(zs)).cpu().numpy()

    # ---- cell types (optionally calibrated to the interpolated composition) ---- #
    ct_idx = None
    if m.n_types >= 2:
        if carried_types is not None:
            # Near-identical planes: keep the real carried types (preserves the niche).
            ct_idx = carried_types
        elif cfg.inference.composition_calibrate:
            p_target = _interp_composition(m, lower, upper, t)
            # Quota assignment: give each type exactly its interpolated share, filling
            # from the highest-confidence cells first, so the composition matches the
            # target while the spatial type arrangement (niche structure) is preserved.
            ct_idx = _quota_assign(typ_prob, p_target)
        else:
            ct_idx = typ_prob.argmax(1)

    # ---- expression decode ---- #
    # The generative decode is a *mean + structured-noise* sampler:
    #   expr = [(1-aw)·mu + aw·anchor] + noise_scale · (L · s_coherent)
    # The mean component is the field mean ``mu`` optionally anchored on a real
    # same-type profile retrieved by the learned layout (spatial coherence + real
    # covariance); the additive factor-analysis noise injects calibrated biological
    # variance (per-gene variance and gene-gene covariance) without smearing the mean.
    decode = cfg.inference.expr_decode
    if decode == "generative":
        if cfg.inference.anchor_weight > 0.0:
            anchor = _nearest_real(m, ct_idx, xy, lower, upper, mu=mu,
                                   expr_weight=cfg.inference.anchor_expr_weight)
            mean_c = (1 - cfg.inference.anchor_weight) * mu + cfg.inference.anchor_weight * anchor
        else:
            mean_c = mu
        ns = cfg.inference.noise_scale
        if ns > 0.0:
            s = _coherent_latent(m, xy, n_target, cfg.expression.n_factors,
                                 cfg.inference.latent_coherence, cfg.seed)
            Lmat = nn_["expr"].loadings.detach().cpu().numpy()
            expr = mean_c + ns * (s @ Lmat.T)
        else:
            expr = mean_c
    elif decode == "residual":
        expr = _add_residual(m, mu, ct_idx, xy, lower, upper)
    else:  # "field"
        expr = mu.copy()

    # Leakage-safe per-gene mean/variance calibration to the interpolated target.
    if cfg.inference.calibrate_gene_stats:
        expr = _calibrate_gene_stats(expr, lower, upper, t)

    expr = np.clip(expr, 0.0, None)
    # The field is trained on log1p-normalized expression; emitting count-like values
    # (expm1) lets the evaluator's per-cell count normalization behave as it does for
    # the raw-count ground truth, which the scale-sensitive mean/variance metrics need.
    # Rank-based primary metrics are invariant to this monotonic map.
    if cfg.inference.output_counts:
        expr = np.expm1(np.clip(expr, 0.0, 20.0))
    expr = expr.astype(np.float32)
    coords = np.column_stack([m._dxy(xy), np.full(n_target, float(z))]).astype(np.float32)
    labels = (np.array([m.cell_type_names[i] for i in ct_idx]) if (ct_idx is not None and m.cell_type_names)
              else (ct_idx.astype(str) if ct_idx is not None else None))
    return VirtualSlice(coords, expr, labels, ct_idx)


# --------------------------------------------------------------------------- #
# Inference helpers                                                            #
# --------------------------------------------------------------------------- #
def _slice_state(m, s):
    for st, sl in zip(m._S, m.stack.slices):
        if sl is s:
            return st
    return m._S[0]


def tp_frac(z, zl, zh):
    if zh == zl:
        return 0.5
    return float(np.clip((float(z) - zl) / (zh - zl), 0.0, 1.0))


def _planes_near_identical(m, lower, upper, ratio_thresh=0.85):
    """True if the flanking slices sit on (nearly) the same geometry — i.e. the median
    cross-slice nearest-neighbour distance is small relative to the within-slice spacing.
    Serial near-identical planes (volumetric) pass; distinct/drifting tissue does not."""
    from scipy.spatial import cKDTree
    lo = np.asarray(m._nxy(lower.coords_xy), np.float64)
    hi = np.asarray(m._nxy(upper.coords_xy), np.float64)
    if lo.shape[0] < 3 or hi.shape[0] < 3:
        return False
    d_self, _ = cKDTree(lo).query(lo, k=2)
    s0 = np.median(d_self[:, 1]) + 1e-9
    d_cross, _ = cKDTree(hi).query(lo, k=1)
    return float(np.median(d_cross) / s0) < ratio_thresh


def _quota_assign(typ_prob, p_target):
    """Assign each cell a type so the population matches ``p_target`` exactly, filling
    each type's quota from its highest-probability cells (preserves spatial structure).
    """
    n, C = typ_prob.shape
    quota = np.floor(p_target * n).astype(int)
    rem = int(n - quota.sum())
    if rem > 0:
        frac_order = np.argsort(-(p_target * n - quota))
        for i in range(rem):
            quota[frac_order[i % C]] += 1
    assigned = -np.ones(n, dtype=int)
    filled = np.zeros(C, dtype=int)
    # Greedy over all (prob, cell, type) triples, highest confidence first.
    ci, ti = np.meshgrid(np.arange(n), np.arange(C), indexing="ij")
    trip = np.stack([typ_prob.ravel(), ci.ravel(), ti.ravel()], axis=1)
    trip = trip[np.argsort(-trip[:, 0])]
    for p, i, c in trip:
        i = int(i); c = int(c)
        if assigned[i] == -1 and filled[c] < quota[c]:
            assigned[i] = c; filled[c] += 1
    un = assigned == -1
    if un.any():
        assigned[un] = typ_prob[un].argmax(1)
    return assigned


def _interp_composition(m, lower, upper, t):
    """z-interpolated flanking cell-type composition (leakage-safe, training slices)."""
    nt = m.n_types
    def comp(sl):
        c = np.zeros(nt)
        if sl.cell_type_indices is not None and sl.n_spots:
            b = np.bincount(sl.cell_type_indices.astype(int), minlength=nt).astype(float)
            c = b / b.sum()
        else:
            c[:] = 1.0 / nt
        return c
    return (1 - t) * comp(lower) + t * comp(upper)


def _coherent_latent(m, xy, n, r, alpha, seed, k=8, iters=2):
    """Draw a factor code s (n, r) with a spatially-coherent component + white part.

    ``alpha`` of the code is a smooth field over the cells' kNN graph (so spatially
    variable factors stay spatially autocorrelated -> Moran's I preserved); the rest is
    idiosyncratic per-cell noise. Each column is renormalized to ~unit variance so the
    decoder covariance ``L Lᵀ`` stays calibrated."""
    rng = np.random.default_rng(seed + 12345)
    s_indep = rng.standard_normal((n, r))
    if alpha <= 0.0 or n < k + 2:
        return s_indep.astype(np.float32)
    from scipy.spatial import cKDTree
    _, nn = cKDTree(xy).query(xy, k=min(k + 1, n))
    nn = nn[:, 1:]
    white = rng.standard_normal((n, r))
    smooth = white.copy()
    for _ in range(iters):
        smooth = 0.5 * smooth + 0.5 * smooth[nn].mean(axis=1)
    std = smooth.std(axis=0, keepdims=True); std[std == 0] = 1.0
    smooth = smooth / std
    s = alpha * smooth + np.sqrt(max(1.0 - alpha ** 2, 0.0)) * s_indep
    return s.astype(np.float32)


def _nearest_real(m, ct_idx, xy_norm, lower, upper, mu=None, expr_weight=1.0):
    """Real same-type anchor profile per synthesized cell.

    The match combines *position* with the field's predicted *expression state*: each
    synthesized cell is matched to the real same-type flanking cell nearest in the joint
    space ``[xy, w · embed(expr)]``. Using the predicted ``mu`` (not just position) picks
    the real cell whose molecular state the layout/expression field actually predicts, so
    the anchor carries both real gene-gene covariance and the right sub-state — improving
    the co-expression and Moran's structure over a position-only match."""
    from scipy.spatial import cKDTree
    src = np.concatenate([lower.expression, upper.expression], 0).astype(np.float32)
    src_xy = np.concatenate([m._nxy(lower.coords_xy), m._nxy(upper.coords_xy)], 0)
    src_ty = (np.concatenate([lower.cell_type_indices, upper.cell_type_indices])
              if (lower.cell_type_indices is not None and upper.cell_type_indices is not None)
              else np.zeros(src.shape[0], int))
    out = np.zeros((xy_norm.shape[0], src.shape[1]), np.float32)
    types = ct_idx if ct_idx is not None else np.zeros(xy_norm.shape[0], int)
    # Expression-state features (PCA embedding), scaled to be commensurate with xy.
    src_emb = q_emb = None
    if mu is not None and expr_weight > 0.0:
        src_emb = m._embed(src)
        q_emb = m._embed(np.clip(mu, 0.0, None))
        sc = expr_weight / (src_emb.std() + 1e-8)
        src_emb = src_emb * sc; q_emb = q_emb * sc
    for c in np.unique(types):
        q = np.where(types == c)[0]
        s = np.where(src_ty == c)[0]
        if s.size == 0:
            s = np.arange(src.shape[0])
        if src_emb is not None:
            tree = cKDTree(np.hstack([src_xy[s], src_emb[s]]))
            _, nn = tree.query(np.hstack([xy_norm[q], q_emb[q]]), k=1)
        else:
            _, nn = cKDTree(src_xy[s]).query(xy_norm[q], k=1)
        out[q] = src[s][nn]
    return out


def _add_residual(m, mu, ct_idx, xy_norm, lower, upper):
    """v11-style hybrid decode (kept as a fallback): blend the mean field with a real
    same-type cell's profile, matched by predicted type + spatial position."""
    a = m.cfg.inference.residual_weight
    anchor = _nearest_real(m, ct_idx, xy_norm, lower, upper)
    return (1 - a) * mu + a * anchor


def _calibrate_gene_stats(expr, lower, upper, t):
    """Pin per-gene mean/variance to the z-interpolated flanking target (leakage-safe).

    A per-gene affine map preserves every gene-gene correlation (and any rank-based
    metric), so this fixes the scale-sensitive mean/variance metrics without touching
    the covariance/spatial structure the generative decoder produced."""
    lo = np.asarray(lower.expression, np.float64)
    hi = np.asarray(upper.expression, np.float64)
    tgt_mean = (1 - t) * lo.mean(0) + t * hi.mean(0)
    tgt_std = (1 - t) * lo.std(0) + t * hi.std(0)
    cur_mean = expr.mean(0)
    cur_std = expr.std(0); cur_std[cur_std == 0] = 1.0
    return (expr - cur_mean) / cur_std * tgt_std + tgt_mean
