"""
Training and generation for SpatialCPA-v14 / H3D-FLA (PyTorch).

``train_model`` runs the pipeline's two-phase strategy:

* **Phase A** — pre-train the joint molecular-morphological encoder + decoders on the
  real slices (reconstruct the expression latent ``e``, the pseudo-image channels ``m``,
  and cell type). This makes the later latent->expression decoding accurate.
* **Phase B** — train the 3D-attention context module + the conditional flow-matching
  vector field with the CFM loss, **gap-aware** (whole context slices randomly dropped)
  and **z-marginalized** (z jittered), plus the biology-informed regularizers
  (closed-loop consistency, edge-aware adaptive smoothness / interface coherence, and a
  soft hypoxia-gradient directionality term), annealed in over training.

``generate_slice`` performs Stage 5 inference: build the 3D-attention context for a sheet
of query positions at the target z, integrate the conditional ODE from several initial
noises (marginalized), decode the generated joint latents to expression + type + a
continuous deformation, then **ground** each cell in the nearest real training profile
(preserving real gene-gene covariance) and match the cell-type composition to the
interpolated flanking mix.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from .nets import JointEncoder, ContextAttention, VectorField


# --------------------------------------------------------------------------- #
# Device                                                                       #
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
            return "cpu"
    except Exception:
        pass
    return "cuda"


# --------------------------------------------------------------------------- #
# Data prep                                                                    #
# --------------------------------------------------------------------------- #
def _prep(m, dev):
    """Per-slice numpy/tensor state: expression latent e, morphology m, joint h target."""
    from .latents import morphology_features
    S = []
    for s in m.stack.slices:
        e = m.expr_latent.encode(s.expression)               # (n, d_e)
        mo = morphology_features(s.coords_xy, s.cell_type_indices, m.n_types,
                                 k=m.cfg.latent.morph_k,
                                 density_sigma=m.cfg.latent.density_sigma)
        es = (e - m._e_mean) / m._e_std
        ms = (mo - m._m_mean) / m._m_std
        types = (s.cell_type_indices.astype(int) if s.cell_type_indices is not None
                 else np.zeros(s.n_spots, int))
        nxy = m._nxy(s.coords_xy).astype(np.float32)
        S.append(dict(
            e=torch.tensor(es, dtype=torch.float32, device=dev),
            m=torch.tensor(ms, dtype=torch.float32, device=dev),
            types=torch.tensor(types, dtype=torch.long, device=dev),
            nxy=nxy, nz=float(m._nz(s.z_center)),
            radius=_radius(nxy),
        ))
    return S


def _radius(nxy):
    if nxy.shape[0] == 0:
        return np.zeros(0, np.float32)
    c = nxy.mean(0)
    return np.linalg.norm(nxy - c, axis=1).astype(np.float32)


def _knn(query_xy, pool_xy, k):
    k = min(k, pool_xy.shape[0])
    _, idx = cKDTree(pool_xy).query(query_xy, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]
    return idx.astype(np.int64)


# --------------------------------------------------------------------------- #
# Context construction (3D attention tokens)                                   #
# --------------------------------------------------------------------------- #
def _build_pool(S, src_idx):
    """Concatenate source-slice cells into one candidate pool (+ per-slice spans)."""
    nxy, z, owner = [], [], []
    for j, si in enumerate(src_idx):
        s = S[si]
        nxy.append(s["nxy"])
        z.append(np.full(s["nxy"].shape[0], s["nz"], np.float32))
        owner.append(np.full(s["nxy"].shape[0], j, np.int64))
    return (np.vstack(nxy).astype(np.float32),
            np.concatenate(z).astype(np.float32),
            np.concatenate(owner).astype(np.int64))


def _context(m, h_pool, pool_nxy, pool_z, pool_owner, src_idx, S, query_nxy, query_z,
             encoder, ctxmod, dev, drop_owner=None, rng=None):
    """Return context vectors (Q, d_model) for the query positions.

    Tokens = local kNN cells from the pool + one global summary token per source slice.
    ``drop_owner`` (gap-aware) masks one source slice's local tokens and its global token.
    """
    cfg = m.cfg
    Q = query_nxy.shape[0]
    n_local = min(cfg.attn.n_context, pool_nxy.shape[0])
    nbr = _knn(query_nxy, pool_nxy, n_local)                 # (Q, n_local)

    pool_pos = np.column_stack([pool_nxy, pool_z]).astype(np.float32)
    pool_pos_t = torch.tensor(pool_pos, device=dev)
    local_h = h_pool[torch.tensor(nbr, device=dev)]          # (Q, n_local, d_joint)
    local_pos = pool_pos_t[torch.tensor(nbr, device=dev)]    # (Q, n_local, 3)
    local_tok = ctxmod.encode_tokens(
        local_h.reshape(-1, local_h.shape[-1]),
        local_pos.reshape(-1, 3)).reshape(Q, n_local, -1)

    # global per-slice summary tokens (mean h at slice centroid) — long-range context
    g_h, g_pos = [], []
    if cfg.attn.n_global_tokens <= 0:
        src_iter = []
    else:
        src_iter = list(enumerate(src_idx))
    for j, si in src_iter:
        s = S[si]
        idx = np.where(pool_owner == j)[0]
        if idx.size == 0:
            continue
        gh = h_pool[torch.tensor(idx, device=dev)].mean(0, keepdim=True)   # (1, d)
        cen = s["nxy"].mean(0)
        gp = torch.tensor([[cen[0], cen[1], s["nz"]]], dtype=torch.float32, device=dev)
        g_h.append(gh); g_pos.append(gp)
    if g_h:
        g_h = torch.cat(g_h, 0); g_pos = torch.cat(g_pos, 0)
        glob_tok = ctxmod.encode_tokens(g_h, g_pos)[None].expand(Q, -1, -1)  # (Q, G, d)
        tokens = torch.cat([local_tok, glob_tok], dim=1)
        n_glob = glob_tok.shape[1]
    else:
        tokens = local_tok
        n_glob = 0

    # padding mask (gap-aware): drop one source slice's local + global tokens
    pad = None
    if drop_owner is not None:
        local_owner = pool_owner[nbr]                        # (Q, n_local)
        pad_local = torch.tensor(local_owner == drop_owner, device=dev)
        pad_glob = torch.zeros((Q, n_glob), dtype=torch.bool, device=dev)
        if n_glob:
            # global token order follows src_idx order among those with cells
            present = [j for j in range(len(src_idx)) if np.any(pool_owner == j)]
            for gi, j in enumerate(present):
                if j == drop_owner:
                    pad_glob[:, gi] = True
        pad = torch.cat([pad_local, pad_glob], dim=1)
        # never let a whole row be masked
        allmask = pad.all(dim=1)
        if allmask.any():
            pad[allmask, -1] = False

    query_pos = torch.tensor(
        np.column_stack([query_nxy, np.full(Q, float(query_z), np.float32)]),
        dtype=torch.float32, device=dev)
    return ctxmod(query_pos, tokens, key_padding_mask=pad)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_model(m):
    cfg = m.cfg
    torch.manual_seed(cfg.train.seed); np.random.seed(cfg.train.seed)
    dev = _device(cfg, m)
    if cfg.train.verbose:
        print(f"[spatialcpav14] training device: {dev}")

    encoder = JointEncoder(m.expr_latent.dim, m.n_morph, cfg.encoder.joint_dim,
                           m.n_types, cfg.encoder.hidden, cfg.encoder.dropout).to(dev)
    ctxmod = ContextAttention(cfg.encoder.joint_dim, cfg.attn.d_model, cfg.attn.n_heads,
                              cfg.attn.fourier_bands, cfg.attn.dropout).to(dev)
    vfield = VectorField(cfg.encoder.joint_dim, ctxmod.out_dim, cfg.attn.fourier_bands,
                         cfg.flow.hidden, cfg.flow.n_layers, cfg.flow.time_embed_dim).to(dev)
    m._nn = dict(encoder=encoder, ctxmod=ctxmod, vfield=vfield, dev=dev)
    S = _prep(m, dev); m._S = S

    _phase_a(m, S, encoder, dev)
    _phase_b(m, S, encoder, ctxmod, vfield, dev)
    encoder.eval(); ctxmod.eval(); vfield.eval()


def _phase_a(m, S, encoder, dev):
    """Reconstruction pre-training of the joint encoder + decoders."""
    cfg = m.cfg
    params = list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    rng = np.random.default_rng(cfg.train.seed)
    idxs = [i for i, s in enumerate(S) if s["e"].shape[0] > 0]
    for ep in range(cfg.train.pretrain_epochs):
        tot = 0.0; nb = 0
        for i in idxs:
            s = S[i]; n = s["e"].shape[0]
            B = min(cfg.train.batch_cells, n)
            b = torch.tensor(rng.choice(n, B, replace=False), device=dev)
            e, mo, ty = s["e"][b], s["m"][b], s["types"][b]
            h = encoder.encode(e, mo)
            loss = F.mse_loss(encoder.decode_e(h), e) + F.mse_loss(encoder.decode_m(h), mo)
            if m.n_types >= 2:
                loss = loss + F.cross_entropy(encoder.type_head(h), ty)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            tot += float(loss.detach()); nb += 1
        if cfg.train.verbose and (ep % 20 == 0 or ep == cfg.train.pretrain_epochs - 1):
            print(f"    [v14 A] epoch {ep} recon_loss={tot / max(nb, 1):.4f}")
    encoder.eval()
    # freeze encoder for the flow phase (fixed CFM targets); decoders still used (no grad)
    for p in encoder.parameters():
        p.requires_grad_(False)
    # cache joint-latent targets per slice
    with torch.no_grad():
        for s in S:
            s["h"] = encoder.encode(s["e"], s["m"]) if s["e"].shape[0] else s["e"].new_zeros((0, m.cfg.encoder.joint_dim))


def _phase_b(m, S, encoder, ctxmod, vfield, dev):
    """Conditional flow matching + biology-informed regularization."""
    cfg = m.cfg; bio = cfg.bio
    order = np.argsort(m.stack.z_centers())
    interior = order[1:-1] if len(order) >= 3 else order[:1]

    plan = []
    for i in interior:
        i = int(i); pos = int(np.where(order == i)[0][0])
        lo, hi = int(order[pos - 1]), int(order[pos + 1])
        src = [lo, hi]
        pool_nxy, pool_z, pool_owner = _build_pool(S, src)
        # target-cell displacement target = offset from its flanking-neighbor centroid
        nbr = _knn(S[i]["nxy"], pool_nxy, min(cfg.attn.n_context, pool_nxy.shape[0]))
        cen = pool_nxy[nbr].mean(1)                          # (n_i, 2)
        disp_t = (S[i]["nxy"] - cen).astype(np.float32)
        plan.append(dict(i=i, src=src, pool_nxy=pool_nxy, pool_z=pool_z,
                         pool_owner=pool_owner, disp=disp_t))

    params = list(ctxmod.parameters()) + list(vfield.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    rng = np.random.default_rng(cfg.train.seed + 1)
    z_half = 1.0  # z already normalized to ~[-1,1]

    for ep in range(cfg.train.epochs):
        anneal = min(1.0, (ep + 1) / max(bio.anneal_epochs, 1))
        tot = 0.0; nb = 0
        for pl in plan:
            i = pl["i"]; s = S[i]; n = s["h"].shape[0]
            if n == 0:
                continue
            B = min(cfg.train.batch_cells, n)
            b = rng.choice(n, B, replace=False)
            bt = torch.tensor(b, device=dev)
            h1 = s["h"][bt]                                  # (B, d) frozen target
            query_nxy = s["nxy"][b]
            zq = s["nz"] + rng.normal(0, cfg.train.z_sigma) * z_half   # z-marginalization

            pool_h = torch.cat([S[j]["h"] for j in pl["src"]], 0)
            drop = None
            if rng.random() < cfg.train.gap_dropout and len(pl["src"]) > 1:
                drop = int(rng.integers(len(pl["src"])))
            ctx = _context(m, pool_h, pl["pool_nxy"], pl["pool_z"], pl["pool_owner"],
                           pl["src"], S, query_nxy, zq, encoder, ctxmod, dev,
                           drop_owner=drop, rng=rng)

            # conditional flow matching (OT straight-line path)
            h0 = torch.randn_like(h1)
            t = torch.rand(B, device=dev)
            ht = (1 - t)[:, None] * h0 + t[:, None] * h1
            zt = torch.full((B,), float(zq), device=dev)
            v = vfield(ht, t, ctx, zt)
            u = h1 - h0
            loss = F.mse_loss(v, u)

            # ---- biology-informed regularizers (decode the predicted endpoint) ----
            h1_hat = ht + (1 - t)[:, None] * v
            if bio.w_consistency > 0:
                e_hat = encoder.decode_e(h1_hat)
                m_hat = encoder.decode_m(h1_hat)
                h_re = encoder.encode(e_hat, m_hat)
                loss = loss + anneal * bio.w_consistency * F.mse_loss(h_re, h1_hat)
            if bio.w_smooth > 0 or bio.w_interface > 0:
                loss = loss + anneal * _smoothness(m, s, b, h1_hat, encoder, dev,
                                                   bio.w_smooth, bio.w_interface)
            if bio.w_hypoxia > 0 and m.n_types >= 2:
                loss = loss + anneal * bio.w_hypoxia * _hypoxia(s, b, h1_hat, encoder, dev,
                                                                bio.hypoxia_margin, rng)
            # displacement head (learned deformation field)
            disp_pred = vfield.displacement(h1, torch.ones(B, device=dev), ctx, zt)
            disp_tgt = torch.tensor(pl["disp"][b], device=dev)
            loss = loss + 0.5 * F.mse_loss(disp_pred, disp_tgt)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            tot += float(loss.detach()); nb += 1
        if cfg.train.verbose and (ep % 40 == 0 or ep == cfg.train.epochs - 1):
            print(f"    [v14 B] epoch {ep} cfm+bio={tot / max(nb, 1):.4f}")


def _smoothness(m, s, b, h1_hat, encoder, dev, w_smooth, w_interface):
    """Edge-aware latent total variation: strong continuity inside homogeneous regions,
    relaxed across morphological interfaces (adaptive smoothness / interface coherence)."""
    xy = s["nxy"][b]
    if xy.shape[0] < 4:
        return h1_hat.new_zeros(())
    kk = min(5, xy.shape[0])
    _, idx = cKDTree(xy).query(xy, k=kk)
    idx = idx[:, 1:]                                         # (B, k-1)
    mo = s["m"][b]                                           # morphology channels
    with torch.no_grad():
        edge = (mo[:, None, :] - mo[torch.tensor(idx, device=dev)]).abs().mean(-1)  # (B,k-1)
        w = torch.exp(-4.0 * edge)                          # relax at interfaces
    diff = (h1_hat[:, None, :] - h1_hat[torch.tensor(idx, device=dev)]).pow(2).mean(-1)
    tv = (w * diff).mean()
    return (w_smooth + w_interface) * tv


def _hypoxia(s, b, h1_hat, encoder, dev, margin, rng):
    """Soft core->periphery gradient: inner cells score >= outer cells (margin)."""
    r = torch.tensor(s["radius"][b], device=dev)
    sh = encoder.hypoxia_head(h1_hat)[:, 0]
    B = sh.shape[0]
    perm = torch.tensor(rng.permutation(B), device=dev)
    r2, sh2 = r[perm], sh[perm]
    inner = r < r2                                           # first cell is more inner
    d = torch.where(inner, sh2 - sh, sh - sh2)              # outer - inner
    return F.relu(d + margin).mean()


# --------------------------------------------------------------------------- #
# Generation (Stage 5)                                                         #
# --------------------------------------------------------------------------- #
def _tp_frac(z, zl, zh):
    if zh == zl:
        return 0.5
    return float(np.clip((float(z) - zl) / (zh - zl), 0.0, 1.0))


def _planes_near_identical(m, lower, upper, ratio_thresh):
    lo = np.asarray(m._nxy(lower.coords_xy), np.float64)
    hi = np.asarray(m._nxy(upper.coords_xy), np.float64)
    if lo.shape[0] < 3 or hi.shape[0] < 3:
        return False
    d_self, _ = cKDTree(lo).query(lo, k=2)
    s0 = np.median(d_self[:, 1]) + 1e-9
    d_cross, _ = cKDTree(hi).query(lo, k=1)
    return float(np.median(d_cross) / s0) < ratio_thresh


def _resample_layout(m, lower, upper, t, n_target, rng):
    """Resample the flanking supports in the z-interpolated ratio. Returns the jittered
    positions AND the pool index each came from (pool = lower cells then upper cells)."""
    lo_xy = m._nxy(lower.coords_xy).astype(np.float32)
    hi_xy = m._nxy(upper.coords_xy).astype(np.float32)
    props = np.vstack([lo_xy, hi_xy])
    w = np.concatenate([np.full(len(lo_xy), max(1 - t, 1e-3)),
                        np.full(len(hi_xy), max(t, 1e-3))])
    w = w / w.sum()
    # Sample without replacement when the pool is large enough (avoids coincident cells
    # that spike local density / distort neighbourhoods); fall back to replacement only if
    # more cells are requested than the pool holds.
    replace = n_target > props.shape[0]
    sel = rng.choice(props.shape[0], size=n_target, replace=replace, p=w)
    med = np.median([lower.median_spacing(), upper.median_spacing()])
    jit = rng.standard_normal((n_target, 2)) * (0.05 * med / (m._xy_s.mean()))
    return (props[sel] + jit).astype(np.float32), sel.astype(np.int64)


def _state_for(m, s):
    for st, sl in zip(m._S, m.stack.slices):
        if sl is s:
            return st
    return m._S[0]


def generate_slice(m, z):
    from .model import VirtualSlice
    cfg = m.cfg; gcfg = cfg.generation; nn_ = m._nn; dev = nn_["dev"]
    encoder, ctxmod, vfield = nn_["encoder"], nn_["ctxmod"], nn_["vfield"]
    lower, upper = m.stack.pick_flanking_slices(z)
    t = _tp_frac(z, lower.z_center, upper.z_center)
    n_target = max(int(round((1 - t) * lower.n_spots + t * upper.n_spots)), 1)
    rng = np.random.default_rng(cfg.seed)

    li, ui = m.stack.slices.index(lower), m.stack.slices.index(upper)
    n_lo = lower.n_spots
    mode = gcfg.position_mode
    if mode == "nearest" or (mode == "morph" and
                             _planes_near_identical(m, lower, upper, gcfg.near_identical_ratio)):
        near_is_lower = t <= 0.5
        near = lower if near_is_lower else upper
        near_xy = m._nxy(near.coords_xy).astype(np.float32)
        if near_xy.shape[0] > n_target:
            pick = rng.choice(near_xy.shape[0], n_target, replace=False)
        else:
            pick = np.arange(near_xy.shape[0])
        anchor = near_xy[pick]
        anchor_src = (pick if near_is_lower else pick + n_lo).astype(np.int64)
        use_disp = False
    else:
        anchor, anchor_src = _resample_layout(m, lower, upper, t, n_target, rng)
        use_disp = (mode == "morph")

    src = [li, ui]
    lo_st, hi_st = _state_for(m, lower), _state_for(m, upper)
    pool_nxy, pool_z, pool_owner = _build_pool(m._S, src)
    pool_h = torch.cat([m._S[li]["h"], m._S[ui]["h"]], 0)
    zn = m._nz(z)

    with torch.no_grad():
        ctx = _context(m, pool_h, pool_nxy, pool_z, pool_owner, src, m._S,
                       anchor, zn, encoder, ctxmod, dev)
        Q = anchor.shape[0]
        zt = torch.full((Q,), float(zn), device=dev)
        # marginalize over initial noise: integrate the ODE from several noises, average
        h_acc = torch.zeros((Q, cfg.encoder.joint_dim), device=dev)
        n_ens = max(cfg.flow.n_ensemble, 1)
        for _ in range(n_ens):
            h = torch.randn((Q, cfg.encoder.joint_dim), device=dev)
            steps = max(cfg.flow.n_ode_steps, 1)
            for si in range(steps):
                tt = torch.full((Q,), si / steps, device=dev)
                h = h + (1.0 / steps) * vfield(h, tt, ctx, zt)
            h_acc += h
        h_star = h_acc / n_ens
        e_hat = encoder.decode_e(h_star)                    # standardized latent
        type_logits = encoder.type_head(h_star) if m.n_types >= 2 else None
        if use_disp:
            disp = vfield.displacement(h_star, torch.ones(Q, device=dev), ctx, zt)
            disp = torch.clamp(disp * gcfg.displacement_scale, -0.15, 0.15)
            anchor = anchor + disp.cpu().numpy().astype(np.float32)

    e_hat_np = e_hat.cpu().numpy()

    # ---- ground each generated cell in a real training profile ----
    pool_expr = np.vstack([np.asarray(lower.expression), np.asarray(upper.expression)])
    pool_type = np.concatenate([
        lower.cell_type_indices if lower.cell_type_indices is not None else np.zeros(lower.n_spots, int),
        upper.cell_type_indices if upper.cell_type_indices is not None else np.zeros(upper.n_spots, int)])
    pool_e = np.vstack([lo_st["e"].cpu().numpy(), hi_st["e"].cpu().numpy()])  # standardized e

    expr, ct_idx = _ground(m, anchor, anchor_src, e_hat_np, pool_nxy, pool_e, pool_expr,
                           pool_type, gcfg, rng)

    # optional blend toward the flow-decoded profile
    if gcfg.edit_weight > 0.0:
        dec = m.expr_latent.decode(e_hat_np * m._e_std + m._e_mean)
        expr = (1 - gcfg.edit_weight) * expr + gcfg.edit_weight * dec

    # composition match to the interpolated flanking mix
    if gcfg.composition_match and m.n_types >= 2:
        ct_idx, expr = _match_composition(m, lower, upper, t, ct_idx, expr, anchor,
                                          pool_nxy, pool_e, e_hat_np, pool_type, pool_expr, rng)

    expr = np.clip(expr, 0.0, None)
    if gcfg.output_counts:
        expr = np.expm1(np.clip(expr, 0.0, 20.0))
    expr = expr.astype(np.float32)
    coords = np.column_stack([m._dxy(anchor), np.full(len(anchor), float(z))]).astype(np.float32)
    labels = (np.array([m.cell_type_names[i] for i in ct_idx])
              if (ct_idx is not None and m.cell_type_names)
              else (ct_idx.astype(str) if ct_idx is not None else None))
    return VirtualSlice(coords, expr, labels, ct_idx)


def _ground(m, anchor, anchor_src, e_hat, pool_nxy, pool_e, pool_expr, pool_type, gcfg, rng):
    """Emit a real exemplar profile per generated cell.

    ``anchor`` mode (default): each cell inherits the *own* real profile of the flanking
    cell it was resampled from — so spatially-contiguous cells keep an intra-slice-coherent
    neighbourhood (preserving real gene-gene covariance and spatial autocorrelation). A
    minority (``ground_blend_flow``) are re-grounded to the flow-decoded latent's nearest
    local real cell, injecting the flow's z-interpolated molecular signal.

    ``latent`` mode: every cell is grounded by the flow-decoded latent (softmax over local
    molecular similarity).
    """
    n = anchor.shape[0]
    expr = np.empty((n, pool_expr.shape[1]), dtype=np.float32)
    ct = np.empty(n, dtype=np.int64)
    if gcfg.ground_mode == "anchor":
        pick = anchor_src.copy()
        n_flow = int(round(gcfg.ground_blend_flow * n))
        if n_flow > 0:
            K = min(gcfg.ground_k, pool_nxy.shape[0])
            sel = rng.choice(n, size=n_flow, replace=False)
            cand = _knn(anchor[sel], pool_nxy, K)
            for r, i in enumerate(sel):
                ci = cand[r]
                d = np.linalg.norm(pool_e[ci] - e_hat[i], axis=1)
                pick[i] = ci[int(np.argmin(d))]
        expr[:] = pool_expr[pick]
        ct[:] = pool_type[pick]
        return expr, (ct if m.n_types >= 2 else None)

    K = min(gcfg.ground_k, pool_nxy.shape[0])
    cand = _knn(anchor, pool_nxy, K)                        # (Q, K) local real candidates
    for i in range(n):
        ci = cand[i]
        d = np.linalg.norm(pool_e[ci] - e_hat[i], axis=1)
        p = np.exp(-(d - d.min()) / max(gcfg.ground_temp, 1e-3))
        p = p / p.sum()
        pick = ci[rng.choice(len(ci), p=p)]
        expr[i] = pool_expr[pick]
        ct[i] = pool_type[pick]
    return expr, (ct if m.n_types >= 2 else None)


def _interp_comp(m, lower, upper, t):
    nt = m.n_types
    def comp(sl):
        c = np.full(nt, 1.0 / nt)
        if sl.cell_type_indices is not None and sl.n_spots:
            b = np.bincount(sl.cell_type_indices.astype(int), minlength=nt).astype(float)
            if b.sum() > 0:
                c = b / b.sum()
        return c
    return (1 - t) * comp(lower) + t * comp(upper)


def _match_composition(m, lower, upper, t, ct_idx, expr, anchor, pool_nxy, pool_e,
                       e_hat, pool_type, pool_expr, rng):
    """Prior-corrected resampling toward the interpolated flanking cell-type mix, drawing
    replacement exemplars of the needed type from each cell's local molecular candidates."""
    nt = m.n_types; n = len(ct_idx)
    target = _interp_comp(m, lower, upper, t)
    tgt_count = np.floor(target * n).astype(int)
    rem = n - tgt_count.sum()
    if rem > 0:
        for j in np.argsort(-(target * n - tgt_count))[:rem]:
            tgt_count[j] += 1
    cur = np.bincount(ct_idx, minlength=nt)
    K = min(m.cfg.generation.ground_k, pool_nxy.shape[0])
    cand = _knn(anchor, pool_nxy, K)
    over = [c for c in range(nt) if cur[c] > tgt_count[c]]
    for uc in range(nt):
        need = tgt_count[uc] - cur[uc]
        if need <= 0:
            continue
        cells = [i for i in range(n) if ct_idx[i] in over]
        rng.shuffle(cells)
        for i in cells:
            if need <= 0:
                break
            ci = cand[i]
            of_type = ci[pool_type[ci] == uc]
            if of_type.size == 0:
                continue
            d = np.linalg.norm(pool_e[of_type] - e_hat[i], axis=1)
            pick = of_type[int(np.argmin(d))]
            oldc = ct_idx[i]
            ct_idx[i] = uc; expr[i] = pool_expr[pick]
            cur[oldc] -= 1; cur[uc] += 1; need -= 1
            if cur[oldc] <= tgt_count[oldc] and oldc in over:
                over.remove(oldc)
    return ct_idx, expr
