"""
Training and generation for SpatialCPA-v17 (PyTorch) — the fully-generative variant.

Phase A trains the joint VAE (negative-binomial or Gaussian reconstruction + KL, plus
auxiliary morphology/type heads) and freezes it. Phase B trains the conditional
flow-matching field in the frozen latent space (gap-aware + z-marginalized), exactly as in
v14. ``generate_slice`` lays out a coherent sheet, integrates the flow to a latent per
spot, and **decodes** it into expression via the NB decoder (retrieval is available as an
ablation).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from .nets import JointVAE, ContextAttention, VectorField, nb_nll


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


def _knn(query_xy, pool_xy, k):
    k = min(k, pool_xy.shape[0])
    _, idx = cKDTree(pool_xy).query(query_xy, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]
    return idx.astype(np.int64)


# --------------------------------------------------------------------------- #
# Per-slice state                                                              #
# --------------------------------------------------------------------------- #
def _prep(m, dev):
    S = []
    for s in m.stack.slices:
        types = (s.cell_type_indices.astype(int) if s.cell_type_indices is not None
                 else np.zeros(s.n_spots, int))
        S.append(dict(
            xin=torch.tensor(m.xin(s.expression), device=dev),
            counts=torch.tensor(np.asarray(s.expression, np.float32), device=dev),
            lib=torch.tensor(m.library(s.expression), device=dev),
            mo=torch.tensor(m.morph(s), device=dev),
            types=torch.tensor(types, dtype=torch.long, device=dev),
            nxy=m._nxy(s.coords_xy).astype(np.float32),
            nz=float(m._nz(s.z_center)),
        ))
    return S


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_model(m):
    cfg = m.cfg
    torch.manual_seed(cfg.train.seed); np.random.seed(cfg.train.seed)
    dev = _device(cfg, m)
    if cfg.train.verbose:
        print(f"[spatialcpav17] training device: {dev}; likelihood: {m.likelihood}")
    vae = JointVAE(m.n_genes, m.n_morph, m.n_types, cfg.vae, likelihood=m.likelihood).to(dev)
    ctxmod = ContextAttention(cfg.vae.latent_dim, cfg.attn.d_model, cfg.attn.n_heads,
                              cfg.attn.fourier_bands, cfg.attn.dropout).to(dev)
    vfield = VectorField(cfg.vae.latent_dim, ctxmod.out_dim, cfg.attn.fourier_bands,
                         cfg.flow.hidden, cfg.flow.n_layers, cfg.flow.time_embed_dim).to(dev)
    m._nn = dict(vae=vae, ctxmod=ctxmod, vfield=vfield, dev=dev)
    S = _prep(m, dev); m._S = S

    _phase_a(m, S, vae, dev)
    _phase_b(m, S, ctxmod, vfield, dev)
    vae.eval(); ctxmod.eval(); vfield.eval()


def _phase_a(m, S, vae, dev):
    """Train the joint VAE (recon + KL + aux); then freeze and cache the posterior mean."""
    cfg = m.cfg
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    rng = np.random.default_rng(cfg.train.seed)
    idxs = [i for i, s in enumerate(S) if s["xin"].shape[0] > 0]
    for ep in range(cfg.train.pretrain_epochs):
        kl_w = cfg.vae.kl_weight * min(1.0, (ep + 1) / max(cfg.vae.kl_warmup_epochs, 1))
        tot = 0.0; nb = 0
        for i in idxs:
            s = S[i]; n = s["xin"].shape[0]
            B = min(cfg.train.batch_cells, n)
            b = torch.tensor(rng.choice(n, B, replace=False), device=dev)
            xin, mo, ty = s["xin"][b], s["mo"][b], s["types"][b]
            mu, logvar = vae.encode(xin, mo)
            h = vae.reparameterize(mu, logvar)
            if m.likelihood == "nb":
                mu_nb, theta, _ = vae.decode(h, s["lib"][b])
                recon = nb_nll(s["counts"][b], mu_nb, theta, cfg.vae.eps)
            else:
                mean, _, _ = vae.decode(h, s["lib"][b])
                recon = F.mse_loss(mean, xin) * m.n_genes
            kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(-1).mean()
            loss = recon + kl_w * kl
            loss = loss + cfg.vae.w_morph * F.mse_loss(vae.decode_morph(h), mo) * m.n_morph
            if m.n_types >= 2:
                loss = loss + cfg.vae.w_type * F.cross_entropy(vae.type_head(h), ty)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), cfg.train.grad_clip)
            opt.step()
            tot += float(loss.detach()); nb += 1
        if cfg.train.verbose and (ep % 30 == 0 or ep == cfg.train.pretrain_epochs - 1):
            print(f"    [v17 A] epoch {ep} vae_loss={tot / max(nb, 1):.3f} (kl_w={kl_w:.1e})")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        for s in S:
            if s["xin"].shape[0]:
                mu, _ = vae.encode(s["xin"], s["mo"]); s["h"] = mu
            else:
                s["h"] = s["xin"].new_zeros((0, m.cfg.vae.latent_dim))


# ── context construction (identical scheme to v14) ─────────────────────────── #
def _neighbor_slices(order, pos, k_each_side):
    lo = [int(order[j]) for j in range(max(0, pos - k_each_side), pos)]
    hi = [int(order[j]) for j in range(pos + 1, min(len(order), pos + 1 + k_each_side))]
    src = lo + hi
    if not src:
        src = [int(order[j]) for j in range(len(order)) if j != pos] or [int(order[pos])]
    return src


def _build_pool(S, src_idx):
    nxy, z, owner = [], [], []
    for j, si in enumerate(src_idx):
        s = S[si]
        nxy.append(s["nxy"]); z.append(np.full(s["nxy"].shape[0], s["nz"], np.float32))
        owner.append(np.full(s["nxy"].shape[0], j, np.int64))
    return (np.vstack(nxy).astype(np.float32), np.concatenate(z).astype(np.float32),
            np.concatenate(owner).astype(np.int64))


def _context(m, h_pool, pool_nxy, pool_z, pool_owner, src_idx, S, query_nxy, query_z,
             ctxmod, dev, drop_owner=None):
    cfg = m.cfg
    Q = query_nxy.shape[0]
    n_local = min(cfg.attn.n_context, pool_nxy.shape[0])
    nbr = _knn(query_nxy, pool_nxy, n_local)
    pool_pos_t = torch.tensor(np.column_stack([pool_nxy, pool_z]).astype(np.float32), device=dev)
    local_h = h_pool[torch.tensor(nbr, device=dev)]
    local_pos = pool_pos_t[torch.tensor(nbr, device=dev)]
    local_tok = ctxmod.encode_tokens(local_h.reshape(-1, local_h.shape[-1]),
                                     local_pos.reshape(-1, 3)).reshape(Q, n_local, -1)
    g_h = []
    src_iter = [] if cfg.attn.n_global_tokens <= 0 else list(enumerate(src_idx))
    present = []
    for j, si in src_iter:
        idx = np.where(pool_owner == j)[0]
        if idx.size == 0:
            continue
        gh = h_pool[torch.tensor(idx, device=dev)].mean(0, keepdim=True)
        cen = S[si]["nxy"].mean(0)
        gp = torch.tensor([[cen[0], cen[1], S[si]["nz"]]], dtype=torch.float32, device=dev)
        g_h.append(ctxmod.encode_tokens(gh, gp)); present.append(j)
    if g_h:
        glob = torch.cat(g_h, 0)[None].expand(Q, -1, -1)
        tokens = torch.cat([local_tok, glob], dim=1); n_glob = glob.shape[1]
    else:
        tokens = local_tok; n_glob = 0
    pad = None
    if drop_owner is not None:
        pad_local = torch.tensor(pool_owner[nbr] == drop_owner, device=dev)
        pad_glob = torch.zeros((Q, n_glob), dtype=torch.bool, device=dev)
        for gi, j in enumerate(present):
            if j == drop_owner:
                pad_glob[:, gi] = True
        pad = torch.cat([pad_local, pad_glob], dim=1)
        allmask = pad.all(dim=1)
        if allmask.any():
            pad[allmask, -1] = False
    query_pos = torch.tensor(np.column_stack([query_nxy, np.full(Q, float(query_z), np.float32)]),
                             dtype=torch.float32, device=dev)
    return ctxmod(query_pos, tokens, key_padding_mask=pad)


def _phase_b(m, S, ctxmod, vfield, dev):
    cfg = m.cfg
    order = np.argsort(m.stack.z_centers())
    interior = order[1:-1] if len(order) >= 3 else order[:1]
    k_side = max(cfg.attn.context_slices_each_side, 1)
    plan = []
    for i in interior:
        i = int(i); pos = int(np.where(order == i)[0][0])
        src = _neighbor_slices(order, pos, k_side)
        pool_nxy, pool_z, pool_owner = _build_pool(S, src)
        plan.append(dict(i=i, src=src, pool_nxy=pool_nxy, pool_z=pool_z, pool_owner=pool_owner))
    params = list(ctxmod.parameters()) + list(vfield.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    rng = np.random.default_rng(cfg.train.seed + 1)
    for ep in range(cfg.train.epochs):
        tot = 0.0; nb = 0
        for pl in plan:
            i = pl["i"]; s = S[i]; n = s["h"].shape[0]
            if n == 0:
                continue
            B = min(cfg.train.batch_cells, n)
            b = rng.choice(n, B, replace=False); bt = torch.tensor(b, device=dev)
            h1 = s["h"][bt]
            query_nxy = s["nxy"][b]
            zq = s["nz"] + rng.normal(0, cfg.train.z_sigma)
            pool_h = torch.cat([S[j]["h"] for j in pl["src"]], 0)
            drop = int(rng.integers(len(pl["src"]))) if (rng.random() < cfg.train.gap_dropout
                                                          and len(pl["src"]) > 1) else None
            ctx = _context(m, pool_h, pl["pool_nxy"], pl["pool_z"], pl["pool_owner"],
                           pl["src"], S, query_nxy, zq, ctxmod, dev, drop_owner=drop)
            h0 = torch.randn_like(h1)
            t = torch.rand(B, device=dev)
            ht = (1 - t)[:, None] * h0 + t[:, None] * h1
            v = vfield(ht, t, ctx, torch.full((B,), float(zq), device=dev))
            loss = F.mse_loss(v, h1 - h0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            tot += float(loss.detach()); nb += 1
        if cfg.train.verbose and (ep % 40 == 0 or ep == cfg.train.epochs - 1):
            print(f"    [v17 B] epoch {ep} cfm={tot / max(nb, 1):.4f}")


# --------------------------------------------------------------------------- #
# Generation                                                                   #
# --------------------------------------------------------------------------- #
def _tp_frac(z, zl, zh):
    return 0.5 if zh == zl else float(np.clip((float(z) - zl) / (zh - zl), 0.0, 1.0))


def _coherent_mask(lo_xy, hi_xy, t, freq, rng):
    allxy = np.vstack([lo_xy, hi_xy])
    span = (allxy.max(0) - allxy.min(0)) + 1e-6
    xn = (allxy - allxy.min(0)) / span
    f = np.zeros(allxy.shape[0])
    for _ in range(3):
        kx, ky = rng.uniform(0.5, freq, 2); ph = rng.uniform(0, 2 * np.pi)
        f += np.sin(2 * np.pi * (kx * xn[:, 0] + ky * xn[:, 1]) + ph)
    f = (f - f.min()) / (f.max() - f.min() + 1e-9)
    n_lo = lo_xy.shape[0]; is_lower = np.zeros(allxy.shape[0], bool); is_lower[:n_lo] = True
    keep = np.where(is_lower, (f >= t).astype(float), (f < t).astype(float)) + 1e-3
    return keep / keep.sum()


def _layout(m, lower, upper, t, n_target, rng):
    lo_xy = m._nxy(lower.coords_xy).astype(np.float32)
    hi_xy = m._nxy(upper.coords_xy).astype(np.float32)
    props = np.vstack([lo_xy, hi_xy])
    gcfg = m.cfg.generation
    if gcfg.coherent_source:
        w = _coherent_mask(lo_xy, hi_xy, t, gcfg.coherent_freq, rng)
    else:
        w = np.concatenate([np.full(len(lo_xy), max(1 - t, 1e-3)), np.full(len(hi_xy), max(t, 1e-3))])
        w = w / w.sum()
    replace = n_target > props.shape[0]
    sel = rng.choice(props.shape[0], size=n_target, replace=replace, p=w)
    med = np.median([lower.median_spacing(), upper.median_spacing()])
    jit = rng.standard_normal((n_target, 2)) * (0.05 * med / m._xy_s.mean())
    return (props[sel] + jit).astype(np.float32), sel.astype(np.int64)


def _interp_comp(m, lower, upper, t):
    nt = m.n_types
    def comp(sl):
        c = np.full(nt, 1.0 / nt)
        if sl.cell_type_indices is not None and sl.n_spots:
            bb = np.bincount(sl.cell_type_indices.astype(int), minlength=nt).astype(float)
            if bb.sum() > 0:
                c = bb / bb.sum()
        return c
    return (1 - t) * comp(lower) + t * comp(upper)


def _state_for(m, s):
    for st, sl in zip(m._S, m.stack.slices):
        if sl is s:
            return st
    return m._S[0]


def generate_slice(m, z):
    from .model import VirtualSlice
    cfg = m.cfg; gcfg = cfg.generation; nn_ = m._nn; dev = nn_["dev"]
    vae, ctxmod, vfield = nn_["vae"], nn_["ctxmod"], nn_["vfield"]
    lower, upper = m.stack.pick_flanking_slices(z)
    t = _tp_frac(z, lower.z_center, upper.z_center)
    n_target = max(int(round((1 - t) * lower.n_spots + t * upper.n_spots)), 1)
    rng = np.random.default_rng(cfg.seed)

    anchor, anchor_src = _layout(m, lower, upper, t, n_target, rng)
    li, ui = m.stack.slices.index(lower), m.stack.slices.index(upper)
    order = np.argsort(m.stack.z_centers()); zc = m.stack.z_centers()
    k_side = max(cfg.attn.context_slices_each_side, 1)
    below = [int(j) for j in order if zc[j] <= z]; above = [int(j) for j in order if zc[j] > z]
    ctx_src = (below[-k_side:] if below else []) + (above[:k_side] if above else [])
    for j in (li, ui):
        if j not in ctx_src:
            ctx_src.append(j)
    pool_nxy, pool_z, pool_owner = _build_pool(m._S, ctx_src)
    pool_h = torch.cat([m._S[j]["h"] for j in ctx_src], 0)
    zn = m._nz(z)

    # per-spot library size from the coherent-patch anchor's real cell
    pool_counts = np.vstack([np.asarray(lower.expression), np.asarray(upper.expression)])
    if gcfg.library_from_anchor and m.likelihood == "nb":
        lib = pool_counts[anchor_src].clip(0, None).sum(1).astype(np.float32)
        lib[lib == 0] = float(np.median(pool_counts.sum(1)) + 1.0)
    else:
        lib = np.ones(n_target, np.float32)

    with torch.no_grad():
        ctx = _context(m, pool_h, pool_nxy, pool_z, pool_owner, ctx_src, m._S, anchor, zn, ctxmod, dev)
        Q = anchor.shape[0]; zt = torch.full((Q,), float(zn), device=dev)
        h_acc = torch.zeros((Q, cfg.vae.latent_dim), device=dev)
        n_ens = max(cfg.flow.n_ensemble, 1)
        for _ in range(n_ens):
            h = torch.randn((Q, cfg.vae.latent_dim), device=dev)
            steps = max(cfg.flow.n_ode_steps, 1)
            for si in range(steps):
                h = h + (1.0 / steps) * vfield(h, torch.full((Q,), si / steps, device=dev), ctx, zt)
            h_acc += h
        h_star = h_acc / n_ens
        type_logits = vae.type_head(h_star) if m.n_types >= 2 else None

        if gcfg.retrieval:                                    # ── ablation: v14-style retrieval ──
            pool_e = torch.cat([m._S[li]["h"], m._S[ui]["h"]], 0).cpu().numpy()
            gpool_nxy = np.vstack([m._S[li]["nxy"], m._S[ui]["nxy"]]).astype(np.float32)
            K = min(gcfg.ground_k, gpool_nxy.shape[0])
            cand = _knn(anchor, gpool_nxy, K)
            hs = h_star.cpu().numpy()
            pick = np.array([cand[i][int(np.argmin(np.linalg.norm(pool_e[cand[i]] - hs[i], axis=1)))]
                             for i in range(Q)])
            expr = pool_counts[pick].astype(np.float32)
            pool_type = np.concatenate([
                lower.cell_type_indices if lower.cell_type_indices is not None else np.zeros(lower.n_spots, int),
                upper.cell_type_indices if upper.cell_type_indices is not None else np.zeros(upper.n_spots, int)])
            ct_idx = pool_type[pick] if m.n_types >= 2 else None
        else:                                                 # ── decode (fully generative) ──
            lib_t = torch.tensor(lib, device=dev)
            out, _, _ = vae.decode(h_star, lib_t)
            if m.likelihood == "nb":
                expr = out.cpu().numpy().astype(np.float32)   # NB mean = expected counts
            else:
                mean = out.cpu().numpy() * m._xin_std + m._xin_mean
                expr = mean.astype(np.float32)
            ct_idx = (type_logits.argmax(1).cpu().numpy().astype(np.int64)
                      if type_logits is not None else None)

    if gcfg.composition_match and m.n_types >= 2 and ct_idx is not None and type_logits is not None:
        ct_idx = _match_types(m, lower, upper, t, ct_idx, type_logits.softmax(1).cpu().numpy())

    expr = np.clip(expr, 0.0, None)
    if gcfg.output_counts and m.likelihood != "nb":
        expr = np.expm1(np.clip(expr, 0.0, 20.0))
    expr = expr.astype(np.float32)
    coords = np.column_stack([m._dxy(anchor), np.full(len(anchor), float(z))]).astype(np.float32)
    labels = (np.array([m.cell_type_names[i] for i in ct_idx])
              if (ct_idx is not None and m.cell_type_names)
              else (ct_idx.astype(str) if ct_idx is not None else None))
    return VirtualSlice(coords, expr, labels, ct_idx)


def _match_types(m, lower, upper, t, ct, p):
    """Relabel a minimal set of cells so the type composition matches the interpolated mix."""
    nt = m.n_types; n = len(ct)
    target = _interp_comp(m, lower, upper, t)
    tgt = np.floor(target * n).astype(int)
    rem = n - tgt.sum()
    if rem > 0:
        for j in np.argsort(-(target * n - tgt))[:rem]:
            tgt[j] += 1
    cur = np.bincount(ct, minlength=nt); conf = p[np.arange(n), ct]
    over = [c for c in range(nt) if cur[c] > tgt[c]]
    for uc in range(nt):
        need = tgt[uc] - cur[uc]
        if need <= 0:
            continue
        cells = [i for i in range(n) if ct[i] in over]
        cells.sort(key=lambda i: p[i, uc] - conf[i], reverse=True)
        for i in cells:
            if need <= 0:
                break
            oldc = ct[i]
            if cur[oldc] <= tgt[oldc]:
                continue
            ct[i] = uc; cur[oldc] -= 1; cur[uc] += 1; need -= 1
            if cur[oldc] <= tgt[oldc] and oldc in over:
                over.remove(oldc)
    return ct
