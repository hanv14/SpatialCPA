"""
Training and generation for SpatialCPA-v13 (PyTorch), the LLM-based generator.

``train_model`` tokenizes every training cell into a cell-sentence and trains the
self-attention transformer with two objectives: a **masked gene-language-model** loss
(predict masked gene tokens — learn the gene "grammar") and a **spatial in-context**
loss (from a cell's flanking spatial neighbours attended in context, reconstruct its
cell type + expression and align the context embedding to the cell's own embedding, so
retrieval works). ``generate_slice`` performs **retrieval-augmented in-context
generation**: for each query position it builds a context embedding by cross-attending
over the retrieved flanking cells, samples a real exemplar from the LM-similarity
distribution, reads its type, and emits a (lightly LM-edited) expression profile.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import CellTransformer, SpatialContextAttention
from .tokenizer import PAD, MASK, CLS, N_SPECIAL


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
# Per-slice tensors                                                            #
# --------------------------------------------------------------------------- #
def _prep(m, dev):
    tok = m.tokenizer
    S = []
    for s in m.stack.slices:
        toks, pad = tok.encode(s.expression)
        types = (s.cell_type_indices.astype(int) if s.cell_type_indices is not None
                 else np.zeros(s.n_spots, int))
        S.append(dict(
            toks=torch.tensor(toks, device=dev),
            pad=torch.tensor(pad, device=dev),
            types=torch.tensor(types, dtype=torch.long, device=dev),
            expr=torch.tensor(np.asarray(s.expression, np.float32), device=dev),
            nxy=m._nxy(s.coords_xy).astype(np.float32),
            nz=float(m._nz(s.z_center)),
        ))
    return S


def _neighbor_index(tgt_xy, pool_xy, k):
    from scipy.spatial import cKDTree
    k = min(k, pool_xy.shape[0])
    _, idx = cKDTree(pool_xy).query(tgt_xy, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]
    return idx.astype(np.int64)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_model(m):
    cfg = m.cfg
    torch.manual_seed(cfg.train.seed); np.random.seed(cfg.train.seed)
    dev = _device(cfg, m)
    if cfg.train.verbose:
        print(f"[spatialcpav13] training device: {dev}")
    model = CellTransformer(m.tokenizer.vocab_size, m.n_genes, m.n_types, cfg.model).to(dev)
    sctx = SpatialContextAttention(cfg.model.d_model, cfg.model.n_heads,
                                   m.n_genes, m.n_types, cfg.model.dropout).to(dev)
    m._nn = dict(model=model, sctx=sctx, dev=dev)
    S = _prep(m, dev); m._S = S

    order = np.argsort(m.stack.z_centers())
    interior = order[1:-1] if len(order) >= 3 else order[:1]
    # Precompute, per interior slice, the flanking pool and each target's neighbour idx.
    plan = []
    for i in interior:
        i = int(i); pos = int(np.where(order == i)[0][0])
        lo, hi = int(order[pos - 1]), int(order[pos + 1])
        pool_xy = np.vstack([S[lo]["nxy"], S[hi]["nxy"]])
        pool_toks = torch.cat([S[lo]["toks"], S[hi]["toks"]], 0)
        pool_pad = torch.cat([S[lo]["pad"], S[hi]["pad"]], 0)
        nbr = _neighbor_index(S[i]["nxy"], pool_xy, cfg.model.n_context)
        plan.append(dict(i=i, pool_toks=pool_toks, pool_pad=pool_pad,
                         nbr=torch.tensor(nbr, device=dev)))

    params = list(model.parameters()) + list(sctx.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    lc = cfg.loss
    rng = np.random.default_rng(cfg.train.seed)

    for ep in range(cfg.train.epochs):
        tot = 0.0; nb = 0
        for pl in plan:
            i = pl["i"]; s = S[i]
            n = s["toks"].shape[0]
            B = min(cfg.train.batch_cells, n)
            bidx = torch.tensor(rng.choice(n, B, replace=False), device=dev)
            toks = s["toks"][bidx]; pad = s["pad"][bidx]
            xyz = torch.tensor(np.column_stack([s["nxy"][bidx.cpu().numpy()],
                               np.full(B, s["nz"], np.float32)]), device=dev)

            # ---- masked gene-language-model ---- #
            m_toks, mlm_labels = _mask_tokens(toks, pad, cfg.train.mlm_mask_frac,
                                              m.tokenizer.vocab_size, rng, dev)
            hidden, cls = model(m_toks, pad)
            logits = model.mlm_logits(hidden)
            mlm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       mlm_labels.reshape(-1), ignore_index=-100)

            # ---- spatial in-context (RAG) ---- #
            _, cls_clean = model(toks, pad)                       # target cell embeddings
            pool_hidden, pool_cls = model(pl["pool_toks"], pl["pool_pad"])  # (P, d)
            nbr = pl["nbr"][bidx]                                 # (B, N)
            neigh_emb = pool_cls[nbr]                             # (B, N, d)
            ctx = m._nn["sctx"](xyz, neigh_emb)                  # (B, d)
            type_loss = xyz.new_zeros(())
            if m.n_types >= 2:
                type_loss = F.cross_entropy(m._nn["sctx"].type_head(ctx), s["types"][bidx])
            expr_loss = F.mse_loss(m._nn["sctx"].expr_head(ctx), s["expr"][bidx])
            align_loss = (1.0 - F.cosine_similarity(ctx, cls_clean.detach(), dim=-1)).mean()

            loss = (lc.w_mlm * mlm_loss + lc.w_context_type * type_loss
                    + lc.w_context_expr * expr_loss + lc.w_align * align_loss)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            tot += float(loss.detach()); nb += 1
        if cfg.train.verbose and (ep % 40 == 0 or ep == cfg.train.epochs - 1):
            print(f"    [v13] epoch {ep} loss={tot / max(nb, 1):.4f}")

    model.eval(); sctx.eval()


def _mask_tokens(toks, pad, frac, vocab_size, rng, dev):
    """BERT-style masking of gene tokens (keep [CLS]/[PAD] intact)."""
    labels = toks.clone()
    maskable = (~pad) & (toks >= N_SPECIAL)
    probs = torch.rand(toks.shape, device=dev)
    do_mask = maskable & (probs < frac)
    labels[~do_mask] = -100
    out = toks.clone()
    out[do_mask] = MASK
    return out, labels


# --------------------------------------------------------------------------- #
# Generation (retrieval-augmented in-context)                                  #
# --------------------------------------------------------------------------- #
def _planes_near_identical(m, lower, upper, ratio_thresh):
    from scipy.spatial import cKDTree
    lo = np.asarray(m._nxy(lower.coords_xy), np.float64)
    hi = np.asarray(m._nxy(upper.coords_xy), np.float64)
    if lo.shape[0] < 3 or hi.shape[0] < 3:
        return False
    d_self, _ = cKDTree(lo).query(lo, k=2)
    s0 = np.median(d_self[:, 1]) + 1e-9
    d_cross, _ = cKDTree(hi).query(lo, k=1)
    return float(np.median(d_cross) / s0) < ratio_thresh


def _tp_frac(z, zl, zh):
    if zh == zl:
        return 0.5
    return float(np.clip((float(z) - zl) / (zh - zl), 0.0, 1.0))


def _sample_positions(m, lower, upper, z, t, n_target, rng):
    """Retrieval layout: real flanking positions (regime-adaptive), no morph/transport."""
    gcfg = m.cfg.generation
    mode = gcfg.position_mode
    if mode == "auto":
        mode = "nearest" if _planes_near_identical(m, lower, upper, gcfg.near_identical_ratio) else "flanking"
    lo_xy = m._nxy(lower.coords_xy).astype(np.float32)
    hi_xy = m._nxy(upper.coords_xy).astype(np.float32)
    if mode == "nearest":
        near = lower if abs(t - 0.0) <= abs(t - 1.0) else upper
        props = m._nxy(near.coords_xy).astype(np.float32)
        w = np.ones(props.shape[0])
    else:
        props = np.vstack([lo_xy, hi_xy])
        w = np.concatenate([np.full(len(lo_xy), max(1 - t, 1e-3)),
                            np.full(len(hi_xy), max(t, 1e-3))])
    w = w / w.sum()
    sel = rng.choice(props.shape[0], size=n_target, replace=True, p=w)
    med = np.median([lower.median_spacing(), upper.median_spacing()])
    jit = rng.standard_normal((n_target, 2)) * (0.25 * med / (m._xy_s.mean()))
    return (props[sel] + jit).astype(np.float32)


def generate_slice(m, z):
    from .model import VirtualSlice
    cfg = m.cfg; gcfg = cfg.generation; nn_ = m._nn; dev = nn_["dev"]
    model = nn_["model"]; sctx = nn_["sctx"]
    lower, upper = m.stack.pick_flanking_slices(z)
    t = _tp_frac(z, lower.z_center, upper.z_center)
    n_target = max(int(round((1 - t) * lower.n_spots + t * upper.n_spots)), 1)
    rng = np.random.default_rng(cfg.seed)

    # Positions (retrieval layout).
    xy = _sample_positions(m, lower, upper, z, t, n_target, rng)

    # Flanking candidate pool (RAG corpus) + its LM embeddings.
    lo_state = _state(m, lower); hi_state = _state(m, upper)
    pool_xy = np.vstack([lo_state["nxy"], hi_state["nxy"]])
    pool_toks = torch.cat([lo_state["toks"], hi_state["toks"]], 0)
    pool_pad = torch.cat([lo_state["pad"], hi_state["pad"]], 0)
    pool_expr = np.vstack([np.asarray(lower.expression), np.asarray(upper.expression)])
    pool_type = np.concatenate([
        lower.cell_type_indices if lower.cell_type_indices is not None else np.zeros(lower.n_spots, int),
        upper.cell_type_indices if upper.cell_type_indices is not None else np.zeros(upper.n_spots, int)])

    from scipy.spatial import cKDTree
    N = min(cfg.model.n_context, pool_xy.shape[0])
    K = min(gcfg.retrieval_k, pool_xy.shape[0])
    _, nbr = cKDTree(pool_xy).query(xy, k=N)
    if nbr.ndim == 1:
        nbr = nbr[:, None]
    _, cand = cKDTree(pool_xy).query(xy, k=K)
    if cand.ndim == 1:
        cand = cand[:, None]

    with torch.no_grad():
        _, pool_cls = model(pool_toks, pool_pad)                 # (P, d)
        pool_cls_n = F.normalize(pool_cls, dim=-1)
        xyz = torch.tensor(np.column_stack([xy, np.full(len(xy), m._nz(z), np.float32)]), device=dev)
        neigh_emb = pool_cls[torch.tensor(nbr, device=dev)]       # (n, N, d)
        ctx = sctx(xyz, neigh_emb)                               # (n, d)
        ctx_n = F.normalize(ctx, dim=-1)
        # LM-decoded profile + type from the context embedding.
        decoded = m._nn["sctx"].expr_head(ctx).cpu().numpy()
        type_logits = m._nn["sctx"].type_head(ctx).cpu().numpy() if m.n_types >= 2 else None
        # Retrieval distribution over the candidate real cells (LM-embedding similarity).
        cand_t = torch.tensor(cand, device=dev)
        cand_emb = pool_cls_n[cand_t]                            # (n, K, d)
        sim = (cand_emb * ctx_n[:, None, :]).sum(-1)            # (n, K)
        probs = torch.softmax(sim / max(gcfg.retrieval_temp, 1e-3), dim=-1).cpu().numpy()

    # Sample one real exemplar per query from the LM-similarity distribution.
    ex_local = np.array([rng.choice(K, p=probs[i]) for i in range(len(xy))])
    ex_idx = cand[np.arange(len(xy)), ex_local]                 # index into pool
    expr = pool_expr[ex_idx].astype(np.float32).copy()          # real exemplar profiles
    ct_idx = pool_type[ex_idx].astype(int) if m.n_types >= 2 else None

    # Optional LM gene-sentence edit: nudge the exemplar toward the LM-decoded profile.
    if gcfg.gene_edit and gcfg.edit_weight > 0.0:
        w = gcfg.edit_weight
        expr = (1 - w) * expr + w * np.clip(decoded, 0.0, None).astype(np.float32)

    # Composition match to the interpolated flanking mix (prior-corrected resampling).
    if gcfg.composition_match and m.n_types >= 2 and type_logits is not None:
        ct_idx, expr = _match_composition(m, lower, upper, t, ctx_probs=None,
                                          ct_idx=ct_idx, expr=expr, cand=cand,
                                          probs=probs, pool_type=pool_type,
                                          pool_expr=pool_expr, rng=rng)

    expr = np.clip(expr, 0.0, None)
    if gcfg.output_counts:
        expr = np.expm1(np.clip(expr, 0.0, 20.0))
    expr = expr.astype(np.float32)
    coords = np.column_stack([m._dxy(xy), np.full(len(xy), float(z))]).astype(np.float32)
    labels = (np.array([m.cell_type_names[i] for i in ct_idx]) if (ct_idx is not None and m.cell_type_names)
              else (ct_idx.astype(str) if ct_idx is not None else None))
    return VirtualSlice(coords, expr, labels, ct_idx)


def _state(m, s):
    for st, sl in zip(m._S, m.stack.slices):
        if sl is s:
            return st
    return m._S[0]


def _interp_comp(m, lower, upper, t):
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


def _match_composition(m, lower, upper, t, ctx_probs, ct_idx, expr, cand, probs,
                       pool_type, pool_expr, rng):
    """Nudge the exemplar-derived type composition to the interpolated flanking mix by
    re-sampling exemplars for a minimal set of cells (prior correction, not a hard quota).
    Each re-sampled cell draws a new exemplar restricted to the needed type, weighted by
    the LM-similarity distribution — so the swap stays LM-driven and keeps a real profile.
    """
    nt = m.n_types
    n = len(ct_idx)
    target = _interp_comp(m, lower, upper, t)
    tgt_count = np.floor(target * n).astype(int)
    rem = n - tgt_count.sum()
    if rem > 0:
        for j in np.argsort(-(target * n - tgt_count))[:rem]:
            tgt_count[j] += 1
    cur = np.bincount(ct_idx, minlength=nt)
    # Types that are over-represented give up cells to under-represented types.
    over = [c for c in range(nt) if cur[c] > tgt_count[c]]
    under = [c for c in range(nt) if cur[c] < tgt_count[c]]
    for uc in under:
        need = tgt_count[uc] - cur[uc]
        # Cells currently in over-represented types, ranked by low confidence for their type.
        pool_cells = [i for i in range(n) if ct_idx[i] in over]
        if not pool_cells:
            break
        rng.shuffle(pool_cells)
        for i in pool_cells:
            if need <= 0:
                break
            # find a candidate exemplar of type uc among this cell's retrieval candidates
            cand_i = cand[i]
            of_type = [(k, cand_i[k]) for k in range(len(cand_i)) if pool_type[cand_i[k]] == uc]
            if not of_type:
                continue
            ks = np.array([k for k, _ in of_type])
            pr = probs[i][ks]; pr = pr / pr.sum() if pr.sum() > 0 else None
            pick = of_type[int(rng.choice(len(of_type), p=pr))][1]
            oldc = ct_idx[i]
            ct_idx[i] = uc
            expr[i] = pool_expr[pick].astype(np.float32)
            cur[oldc] -= 1; cur[uc] += 1; need -= 1
            if cur[oldc] <= tgt_count[oldc] and oldc in over:
                over.remove(oldc)
    return ct_idx, expr
