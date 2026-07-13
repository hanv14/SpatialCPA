"""
Neural cross-slice flow-matching bridge for SpatialCPA-v9 (PyTorch).

A conditional **rectified-flow / OT-CFM** model learns a velocity field that
transports the cell distribution of one flanking slice to the other in a joint
``(position, expression-latent)`` space. Training supervision is the *optimal-
transport coupling* between adjacent training slices (minibatch-OT conditional
flow matching, Tong et al. 2023): pairs ``(x0, x1)`` are drawn proportional to the
entropic-OT plan, the straight path ``x_s = (1-s)x0 + s x1`` is followed, and the
network learns ``v_theta(x_s, s, context) ≈ x1 - x0``. Because supervision comes
from OT couplings, the learned marginal field is a *distribution-correct* transport
(a learned generalization of the McCann displacement interpolation), not a mere
cell-to-cell regression.

At inference the flanking slices of the target z define the boundary
distributions; integrating the learned probability-flow ODE from the lower slice
to the fractional time ``t = (z − z_lo)/(z_hi − z_lo)`` yields the virtual slice.
A configurable fraction of the coherent OT-morph displacement is blended in as a
structural prior, which regularizes the learned field toward a spatially coherent
tissue deformation (and makes v9 degrade smoothly toward the strong v8 morph).

All training uses the training slices only; the query z never reveals held-out
content. If PyTorch is missing or training fails, the caller falls back to the OT
morph.
"""

from __future__ import annotations

import numpy as np

from . import transport as tp


class NeuralBridge:
    """Fit-once, generate-anywhere neural slice bridge."""

    def __init__(self, cfg, gene_names):
        self.cfg = cfg
        self.gene_names = list(gene_names)
        self.ok = False           # set True after a successful fit
        self._torch = None

    # ------------------------------------------------------------------ #
    def _device(self):
        import torch
        d = self.cfg.train.device
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _featurize(self, xy, latent):
        """(xy, latent) -> normalized joint feature; stores the transform."""
        f = np.concatenate([np.asarray(xy, np.float32), np.asarray(latent, np.float32)], axis=1)
        return (f - self._fmean) / self._fstd

    def _defeaturize(self, f):
        g = f * self._fstd + self._fmean
        return g[:, :2], g[:, 2:]

    # ------------------------------------------------------------------ #
    def fit(self, stack, gene_embedding=None):
        """Train the AE and the conditional flow on the training stack."""
        import torch
        import torch.nn as nn
        from .nets import ExpressionAE, ContextEncoder, FlowNet

        cfg = self.cfg
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        dev = self._device()

        slices = stack.slices
        X_all = np.concatenate([s.expression for s in slices], axis=0).astype(np.float32)
        n_genes = X_all.shape[1]
        # Standardize expression for the AE (per-gene, training-only).
        self._xmean = X_all.mean(0, keepdims=True)
        self._xstd = X_all.std(0, keepdims=True); self._xstd[self._xstd == 0] = 1.0
        Xn = (X_all - self._xmean) / self._xstd

        # ---- 1) expression autoencoder ---------------------------------- #
        ae = ExpressionAE(n_genes, cfg.model.latent_dim, cfg.model.ae_hidden,
                          dropout=cfg.model.dropout, gene_embedding=gene_embedding).to(dev)
        opt = torch.optim.Adam(ae.parameters(), lr=cfg.train.lr,
                               weight_decay=cfg.train.weight_decay)
        Xt = torch.tensor(Xn, device=dev)
        n = Xt.shape[0]
        bs = min(cfg.train.batch_size, n)
        for ep in range(cfg.train.ae_epochs):
            perm = torch.randperm(n, device=dev)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb = Xt[idx]
                rec, _ = ae(xb)
                loss = ((rec - xb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(idx)
            if cfg.train.verbose and (ep % 50 == 0 or ep == cfg.train.ae_epochs - 1):
                print(f"    [AE] epoch {ep} recon={tot / n:.4f}")
        ae.eval()
        with torch.no_grad():
            self._latent = {id(s): ae.encode(
                torch.tensor((s.expression - self._xmean) / self._xstd, device=dev)
            ).cpu().numpy() for s in slices}
        self.ae = ae

        # ---- feature normalization over all training cells -------------- #
        allf = np.concatenate([
            np.concatenate([s.coords_xy.astype(np.float32), self._latent[id(s)]], axis=1)
            for s in slices], axis=0)
        self._fmean = allf.mean(0, keepdims=True)
        self._fstd = allf.std(0, keepdims=True); self._fstd[self._fstd == 0] = 1.0
        feat_dim = allf.shape[1]

        # ---- 2) build OT-coupled adjacent-slice supervision ------------- #
        self._pairs = []   # list of (lo_feat_full, up_feat_full, plan, gap_norm)
        zc = stack.z_centers()
        gap_scale = float(np.median(np.diff(np.sort(zc)))) if len(zc) > 1 else 1.0
        gap_scale = gap_scale if gap_scale > 0 else 1.0
        order = np.argsort(zc)
        for a, b in zip(order[:-1], order[1:]):
            sl, su = slices[a], slices[b]
            if sl.n_spots < 3 or su.n_spots < 3:
                continue
            lof = self._featurize(sl.coords_xy, self._latent[id(sl)])
            upf = self._featurize(su.coords_xy, self._latent[id(su)])
            plan = _ot_plan(lof, upf, cfg.transport)
            gap = abs(su.z_center - sl.z_center) / gap_scale
            self._pairs.append((lof.astype(np.float32), upf.astype(np.float32),
                                plan.astype(np.float32), np.float32(gap)))
        if not self._pairs:
            raise RuntimeError("no adjacent training slice pairs for flow supervision")

        # ---- 3) train the conditional flow ------------------------------ #
        ctx_enc = ContextEncoder(feat_dim, cfg.model.context_dim).to(dev)
        flow = FlowNet(feat_dim, cfg.model.context_dim, cfg.model.flow_hidden,
                       cfg.model.flow_layers, cfg.model.time_embed_dim,
                       dropout=cfg.model.dropout).to(dev)
        params = list(ctx_enc.parameters()) + list(flow.parameters())
        opt = torch.optim.Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        pairs_t = [(torch.tensor(lo, device=dev), torch.tensor(up, device=dev),
                    torch.tensor(pl, device=dev), float(g))
                   for (lo, up, pl, g) in self._pairs]

        n_per = max(1, cfg.train.max_pairs_per_epoch // len(pairs_t))
        for ep in range(cfg.train.flow_epochs):
            tot = 0.0; cnt = 0
            for lo, up, pl, gap in pairs_t:
                gaps = torch.tensor([gap, 0.0, 0.0], device=dev, dtype=torch.float32)
                context = ctx_enc(lo, up, gaps)
                # sample OT-coupled pairs (i, j) ~ plan
                flat = pl.flatten()
                flat = flat / flat.sum()
                draws = torch.multinomial(flat, n_per, replacement=True)
                m = up.shape[0]
                i = torch.div(draws, m, rounding_mode="floor")
                j = draws % m
                x0, x1 = lo[i], up[j]
                s = torch.rand(x0.shape[0], device=dev)
                xs = (1 - s)[:, None] * x0 + s[:, None] * x1
                target = x1 - x0
                pred = flow(xs, s, context)
                loss = ((pred - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                if cfg.train.grad_clip:
                    nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                opt.step()
                tot += float(loss); cnt += 1
            if cfg.train.verbose and (ep % 100 == 0 or ep == cfg.train.flow_epochs - 1):
                print(f"    [flow] epoch {ep} fm_loss={tot / max(cnt,1):.4f}")

        self.ctx_enc = ctx_enc.eval()
        self.flow = flow.eval()
        self._dev = dev
        self._gap_scale = gap_scale
        self.ok = True
        return self

    # ------------------------------------------------------------------ #
    def generate(self, lower, upper, t, n_target, seed=0):
        """Integrate the learned flow from the lower slice to fraction t.

        Returns ``(coords_xy, latent)`` for ``n_target`` synthesized cells. The
        expression is decoded from ``latent`` by the caller (via the AE decoder or
        a nearest-real-cell snap).
        """
        import torch
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        dev = self._dev

        lof = self._featurize(lower.coords_xy, self._latent[id(lower)])
        upf = self._featurize(upper.coords_xy, self._latent[id(upper)])
        gap = abs(upper.z_center - lower.z_center) / self._gap_scale

        # Start cloud = ALL lower cells (one synthesized cell per source cell, no
        # duplication — matching the coherent morph; the emergent count is the lower
        # slice's, like a single-sheet deformation). Subsample WITHOUT replacement
        # only when a strictly smaller count is requested.
        n_lo = lof.shape[0]
        if n_target < n_lo:
            idx = np.sort(rng.choice(n_lo, n_target, replace=False))
        else:
            idx = np.arange(n_lo)
        x = torch.tensor(lof[idx], device=dev)

        lo_t = torch.tensor(lof, device=dev)
        up_t = torch.tensor(upf, device=dev)
        gaps = torch.tensor([gap, 0.0, 0.0], device=dev, dtype=torch.float32)
        with torch.no_grad():
            context = self.ctx_enc(lo_t, up_t, gaps)
            n_steps = max(1, cfg.flow.n_steps)
            ds = t / n_steps
            s = 0.0
            for _ in range(n_steps):
                sv = torch.full((x.shape[0],), s, device=dev)
                v = self.flow(x, sv, context)
                x = x + ds * v
                if cfg.flow.stochastic and cfg.flow.noise_scale > 0:
                    x = x + (cfg.flow.noise_scale * np.sqrt(abs(ds))) * torch.randn_like(x)
                s += ds
        xf = x.cpu().numpy()

        coords_xy, latent = self._defeaturize(xf)

        # Structural prior: blend a fraction of the coherent OT-morph displacement
        # of the SAME start cells, regularizing the learned field toward a coherent
        # tissue deformation (and making v9 degrade smoothly toward the strong v8
        # morph as morph_prior -> 1).
        if cfg.flow.morph_prior > 0:
            morph_xy, sub_a, _ = tp.smooth_morph(
                lower.coords_xy.astype(np.float64), upper.coords_xy.astype(np.float64),
                self._latent[id(lower)], self._latent[id(upper)],
                t, cfg.transport, cfg.transport, seed=seed)
            # Map morph positions back to full lower-cell indexing (smooth_morph may
            # subsample the anchor); cells outside the subsample keep their own xy.
            full = lower.coords_xy.astype(np.float64).copy()
            full[sub_a] = morph_xy
            morph_sel = full[idx]
            a = float(cfg.flow.morph_prior)
            coords_xy = (1 - a) * coords_xy + a * morph_sel
        return coords_xy.astype(np.float32), latent.astype(np.float32), idx

    def decode_expression(self, latent):
        import torch
        with torch.no_grad():
            rec = self.ae.decode(torch.tensor(latent, device=self._dev)).cpu().numpy()
        return (rec * self._xstd + self._xmean).astype(np.float32)


def _ot_plan(lof, upf, tcfg):
    """Entropic-OT plan between two feature clouds (joint position+latent cost)."""
    from scipy.spatial.distance import cdist
    n = min(lof.shape[0], tcfg.max_ot_cells)
    m = min(upf.shape[0], tcfg.max_ot_cells)
    # (already whole slices here; OT on full sets up to the cap)
    C = cdist(lof[:n], upf[:m], metric="sqeuclidean")
    C = C / (np.median(C) + 1e-9)
    return tp.sinkhorn_plan(C, tcfg.epsilon, tcfg.n_iter)
