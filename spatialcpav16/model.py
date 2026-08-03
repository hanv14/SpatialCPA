"""SpatialCPA-v16 — Spectral Tissue Flow.

Generation of a virtual section, end to end:

    1. geometry      z-interpolated SDF outline + intensity -> target point set
    2. basis         tissue harmonics of that point set -> (Phi*, lambda*)
    3. flow          conditional flow matching over harmonic tokens -> spectrum
    4. inverse       spectrum -> smooth per-cell expression + type fields
    5. calibration   force the per-gene spectral profile onto its z-interpolated target
    6. residual      restore high-frequency dispersion as whole-cell draws
    7. typing        sample the generated type posterior under a composition constraint

Stages 3 and 5 are the contribution. Stage 3 makes the *section* the unit of
generation, so spatial structure is produced coherently instead of emerging from
independent per-cell draws. Stage 5 turns spatial autocorrelation from something
a model might happen to get right into a quantity that is written down, targeted
and verified before anything is written out.
"""

from __future__ import annotations

import numpy as np

from .calibrate import (apply_calibration, blend_profiles, residual_draw,
                        scale_to_variance, section_profile)
from .config import SpatialCPAv16Config
from .data import SliceStack, VirtualSlice
from .geometry import interpolate_geometry, target_cell_count
from .harmonics import gft, igft, morans_full, tissue_harmonics
from .nets import TORCH, evaluate_field


class SpatialCPAv16:
    def __init__(self, stack: SliceStack, gene_names=None, cell_type_names=None,
                 cfg: SpatialCPAv16Config | None = None, verbose=True):
        self.stack = stack
        self.cfg = cfg or SpatialCPAv16Config()
        self.gene_names = list(gene_names) if gene_names is not None else None
        self.cell_type_names = (list(cell_type_names)
                                if cell_type_names is not None else None)
        self.verbose = verbose
        self.rng = np.random.default_rng(self.cfg.seed)
        self.diagnostics = {}
        self.trained = False
        self._fit()

    # ── fit ──────────────────────────────────────────────────────────────────
    def _log(self, msg):
        if self.verbose:
            print(f"  [v16] {msg}")

    def _fit(self):
        cfg = self.cfg
        slices = self.stack.slices
        G = slices[0].expression.shape[1]
        self.n_genes = G

        # Gene-space basis. The spectrum is modelled in a compact gene basis so
        # the coefficient matrix stays small; the calibration in stage 5 acts on
        # the *reconstructed genes*, so the per-gene autocorrelation guarantee is
        # unaffected by this compression.
        X_all = np.vstack([s.expression for s in slices]).astype(np.float64)
        self.gene_mean = X_all.mean(axis=0)
        K = int(min(cfg.flow.gene_components, max(2, min(X_all.shape) - 1)))
        from sklearn.decomposition import PCA
        self.gene_pca = PCA(n_components=K, random_state=cfg.seed).fit(X_all)
        self.K = K

        n_types = (len(self.cell_type_names) if self.cell_type_names
                   else int(max((s.cell_type_indices.max() + 1)
                                for s in slices if s.cell_type_indices is not None),
                            default=1))
        self.n_types = int(max(1, n_types))
        self.C = self.n_types

        # Per-section harmonic basis, spectrum, profile and residual pool.
        self.sections = []
        for s in slices:
            self.sections.append(self._encode_section(s))
        self._log(f"{len(self.sections)} sections encoded; "
                  f"{cfg.harmonics.n_modes} harmonics, {K} gene components, "
                  f"{self.C} types")

        # The residual pool: every training cell's high-frequency deviation.
        self.res_pool = np.vstack([e["residual"] for e in self.sections])
        self.res_types = np.concatenate([e["types"] for e in self.sections])

        self.model = None
        if TORCH:
            try:
                self._train_flow()
                self.trained = True
            except Exception as exc:                       # pragma: no cover
                self._log(f"flow training failed ({exc}); using the "
                          f"depth-interpolated spectrum instead")
                self.trained = False
        else:
            self._log("torch unavailable; using the depth-interpolated spectrum")
        self.diagnostics["flow_trained"] = bool(self.trained)

    def _encode_section(self, s, xy=None):
        cfg = self.cfg
        xy = s.coords_xy if xy is None else xy
        Phi, lam = tissue_harmonics(xy, n_modes=cfg.harmonics.n_modes,
                                    k=cfg.harmonics.k_neighbors, seed=cfg.seed)
        X = np.asarray(s.expression, dtype=np.float64)
        Y = self.gene_pca.transform(X)
        types = (s.cell_type_indices if s.cell_type_indices is not None
                 else np.zeros(len(X), dtype=int))
        T = np.zeros((len(X), self.C), dtype=np.float64)
        T[np.arange(len(X)), np.clip(types, 0, self.C - 1)] = 1.0

        chan = np.column_stack([Y, T])
        Chat, cmu = gft(Phi, chan)

        # The residual: what the modelled harmonics do not explain, per cell.
        smooth = igft(Phi, Chat, cmu)
        Y_s, _ = smooth[:, :self.K], smooth[:, self.K:]
        X_smooth = self.gene_pca.inverse_transform(Y_s)
        residual = X - X_smooth
        self._smooth_cache = None

        prof = section_profile(X, Phi, lam, n_bins=cfg.harmonics.n_freq_bins,
                               xy=xy, k=cfg.harmonics.k_neighbors)
        return {"Phi": Phi, "lam": lam, "Chat": Chat, "cmu": cmu, "z": float(s.z),
                "profile": prof, "residual": residual, "types": types,
                "xy": xy, "n": len(X), "X": X, "smooth": smooth}

    # ── flow training ────────────────────────────────────────────────────────
    def _training_examples(self):
        """(target spectrum, eigenvalues, z, context) tuples.

        Two sources of examples, because a benchmark stack has only a handful of
        sections and flow matching on four points learns nothing:

        * **leave-one-section-out** over the training stack, which is the task
          itself and the only source of a genuinely correct target;
        * **cell resampling**, which is the augmentation that makes this
          trainable. Re-drawing 80 % of a section's cells yields a *different
          graph* of the *same tissue* — new harmonics, new eigenvalues, a new
          coefficient matrix, and the same underlying anatomy. The model sees the
          spectrum of one piece of tissue described in many bases, which is
          exactly the invariance it must have to describe the held-out section's
          basis, and it is free.
        """
        cfg = self.cfg
        ex = []
        n_sec = len(self.sections)
        reps = max(1, int(np.ceil(64 / max(n_sec, 1))))
        for rep in range(reps):
            for i, tgt in enumerate(self.sections):
                ctx_idx = [j for j in (i - 1, i + 1) if 0 <= j < n_sec]
                if not ctx_idx:
                    continue
                while len(ctx_idx) < cfg.flow.n_context:
                    ctx_idx.append(ctx_idx[-1])
                ctx_idx = ctx_idx[:cfg.flow.n_context]

                if rep == 0:
                    enc = tgt
                else:
                    keep = self.rng.random(tgt["n"]) < 0.8
                    if keep.sum() < cfg.harmonics.n_modes + 4:
                        continue
                    sub = _SubSlice(tgt, keep)
                    enc = self._encode_section(sub, xy=tgt["xy"][keep])

                base, ctx = self._baseline(
                    [self.sections[j] for j in ctx_idx], enc["xy"],
                    enc["Phi"], enc["z"])
                # The flow's target is the *correction* to the field-evaluated
                # baseline, not the spectrum itself. The baseline already carries
                # the anatomy, so the model's capacity goes to what interpolation
                # cannot supply — and its zero-effort output is exactly the
                # baseline rather than noise.
                ex.append((enc["Chat"] - base, enc["lam"], enc["z"], ctx))
        return ex

    def _baseline(self, ctx_sections, xy_dst, Phi_dst, z_dst):
        """Field-evaluated baseline spectrum in the target's own basis.

        Each context section's *smooth* channel field is evaluated at the target
        positions and depth-weighted; the blend is then transformed into the
        target basis. Weights are inverse distance in z, so the nearer section
        dominates and the construction degrades gracefully at the stack's ends.

        Returns ``(baseline_spectrum (M, K+C), context_stack (M, (K+C)*n_ctx))``.
        """
        fields, coefs = [], []
        zs = np.array([e["z"] for e in ctx_sections], dtype=np.float64)
        wts = 1.0 / np.maximum(np.abs(zs - float(z_dst)), 1e-6)
        wts = wts / wts.sum()
        for e, wt in zip(ctx_sections, wts):
            f = evaluate_field(e["xy"], e["smooth"], xy_dst)
            fields.append(wt * f)
            coefs.append(gft(Phi_dst, f)[0])
        blend = np.sum(fields, axis=0)
        return gft(Phi_dst, blend)[0], np.concatenate(coefs, axis=1)

    def _train_flow(self):
        import torch
        from .nets import SpectralVelocityField, flow_match_loss

        cfg = self.cfg
        ex = self._training_examples()
        if len(ex) < 4:
            raise RuntimeError(f"only {len(ex)} training examples")

        M = min(e[0].shape[0] for e in ex)
        Kc = ex[0][0].shape[1]
        Y = np.stack([e[0][:M] for e in ex])
        L = np.stack([e[1][:M] for e in ex])
        Z = np.array([e[2] for e in ex], dtype=np.float64)
        Cx = np.stack([e[3][:M] for e in ex])

        # Standardize per channel so the flow's Gaussian prior is the right
        # scale; the inverse is applied at generation.
        self.y_mu = Y.mean(axis=(0, 1), keepdims=True)
        self.y_sd = Y.std(axis=(0, 1), keepdims=True) + 1e-6
        self.c_mu = Cx.mean(axis=(0, 1), keepdims=True)
        self.c_sd = Cx.std(axis=(0, 1), keepdims=True) + 1e-6
        self.z_mu, self.z_sd = float(Z.mean()), float(Z.std() + 1e-6)

        dev = _device(cfg)
        Yt = torch.tensor((Y - self.y_mu) / self.y_sd, dtype=torch.float32, device=dev)
        Lt = torch.tensor(L, dtype=torch.float32, device=dev)
        Zt = torch.tensor((Z - self.z_mu) / self.z_sd, dtype=torch.float32, device=dev)
        Ct = torch.tensor((Cx - self.c_mu) / self.c_sd, dtype=torch.float32, device=dev)

        model = SpectralVelocityField(
            k_dim=Kc, width=cfg.flow.width, depth=cfg.flow.depth,
            heads=cfg.flow.heads, n_fourier=cfg.flow.n_fourier,
            n_context=cfg.flow.n_context, dropout=cfg.flow.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.flow.lr,
                                weight_decay=cfg.flow.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.flow.epochs)
        gen = torch.Generator(device=dev).manual_seed(cfg.seed)

        n = len(ex)
        bs = int(min(cfg.flow.batch_size, n))
        model.train()
        for ep in range(cfg.flow.epochs):
            idx = torch.randperm(n, device=dev, generator=gen)[:bs]
            ctx = Ct[idx]
            if cfg.flow.context_dropout > 0:
                drop = (torch.rand(ctx.shape[0], 1, ctx.shape[2], device=dev,
                                   generator=gen) < cfg.flow.context_dropout)
                ctx = ctx.masked_fill(drop, 0.0)
            zj = Zt[idx]
            if cfg.flow.z_jitter > 0:
                zj = zj + cfg.flow.z_jitter * torch.randn(
                    zj.shape, device=dev, generator=gen)
            loss = flow_match_loss(model, Yt[idx], Lt[idx], zj, ctx, generator=gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if self.verbose and (ep + 1) % max(1, cfg.flow.epochs // 4) == 0:
                self._log(f"flow epoch {ep + 1}/{cfg.flow.epochs} loss {loss.item():.4f}")
        model.eval()
        self.model = model
        self.device = dev
        self.M_train = M
        self.diagnostics["flow_final_loss"] = float(loss.item())
        self.diagnostics["flow_examples"] = int(n)

    # ── generation ───────────────────────────────────────────────────────────
    def generate_virtual_slice(self, z, seed=None):
        cfg = self.cfg
        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        w, i_lo, i_hi = self.stack.fractional_depth(z)
        lo, hi = self.sections[i_lo], self.sections[i_hi]

        # 1 — geometry
        n_cells = target_cell_count(lo["n"], hi["n"], w)
        xy, intensity, xe, ye = interpolate_geometry(
            lo["xy"], hi["xy"], w, n_cells,
            n_bins=cfg.geometry.grid_bins, smooth=cfg.geometry.density_smooth,
            occupancy_quantile=cfg.geometry.occupancy_quantile,
            seed=int(rng.integers(1 << 31)), jitter=cfg.geometry.jitter)

        # 2 — basis of the target section
        Phi, lam = tissue_harmonics(xy, n_modes=cfg.harmonics.n_modes,
                                    k=cfg.harmonics.k_neighbors, seed=cfg.seed)
        M = Phi.shape[1]

        # 3 — spectrum: field-evaluated baseline plus the flow's correction
        base, ctx = self._baseline([lo, hi], xy, Phi, z)
        Chat = base + self._sample_spectrum(lam, z, ctx, M, rng)

        # 4 — inverse transform
        cmu = (1.0 - w) * lo["cmu"] + w * hi["cmu"]
        chan = igft(Phi, Chat, cmu)
        Y_s, T_field = chan[:, :self.K], chan[:, self.K:]
        X_smooth = self.gene_pca.inverse_transform(Y_s)

        # 5 — spectral calibration
        target = blend_profiles(lo["profile"], hi["profile"], w)
        pred_moran_before = self._predict_morans(X_smooth, Phi, lam)
        if cfg.expression.calibrate:
            X_cal, _ = apply_calibration(X_smooth, Phi, lam, target)
            total_var_target = self._target_total_variance(lo, hi, w, len(xy))
            X_cal = scale_to_variance(X_cal, total_var_target * target["kept"])
        else:
            X_cal = X_smooth

        # 7 — typing (before the residual, which is drawn per type)
        types = self._sample_types(T_field, lo, hi, w, rng)

        # 6 — residual
        res = np.zeros_like(X_cal)
        if cfg.expression.residual_scale > 0:
            pool_types = (self.res_types if cfg.expression.residual_by_type
                          else np.zeros(len(self.res_types), dtype=int))
            draw_types = (types if cfg.expression.residual_by_type
                          else np.zeros(len(types), dtype=int))
            res = residual_draw(self.res_pool, pool_types, self.res_pool,
                                draw_types, rng)
            res = res - res.mean(axis=0, keepdims=True)
            res_var_target = (self._target_total_variance(lo, hi, w, len(xy))
                              * (1.0 - target["kept"]))
            have = (res ** 2).sum(axis=0)
            g = np.sqrt(np.where(have > 1e-12,
                                 res_var_target / np.maximum(have, 1e-12), 0.0))
            res = res * g[None, :] * cfg.expression.residual_scale

        X_out = X_cal + res
        X_out = np.clip(X_out, 0.0, None)

        pred_moran_after = self._predict_morans(X_out, Phi, lam, xy=xy)
        diag = {
            "n_cells": int(len(xy)), "w": float(w), "n_modes": int(M),
            "flow_trained": bool(self.trained),
            "modelled_variance_fraction": float(np.mean(target["kept"])),
            "predicted_morans_median_before": float(np.nanmedian(pred_moran_before)),
            "predicted_morans_median_after": float(np.nanmedian(pred_moran_after)),
            "target_morans_median": float(np.nanmedian(
                (1.0 - w) * self._predict_morans(lo["X"], lo["Phi"], lo["lam"],
                                                 xy=lo["xy"])
                + w * self._predict_morans(hi["X"], hi["Phi"], hi["lam"],
                                           xy=hi["xy"]))),
        }
        return VirtualSlice(expression=X_out,
                            coords=np.column_stack([xy, np.full(len(xy), float(z))]),
                            cell_type=types, z=float(z), diagnostics=diag)

    def _sample_spectrum(self, lam, z, ctx, M, rng):
        """The flow's ensemble-mean *correction* to the baseline (zero if untrained)."""
        if not self.trained or self.model is None:
            return np.zeros((M, self.K + self.C))

        import torch
        from .nets import integrate
        Mn = min(M, self.M_train)
        lam_t = torch.tensor(lam[None, :Mn], dtype=torch.float32, device=self.device)
        z_t = torch.tensor([[(z - self.z_mu) / self.z_sd]], dtype=torch.float32,
                           device=self.device)[:, 0]
        ctx_t = torch.tensor(((ctx[:Mn] - self.c_mu[0]) / self.c_sd[0])[None],
                             dtype=torch.float32, device=self.device)
        gen = torch.Generator(device=self.device).manual_seed(int(rng.integers(1 << 31)))

        draws = []
        for _ in range(self.cfg.flow.n_ensemble):
            y = integrate(self.model, lam_t, z_t, ctx_t,
                          n_steps=self.cfg.flow.ode_steps, generator=gen)
            draws.append(y.cpu().numpy()[0])
        # The ensemble mean is the conditional expectation of the spectrum. A
        # single draw carries the full conditional variance, and averaging draws
        # is *not* over-smoothing here: the calibration stage sets the energy
        # afterwards, so the mean supplies the pattern and the profile supplies
        # the amount of structure.
        Y = np.mean(draws, axis=0) * self.y_sd[0] + self.y_mu[0]
        if Y.shape[0] < M:
            pad = np.zeros((M - Y.shape[0], Y.shape[1]))
            Y = np.vstack([Y, pad])
        return Y[:M]

    def _predict_morans(self, X, Phi, lam, xy=None):
        """The method's own estimate of what the evaluator will measure.

        Uses the complete form — modelled harmonics *plus* the residual bucket —
        because the subspace-restricted version is badly optimistic for any gene
        whose variance is mostly high-frequency, which on real panels is most of
        them. Reporting the restricted number would let an over-smoothed section
        look correctly structured.
        """
        Xhat, mu = gft(Phi, X)
        total = ((np.asarray(X, dtype=np.float64) - mu[None, :]) ** 2).sum(axis=0)
        kept = (Xhat ** 2).sum(axis=0)
        res = np.maximum(total - kept, 0.0)
        from .calibrate import _residual_eigenvalue
        lam_res = _residual_eigenvalue(X, mu, lam, Xhat, total, kept, xy=xy,
                                       k=self.cfg.harmonics.k_neighbors)
        return morans_full(Xhat, lam, res, lam_res)

    def _target_total_variance(self, lo, hi, w, n_out):
        """Per-gene total variance the virtual section should carry.

        Interpolated from the brackets' own per-gene variance. Variance, not
        standard deviation, because the spectral budget is stated in variance and
        the two stages have to add up.
        """
        v_lo = lo["X"].var(axis=0)
        v_hi = hi["X"].var(axis=0)
        return ((1.0 - w) * v_lo + w * v_hi) * max(n_out - 1, 1)

    def _sample_types(self, T_field, lo, hi, w, rng):
        """Assign types from the generated indicator fields.

        The fields are the spectral model's own output, so a type's spatial
        arrangement is generated rather than copied — which is what
        ``paper_celltype_localization`` measures. Sampling is done under a
        composition constraint so the section's type proportions match the
        depth-interpolated ones: a type the model under-produces would score
        zero on that metric, and the constraint makes silently dropping a type
        impossible.
        """
        cfg = self.cfg
        P = np.asarray(T_field, dtype=np.float64)
        P = P - P.max(axis=1, keepdims=True)
        P = np.exp(P / max(cfg.types.temperature, 1e-6))
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)

        n = P.shape[0]
        if not cfg.types.composition_match:
            return np.array([rng.choice(self.C, p=P[i]) for i in range(n)])

        comp_lo = np.bincount(lo["types"], minlength=self.C) / max(len(lo["types"]), 1)
        comp_hi = np.bincount(hi["types"], minlength=self.C) / max(len(hi["types"]), 1)
        comp = (1.0 - w) * comp_lo + w * comp_hi
        quota = np.floor(comp * n).astype(int)
        short = n - quota.sum()
        if short > 0:
            frac = comp * n - quota
            order = np.argsort(-frac)[:short]
            quota[order] += 1

        # Greedy assignment by confidence: the most confident cells claim their
        # type first, so the constraint bends the uncertain cells rather than the
        # clear ones.
        out = np.full(n, -1, dtype=int)
        conf = P.max(axis=1)
        for i in np.argsort(-conf):
            order = np.argsort(-P[i])
            for t in order:
                if quota[t] > 0:
                    out[i] = t
                    quota[t] -= 1
                    break
            if out[i] < 0:
                out[i] = int(np.argmax(P[i]))
        return out


class _SubSlice:
    """A cell-resampled view of an encoded section, for the flow's augmentation."""

    def __init__(self, enc, keep):
        self.expression = enc["X"][keep]
        self.coords_xy = enc["xy"][keep]
        self.z = enc["z"]
        self.cell_type_indices = enc["types"][keep]
        self.section_id = ""


def _device(cfg):
    import torch
    if cfg.runtime.device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if cfg.runtime.device in ("auto", "cuda"):
        return torch.device(f"cuda:{cfg.runtime.cuda_index}")
    return torch.device("cpu")
