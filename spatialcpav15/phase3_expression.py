"""Phase 3 — the expression model.

**3.1** An expression VAE trained on *all cells pooled* (this stage ignores z),
with a proper count likelihood (NB, or ZINB / Gaussian when asked), library size
as an **explicit scalar** kept out of the profile, and a deliberately low KL
weight — it is a likelihood and a coordinate system, not the generator.  Posterior
collapse is checked, not assumed.

**3.2** A conditional latent diffusion in the *frozen* VAE latent.  The denoiser
is conditioned on the continuous hierarchical type embedding from Phase 1.2 (not
a bare label — that is the bottleneck that makes a generator emit
``type-mean + noise``), normalized position, ``dz`` to the nearest real slice,
and cross-attention over the encoded latents of nearby cells in the **real**
bracketing slices.  Training uses slice dropout of consecutive runs and
randomized ``dz`` so the model learns to widen its distribution as context
recedes, and the whole gap block is denoised jointly with attention across cells
and across z — never independently per slice.

**3.3** The gate: with ground-truth layouts on an internally held-out *training*
slice, the model must beat nearest-slice copy, linear interpolation between
brackets, and cell-type-mean assignment.  Failing the type-mean bar means the
model is a label propagator; the score is reported either way, and it selects the
decoder readout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

from . import runtime
from .config import ExprDiffusionConfig, ExprVAEConfig, RuntimeConfig


def _looks_like_counts(X: np.ndarray) -> bool:
    sub = X[: min(2000, X.shape[0])]
    return bool(np.all(sub >= -1e-6) and np.allclose(sub, np.round(sub), atol=1e-4))


# ── Phase 3.1 ─────────────────────────────────────────────────────────────────

@dataclass
class LatentSpace:
    """The frozen VAE and the cached latents of every real (training) cell."""

    vae: object
    device: object
    latents: np.ndarray                 # (n_cells, L)
    log_lib: np.ndarray                 # (n_cells,)
    likelihood: str
    lib_scale: float
    diagnostics: Dict = field(default_factory=dict)

    @property
    def latent_dim(self) -> int:
        return int(self.latents.shape[1])


def train_expression_vae(counts: np.ndarray, log_expr: np.ndarray,
                         cfg: ExprVAEConfig, seed: int = 0,
                         verbose: bool = True,
                         rt: RuntimeConfig = None) -> LatentSpace:
    """Phase 3.1 — fit the VAE on all cells pooled and cache their latents."""
    import torch

    from .nets import ExpressionVAE

    torch.manual_seed(seed)
    rt = rt or RuntimeConfig()
    device = runtime.configure(rt)
    n, G = log_expr.shape

    likelihood = cfg.likelihood
    if likelihood == "auto":
        likelihood = "nb" if _looks_like_counts(counts) else "gaussian"

    lib = np.asarray(counts, dtype=np.float64).sum(axis=1)
    lib_scale = 1.0
    if likelihood == "gaussian":
        # A Gaussian head models the (already normalized) profile directly; the
        # "library" is then just a scale we keep out of the profile.
        lib = np.maximum(lib, 1e-6)
    lib = np.maximum(lib, 1.0)
    log_lib = np.log(lib)

    X_log = torch.from_numpy(np.asarray(log_expr, dtype=np.float32)).to(device)
    X_cnt = torch.from_numpy(np.asarray(counts, dtype=np.float32)).to(device)
    LL = torch.from_numpy(log_lib.astype(np.float32)).to(device)

    vae = ExpressionVAE(G, latent_dim=cfg.latent_dim, hidden=cfg.hidden,
                        n_layers=cfg.n_layers, likelihood=likelihood,
                        per_gene_variance=cfg.gaussian_per_gene_variance).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.lr, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)

    n_steps = max(int(cfg.epochs),
                  int(cfg.min_passes * np.ceil(n / max(cfg.batch_size, 1))))
    vae.train()
    for ep in range(n_steps):
        idx = torch.randint(0, n, (min(cfg.batch_size, n),), generator=g).to(device)
        loss, rec, kl, _ = vae(X_log[idx], X_cnt[idx], LL[idx], cfg.kl_weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
        opt.step()
        if verbose and (ep + 1) % max(n_steps // 4, 1) == 0:
            print(f"      [phase3.1] step {ep + 1}/{n_steps} "
                  f"rec={float(rec):.3f} kl={float(kl):.3f}")
    vae.eval()

    with torch.no_grad():
        mu, lv = vae.encode(X_log, LL)
        Z = mu.cpu().numpy()

    if rt.release_cache_between_stages:
        runtime.release(device)

    active = float(np.mean(Z.std(axis=0) > 0.1 * Z.std()))
    diag = {"likelihood": likelihood, "latent_dim": cfg.latent_dim,
            "n_steps": n_steps,
            "active_units": active,
            "posterior_collapse": bool(active < cfg.min_active_units),
            "latent_std_median": float(np.median(Z.std(axis=0)))}
    if verbose:
        flag = " (POSTERIOR COLLAPSE)" if diag["posterior_collapse"] else ""
        print(f"      [phase3.1] likelihood={likelihood} "
              f"active_units={active:.2f}{flag}")
    return LatentSpace(vae=vae, device=device, latents=Z, log_lib=log_lib,
                       likelihood=likelihood, lib_scale=lib_scale, diagnostics=diag)


def decode_latents(space: LatentSpace, Z: np.ndarray, log_lib: np.ndarray,
                   readout: str, rng: np.random.Generator) -> np.ndarray:
    """Latents -> expression in the caller's units, through the frozen decoder.

    ``mean`` returns the fitted rate (a denoised profile); ``sample`` draws from
    the fitted NB, which restores realistic per-gene dispersion.  Which one is
    used is decided by the Phase 3.3 gate, not by hand.  A Gaussian head models
    the log1p scale, so its output is mapped back to the data scale and the
    ``sample`` readout does not apply.
    """
    import torch

    with torch.no_grad():
        z = torch.from_numpy(np.asarray(Z, dtype=np.float32)).to(space.device)
        ll = torch.from_numpy(np.asarray(log_lib, dtype=np.float32)).to(space.device)
        mu_t, _, _ = space.vae.decode(z, ll)
        mu = np.clip(mu_t.cpu().numpy(), 0.0, None)
        if space.vae.is_continuous:
            return mu          # continuous data has no count draw to take
        theta = space.vae.log_theta.exp().clamp(1e-4, 1e6).cpu().numpy()
    if readout == "mean":
        return np.clip(mu, 0.0, None)
    p = theta / (theta + np.maximum(mu, 1e-8))
    lam = rng.gamma(shape=np.broadcast_to(theta, mu.shape),
                    scale=np.maximum((1 - p) / np.maximum(p, 1e-8), 1e-12))
    return rng.poisson(np.clip(lam, 0, 1e9)).astype(np.float32)


# ── Phase 3.2 — conditional latent diffusion ─────────────────────────────────

@dataclass
class CellIndex:
    """Flat view of every real cell, for context lookup and training."""

    u: np.ndarray            # (n, 2) normalized in-plane
    z: np.ndarray            # (n,) raw z
    z_norm: np.ndarray       # (n,) normalized z
    types: np.ndarray        # (n,)
    section: np.ndarray      # (n,) int index into the stack
    latents: np.ndarray      # (n, L)
    log_lib: np.ndarray      # (n,)


def build_cell_index(stack, frame, latents, log_lib) -> CellIndex:
    us, zs, zn, ts, ss = [], [], [], [], []
    for si, s in enumerate(stack.slices):
        u, w = frame.to_normalized(s.coords_xy, s.z_values, s.section_id)
        us.append(u)
        zs.append(s.z_values)
        zn.append(w)
        ts.append(s.cell_type_indices if s.cell_type_indices is not None
                  else np.zeros(s.n_spots, dtype=np.int64))
        ss.append(np.full(s.n_spots, si, dtype=np.int64))
    return CellIndex(u=np.vstack(us), z=np.concatenate(zs),
                     z_norm=np.concatenate(zn), types=np.concatenate(ts),
                     section=np.concatenate(ss), latents=latents, log_lib=log_lib)


def gather_context(index: CellIndex, u_query: np.ndarray, z_query: float,
                   allowed_sections: np.ndarray, k: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest real cells in the *allowed* (bracketing) sections, per query cell.

    Returns ``(latents, positions, types, dz)`` shaped ``(N, K, ...)``.  The
    context is always drawn from real observed slices — never from generated
    material — which is what makes it a grounded conditioning signal.
    """
    pool = np.where(np.isin(index.section, allowed_sections))[0]
    if pool.size == 0 or k <= 0:
        n = u_query.shape[0]
        return (np.zeros((n, 0, index.latents.shape[1])), np.zeros((n, 0, 3)),
                np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0)))
    kk = int(min(k, pool.size))
    _, nn = cKDTree(index.u[pool]).query(u_query, k=kk)
    nn = np.atleast_2d(nn)
    if nn.shape[0] != u_query.shape[0]:
        nn = nn.T
    gid = pool[nn]
    lat = index.latents[gid]
    pos = np.concatenate([index.u[gid],
                          index.z_norm[gid][..., None]], axis=-1)
    typ = index.types[gid]
    dz = np.abs(index.z[gid] - z_query)
    return lat, pos, typ, dz


@dataclass
class ExpressionDiffusion:
    """The trained Phase 3.2 denoiser (or the disabled stand-in)."""

    net: object = None
    schedule: object = None
    device: object = None
    cfg: ExprDiffusionConfig = None
    trained: bool = False
    readout: str = "mean"
    lat_mean: Optional[np.ndarray] = None
    lat_scale: Optional[np.ndarray] = None
    diagnostics: Dict = field(default_factory=dict)

    def standardize(self, Z: np.ndarray) -> np.ndarray:
        return (Z - self.lat_mean) / self.lat_scale

    def unstandardize(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.lat_scale + self.lat_mean


def train_expression_diffusion(index: CellIndex, type_rep, stack,
                               space: LatentSpace, cfg: ExprDiffusionConfig,
                               dz_unit: float, seed: int = 0,
                               verbose: bool = True,
                               rt: RuntimeConfig = None) -> ExpressionDiffusion:
    """Phase 3.2 — train the conditional latent diffusion."""
    model = ExpressionDiffusion(cfg=cfg)
    n_sections = stack.n_slices
    if not cfg.enabled or n_sections < 3 or index.u.shape[0] < 64:
        model.diagnostics = {"trained": False,
                             "reason": f"{n_sections} sections / "
                                       f"{index.u.shape[0]} cells is too little"}
        return model

    import torch

    from .nets import ContextDenoiser, CosineSchedule

    torch.manual_seed(seed)
    rt = rt or RuntimeConfig()
    device = runtime.configure(rt)
    L = space.latent_dim
    hier = type_rep.hierarchical(index.types).astype(np.float32)
    T = hier.shape[1]

    net = ContextDenoiser(L, T, hidden=cfg.hidden, heads=cfg.n_heads,
                          n_blocks=cfg.n_blocks, seed=seed).to(device)
    sched = CosineSchedule(cfg.n_timesteps, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)

    # Per-dimension standardization.  The diffusion prior is N(0, I): if the
    # latent cloud is not centred and unit-scaled per dimension, a sample started
    # from pure noise lands in the wrong region of the latent space and the frozen
    # decoder turns it into nonsense.
    lat_mean = index.latents.mean(axis=0)
    lat_scale = np.maximum(index.latents.std(axis=0), 1e-3)
    model.lat_mean, model.lat_scale = lat_mean, lat_scale
    lat_n = (index.latents - lat_mean) / lat_scale
    hier_t = torch.from_numpy(hier).to(device)

    def make_batch():
        """One training step: a target block + slice-dropped real context."""
        target_sec = int(rng.integers(0, n_sections))
        # Slice dropout: hold out a *consecutive run* including the target, so
        # the model must generate with the context receded by a variable dz.
        run = 1
        if rng.random() < cfg.slice_dropout:
            run = int(rng.integers(1, cfg.max_dropout_run + 1))
        lo = max(target_sec - int(rng.integers(0, run)), 0)
        held = set(range(lo, min(lo + run, n_sections)))
        allowed = np.array([s for s in range(n_sections) if s not in held])
        if allowed.size == 0:
            allowed = np.array([(target_sec + 1) % n_sections])

        tm = np.where(index.section == target_sec)[0]
        if tm.size < 8:
            return None
        take = rng.choice(tm, size=min(cfg.block_size, tm.size), replace=False)
        z_q = float(np.median(index.z[take]))
        lat, pos, typ, dz = gather_context(index, index.u[take], z_q,
                                           allowed, cfg.n_context)
        near = float(np.min(np.abs(
            np.array([np.median(index.z[index.section == s]) for s in allowed]) - z_q)))
        tfrac = np.full(take.size, 0.5)
        return dict(
            x0=torch.from_numpy(lat_n[take][None].astype(np.float32)).to(device),
            type_emb=hier_t[take][None],
            pos=torch.from_numpy(np.column_stack(
                [index.u[take], index.z_norm[take]])[None].astype(np.float32)).to(device),
            dz=torch.full((1, take.size), near / dz_unit, device=device),
            log_lib=torch.from_numpy(
                (index.log_lib[take] - index.log_lib.mean())[None].astype(np.float32)
            ).to(device),
            tfrac=torch.from_numpy(tfrac[None].astype(np.float32)).to(device),
            ctx_lat=torch.from_numpy(
                ((lat - lat_mean) / lat_scale)[None].astype(np.float32)).to(device),
            ctx_pos=torch.from_numpy(pos[None].astype(np.float32)).to(device),
            ctx_type=torch.from_numpy(type_rep.hierarchical(typ.ravel()).reshape(
                1, typ.shape[0], typ.shape[1], -1).astype(np.float32)).to(device),
            ctx_dz=torch.from_numpy((dz / dz_unit)[None].astype(np.float32)).to(device),
        )

    net.train()
    losses = []
    for ep in range(cfg.epochs):
        b = make_batch()
        if b is None:
            continue
        x0 = b["x0"]
        step = torch.randint(0, cfg.n_timesteps, (1,), device=device)
        noise = torch.randn_like(x0)
        xt = sched.q_sample(x0, step, noise)
        pred = net(xt, step, b["type_emb"], b["pos"], b["dz"], b["log_lib"],
                   b["tfrac"], b["ctx_lat"], b["ctx_pos"], b["ctx_type"], b["ctx_dz"])
        loss = ((pred - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if verbose and (ep + 1) % max(cfg.epochs // 4, 1) == 0:
            print(f"      [phase3.2] epoch {ep + 1}/{cfg.epochs} "
                  f"loss={np.mean(losses[-30:]):.4f}")
    net.eval()

    model.net = net
    model.schedule = sched
    model.device = device
    model.trained = True
    if rt.release_cache_between_stages:
        runtime.release(device)
    model.diagnostics = {"trained": True, "final_loss": float(np.mean(losses[-30:])),
                         "latent_scale_median": float(np.median(lat_scale)),
                         "type_dim": T}
    return model


def sample_expression_latents(model: ExpressionDiffusion, index: CellIndex,
                              type_rep, u: np.ndarray, z_norm: np.ndarray,
                              types: np.ndarray, log_lib: np.ndarray,
                              allowed_sections: np.ndarray, z_query: float,
                              dz_unit: float, lib_mean: float,
                              seed: int = 0) -> np.ndarray:
    """Generate the whole gap block jointly (Phase 3.2 / Phase 4.5).

    Every target cell is denoised in one pass with attention across the block —
    including across z when the block spans several depths — so there is no
    per-slice independence and therefore no z-flicker.
    """
    import torch

    n = u.shape[0]
    lat, pos, typ, dz = gather_context(index, u, z_query, allowed_sections,
                                       model.cfg.n_context)
    near = dz.min() if dz.size else 0.0
    device = model.device
    hier = type_rep.hierarchical(types).astype(np.float32)
    ctx_type = type_rep.hierarchical(typ.ravel()).reshape(
        1, typ.shape[0], typ.shape[1], -1).astype(np.float32)
    args = dict(
        type_emb=torch.from_numpy(hier[None]).to(device),
        pos=torch.from_numpy(np.column_stack([u, z_norm])[None].astype(np.float32)).to(device),
        dz=torch.full((1, n), float(near / dz_unit), device=device),
        log_lib=torch.from_numpy((log_lib - lib_mean)[None].astype(np.float32)).to(device),
        tfrac=torch.full((1, n), 0.5, device=device),
        ctx_lat=torch.from_numpy(
            model.standardize(lat)[None].astype(np.float32)).to(device),
        ctx_pos=torch.from_numpy(pos[None].astype(np.float32)).to(device),
        ctx_type=torch.from_numpy(ctx_type).to(device),
        ctx_dz=torch.from_numpy((dz / dz_unit)[None].astype(np.float32)).to(device),
    )

    def eps(x, step):
        return model.net(x, step[:1], **args)

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    L = model.net.out.out_features
    noise = torch.randn((1, n, L), generator=g).to(device)
    with torch.no_grad():
        z = model.schedule.ddim(eps, (1, n, L), device, model.cfg.ddim_steps,
                                noise=noise)
    return model.unstandardize(z[0].cpu().numpy())


# ── Phase 3.3 — the gate ─────────────────────────────────────────────────────

def _coexpr(X: np.ndarray) -> np.ndarray:
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    return C[iu]


def _agree(a: np.ndarray, b: np.ndarray) -> float:
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < 3 or a[v].std() == 0 or b[v].std() == 0:
        return float("nan")
    return float(pearsonr(a[v], b[v])[0])


def _within_type_variation(X: np.ndarray, types: np.ndarray) -> float:
    """Mean within-type variance — the quantity a label propagator destroys."""
    vs = []
    for c in np.unique(types):
        m = types == c
        if m.sum() >= 3:
            vs.append(X[m].var(axis=0).mean())
    return float(np.mean(vs)) if vs else float("nan")


def expression_gate(model: ExpressionDiffusion, space: LatentSpace,
                    index: CellIndex, type_rep, stack, dz_unit: float,
                    seed: int = 0, verbose: bool = True) -> Dict:
    """Phase 3.3 — beat nearest-copy, bracket-linear and type-mean, on real layouts.

    The evaluation slice is an internally held-out **training** slice, so this is
    leakage-safe.  The layout (positions and types) is the real one, which is the
    proposal's requirement: this measures the expression model alone, with the
    structure stage factored out.
    """
    n_sec = stack.n_slices
    if not model.trained or n_sec < 4:
        return {"ran": False, "reason": "not enough sections for an internal gate"}

    rng = np.random.default_rng(seed)
    target_sec = n_sec // 2
    allowed = np.array([s for s in range(n_sec) if s != target_sec])
    tm = np.where(index.section == target_sec)[0]
    if tm.size < 32:
        return {"ran": False, "reason": "held-out training slice too small"}
    take = tm if tm.size <= 512 else rng.choice(tm, 512, replace=False)

    real_log = _log_norm(space, take)
    z_q = float(np.median(index.z[take]))
    lib_mean = float(index.log_lib.mean())

    scores = {}
    # candidate: the pipeline itself, both readouts
    Zg = sample_expression_latents(model, index, type_rep, index.u[take],
                                   index.z_norm[take], index.types[take],
                                   index.log_lib[take], allowed, z_q, dz_unit,
                                   lib_mean, seed=seed)
    for readout in ("mean", "sample"):
        Xg = decode_latents(space, Zg, index.log_lib[take], readout, rng)
        scores[f"v15_{readout}"] = _score_expr(Xg, real_log, index.u[take],
                                               index.types[take])

    # baselines the proposal names explicitly
    near_sec = min(allowed, key=lambda s: abs(
        np.median(index.z[index.section == s]) - z_q))
    scores["nearest_slice_copy"] = _score_expr(
        _copy_from(index, space, take, np.array([near_sec])), real_log,
        index.u[take], index.types[take])
    a = allowed[allowed < target_sec]
    b = allowed[allowed > target_sec]
    if a.size and b.size:
        Xa = _copy_from(index, space, take, np.array([a.max()]))
        Xb = _copy_from(index, space, take, np.array([b.min()]))
        scores["bracket_linear"] = _score_expr(0.5 * (Xa + Xb), real_log,
                                               index.u[take], index.types[take])
    scores["celltype_mean"] = _score_expr(
        _type_mean(index, space, take), real_log, index.u[take], index.types[take])

    ours = max(("v15_mean", "v15_sample"), key=lambda k: scores[k]["composite"])
    baselines = {k: v for k, v in scores.items() if not k.startswith("v15_")}
    best_base = max(baselines, key=lambda k: baselines[k]["composite"])
    out = {
        "ran": True,
        "scores": scores,
        "chosen_readout": ours.split("_", 1)[1],
        "best_baseline": best_base,
        "beats_type_mean": bool(scores[ours]["composite"]
                                > scores["celltype_mean"]["composite"]),
        "beats_best_baseline": bool(scores[ours]["composite"]
                                    > baselines[best_base]["composite"]),
    }
    if verbose:
        print(f"      [phase3 gate] readout={out['chosen_readout']} "
              f"composite={scores[ours]['composite']:.3f} vs "
              f"type_mean={scores['celltype_mean']['composite']:.3f}, "
              f"{best_base}={baselines[best_base]['composite']:.3f} "
              f"| beats_type_mean={out['beats_type_mean']}")
    return out


def _log_norm(space: LatentSpace, idx: np.ndarray) -> np.ndarray:
    """VAE reconstruction of real cells, on the comparison (log-normalized) scale."""
    import torch
    with torch.no_grad():
        z = torch.from_numpy(space.latents[idx].astype(np.float32)).to(space.device)
        ll = torch.from_numpy(space.log_lib[idx].astype(np.float32)).to(space.device)
        mu, _, _ = space.vae.decode(z, ll)
        X = space.vae.to_data_scale(mu).cpu().numpy()
    X = np.clip(X, 0.0, None)
    return np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1e-9) * 1e4)


def _copy_from(index: CellIndex, space: LatentSpace, take: np.ndarray,
               sections: np.ndarray) -> np.ndarray:
    pool = np.where(np.isin(index.section, sections))[0]
    _, nn = cKDTree(index.u[pool]).query(index.u[take], k=1)
    return _log_norm(space, pool[np.atleast_1d(nn)])


def _type_mean(index: CellIndex, space: LatentSpace, take: np.ndarray) -> np.ndarray:
    """Cell-type-mean assignment — the bar a label propagator cannot clear."""
    all_log = _log_norm(space, np.arange(index.latents.shape[0]))
    out = np.zeros((take.size, all_log.shape[1]))
    for c in np.unique(index.types[take]):
        m_all = index.types == c
        prof = all_log[m_all].mean(axis=0) if m_all.any() else all_log.mean(axis=0)
        out[index.types[take] == c] = prof
    return out


def _score_expr(pred: np.ndarray, real: np.ndarray, u: np.ndarray,
                types: np.ndarray) -> Dict:
    """Judge on spatially variable genes and on within-type variation."""
    predL = np.log1p(np.clip(pred, 0, None)
                     / np.maximum(np.clip(pred, 0, None).sum(1, keepdims=True), 1e-9) * 1e4) \
        if pred.max() > 50 else pred
    co = _agree(_coexpr(predL), _coexpr(real))
    mp, mr = _morans(u, predL), _morans(u, real)
    mo = _agree(mp, mr)
    wv_p, wv_r = _within_type_variation(predL, types), _within_type_variation(real, types)
    ratio = float(wv_p / wv_r) if wv_r and np.isfinite(wv_r) and wv_r > 0 else float("nan")
    # penalize *both* under- and over-dispersion relative to the real slice
    var_term = float(np.exp(-abs(np.log(max(ratio, 1e-6))))) if np.isfinite(ratio) else 0.0
    parts = [v for v in (co, mo) if np.isfinite(v)]
    composite = float(np.mean(parts) * 0.7 + var_term * 0.3) if parts else var_term
    return {"coexpression": co, "morans": mo, "within_type_var_ratio": ratio,
            "composite": composite}


def _morans(u: np.ndarray, X: np.ndarray, k: int = 10) -> np.ndarray:
    n = X.shape[0]
    if n < k + 1:
        return np.full(X.shape[1], np.nan)
    _, idx = cKDTree(u).query(u, k=k + 1)
    idx = idx[:, 1:]
    Z = X - X.mean(axis=0, keepdims=True)
    denom = (Z ** 2).sum(axis=0)
    numer = (Z * Z[idx].mean(axis=1)).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, numer / denom, np.nan)
