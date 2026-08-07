"""
Diagnostics for SpatialCPA-v17 — is there a *manifold gap*?

The fully-generative decode path only works if the flow's generated latents ``h*`` land
where the decoder was trained (the encoder's aggregate posterior). If the flow's pushforward
drifts off that manifold, the decoder extrapolates and generation degrades — the case where a
Phase-C decoder fine-tune (training the decoder on flow-produced latents) would help.

:func:`reconstruction_gap` measures this without ever touching the held-out slice. For each
**interior training slice** (positions/z the flow was trained on) it compares, at the real
cells' own positions:

* **encode→decode** ``decode(encode(x))`` — the decoder's best case (no flow), and
* **flow→decode**   ``decode(flow @ z)``    — what generation actually does,

in two spaces:

* **latent** — a kNN off-manifold ratio ``d_flow / d_self``: mean distance from each flow
  latent to the nearest encoder latent, divided by the encoder cloud's own nearest-neighbor
  distance. ``~1`` means the flow lands inside the encoder cloud; ``≫1`` means off-manifold.
* **expression** — how well each path reproduces the real per-gene mean and variance (Pearson
  r, in the decoder's native target space). The *gap* is ``r(recon) − r(gen)``: near zero means
  the flow is not the bottleneck; large means it is.
"""

from __future__ import annotations

import numpy as np


def _pearson(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _knn_offmanifold(h_flow, h_enc):
    """mean NN distance (flow→encoder) / (encoder→encoder), a scale-free off-manifold ratio."""
    from scipy.spatial import cKDTree
    if h_enc.shape[0] < 2:
        return float("nan")
    tree = cKDTree(h_enc)
    d_flow, _ = tree.query(h_flow, k=1)
    d_self, _ = tree.query(h_enc, k=2)          # column 0 is the point itself (dist 0)
    return float(d_flow.mean() / (d_self[:, 1].mean() + 1e-8))


def reconstruction_gap(m, verbose=True):
    """Return per-slice + aggregate manifold-gap metrics for a trained SpatialCPAv17.

    Requires ``m.trained``. Does not modify the model. Compares encode→decode vs flow→decode
    on each interior training slice in latent and expression space.
    """
    if not getattr(m, "trained", False):
        raise RuntimeError("model is not trained (decode fallback active); no gap to measure.")
    import torch
    from .trainer import _neighbor_slices, _build_pool, _context

    cfg = m.cfg
    nn_ = m._nn; dev = nn_["dev"]
    vae, ctxmod, vfield = nn_["vae"], nn_["ctxmod"], nn_["vfield"]
    order = np.argsort(m.stack.z_centers())
    interior = order[1:-1] if len(order) >= 3 else order[:1]
    k_side = max(cfg.attn.context_slices_each_side, 1)

    rows = []
    for i in interior:
        i = int(i)
        s = m._S[i]
        n = s["h"].shape[0]
        if n == 0:
            continue
        pos = int(np.where(order == i)[0][0])
        src = _neighbor_slices(order, pos, k_side)             # flanking slices, i itself excluded
        pool_nxy, pool_z, pool_owner = _build_pool(m._S, src)
        pool_h = torch.cat([m._S[j]["h"] for j in src], 0)
        query_nxy = s["nxy"]; zq = float(s["nz"])

        with torch.no_grad():
            # flow latent at the real cells' own positions (no gap-dropout / no z-jitter)
            ctx = _context(m, pool_h, pool_nxy, pool_z, pool_owner, src, m._S,
                           query_nxy, zq, ctxmod, dev)
            Q = query_nxy.shape[0]; zt = torch.full((Q,), zq, device=dev)
            steps = max(cfg.flow.n_ode_steps, 1)
            h = torch.randn((Q, cfg.vae.latent_dim), device=dev)
            for si in range(steps):
                h = h + (1.0 / steps) * vfield(h, torch.full((Q,), si / steps, device=dev), ctx, zt)
            h_flow = h
            h_enc = s["h"]

            # decoder means in the decoder's native target space (fair, transform-free)
            lib = s["lib"]
            mu_enc, _, _ = vae.decode(h_enc, lib)
            mu_flow, _, _ = vae.decode(h_flow, lib)
            if m.likelihood == "nb":
                real = s["counts"].cpu().numpy()               # counts
            else:
                real = s["xin"].cpu().numpy()                  # standardized log-intensity
            mu_enc = mu_enc.cpu().numpy(); mu_flow = mu_flow.cpu().numpy()
            h_flow = h_flow.cpu().numpy(); h_enc = h_enc.cpu().numpy()

        gm_real, gv_real = real.mean(0), real.var(0)
        rows.append(dict(
            slice=m.stack.slices[i].section_id, n=n,
            offmanifold=_knn_offmanifold(h_flow, h_enc),
            latent_mean_shift=float(np.linalg.norm(h_flow.mean(0) - h_enc.mean(0))),
            latent_std_ratio=float(np.mean(h_flow.std(0) / (h_enc.std(0) + 1e-8))),
            r_mean_recon=_pearson(mu_enc.mean(0), gm_real),
            r_mean_gen=_pearson(mu_flow.mean(0), gm_real),
            r_var_recon=_pearson(mu_enc.var(0), gv_real),
            r_var_gen=_pearson(mu_flow.var(0), gv_real),
        ))

    if not rows:
        raise RuntimeError("no interior slices with cells to evaluate.")

    def _agg(key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {k: _agg(k) for k in
           ("offmanifold", "latent_mean_shift", "latent_std_ratio",
            "r_mean_recon", "r_mean_gen", "r_var_recon", "r_var_gen")}

    # Verdict is driven by whether landing off-manifold actually *hurts* the decoded output
    # (the fidelity drop recon->gen), not by the raw off-manifold ratio: that ratio is >1 partly
    # by construction (encoder latents are deterministic posterior means -> a tight cloud, while
    # the flow adds sampling spread), so a robust decoder can absorb it with no fidelity loss.
    mean_drop = agg["r_mean_recon"] - agg["r_mean_gen"]
    var_drop = agg["r_var_recon"] - agg["r_var_gen"]
    gap = (np.isfinite(mean_drop) and mean_drop > 0.10) or (np.isfinite(var_drop) and var_drop > 0.10)
    agg["mean_fidelity_drop"] = float(mean_drop)
    agg["var_fidelity_drop"] = float(var_drop)
    agg["gap_detected"] = bool(gap)

    if verbose:
        print("per-slice manifold-gap diagnostic:")
        print(f"  {'slice':<12} {'n':>5} {'offman':>7} {'mshift':>7} {'stdrat':>7} "
              f"{'rμ_rec':>7} {'rμ_gen':>7} {'rσ²_rec':>8} {'rσ²_gen':>8}")
        for r in rows:
            print(f"  {str(r['slice']):<12} {r['n']:>5} {r['offmanifold']:>7.2f} "
                  f"{r['latent_mean_shift']:>7.2f} {r['latent_std_ratio']:>7.2f} "
                  f"{r['r_mean_recon']:>7.3f} {r['r_mean_gen']:>7.3f} "
                  f"{r['r_var_recon']:>8.3f} {r['r_var_gen']:>8.3f}")
        print(f"\naggregate: off-manifold ratio={agg['offmanifold']:.2f} (context only; >1 is "
              f"partly structural)")
        print(f"  gene-mean fidelity: recon r={agg['r_mean_recon']:.3f} vs gen r={agg['r_mean_gen']:.3f} "
              f"(drop={mean_drop:+.3f})")
        print(f"  gene-var  fidelity: recon r={agg['r_var_recon']:.3f} vs gen r={agg['r_var_gen']:.3f} "
              f"(drop={var_drop:+.3f})")
        if gap:
            print("VERDICT: manifold gap detected — flow-decoded fidelity is meaningfully worse "
                  "than encode-decoded; a Phase-C decoder fine-tune (--finetune-decoder) is likely "
                  "to help.")
        else:
            print("VERDICT: no significant manifold gap — flow-decoded matches encode-decoded "
                  "fidelity; the decoder absorbs the flow's spread. Phase-C fine-tune unlikely to help.")

    return {"per_slice": rows, "aggregate": agg}
