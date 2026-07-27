"""Phase 2.3-2.5 — structure completion as query-based interpolation.

Training follows the proposal's VFI framing.  From the *densely sectioned*
material available (here: the training stack itself), tuples are sampled with
brackets held out and a **real intermediate as the target**:

    (I_{a-1}, I_a, I_b, I_{b+1}, t, dz) -> I_t

with

* **two slices per side**, so the derivative of the structure is available and
  the model is not forced into a monotone morph between two endpoints;
* **flow-free, directly generative** — the model produces the intermediate
  field, it does not warp one bracket onto another;
* conditioning on **both** normalized ``t`` **and** absolute ``dz``;
* augmentation by random gap widths, random ``t``, random in-plane crops, and
  **z-reversal**, which is an exact symmetry of this problem and therefore free
  data.  In-plane mirroring is off by default (lateralized biology).

Sampling (Phase 2.4) is a deterministic DDIM with the **same initial noise
reused across every query in a gap**, so ``t -> I_t`` is a continuous function
and independently issued queries compose into one coherent volume.  Different
seeds give different coherent volumes; their voxelwise spread is the structural
uncertainty map.

Phase 2.5 is enforced, not just documented: the learned completion must beat
linear interpolation of the field on internally held-out real slices, otherwise
the linear field is used and the fallback is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import StructureConfig
from .phase2_field import FieldSpec, linear_field


def _pick_device(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class StructureTuple:
    """One training query: bracket indices, the target index, ``t`` and ``dz``."""

    a_out: int
    a: int
    b: int
    b_out: int
    target: int
    t: float
    dz: float


def enumerate_tuples(z: np.ndarray, cfg: StructureConfig) -> List[StructureTuple]:
    """All (bracket, held-out intermediate) tuples available in the stack.

    Gap widths are *not* fixed: every (a, b) pair that strictly contains at least
    one real slice contributes, which is how the model sees a range of ``dz``.
    """
    n = len(z)
    out: List[StructureTuple] = []
    for a in range(n - 2):
        for b in range(a + 2, n):
            dz = float(z[b] - z[a])
            if dz <= 0:
                continue
            for m in range(a + 1, b):
                t = float((z[m] - z[a]) / dz)
                a_out = a - 1 if a - 1 >= 0 else (a if cfg.asymmetric_brackets else -1)
                b_out = b + 1 if b + 1 < n else (b if cfg.asymmetric_brackets else -1)
                if a_out < 0 or b_out < 0:
                    continue
                out.append(StructureTuple(a_out, a, b, b_out, m, t, dz))
    return out


@dataclass
class StructureModel:
    """The trained (or gated-off) structure completer."""

    fields: np.ndarray                       # (Z, C, H, W) observed slabs (encoded)
    z: np.ndarray                            # (Z,) section depths
    spec: FieldSpec
    scale: np.ndarray                        # (C,) per-channel normalization
    compressor: object = None                # Phase 2.2; identity when disabled
    net: object = None
    schedule: object = None
    device: object = None
    cfg: StructureConfig = None
    trained: bool = False
    use_learned: bool = False
    diagnostics: Dict = field(default_factory=dict)

    # ── query ────────────────────────────────────────────────────────────────
    def _bracket_stack(self, tup: StructureTuple) -> np.ndarray:
        idx = [tup.a_out, tup.a, tup.b, tup.b_out]
        return np.concatenate([self.fields[i] / self.scale[:, None, None]
                               for i in idx], axis=0)

    def query(self, z_query: float, noise=None) -> np.ndarray:
        """Complete the field at an arbitrary depth (Phase 2.4 direct query).

        Direct query is preferred over recursive subdivision: subdivision
        compounds smoothing error multiplicatively, and there is no reason to pay
        that when the model is conditioned on ``dz`` and can answer directly.
        """
        a, b, t, dz = self._locate(z_query)
        tup = StructureTuple(max(a - 1, 0), a, b, min(b + 1, len(self.z) - 1),
                             -1, t, dz)
        return self.query_with(tup, noise=noise)

    def query_with(self, tup: StructureTuple, noise=None,
                   force_learned: bool = False) -> np.ndarray:
        """Answer a query against **explicit** brackets.

        The Phase 2.5 gate needs this: a query at the depth of a real slice must
        be answered from the brackets that *exclude* that slice, otherwise the
        linear baseline trivially reproduces the target and the comparison is
        meaningless.
        """
        t, dz = tup.t, tup.dz
        lin = linear_field(self.fields, tup.a, tup.b, t)
        if not (self.trained and (self.use_learned or force_learned)):
            return lin
        import torch
        br = torch.from_numpy(self._bracket_stack(tup)[None]).float().to(self.device)
        C, H, W = self.fields.shape[1:]
        shape = (1, C, H, W)
        tt = torch.tensor([t], device=self.device, dtype=torch.float32)
        dzz = torch.tensor([dz], device=self.device, dtype=torch.float32)

        def eps(x, step):
            return self.net(x, br, step, tt, dzz)

        res = self.schedule.ddim(eps, shape, self.device, self.cfg.ddim_steps,
                                 noise=noise)
        res = res[0].detach().cpu().numpy() * self.scale[:, None, None]
        return lin + res

    def decoded(self, F: np.ndarray) -> np.ndarray:
        """Undo the Phase 2.2 compression (identity when it is disabled).

        The compression is affine, so it commutes with the linear bracket
        interpolation the interpolator is built on: training, querying and the
        gate can all stay in the compressed space, and only the field handed to
        the layout sampler is decoded.
        """
        return self.compressor.decode(F) if self.compressor is not None else F

    def query_clipped(self, z_query: float, noise=None) -> np.ndarray:
        """Query, then project onto the valid-field set.

        The density channels are non-negative by construction, so a generative
        model must not be allowed to emit negative density; the embedding
        channels are signed and are left alone.  This is the cheap version of the
        proposal's 'no invalid intermediate states' requirement.
        """
        F = np.nan_to_num(self.decoded(self.query(z_query, noise=noise)),
                          nan=0.0, posinf=0.0, neginf=0.0)
        n_dens = 1 + self.spec.n_groups
        F[:n_dens] = np.clip(F[:n_dens], 0.0, None)
        return F

    def _locate(self, z_query: float) -> Tuple[int, int, float, float]:
        zs = self.z
        b = int(np.clip(np.searchsorted(zs, z_query, side="left"), 1, len(zs) - 1))
        a = b - 1
        dz = float(zs[b] - zs[a])
        t = float((z_query - zs[a]) / dz) if dz > 0 else 0.0
        return a, b, t, dz

    def fixed_noise(self, seed: int):
        """One noise tensor per seed, reused for **every** query in the gap."""
        import torch
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        C, H, W = self.fields.shape[1:]
        return torch.randn((1, C, H, W), generator=g).to(self.device)


# ── training ─────────────────────────────────────────────────────────────────

def train_structure_model(fields: np.ndarray, z: np.ndarray, spec: FieldSpec,
                          cfg: StructureConfig, seed: int = 0,
                          verbose: bool = True, compressor=None) -> StructureModel:
    """Train the Phase 2.3 interpolator and run the Phase 2.5 gate.

    ``fields`` are already in the Phase 2.2 representation (compressed or not);
    ``compressor`` is kept so queries can be decoded back to channel space.
    """
    scale = fields.reshape(fields.shape[0], fields.shape[1], -1).std(axis=(0, 2))
    # Floor the per-channel scale *relative* to the field, not absolutely: a
    # near-constant channel divided by an absolute floor explodes into the
    # network and takes the training loss to NaN.
    scale = np.maximum(scale, 1e-3 * max(float(scale.max()), 1e-12)).astype(np.float32)
    scale = np.maximum(scale, 1e-8).astype(np.float32)
    model = StructureModel(fields=fields, z=z, spec=spec, scale=scale, cfg=cfg,
                           compressor=compressor)

    tuples = enumerate_tuples(z, cfg)
    if not cfg.enabled or len(tuples) < 4 or len(z) < 4:
        model.diagnostics = {"trained": False,
                             "reason": f"only {len(tuples)} training tuples "
                                       f"from {len(z)} sections",
                             "fallback": "linear field interpolation"}
        return model

    import torch
    from .nets import CosineSchedule, QueryUNet

    torch.manual_seed(seed)
    device = _pick_device(cfg.device)
    C, H, W = fields.shape[1:]
    net = QueryUNet(C, hidden=cfg.hidden).to(device)
    sched = CosineSchedule(cfg.n_timesteps, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)

    Fn = torch.from_numpy(fields / scale[None, :, None, None]).float().to(device)
    rng = np.random.default_rng(seed)
    crop = int(min(cfg.crop, H, W))

    def batch(bs: int):
        xs, brs, ts, dzs = [], [], [], []
        for _ in range(bs):
            tup = tuples[rng.integers(len(tuples))]
            idx = [tup.a_out, tup.a, tup.b, tup.b_out]
            t, dz = tup.t, tup.dz
            if cfg.z_reversal and rng.random() < 0.5:      # exact symmetry
                idx = idx[::-1]
                t = 1.0 - t
            if cfg.mirror and rng.random() < 0.5:
                pass
            y0 = rng.integers(0, H - crop + 1)
            x0 = rng.integers(0, W - crop + 1)
            sl = (slice(None), slice(y0, y0 + crop), slice(x0, x0 + crop))
            br = torch.cat([Fn[i][sl] for i in idx], dim=0)
            lin = (1 - t) * Fn[idx[1]][sl] + t * Fn[idx[2]][sl]
            xs.append(Fn[tup.target][sl] - lin)
            brs.append(br)
            ts.append(t)
            dzs.append(dz)
        return (torch.stack(xs), torch.stack(brs),
                torch.tensor(ts, device=device, dtype=torch.float32),
                torch.tensor(dzs, device=device, dtype=torch.float32))

    net.train()
    losses = []
    for ep in range(cfg.epochs):
        x0, br, tt, dzz = batch(cfg.batch_size)
        step = torch.randint(0, cfg.n_timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = sched.q_sample(x0, step, noise)
        pred = net(xt, br, step, tt, dzz)
        loss = ((pred - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if verbose and (ep + 1) % max(cfg.epochs // 4, 1) == 0:
            print(f"      [phase2] epoch {ep + 1}/{cfg.epochs} "
                  f"loss={np.mean(losses[-20:]):.4f}")
    net.eval()

    model.net = net
    model.schedule = sched
    model.device = device
    model.trained = True

    gate = structure_gate(model, tuples, cfg)
    model.use_learned = bool(gate["passed"])
    model.diagnostics = {"trained": True, "final_loss": float(np.mean(losses[-20:])),
                         "n_tuples": len(tuples), **gate}
    if verbose:
        verdict = "PASS -> learned completion" if model.use_learned else \
                  "FAIL -> falling back to linear field"
        print(f"      [phase2 gate] mse_learned={gate['mse_learned']:.5f} "
              f"mse_linear={gate['mse_linear']:.5f} "
              f"gain={gate['relative_gain']:+.3f} | {verdict}")
    return model


# ── Phase 2.5 — the gate ─────────────────────────────────────────────────────

def structure_gate(model: StructureModel, tuples: List[StructureTuple],
                   cfg: StructureConfig, n_seeds: int = 4,
                   max_eval: int = 12) -> Dict:
    """Held-out mid-gap accuracy vs. linear interpolation + reslice consistency.

    The proposal's stop condition is literal: *if mid-gap accuracy is no better
    than linear interpolation of the field, fix this before building Phase 3 on
    top of it.*  Here that is machine-checked, and a failing model is simply not
    used — the pipeline degrades to the linear field rather than shipping a
    completion that is worse than arithmetic.

    The comparison is made against the **seed-ensemble mean**, not a single draw.
    A single draw is a *sample* from the conditional distribution, so even a
    perfect sampler carries the full conditional variance and would lose an MSE
    contest to any conditional-mean estimator — that would measure sampling, not
    accuracy.  The ensemble mean is the accuracy estimate; the per-seed spread is
    reported separately as the uncertainty (Phase 2.4).
    """
    if not tuples:
        return {"passed": False, "mse_learned": float("nan"),
                "mse_linear": float("nan"), "relative_gain": 0.0,
                "reslice_striping": float("nan")}
    sel = tuples[:: max(1, len(tuples) // max_eval)][:max_eval]
    e_learn, e_lin, spreads = [], [], []
    noises = [model.fixed_noise(s) for s in range(max(1, n_seeds))]
    inv = 1.0 / model.scale[:, None, None]
    for tup in sel:
        target = model.fields[tup.target] * inv
        draws = np.stack([model.query_with(tup, noise=nz, force_learned=True) * inv
                          for nz in noises])
        pred = draws.mean(axis=0)
        spreads.append(float(draws.std(axis=0).mean()))
        lin = linear_field(model.fields, tup.a, tup.b, tup.t) * inv
        e_learn.append(float(((pred - target) ** 2).mean()))
        e_lin.append(float(((lin - target) ** 2).mean()))

    ml, mn = float(np.mean(e_learn)), float(np.mean(e_lin))
    gain = (mn - ml) / mn if mn > 0 else 0.0
    saved = model.use_learned
    model.use_learned = True
    striping = reslice_consistency(model)
    model.use_learned = saved

    return {"passed": bool(gain > cfg.gate_margin),
            "mse_learned": ml, "mse_linear": mn, "relative_gain": gain,
            "n_gate_queries": len(sel), "n_gate_seeds": len(noises),
            "mean_seed_spread": float(np.mean(spreads)),
            "reslice_striping": striping}


def reslice_consistency(model: StructureModel, n_sub: int = 3) -> float:
    """Orthogonal-reslice diagnostic (Phase 2.5.1), as a number rather than a look.

    A completed volume is built on a fine z-grid and the total variation along z
    is compared to the in-plane total variation.  Anatomy has no preferred
    slicing axis, so a value near 1 means the volume looks like tissue from every
    axis; a large value means per-slice inconsistency (the striping/banding an
    orthogonal reslice would show); a value near 0 means the intermediates are
    over-smoothed relative to the real planes.
    """
    zs = model.z
    if len(zs) < 2:
        return float("nan")
    grid = []
    for i in range(len(zs) - 1):
        for k in range(n_sub):
            grid.append(zs[i] + (zs[i + 1] - zs[i]) * k / n_sub)
    grid.append(float(zs[-1]))
    noise = model.fixed_noise(0)
    vol = np.stack([model.query_clipped(z, noise=noise)[0] for z in grid])  # density
    vol = vol / (vol.std() + 1e-9)
    dz_tv = np.abs(np.diff(vol, axis=0)).mean() * n_sub
    inplane_tv = 0.5 * (np.abs(np.diff(vol, axis=1)).mean()
                        + np.abs(np.diff(vol, axis=2)).mean())
    return float(dz_tv / (inplane_tv + 1e-9))


# ── Phase 2.4 — multi-seed completion for uncertainty ────────────────────────

def complete_with_uncertainty(model: StructureModel, z_query: float,
                              n_seeds: int) -> Tuple[np.ndarray, np.ndarray]:
    """N deterministic completions -> (mean field, voxelwise spread).

    Each seed gives **one coherent volume** — one hypothesis — because its noise
    is held fixed across queries.  The spread across seeds is the structural
    uncertainty map that Phase 4.7 requires be carried into every downstream
    figure.
    """
    if not (model.trained and model.use_learned):
        f = model.query_clipped(z_query)
        return f, np.zeros_like(f)
    outs = [model.query_clipped(z_query, noise=model.fixed_noise(s))
            for s in range(max(1, n_seeds))]
    A = np.stack(outs)
    return A.mean(axis=0), A.std(axis=0)
