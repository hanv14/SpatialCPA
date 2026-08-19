# Config selection — synthetic

Selected by internal LOSO over **training sections only**, seed 20260819. Folds: synthetic_s01, synthetic_s05, synthetic_s07. No held-out section is reachable from `select_config`: it takes a `TrainingVolume`, which only `split_holdout` produces.

## Selected configuration

| gate | selected |
|---|---|
| `layout_mode` | field |
| `prior_mode` | iid |
| `expr_mode` | cross-mix |
| `text_emb_mode` | medcpt |
| `train_steps` | 2400 |
| `w_autocorr` | 0.5 |
| `w_profile` | 0.5 |
| `w_distribution` | 0.5 |
| config hash | `fe49ea9f8ad54bb2` |

## The joint gate: `train_steps` x metric-aware weights

All four cells, each **fitted at the budget it names**. Reported in full because the gate exists for an interaction, and coordinate descent over these two would pick `weights off` from a 1x incumbent and never reach the cell that wins (`specs/09` §3).

| cell | steps | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization | median rank |
|---|---|---|---|---|---|---|---|---|
| 1x, weights off | 1200 | 0.9197 | 0.8710 | 0.8501 | -0.0527 | 0.0382 | -0.0595 | 3.0 |
| 1x, weights on | 1200 | 0.8813 | 0.8589 | 0.6560 | -0.0387 | 0.0599 | -0.0577 | 3.5 |
| 2x, weights off | 2400 | 0.9316 | 0.9022 | 0.9145 | -0.0519 | 0.0195 | -0.0502 | 2.0 |
| 2x, weights on | 2400 | 0.9368 | 0.9123 | 0.9163 | -0.0370 | 0.0554 | -0.0410 | 1.0 |

## Coordinate descent

| gate | candidate | steps | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization | median rank |
|---|---|---|---|---|---|---|---|---|---|
| layout_mode | layout_mode=field | 600 | 0.6101 | 0.5370 | 0.1704 | 0.0221 | 0.0910 | -0.0631 | 1.5 |
| layout_mode | layout_mode=hybrid | 600 | 0.6009 | 0.5160 | 0.1749 | -0.0024 | 0.1048 | -0.0569 | 2.0 |
| layout_mode | layout_mode=resample | 600 | 0.6010 | 0.4791 | 0.1754 | 0.0101 | 0.0751 | -0.0660 | 2.5 |
| prior_mode | prior_mode=correlated | 600 | 0.6101 | 0.5370 | 0.1704 | 0.0221 | 0.0910 | -0.0631 | 2.0 |
| prior_mode | prior_mode=iid | 600 | 0.6494 | 0.5769 | 0.2057 | -0.0196 | 0.1061 | -0.0644 | 1.0 |
| expr_mode | expr_mode=zinb-flow | 600 | 0.6494 | 0.5769 | 0.2057 | -0.0196 | 0.1061 | -0.0644 | 2.5 |
| expr_mode | expr_mode=cross-mix | 600 | 0.8495 | 0.7914 | 0.6811 | -0.0426 | 0.0680 | -0.0644 | 1.5 |
| expr_mode | expr_mode=auto-blend | 600 | 0.8425 | 0.7773 | 0.6808 | -0.0487 | 0.0536 | -0.0644 | 2.0 |
| text_emb_mode | text_emb_mode=medcpt | 600 | 0.8495 | 0.7914 | 0.6811 | -0.0426 | 0.0680 | -0.0644 | 1.0 |
| text_emb_mode | text_emb_mode=lookup | 600 | 0.8247 | 0.7736 | 0.6752 | -0.0408 | 0.0312 | -0.0558 | 2.0 |
| layout_mode | layout_mode=field | 600 | 0.8495 | 0.7914 | 0.6811 | -0.0426 | 0.0680 | -0.0644 | 1.5 |
| layout_mode | layout_mode=hybrid | 600 | 0.8336 | 0.7688 | 0.6891 | -0.0300 | 0.0313 | -0.0506 | 1.5 |
| layout_mode | layout_mode=resample | 600 | 0.8278 | 0.7683 | 0.6844 | -0.0374 | 0.0195 | -0.0660 | 3.0 |
| prior_mode | prior_mode=correlated | 600 | 0.8430 | 0.7613 | 0.6810 | -0.0433 | 0.0388 | -0.0631 | 2.0 |
| prior_mode | prior_mode=iid | 600 | 0.8495 | 0.7914 | 0.6811 | -0.0426 | 0.0680 | -0.0644 | 1.0 |
| expr_mode | expr_mode=zinb-flow | 600 | 0.6494 | 0.5769 | 0.2057 | -0.0196 | 0.1061 | -0.0644 | 2.5 |
| expr_mode | expr_mode=cross-mix | 600 | 0.8495 | 0.7914 | 0.6811 | -0.0426 | 0.0680 | -0.0644 | 1.5 |
| expr_mode | expr_mode=auto-blend | 600 | 0.8425 | 0.7773 | 0.6808 | -0.0487 | 0.0536 | -0.0644 | 2.0 |
| text_emb_mode | text_emb_mode=medcpt | 600 | 0.8495 | 0.7914 | 0.6811 | -0.0426 | 0.0680 | -0.0644 | 1.0 |
| text_emb_mode | text_emb_mode=lookup | 600 | 0.8247 | 0.7736 | 0.6752 | -0.0408 | 0.0312 | -0.0558 | 2.0 |

Fits issued: 19 (1200 steps, 1200 steps, 2400 steps, 2400 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps, 600 steps).

The six metrics are computed with T08's kernels rather than `bench3`'s vendored implementations — `eval/metrics.py` is T10's module. The names match, and T10 re-scores the selected config with the vendored code.

## Calibration (leakage-free, flanking training sections only)

Run as a separate **arm** with `prior_mode = "correlated"` restored over the selected config
(`train_steps = 2400`, `expr_mode = "cross-mix"`). The selector chose `prior_mode = "iid"`, under
which the prior never queries the GRF and `ell` has no effect at all; `calibrate_lengthscale` and
`calibrate_ell_z` now refuse that combination rather than bisecting a flat objective. Targets
measured on flanking **training** sections `synthetic_s03`, `synthetic_s07`.

| quantity | value | status |
|---|---|---|
| `ell_xy` | 7.0 um | target_unreachable |
| `ell_z` | 25.0 um | target_unreachable (upper bound, R1) |
| fitted `ell` | 136.8 / 364.6 um | variogram |
| Moran's I | gen 0.2390 vs flanking 0.4102 | 0 iterations |
| between-section r | gen 0.6734 vs observed 0.9182 | R1 remedy 2 |

Both axes are `target_unreachable`, and `0 iterations` says why: the objective's maximum over the
log grid is already below the target, so there is nothing to bisect towards. `ell_z = 25.0 um` is
the bracket's **lower** endpoint (`calibration_ell_z_min_factor` x the 100 um median spacing), not
its upper one — the generated between-section correlation is highest at the short end of the
bracket and falls as `ell_z` grows, which is the opposite of what a GRF prior should do. R1's
remedy 3 still holds in the sense it was written for: the variogram's own `ell_z` (364.6 um here,
561 um on the gate fixture against a 200 um truth) enters only as the bracket's upper endpoint and
is never returned as a value.

The reason both axes are unreachable is the expression path the selector chose, and it bounds what
any `ell` could have done: under `expr_mode = "cross-mix"` every emitted count is taken **verbatim**
from a donor cell, so the GRF reaches the output only through the donor *weights*, never through the
counts. Restoring `prior_mode = "correlated"` gives the calibrator a live `ell`, but not enough
leverage on the emitted expression to move Moran's I from 0.2390 to 0.4102. Calibration headroom is
a property of the `prior_mode` x `expr_mode` pair, not of `ell` alone — recorded as R8.

Derived `retrieval_z_window` = **3** spacings (largest section gap 100 um). `Config.retrieval_z_window` remains the fallback and the ablation handle.

## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)

One **global** `ell` is calibrated; this table says whether it serves every gene module equally. It is not a target, and a poor table is evidence for the per-channel-group escalation, which is a design change to be decided explicitly.

| module | genes | I_gen | I_real | |diff| |
|---|---|---|---|---|
| 0 | 55 | 0.2336 | 0.3538 | 0.1241 |
| 1 | 55 | 0.2852 | 0.3849 | 0.1056 |
| 2 | 54 | 0.2730 | 0.4175 | 0.1467 |
| 3 | 36 | 0.3097 | 0.4296 | 0.1298 |

Measured on the same `prior_mode = "correlated"` arm as the calibration above. The deficit is
**global rather than per-module**: every module misses in the same direction by 0.11-0.15, so there
is no module the single `ell` serves badly and the others well. That is evidence against the
per-channel-group escalation, not for it.

## Open risk R3 — the boundary is a different regime

Mean latent variance of the uncertainty gate at the stack's ends against its interior, on real cells with each section excluded from its own retrieval pool. Measured on the **selected** config (`prior_mode = "iid"`).

| position | mean latent variance | vs middle |
|---|---|---|
| first | 0.047621 | +13.2% |
| middle | 0.042071 | — |
| last | 0.045761 | +8.8% |

The gate notices the boundary and elevates there, which is the direction `specs/09` §1 asked for.
On the `prior_mode = "correlated"` calibration arm the same measurement **inverts** — first 0.783784,
middle 0.846727, last 0.820789, i.e. the ends run 7.4% and 3.1% *below* the interior — and the
absolute variance is ~20x larger because the prior itself is correlated. So the sign of the boundary
effect is a property of the prior, not a fixed property of the gate; only the `iid` row above
describes what ships. Every emitted section carries `uns["boundary"]` in either regime.

## Definition of done — the selected config against the two fallbacks

Selected config (`prior_mode = "iid"`):

| arm | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization |
|---|---|---|---|---|---|---|
| method | 0.8329 | 0.7527 | 0.7028 | -0.0293 | 0.0254 | -0.0480 |
| resample | 0.8278 | 0.7683 | 0.6844 | -0.0374 | 0.0195 | -0.0660 |
| donor | 0.8454 | 0.8218 | 0.7358 | -0.0345 | 0.0218 | -0.0480 |

Calibration arm (`prior_mode = "correlated"`), for comparison:

| arm | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization |
|---|---|---|---|---|---|---|
| method | 0.8215 | 0.7285 | 0.6923 | -0.0355 | 0.0393 | -0.0532 |
| resample | 0.8278 | 0.7683 | 0.6844 | -0.0374 | 0.0195 | -0.0660 |
| donor | 0.8599 | 0.8220 | 0.7370 | -0.0386 | 0.0313 | -0.0532 |

The method beats `resample` on **5 of 6** in the selected (`iid`) arm — everything but
`gearys_pearson` — and on 4 of 6 in the `correlated` arm. Against the independent-donor sampler it
wins **2 of 6** in both arms (`marker_field_r`, `marker_depth_r`), ties `celltype_localization`, and
loses the three distribution-level statistics. The definition of done is therefore **half met**: the
no-regression guarantee holds against v20's layout path, the donor bar does not.

Detection / dispersion calibration fitted on ['synthetic_s00', 'synthetic_s01', 'synthetic_s03']; per-gene detection MAD before it 0.0093 (`iid`) and 0.0091 (`correlated` arm) — there is no headroom to correct on this fixture, which is why the correction ships off by default.

Anchor weight w(v) is **0.0 at every knot** in both arms — at variances [0.02776, 0.0336, 0.03732, 0.04088, 0.04451, 0.04888, 0.05495, 0.06937] under `iid` and [0.50616, 0.6106, 0.67725, 0.7393, 0.80093, 0.87268, 0.96339, 1.15094] under `correlated`, non-increasing by construction. The isotonic fit finds no variance band on this fixture where blending toward the retrieved anchor beats the generated value, so the gate is a no-op here rather than untested: `test_anchor_weight_is_monotone_non_increasing` and the randomised PAVA test cover the mechanism.
