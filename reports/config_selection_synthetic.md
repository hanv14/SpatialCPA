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

Run as a separate **arm** over the selected config with the two gates that sever `ell` from the
output put back: `prior_mode = "correlated"` (the selector chose `iid`, under which the prior never
queries the field) and `expr_mode = "zinb-flow"` (it chose `cross-mix`, which copies donor counts
verbatim and never evaluates the flow). Both calibrators now **refuse** those configurations rather
than reporting a tie-break on a flat objective as a measurement. `train_steps = 2400`; targets
measured on flanking **training** sections `synthetic_s03`, `synthetic_s07`.

**This is the first arm in which the objective is live, and so the first real measurement.**

| quantity | value | status | applied? |
|---|---|---|---|
| `ell_xy` | **86.4 um** | `converged` in 2 iterations | **yes** — `Config.ell_xy` 100 → 86.4 um |
| `ell_z` | 364.6 um | `target_unreachable` | no — dropped, `Config.ell_z` stays 100 um |
| fitted `ell` (variogram) | 136.8 / 364.6 um | bracket endpoint | — |
| Moran's I | gen **0.4051** vs flanking **0.4102** | |gap| 0.0051 vs a 0.02 tolerance | — |
| between-section r | gen **0.8706** vs observed **0.9182** | R1 remedy 2 | — |

`ell_xy` converges cleanly and lands **13.6% below** the variogram's own 136.8 um, in the direction
T03 predicted (the fit is window-biased). It is written into the config by `apply_lengthscale` and
reaches the prior through `with_lengthscale`.

`ell_z` does not converge, and the *direction* it fails in is the informative part. The objective is
monotone increasing in `ell_z`, so the search terminates at the bracket's **top** — 364.6 um, the
variogram fit — and still undershoots, 0.8706 against 0.9182. What the data support is therefore
`ell_z >= 364.6 um`: a **lower** bound on the tissue, even though the number is the largest the
bracket allowed. (The earlier "upper bound" phrasing in the code and in R1's notes has been
corrected accordingly; it described the search's endpoint, not the parameter.) `apply_lengthscale`
drops it and `Config.ell_z` stands at 100 um, so nothing about R1 propagates into a shipped value.

**Why the `ell_z` target may be unreachable by construction, not by tuning.** The objective
deliberately generates both sections with **both excluded from retrieval**, so their correlation
comes from the field alone — otherwise a shared donor pool would correlate them at any `ell_z` and
the curve would be flat. But real adjacent sections get much of their 0.9182 from exactly that
shared anatomy. Remedy 2's target and its objective are therefore not measuring the same quantity,
and no `ell_z` need exist that closes the gap. That is a question for the spec owner rather than a
tuning knob, and it is the reason R1 stays open on the measurement even though every mechanism it
asked for now works.

**The anisotropy the oblique claim rests on.** Calibrated in-plane, bounded along z: the ratio is
`ell_z / ell_xy >= 364.6 / 86.4 = 4.2`, against the fixture's generative truth of 200/120 = 1.7 and
the variogram's own 364.6/136.8 = 2.7. All three agree the field is elongated along z by a factor of
at least two, which is what makes an oblique cut a different sampling problem from an axis-aligned
one. The lower bound is the honest form of the statement: this stack constrains the ratio from
below and not from above.

### The `cross-mix` arm, recorded as the negative result

The same calibration on the config the selector actually chose, kept because it is the measurement
that motivated the guard:

| sweep | ell_z = 25 | 60 | 137 | 200 | 275 | 364.6 |
|---|---|---|---|---|---|---|
| GRF field, r between planes 100 um apart | −0.005 | 0.202 | 0.654 | 0.794 | 0.874 | **0.921** |
| generated sections, `expr_mode="zinb-flow"` | 0.887 | 0.913 | 0.919 | 0.924 | 0.927 | **0.932** |
| generated sections, `expr_mode="cross-mix"` | 0.6728 | 0.6728 | 0.6728 | 0.6728 | 0.6728 | 0.6728 |

Under `cross-mix` the objective is identical to **ten decimal places** across a 15x sweep, and the
calibration returned `ell_xy = 7.0 um` / `ell_z = 25.0 um`, both `target_unreachable` in 0
iterations — the bracket's lower endpoints, which `specs/09` §2 says an unreachable target must
never return. That was the tell. `_cross_mix` copies each emitted count verbatim from a donor cell
and the donor weights come from the retrieval score rather than the latent, so the GRF is absent
from the expression path and `ell` cannot move a single count; the grid argmax of a constant
function is whichever point ties first. The field is sound (row 1, strictly monotone, reproduced
exactly through `with_lengthscale`) and so is the 400 um stack (row 2), so neither was ever the
limit.

Derived `retrieval_z_window` = **3** spacings (largest section gap 100 um). `Config.retrieval_z_window` remains the fallback and the ablation handle.

## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)

One **global** `ell` is calibrated; this table says whether it serves every gene module equally. It is not a target, and a poor table is evidence for the per-channel-group escalation, which is a design change to be decided explicitly.

| module | genes | I_gen | I_real | |diff| |
|---|---|---|---|---|
| 0 | 55 | 0.3996 | 0.3538 | 0.0585 |
| 1 | 55 | 0.4129 | 0.3849 | 0.0577 |
| 2 | 54 | 0.4584 | 0.4175 | 0.0586 |
| 3 | 36 | 0.4698 | 0.4296 | 0.0491 |

Measured on the live (`zinb-flow`) arm, under the calibrated `ell_xy`. The spread is **0.0491 to
0.0586 — flat across modules**, and now slightly *over*-shooting rather than under-shooting, which
is what a global `ell` tuned to the mean should look like. There is no module the single `ell`
serves badly and the others well, so this is evidence **against** the per-channel-group escalation
`specs/09` §2 describes. (On the dead `cross-mix` arm the same table read 0.1241 / 0.1056 / 0.1467 /
0.1298 — uniformly worse, and uniformly, which is the signature of a global deficit rather than a
per-module one.)

## Open risk R3 — the boundary is a different regime

Mean latent variance of the uncertainty gate at the stack's ends against its interior, on real cells with each section excluded from its own retrieval pool. Measured on the **selected** config (`prior_mode = "iid"`).

| position | mean latent variance | vs middle |
|---|---|---|
| first | 0.047621 | +13.2% |
| middle | 0.042071 | — |
| last | 0.045761 | +8.8% |

The gate notices the boundary and elevates there, which is the direction `specs/09` §1 asked for.
On the `correlated` calibration arms the same measurement **inverts** — first 0.783784, middle
0.846727, last 0.820789 under `cross-mix`, and first 0.741742, middle 0.782572, last 0.753801 under
`zinb-flow` (ends 5.2% and 3.7% *below* the interior) — with the absolute variance ~20x larger
because the prior itself is correlated. Two independent arms agree on the inversion, so it is the
prior's signature and not noise. So the sign of the boundary
effect is a property of the prior, not a fixed property of the gate; only the `iid` row above
describes what ships. Every emitted section carries `uns["boundary"]` in either regime.

## Definition of done — the selected config against the two fallbacks

Selected config (`prior_mode = "iid"`):

| arm | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization |
|---|---|---|---|---|---|---|
| method | 0.8329 | 0.7527 | 0.7028 | -0.0293 | 0.0254 | -0.0480 |
| resample | 0.8278 | 0.7683 | 0.6844 | -0.0374 | 0.0195 | -0.0660 |
| donor | 0.8454 | 0.8218 | 0.7358 | -0.0345 | 0.0218 | -0.0480 |

Calibration arms, `prior_mode = "correlated"`:

| arm | expr_mode | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization |
|---|---|---|---|---|---|---|---|
| method | cross-mix | 0.8215 | 0.7285 | 0.6923 | -0.0355 | 0.0393 | -0.0532 |
| **method** | **zinb-flow** | **0.9386** | **0.9067** | **0.9211** | -0.0467 | 0.0350 | -0.0502 |
| resample | cross-mix | 0.8278 | 0.7683 | 0.6844 | -0.0374 | 0.0195 | -0.0660 |
| donor | — | 0.8694 | 0.8347 | 0.7256 | -0.0373 | 0.0255 | -0.0502 |

**The definition of done depends on which config you ask about, and that is the finding.** As
selected (`iid` + `cross-mix`) the method beats `resample` on 5 of 6 and the independent-donor
sampler on only **2 of 6** — half met. On the live arm (`correlated` + `zinb-flow`) it beats
`resample` on 5 of 6 and the donor baseline on **4 of 6** (`morans_pearson` 0.9386 vs 0.8694,
`gearys_pearson` 0.9067 vs 0.8347, `umap_mixing` 0.9211 vs 0.7256, `marker_depth_r` 0.0350 vs
0.0255), ties `celltype_localization`, and loses only `marker_field_r`. The three distribution-level
statistics the method loses as selected are the three it wins by the widest margin when the flow is
switched on.

So the definition of done is **met on the configuration that exercises the method and not on the one
the selector shipped** — which is R8 with a price attached, not a separate result: the 25% reduced
budget picks `iid` and `cross-mix`, and both choices switch off the machinery the headline claim
rests on.

Detection / dispersion calibration fitted on ['synthetic_s00', 'synthetic_s01', 'synthetic_s03']; per-gene detection MAD before it 0.0093 (`iid`), 0.0091 (`correlated`+`cross-mix`) and 0.0084 (`correlated`+`zinb-flow`) — there is no headroom to correct on this fixture, which is why the correction ships off by default.

Anchor weight w(v) is **0.0 at every knot** in both arms — at variances [0.02776, 0.0336, 0.03732, 0.04088, 0.04451, 0.04888, 0.05495, 0.06937] under `iid` and [0.50616, 0.6106, 0.67725, 0.7393, 0.80093, 0.87268, 0.96339, 1.15094] under `correlated`, non-increasing by construction. The isotonic fit finds no variance band on this fixture where blending toward the retrieved anchor beats the generated value, so the gate is a no-op here rather than untested: `test_anchor_weight_is_monotone_non_increasing` and the randomised PAVA test cover the mechanism.
