# Cost correction — claim 3's fix was not "seconds"

I costed it as *"small code change, seconds to test"*. The **test** is seconds. The **change** was
not, and the reason is worth recording because it is a design constraint rather than an oversight.

## The trap: every naive route breaks something

| route | breaks |
|---|---|
| hash the coordinates | **discontinuous.** Two plane pathways agree to ~1e-13 um (GATE 1 G1.2a), and a hash turns that into an entirely different donor. Bitwise under supplied coordinates, catastrophic under derived ones |
| project the GRF's latent to `G` channels | the GRF has `latent_dim` = 64 channels and this needs one per gene. Projecting 64 up to 1017 **correlates the genes**, making the mix quietly more copy-like and undermining T06's `test_per_gene_independence_destroys_covariance` |
| fractional part of a phase | wraps, so it is discontinuous at the wrap points |

## What works

`u_g(x) = arccos(cos(w_g . x + b_g)) / pi` — a triangle wave of a random-Fourier phase. Uniform
exactly (a triangle wave of a uniform phase is uniform), continuous everywhere, and near-independent
across genes because the `w_g` are drawn independently.

## And the default I picked first was wrong

The frequency scale is squeezed from both sides: short enough that **neighbouring cells** do not
share donors, long enough that a 1e-13 um difference does not flip one. My first value, 40 um,
failed the first constraint and the measurement caught it:

| `cross_mix_key_frequency_um` | mean cell-cell correlation < 15 um apart |
|---|---|
| 40 um (first tried) | **+0.26** |
| 20 um | +0.03 |
| 10 um | +0.005 |
| **5 um (shipped default)** | **−0.000** |

15 um is nearest-neighbour distance in this tissue, so at 40 um the key would have made adjacent
cells share donor selections — the exact defect its own docstring warned about. At 5 um the uniform
moves by ~1e-12 under a 1e-13 um coordinate difference, so the continuity constraint has twelve
orders of margin.

`tests/test_expression.py::test_position_key_does_not_correlate_NEIGHBOURING_CELLS` is that
measurement, with a 2000 um key as its non-vacuity control.

## What is shipped

`Config.cross_mix_position_keyed` defaults to **off**: turning it on changes the emission's
statistics and the framing that needs it is not decided. The default `cross-mix` path is untouched
and `test_cross_mix_matches_v20` still holds bitwise.

## The lesson

*A cost estimate for a change that must preserve a property is not an estimate of the edit.* The
edit is small. Establishing that it does not break per-gene independence, spatial independence, the
marginal distribution, or continuity is four measurements, and one of them rejected my first
default. I should have costed the properties, not the diff.
