# Step 1 — splitting the layout deficit into placement and typing

**Read `reports/layout_split_preregistration.md` first.** The arms, bands, preconditions
and four predictions were committed before this ran.

`layout_mode=field`, grid sampler, shippable cell count, seed 1,
weights `runs/pilot/model_exp_2400.pt`. **Zero fits.** Scored through the pinned `evaluate_paper`.

Every arm carries a coherent expression field matched to its own positions, because the
pose is aligned **by expression** — arms that share positions share a pose exactly, and
`align_rotation_deg` is a precondition rather than a footnote.

| arm | positions | types | section_2 | section_4 | section_6 | **median** | pose |
|---|---|---|---|---|---|---|---|
| `base` | model | model | +0.5371 | +0.1863 | +0.6603 | **+0.5371** | 1.500° |
| `fix_types` | model | **ground truth** | +0.6379 | +0.2972 | +0.7278 | **+0.6379** | 1.500° |
| `fix_positions` | **the copy's** | model | +0.5888 | +0.6075 | +0.5974 | **+0.5974** | 0.000° |
| `both_oracle` | the copy's | ground truth | +0.8303 | +0.8375 | +0.8443 | **+0.8375** | 0.000° |
| `null_types` | model | *permuted* | +0.0293 | +0.0832 | +0.0518 | **+0.0518** | 1.500° |

Reference: `flanking_copy` **0.7765**, `resample` 0.7546, `oracle` 0.9808.

## The split

`base` = **+0.5371**, denominator `floor - base` = **+0.2394**.

| | recovered fraction of the deficit |
|---|---|
| fixing the **types** (`fix_types`) | **42.1%** |
| fixing the **positions** (`fix_positions`) | **25.2%** |
| sum — its distance from 100% is the **interaction** | 67.3% |

## **MIXED**

neither mechanism dominates; both shares are reported and no single-mechanism claim may be made.

⚠️ **One seed, one fit.** The r11 arms are one fit and one seed, so no across-seed spread
exists for this metric on the field arms (`envelope_correction.md` §3). Raw deficits only;
**no multiple-of-envelope may be quoted.**

⚠️ This splits **one metric at axis-aligned planes**. It says nothing about oblique planes
— see `reports/oblique_layout_cost.md` for why those need a different evaluation set.
