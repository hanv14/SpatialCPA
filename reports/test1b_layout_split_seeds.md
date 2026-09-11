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
| `base` | model | model | +0.5371 | +0.2007 | +0.6603 | **+0.5371** | 6.000° |
| `fix_types` | model | **ground truth** | +0.6379 | +0.2972 | +0.7278 | **+0.6379** | 6.000° |
| `fix_positions` | **the copy's** | model | +0.6221 | +0.6075 | +0.5974 | **+0.6075** | 0.000° |
| `both_oracle` | the copy's | ground truth | +0.8303 | +0.8375 | +0.8443 | **+0.8375** | 0.000° |
| `null_types` | model | *permuted* | +0.0293 | +0.0832 | +0.0518 | **+0.0518** | 6.000° |

Reference: `flanking_copy` **0.7765**, `resample` 0.7546, `oracle` 0.9808.

## The split

`base` = **+0.5371**, denominator `floor - base` = **+0.2394**.

| | recovered fraction of the deficit |
|---|---|
| fixing the **types** (`fix_types`) | **42.1%** |
| fixing the **positions** (`fix_positions`) | **29.4%** |
| sum — its distance from 100% is the **interaction** | 71.5% |
| fixing **both** (`both_oracle`) | **125.5%** |

⚠️ **The median at n = 3 IS a section**, and the three can disagree in sign. The verdict
above is the pre-registered one; the table below is why it is not the whole story.

| section | base | deficit | types | positions | both | verdict |
|---|---|---|---|---|---|---|
| section_2 | +0.5371 | 0.2394 | 42.1% | 35.5% | 122.5% | MIXED |
| section_4 | +0.2007 | 0.5758 | 16.8% | 70.7% | 110.6% | PLACEMENT |
| section_6 | +0.6603 | 0.1162 | 58.1% | -54.1% | 158.3% | MIXED |

## Across sections against across seeds

Seeds: [1, 2, 3]. **Reported, not verdicted** — the observation this tests was read off
a table already seen, so nothing here claims it.

| arm | positions | across-section spread | worst across-seed spread | ratio |
|---|---|---|---|---|
| `base` | model | 0.4596 | 0.0453 | 10.1x |
| `fix_types` | model | 0.4306 | 0.0952 | 4.5x |
| `null_types` | model | 0.0539 | 0.0194 | 2.8x |
| `fix_positions` | copy | 0.0247 | 0.0697 | 0.4x |
| `both_oracle` | copy | 0.0140 | 0.0000 | infx |

## **NOT READABLE**

Preconditions failed (§6):

- the arms' alignments span 6.00°, above 5.0°: the comparison is across poses

The split is not reported. The preconditions were committed before the run.

✅ **3 generation seeds** — the one-seed caveat that stood on every field-layout number since R11 is retired for this measurement.

⚠️ This splits **one metric at axis-aligned planes**. It says nothing about oblique planes
— see `reports/oblique_layout_cost.md` for why those need a different evaluation set.
