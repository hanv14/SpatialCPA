# R11, split in two: a wrong integral and a starved sampler

No fit — the two saved coupling models re-run their layouts with
`ProposalBudgetWarning` captured rather than ignored.

`integral` is `n_expected / ground truth`: how wrong the learned intensity's *scale* is.
`placed` is `drawn / n_expected`: how much of what it asked for the rejection sampler
managed to place. They are independent failures and `cell_count_ratio` conflates them.

| link | section | `n_expected` | drawn | GT | integral | placed | starved | mid-plane max/mean | acceptance bound |
|---|---|---|---|---|---|---|---|---|---|
| `softplus` | section_2 | 11225.8 | 11168 | 4187 | **2.68x** | **99.5%** | no | 6.7 | **13.524%** |
| `softplus` | section_4 | 6203.2 | 146 | 4102 | **1.51x** | **2.4%** | **yes** | 32.4 | **2.803%** |
| `softplus` | section_6 | 3821.3 | 3788 | 4162 | **0.92x** | **99.1%** | no | 3.4 | **27.125%** |
| `exp` | section_2 | 70046.1 | 48343 | 4187 | **16.73x** | **69.0%** | **yes** | 15.6 | **5.842%** |
| `exp` | section_4 | 8762.9 | 92 | 4102 | **2.14x** | **1.0%** | **yes** | 532.3 | **0.171%** |
| `exp` | section_6 | 3752.7 | 3719 | 4162 | **0.90x** | **99.1%** | no | 4.1 | **22.237%** |

## Reading it: the envelope is dominant but does not account for all of the shortfall

**The envelope separates the cases perfectly.** Ranked by mid-plane `max/mean`, the three starved
cells are the three highest (15.6, 32.4, 532.3) and the three healthy ones the three lowest
(3.4, 4.1, 6.7), with no overlap. And the `exp` link raises the dynamic range on both problem
sections — `section_2` 6.7 -> 15.6, `section_4` **32.4 -> 532.3, a 16x worsening** — which is how a
change to the *decoder's* link function reaches the layout at all.

**But the arithmetic does not close.** Proposals are uniform on the plane and accepted with
probability `lam(x) / (max(lam) * slack)`, so the expected acceptance is exactly `1 / (dyn * slack)`
— the `acceptance bound` column. With `layout_max_proposal_factor = 20`, `softplus`/`section_4`
should therefore place about `20 * 2.803% = 56%` of `n_target`; it placed **2.4%**, twenty-four
times fewer. `exp`/`section_2` should have saturated (`20 * 5.842% > 100%`); it placed 69%.

So a **second thinning factor** is active besides the envelope, and the interaction is the only
other one in `_propose_points`. The warning names `r0 = 0.859 um`, which at 0.00307 cells/um^2
(~18 um mean spacing) cannot possibly bind — but `r0` is not the whole of the Strauss interaction,
and the warning prints **neither `R` nor `gamma`**. That is the gap: the message reports the one
interaction parameter that is provably irrelevant here and omits the two that could explain a 24x
residual.

**Next measurement**, before anything is attributed: report the fitted `R` and `gamma` beside `r0`,
and measure the acceptance rate with the interaction disabled (`Config.replace(repulsion=False)`,
which is ablation A4 and already supported). That separates envelope from interaction directly. It
is a layout-only run on the two saved models — no fit.
