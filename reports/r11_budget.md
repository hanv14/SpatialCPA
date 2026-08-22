# R11, split in two: a wrong integral and a starved sampler

No fit — the two saved coupling models re-run their layouts with
`ProposalBudgetWarning` captured rather than ignored.

`integral` is `n_expected / ground truth`: how wrong the learned intensity's *scale* is.
`placed` is `drawn / n_expected`: how much of what it asked for the rejection sampler
managed to place. They are independent failures and `cell_count_ratio` conflates them.

| link | section | `n_expected` | GT | integral | max/mean | bound | drawn | placed | drawn, no repulsion | placed | starved on/off |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `exp` | section_2 | 70046.1 | 4187 | **16.73x** | 15.6 | 5.842% | 48343 | **69.0%** | 51740 | **73.9%** | yes / yes |
| `exp` | section_4 | 8762.9 | 4102 | **2.14x** | 532.3 | 0.171% | 92 | **1.0%** | 103 | **1.2%** | yes / yes |
| `exp` | section_6 | 3752.7 | 4162 | **0.90x** | 4.1 | 22.237% | 3719 | **99.1%** | 3719 | **99.1%** | no / no |

## The fitted Strauss interaction, which the warning does not print

`r0` **0.859 um**, `R` **1.186 um**, `gamma` **0.5250**. `gamma = 1` is no soft repulsion at all.

`ProposalBudgetWarning` names `r0` alone. `r0` is the hard core; `R` and `gamma` are the
soft part, and they are what a rejection sampler actually spends its budget on. The
`no repulsion` columns above are ablation A4 on the same weights and the same seed: the
envelope is identical between the two arms, so the difference between them is the
interaction's thinning and nothing else.
