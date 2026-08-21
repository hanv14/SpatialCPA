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
