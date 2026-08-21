# R11 — is the layout's count error coupled to the `mu` link? (2400 steps)

Two fits identical in every respect except `decoder_mu_link`. `IntensityHead` takes only
coordinates and field features, so there is no direct path; the shared triplane field and
the joint objective are the indirect one.

| link | section | `n_expected` | drawn | ground truth | ratio |
|---|---|---|---|---|---|
| `softplus` | section_2 | 11225.8 | 11168 | 4187 | **2.67x** |
| `softplus` | section_4 | 6203.2 | 146 | 4102 | **0.04x** |
| `softplus` | section_6 | 3821.3 | 3788 | 4162 | **0.91x** |
| `exp` | section_2 | 70046.1 | 48343 | 4187 | **11.55x** |
| `exp` | section_4 | 8762.9 | 92 | 4102 | **0.02x** |
| `exp` | section_6 | 3752.7 | 3719 | 4162 | **0.89x** |

| link | intensity bands | field abs-mean | field sd | field p99 abs | fit s |
|---|---|---|---|---|---|
| `softplus` | 4 | 1.0087 | 1.0797 | 2.2549 | 6533 |
| `exp` | 4 | 0.8880 | 1.0137 | 2.7016 | 3901 |
