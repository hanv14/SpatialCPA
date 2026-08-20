# R8 — does the reduced-epoch selection budget underrate the trained paths?

The selector chose `prior_mode = "iid"` then `expr_mode = "cross-mix"` by coordinate descent at
**600** steps (`selection_reduced_epoch_frac = 0.25` x the selected 2400) and applied both to a
2400-step model. The hypothesis under test: **the reduced budget favours the copying path, because
`cross-mix` emits real donor counts and needs no training, while `zinb-flow` has to learn a flow
before it emits anything sensible** — so a quarter-budget scoring run systematically underrates it.

Method: the same `prior_mode` x `expr_mode` grid scored at **both** budgets with the production
scorer (`train.select.FitScorer`), so the two halves differ only in `steps`. Incumbents held at the
values coordinate descent held while scoring these gates (`layout_mode="field"`,
`text_emb_mode="medcpt"`, all three metric weights 0.5), three internal LOSO folds, seed 20260819.
12 fits, 12 630 s. Nothing here touches a held-out section.

## The grid

| budget | prior_mode | expr_mode | morans | gearys | umap_mixing | field_r | depth_r | ct_loc |
|---|---|---|---|---|---|---|---|---|
| 600 | correlated | zinb-flow | 0.5925 | 0.5070 | 0.1650 | 0.0070 | 0.1109 | −0.0586 |
| 600 | correlated | cross-mix | 0.8416 | 0.7707 | 0.6749 | −0.0324 | 0.0440 | −0.0586 |
| 600 | correlated | auto-blend | 0.8368 | 0.7571 | 0.6718 | −0.0466 | 0.0333 | −0.0586 |
| 600 | iid | zinb-flow | 0.6682 | 0.6139 | 0.2076 | −0.0011 | 0.0618 | −0.0566 |
| 600 | iid | cross-mix | 0.8304 | 0.7694 | 0.6774 | −0.0480 | 0.0821 | −0.0566 |
| 600 | iid | auto-blend | 0.8367 | 0.7561 | 0.6747 | −0.0395 | 0.0713 | −0.0566 |
| 2400 | **correlated** | **zinb-flow** | **0.9357** | **0.8933** | **0.9295** | −0.0388 | 0.0429 | −0.0457 |
| 2400 | correlated | cross-mix | 0.8236 | 0.7365 | 0.7141 | −0.0462 | 0.0222 | −0.0457 |
| 2400 | correlated | auto-blend | 0.9357 | 0.8933 | 0.9295 | −0.0388 | 0.0429 | −0.0457 |
| 2400 | iid | zinb-flow | 0.9183 | 0.8709 | 0.8753 | −0.0435 | 0.0416 | −0.0519 |
| 2400 | iid | cross-mix | 0.8392 | 0.7722 | 0.6885 | −0.0463 | 0.0317 | −0.0519 |
| 2400 | iid | auto-blend | 0.9204 | 0.9002 | 0.8812 | −0.0446 | 0.0713 | −0.0519 |

## The ranking reverses, on both gates

Median rank across the six metrics, scored within the comparison group the selector forms:

| gate | 600 steps | 2400 steps |
|---|---|---|
| `expr_mode`, prior=correlated | zinb 2.5, **cross 1.5**, auto 2.0 | **zinb 1.5**, cross 3.0, **auto 1.5** |
| `expr_mode`, prior=iid | zinb 3.0, **cross 1.5**, auto 2.0 | zinb 2.0, cross 3.0, **auto 1.0** |
| `prior_mode`, expr=zinb-flow | corr 2.0, **iid 1.0** | **corr 1.0**, iid 2.0 |
| `prior_mode`, expr=cross-mix | **corr 1.5**, iid 1.5 | **corr 1.5**, iid 1.5 |
| `prior_mode`, expr=auto-blend | corr 2.0, **iid 1.0** | **corr 1.0**, iid 2.0 |

`cross-mix` wins the `expr_mode` gate at 600 under **both** priors and comes **last** at 2400 under
both. `iid` wins the `prior_mode` gate at 600 on exactly the two expression paths where the prior
can act, and loses on both at 2400. **Both of the selector's two wrong choices are reduced-budget
artefacts**, not one — R8 was recorded against `prior_mode` alone and is wider than that.

## The mechanism, measured

Gain in `morans_pearson` from 600 to 2400 steps:

| prior_mode | expr_mode | 600 | 2400 | gain |
|---|---|---|---|---|
| correlated | zinb-flow | 0.5925 | 0.9357 | **+0.3432** |
| correlated | auto-blend | 0.8368 | 0.9357 | +0.0989 |
| correlated | cross-mix | 0.8416 | 0.8236 | **−0.0180** |
| iid | zinb-flow | 0.6682 | 0.9183 | **+0.2501** |
| iid | auto-blend | 0.8367 | 0.9204 | +0.0837 |
| iid | cross-mix | 0.8304 | 0.8392 | +0.0088 |

This is the hypothesis in one column. `cross-mix` is **flat in budget** — it copies donor counts, so
four times the training buys it nothing and on the correlated prior it even drifts down. The flow
paths gain **+0.25 to +0.34**. Scoring them against each other at 25 % of the budget compares a
path that is finished with paths that have barely started.

## The two gates compound, because coordinate descent is ordered

Descent fixed `prior_mode = "iid"` **before** scoring `expr_mode`. At 600 steps `zinb-flow` ranks
2.5 under `correlated` and **3.0 under `iid`** — switching the field off makes the flow path look
worse still, and the gate that follows sees the degraded version. A wrong choice on an earlier gate
does not merely cost its own metric; it biases every gate scored after it.

## What full-budget scoring actually selects

All six cells ranked together at 2400 steps:

| rank | config | morans | gearys | umap_mixing |
|---|---|---|---|---|
| 1.8 | correlated + zinb-flow | 0.9357 | 0.8933 | 0.9295 |
| 1.8 | correlated + auto-blend | 0.9357 | 0.8933 | 0.9295 |
| 3.0 | iid + auto-blend | 0.9204 | 0.9002 | 0.8812 |
| 4.0 | iid + zinb-flow | 0.9183 | 0.8709 | 0.8753 |
| 5.0 | **iid + cross-mix (what shipped)** | 0.8392 | 0.7722 | 0.6885 |
| 5.5 | correlated + cross-mix | 0.8236 | 0.7365 | 0.7141 |

**The shipped configuration ranks fifth of six at the budget it is actually trained at.**

`auto-blend` under `correlated` is bit-identical to `zinb-flow`, because the fitted `w(v)` is 0 at
every knot on this fixture, so the blend passes the flow's draw through unmixed. It is not a third
behaviour here; it is `zinb-flow` with a gate that never fires.

## The gap on record — the selected config against the full-budget winner

Both at 2400 steps, same folds, same metric code:

| config | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization |
|---|---|---|---|---|---|---|
| `iid` + `cross-mix` (selected, shipped) | 0.8392 | 0.7722 | 0.6885 | −0.0463 | 0.0317 | −0.0519 |
| `correlated` + `zinb-flow` (full-budget winner) | **0.9357** | **0.8933** | **0.9295** | **−0.0388** | **0.0429** | **−0.0457** |
| difference | +0.0965 | +0.1211 | +0.2410 | +0.0075 | +0.0112 | +0.0062 |

The winner is ahead on **all six**. Against the external baselines (measured separately, in
`config_selection_synthetic.md`): the selected config beats the independent-donor sampler on 2 of 6,
the full-budget winner on 4 of 6.
