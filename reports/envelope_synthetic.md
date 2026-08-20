# R10 — the reproducibility envelope, and which gate choices were decided inside it

`specs/09` §3's repeated-seed rule and its capability tie-break both need a number: how far apart
two measurements must be before the difference is real. This measures it, and then re-checks the
18-cell selection against it.

## Method

Three representative cells of the merged full-budget gate, each fitted at **three seeds**
(20260819, 20260820, 20260821), all nine fits in one process so the spread is across *seeds* and is
not confounded with the cross-process nondeterminism R10 also observed. 9 fits, ~3.4 h,
checkpointed per fit (`reports/envelope_synthetic.csv`).

* **winner** — `resample` + `correlated` + `zinb-flow` (score level 0.95)
* **closest rival** — `hybrid` + `correlated` + `zinb-flow`, 0.0344 from the winner (0.94)
* **far arm** — `resample` + `correlated` + `cross-mix`, 0.2900 away (0.83), included to test
  whether the envelope is score-dependent

## The envelope

Across-seed spread (max − min over three seeds):

| cell | morans | gearys | umap_mixing | field_r | depth_r | ct_loc | max |
|---|---|---|---|---|---|---|---|
| winner (0.95) | 0.0049 | 0.0040 | 0.0115 | 0.0087 | **0.0299** | 0.0000 | 0.0299 |
| closest rival (0.94) | 0.0036 | **0.0335** | 0.0050 | 0.0054 | 0.0215 | 0.0068 | **0.0335** |
| far arm (0.83) | 0.0160 | 0.0270 | 0.0097 | 0.0162 | 0.0068 | 0.0000 | 0.0270 |
| **envelope** | 0.0160 | **0.0335** | 0.0115 | 0.0162 | 0.0299 | 0.0068 | **0.0335** |

**The envelope is 0.0335**, and three things about it matter more than the number:

* **It is nearly 3x the n = 2 figure.** Two fits had said 0.0120. Nine say 0.0335. A maximum over
  nine samples is still a *lower* bound, which is why `Config.claim_tie_break_envelope` is set to
  **0.04** — rounded up, so the threshold is not tighter than the evidence supports.
* **It is not score-dependent.** 0.0299 at a score level of 0.95, 0.0335 at 0.94, 0.0270 at 0.83.
  One threshold serves every cell; this is what the far arm was included to test.
* **It is strongly metric-dependent.** `celltype_localization` reproduces to 0.0068 while
  `gearys_pearson` moves by 0.0335 and `marker_depth_r` by 0.0299 — a 5x range. A per-metric
  envelope is available in the table above and is the right thing for T10 to quote per claim.

## Which gate choices were decided inside the noise

Margin = the largest per-metric difference between the selected cell and the closest cell that
differs on that gate, against the 0.04 threshold:

| gate | selected | closest rival | margin | verdict |
|---|---|---|---|---|
| `layout_mode` | resample | **hybrid** | **0.0344** | **inside the envelope** |
| `prior_mode` | correlated | iid | 0.0731 | safe (1.8x) |
| `expr_mode` | zinb-flow | auto-blend | **0.0000** | **inside — identical** |
| `expr_mode` | zinb-flow | cross-mix | 0.2900 | safe (7.3x) |
| `text_emb_mode` | lookup | **medcpt** | **0.0110** | **inside the envelope** |

**Two real gate choices were decided inside the noise, and one is a labelling artefact.**

* **`layout_mode` is the one that matters, and it was not previously suspected.** `resample` beat
  `hybrid` by 0.0344 against an envelope of 0.0335 — a margin 1.03x the noise floor, which is not a
  decision. The tie-break selects **`hybrid`**: `resample` reuses real cell positions and is the v20
  fallback, so shipping it switches the learned continuous layout *off*, while `hybrid` exercises
  it. This is a borderline call and is flagged as such — at the raw measured 0.0335 the margin
  would clear by 3%, and it is only the (justified) rounding up that puts it inside. The
  justification is that 0.0335 is a maximum over nine fits and the same statistic at two fits was
  0.0120, so treating it as an exact ceiling would be wrong in the unsafe direction.
* **`text_emb_mode`** was already known (R9): 0.0110 apart, and `lookup` disables the MedCPT
  channel. The tie-break selects **`medcpt`**.
* **`expr_mode` vs `auto-blend`** is not a noise problem: the two cells are bit-identical because
  the fitted `w(v)` is 0 at every knot, so `auto-blend`'s extra claim is provably inert and
  `zinb-flow` is the honest label for the same model. Against a genuinely different expression
  path (`cross-mix`) the margin is 7.3x the envelope.
* **`prior_mode` stands on measurement** at 1.8x the envelope, as does `expr_mode` against
  `cross-mix`.

## The config after both tie-breaks

| gate | rank winner | **shipped** | decided by |
|---|---|---|---|
| `layout_mode` | resample | **hybrid** | capability tie-break (margin 0.0344 < 0.04) |
| `prior_mode` | correlated | correlated | measurement (0.0731) |
| `expr_mode` | zinb-flow | zinb-flow | measurement vs `cross-mix`; `auto-blend` inert |
| `text_emb_mode` | lookup | **medcpt** | capability tie-break (margin 0.0110 < 0.04) |
| `train_steps` | 2400 | 2400 | joint gate |
| metric weights | 0.5 | 0.5 | joint gate |

Config hash **`00ef4a19a2f576b8`**.

## What this obliges T10 to do

The definition-of-done arms in `config_selection_synthetic.md` were measured under the *previous*
selected config and at **one seed**. They are superseded twice over, and under the repeated-seed
rule a single-seed replacement would not be admissible either. **T10 must produce them under
`00ef4a19a2f576b8` at `Config.claim_min_seeds` seeds**, reporting min–max beside every median, and
must treat any effect smaller than the campaign's own envelope as a tie rather than a win.
