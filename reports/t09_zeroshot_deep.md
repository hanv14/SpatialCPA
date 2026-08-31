# The four-arm zero-shot comparison on `deep_starmap`

Seeds [2, 3, 4], folds ['section_3', 'section_5'], arms ['A1', 'A2', 'A3', 'A4']. 72 scored cells. The pre-registration is in `progress/t09_inference_and_calibration.md`; this file applies it and nothing else.

| arm | `text_emb_mode` | `use_distill` | unseen gene's embedding |
|---|---|---|---|
| A1 | `medcpt` | yes | `norm(W t + gamma psi(t))` — the full claim |
| A2 | `medcpt` | no | `norm(W t)` — the designed channel alone |
| A3 | `lookup` | yes | `norm(gamma psi(t))` — **the real competitor** |
| A4 | `lookup` | no | `norm(0)` — one vector per gene; the void condition |

## Scores — `held_out` genes

Per-seed fold means, then the arm's own across-seed spread.

| metric | A1 s2 | A1 s3 | A1 s4 | A2 s2 | A2 s3 | A2 s4 | A3 s2 | A3 s3 | A3 s4 | A4 s2 | A4 s3 | A4 s4 | A1 spread | A2 spread | A3 spread | A4 spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.1209 | +0.1266 | +0.1112 | +0.2518 | +0.2681 | +0.2988 | +0.0693 | +0.0188 | -0.0236 | +0.0070 | -0.0420 | -0.0462 | 0.0153 | 0.0470 | 0.0930 | 0.0532 |
| `gearys_pearson` | +0.0239 | +0.0271 | -0.0096 | +0.1170 | +0.0850 | +0.0432 | -0.0192 | -0.0259 | +0.0689 | +0.0373 | -0.0217 | +0.0490 | 0.0367 | 0.0738 | 0.0949 | 0.0707 |
| `umap_mixing` | +0.1068 | +0.1104 | +0.1131 | +0.1431 | +0.1460 | +0.1592 | +0.1017 | +0.1024 | +0.0954 | +0.1115 | +0.1107 | +0.1131 | 0.0063 | 0.0161 | 0.0070 | 0.0024 |
| `marker_field_r` | +0.0681 | +0.0648 | +0.0774 | +0.0844 | +0.0714 | +0.0807 | +0.0393 | +0.0604 | +0.0925 | +0.0495 | +0.0738 | +0.0630 | 0.0126 | 0.0130 | 0.0533 | 0.0243 |
| `marker_depth_r` | +0.0131 | -0.0219 | +0.1054 | +0.0455 | +0.0399 | +0.0329 | +0.0490 | +0.0170 | +0.0438 | -0.0165 | +0.0134 | +0.0545 | 0.1273 | 0.0125 | 0.0320 | 0.0710 |

### Referents

| metric | constant field | shuffled | usable floor |
|---|---|---|---|
| `morans_pearson` | +0.5199 | +0.0382 | `shuffled` (constant field is float32 round-off on a zero-variance input, not a floor) |
| `gearys_pearson` | -0.4771 | -0.0008 | `shuffled` (constant field is float32 round-off on a zero-variance input, not a floor) |
| `umap_mixing` | +0.0021 | +0.1101 | **none** — shuffled is the arm's own score (`_mixing` reads no coordinates); constant field is a degenerate cloud. No usable floor. |
| `marker_field_r` | +0.1649 | +0.0451 | `constant_field`, `shuffled` |
| `marker_depth_r` | +0.0017 | -0.0212 | `constant_field`, `shuffled` |

## Scores — `kept` genes

Per-seed fold means, then the arm's own across-seed spread.

| metric | A1 s2 | A1 s3 | A1 s4 | A2 s2 | A2 s3 | A2 s4 | A3 s2 | A3 s3 | A3 s4 | A4 s2 | A4 s3 | A4 s4 | A1 spread | A2 spread | A3 spread | A4 spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.5044 | +0.5014 | +0.4912 | +0.5088 | +0.4979 | +0.4929 | +0.6430 | +0.6331 | +0.6200 | +0.6394 | +0.6298 | +0.6239 | 0.0132 | 0.0159 | 0.0230 | 0.0155 |
| `gearys_pearson` | +0.3412 | +0.3237 | +0.3266 | +0.3280 | +0.3358 | +0.3447 | +0.4625 | +0.4233 | +0.4672 | +0.4667 | +0.4316 | +0.4469 | 0.0176 | 0.0167 | 0.0439 | 0.0351 |
| `umap_mixing` | +0.5898 | +0.5880 | +0.5893 | +0.5887 | +0.5886 | +0.5917 | +0.6097 | +0.6083 | +0.6160 | +0.6108 | +0.6094 | +0.6197 | 0.0018 | 0.0031 | 0.0077 | 0.0102 |
| `marker_field_r` | +0.2187 | +0.2207 | +0.2301 | +0.2306 | +0.2314 | +0.2388 | +0.2879 | +0.3129 | +0.2878 | +0.3167 | +0.2985 | +0.2876 | 0.0113 | 0.0082 | 0.0251 | 0.0291 |
| `marker_depth_r` | +0.2877 | +0.3659 | +0.2944 | +0.3052 | +0.3569 | +0.3261 | +0.4098 | +0.3473 | +0.3247 | +0.4394 | +0.3406 | +0.2814 | 0.0781 | 0.0517 | 0.0851 | 0.1580 |

### Referents

| metric | constant field | shuffled | usable floor |
|---|---|---|---|
| `morans_pearson` | +0.1393 | +0.0023 | `shuffled` (constant field is float32 round-off on a zero-variance input, not a floor) |
| `gearys_pearson` | -0.1521 | -0.0051 | `shuffled` (constant field is float32 round-off on a zero-variance input, not a floor) |
| `umap_mixing` | +0.0010 | +0.5890 | **none** — shuffled is the arm's own score (`_mixing` reads no coordinates); constant field is a degenerate cloud. No usable floor. |
| `marker_field_r` | +0.1128 | +0.0651 | `constant_field`, `shuffled` |
| `marker_depth_r` | -0.0140 | -0.0330 | `constant_field`, `shuffled` |

## `A1` - `A3` — PRIMARY — does W.t add anything over a distillation head that sees the text?

| side | metric | s2 | s3 | s4 | mean | envelope | vs it | signs | fold balance | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `held_out` | `morans_pearson` | +0.0516 | +0.1078 | +0.1349 | **+0.0981** | 0.0930 | 1.1x | **disagree** | 0.05 ⚠ | signs disagree |
| `held_out` | `gearys_pearson` | +0.0431 | +0.0530 | -0.0786 | **+0.0058** | 0.0949 | 0.1x | **disagree** | 0.21 ⚠ | signs disagree |
| `held_out` | `umap_mixing` | +0.0051 | +0.0080 | +0.0177 | **+0.0103** | 0.0070 | 1.5x | agree | 0.00 ⚠ | one fold carries it |
| `held_out` | `marker_field_r` | +0.0288 | +0.0044 | -0.0152 | **+0.0060** | 0.0533 | 0.1x | **disagree** | 0.25 ⚠ | signs disagree |
| `held_out` | `marker_depth_r` | -0.0359 | -0.0389 | +0.0616 | **-0.0044** | 0.1273 | 0.0x | **disagree** | 0.22 ⚠ | signs disagree |
| `kept` | `morans_pearson` | -0.1386 | -0.1317 | -0.1288 | **-0.1330** | 0.0230 | 5.8x | agree | 0.83 | **STANDS** |
| `kept` | `gearys_pearson` | -0.1212 | -0.0996 | -0.1405 | **-0.1205** | 0.0439 | 2.7x | agree | 0.41 | **STANDS** |
| `kept` | `umap_mixing` | -0.0199 | -0.0203 | -0.0267 | **-0.0223** | 0.0077 | 2.9x | agree | 0.48 | **STANDS** |
| `kept` | `marker_field_r` | -0.0692 | -0.0922 | -0.0578 | **-0.0731** | 0.0251 | 2.9x | agree | 0.45 | **STANDS** |
| `kept` | `marker_depth_r` | -0.1220 | +0.0186 | -0.0303 | **-0.0446** | 0.0851 | 0.5x | **disagree** | 0.29 | signs disagree |

## `A2` - `A3` — the two routes to an unseen gene, head to head: text alone vs distillation alone. Neither reads the free residual, which is zero for a held-out gene either way

| side | metric | s2 | s3 | s4 | mean | envelope | vs it | signs | fold balance | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `held_out` | `morans_pearson` | +0.1825 | +0.2493 | +0.3225 | **+0.2514** | 0.0930 | 2.7x | agree | 0.02 ⚠ | one fold carries it |
| `held_out` | `gearys_pearson` | +0.1362 | +0.1109 | -0.0257 | **+0.0738** | 0.0949 | 0.8x | **disagree** | 0.07 ⚠ | signs disagree |
| `held_out` | `umap_mixing` | +0.0414 | +0.0436 | +0.0638 | **+0.0496** | 0.0161 | 3.1x | agree | 0.89 | **STANDS** |
| `held_out` | `marker_field_r` | +0.0451 | +0.0110 | -0.0119 | **+0.0148** | 0.0533 | 0.3x | **disagree** | 0.48 | signs disagree |
| `held_out` | `marker_depth_r` | -0.0035 | +0.0229 | -0.0109 | **+0.0028** | 0.0320 | 0.1x | **disagree** | 0.20 ⚠ | signs disagree |
| `kept` | `morans_pearson` | -0.1343 | -0.1352 | -0.1271 | **-0.1322** | 0.0230 | 5.7x | agree | 0.85 | **STANDS** |
| `kept` | `gearys_pearson` | -0.1344 | -0.0876 | -0.1225 | **-0.1148** | 0.0439 | 2.6x | agree | 0.65 | **STANDS** |
| `kept` | `umap_mixing` | -0.0210 | -0.0198 | -0.0243 | **-0.0217** | 0.0077 | 2.8x | agree | 0.41 | **STANDS** |
| `kept` | `marker_field_r` | -0.0573 | -0.0815 | -0.0490 | **-0.0626** | 0.0251 | 2.5x | agree | 0.35 | **STANDS** |
| `kept` | `marker_depth_r` | -0.1045 | +0.0097 | +0.0014 | **-0.0311** | 0.0851 | 0.4x | **disagree** | 0.47 | signs disagree |

## `A2` - `A4` — pure text vs no text at all: the text channel against a gene-blind arm

| side | metric | s2 | s3 | s4 | mean | envelope | vs it | signs | fold balance | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `held_out` | `morans_pearson` | +0.2448 | +0.3100 | +0.3450 | **+0.2999** | 0.0532 | 5.6x | agree | 0.38 | **STANDS** |
| `held_out` | `gearys_pearson` | +0.0797 | +0.1067 | -0.0058 | **+0.0602** | 0.0738 | 0.8x | **disagree** | 0.29 | signs disagree |
| `held_out` | `umap_mixing` | +0.0316 | +0.0353 | +0.0461 | **+0.0377** | 0.0161 | 2.3x | agree | 0.80 | **STANDS** |
| `held_out` | `marker_field_r` | +0.0349 | -0.0024 | +0.0176 | **+0.0167** | 0.0243 | 0.7x | **disagree** | 0.79 | signs disagree |
| `held_out` | `marker_depth_r` | +0.0620 | +0.0265 | -0.0215 | **+0.0223** | 0.0710 | 0.3x | **disagree** | 0.12 ⚠ | signs disagree |
| `kept` | `morans_pearson` | -0.1306 | -0.1319 | -0.1310 | **-0.1312** | 0.0159 | 8.3x | agree | 0.71 | **STANDS** |
| `kept` | `gearys_pearson` | -0.1387 | -0.0958 | -0.1022 | **-0.1122** | 0.0351 | 3.2x | agree | 0.92 | **STANDS** |
| `kept` | `umap_mixing` | -0.0221 | -0.0209 | -0.0280 | **-0.0236** | 0.0102 | 2.3x | agree | 0.34 | **STANDS** |
| `kept` | `marker_field_r` | -0.0860 | -0.0670 | -0.0487 | **-0.0673** | 0.0291 | 2.3x | agree | 0.33 | **STANDS** |
| `kept` | `marker_depth_r` | -0.1342 | +0.0163 | +0.0447 | **-0.0244** | 0.1580 | 0.2x | **disagree** | 0.11 ⚠ | signs disagree |

## `A1` - `A2` — what the distillation head does to the text arm

| side | metric | s2 | s3 | s4 | mean | envelope | vs it | signs | fold balance | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `held_out` | `morans_pearson` | -0.1309 | -0.1415 | -0.1876 | **-0.1533** | 0.0470 | 3.3x | agree | 0.06 ⚠ | one fold carries it |
| `held_out` | `gearys_pearson` | -0.0931 | -0.0580 | -0.0528 | **-0.0680** | 0.0738 | 0.9x | **disagree** | 0.03 ⚠ | signs disagree |
| `held_out` | `umap_mixing` | -0.0363 | -0.0356 | -0.0460 | **-0.0393** | 0.0161 | 2.4x | agree | 0.87 | **STANDS** |
| `held_out` | `marker_field_r` | -0.0163 | -0.0066 | -0.0033 | **-0.0087** | 0.0130 | 0.7x | **disagree** | 0.27 | signs disagree |
| `held_out` | `marker_depth_r` | -0.0324 | -0.0618 | +0.0725 | **-0.0072** | 0.1273 | 0.1x | **disagree** | 0.27 | signs disagree |
| `kept` | `morans_pearson` | -0.0043 | +0.0035 | -0.0017 | **-0.0008** | 0.0159 | 0.1x | **disagree** | 0.21 ⚠ | signs disagree |
| `kept` | `gearys_pearson` | +0.0132 | -0.0121 | -0.0180 | **-0.0057** | 0.0176 | 0.3x | **disagree** | 0.35 | signs disagree |
| `kept` | `umap_mixing` | +0.0011 | -0.0005 | -0.0024 | **-0.0006** | 0.0031 | 0.2x | **disagree** | 0.42 | signs disagree |
| `kept` | `marker_field_r` | -0.0119 | -0.0107 | -0.0088 | **-0.0105** | 0.0113 | 0.9x | **disagree** | 0.11 ⚠ | signs disagree |
| `kept` | `marker_depth_r` | -0.0175 | +0.0089 | -0.0317 | **-0.0134** | 0.0781 | 0.2x | **disagree** | 0.02 ⚠ | signs disagree |

## `A3` - `A4` — the distillation head with no text channel to help it

| side | metric | s2 | s3 | s4 | mean | envelope | vs it | signs | fold balance | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `held_out` | `morans_pearson` | +0.0623 | +0.0608 | +0.0225 | **+0.0485** | 0.0930 | 0.5x | **disagree** | 0.01 ⚠ | signs disagree |
| `held_out` | `gearys_pearson` | -0.0565 | -0.0042 | +0.0200 | **-0.0136** | 0.0949 | 0.1x | **disagree** | 0.23 ⚠ | signs disagree |
| `held_out` | `umap_mixing` | -0.0098 | -0.0083 | -0.0177 | **-0.0119** | 0.0070 | 1.7x | agree | 0.24 ⚠ | one fold carries it |
| `held_out` | `marker_field_r` | -0.0102 | -0.0135 | +0.0295 | **+0.0019** | 0.0533 | 0.0x | **disagree** | 0.08 ⚠ | signs disagree |
| `held_out` | `marker_depth_r` | +0.0655 | +0.0035 | -0.0107 | **+0.0195** | 0.0710 | 0.3x | **disagree** | 0.08 ⚠ | signs disagree |
| `kept` | `morans_pearson` | +0.0037 | +0.0033 | -0.0039 | **+0.0010** | 0.0230 | 0.0x | **disagree** | 0.08 ⚠ | signs disagree |
| `kept` | `gearys_pearson` | -0.0043 | -0.0083 | +0.0203 | **+0.0026** | 0.0439 | 0.1x | **disagree** | 0.13 ⚠ | signs disagree |
| `kept` | `umap_mixing` | -0.0010 | -0.0011 | -0.0037 | **-0.0019** | 0.0102 | 0.2x | **disagree** | 0.02 ⚠ | signs disagree |
| `kept` | `marker_field_r` | -0.0288 | +0.0144 | +0.0002 | **-0.0047** | 0.0291 | 0.2x | **disagree** | 0.12 ⚠ | signs disagree |
| `kept` | `marker_depth_r` | -0.0296 | +0.0066 | +0.0433 | **+0.0068** | 0.1580 | 0.0x | **disagree** | 0.33 | signs disagree |

## The pre-registered verdict

**Primary**: `marker_depth_r` on the `held_out` genes, A1 - A3 = **-0.0044**, envelope 0.1273 (0.0x), signs **disagree**, fold balance 0.22 -> **signs disagree**.

**Against the constant-field band** (+0.0017), read against the shared envelope **0.1273** — the largest across-seed spread in the comparison, arms and referents together (`specs/10` §4.2b). Each arm's own spread is shown, and does not set its threshold:

| arm | mean | over band | own spread | shared envelope | over band / envelope |
|---|---|---|---|---|---|
| A1 (medcpt + distill) | +0.0322 | +0.0305 | 0.1273 | 0.1273 | **0.24x** |
| A3 (lookup + distill) | +0.0366 | +0.0349 | 0.0320 | 0.1273 | **0.27x** |
| A4 (lookup, pure text) | +0.0171 | +0.0155 | 0.0710 | 0.1273 | **0.12x** |

**SUPPORT** requires A1 > A3 with signs agreeing, the margin over the envelope and over the fold spread, and A1 clearing the band by more than the envelope: **not met**.

**REFUTATION of the architecture** (text works, `W.t` is redundant) would be A1 - A3 inside its envelope while *both* clear the band: **not the case**.

**REFUTATION of the idea** (no route from text to an unseen gene) is neither arm clearing the band by more than the envelope: **this is the case**.

**Void condition** — A4 must sit inside the band. A4 is +0.0155 from it against the shared 0.1273 envelope (0.12x): **holds**, no leak detected.

