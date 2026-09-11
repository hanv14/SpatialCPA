# Test 1 — the field layout with the cell count supplied externally

**Read `reports/reframing_tests_preregistration.md` §1 first.** The bands, the two count
sources and the rule that the shippable one governs were committed before this ran.

This is the **existence test** for the copy-based-field framing's novelty claim, not an
ablation rescue: `_resample_layout` cannot produce a meaningful oblique section (it pastes
a coronal point pattern onto the new plane), so if the field layout cannot be made to work
then no configuration generates one and claim 2 cannot be made at all.

`layout_mode=field`, grid sampler, seed 1, weights `runs/pilot/model_exp_2400.pt`. **Zero fits** — `layout_mode` is fit-invariant.

| reference | `paper_celltype_localization` |
|---|---|
| `field`, count from the broken integral | 0.6607 |
| `resample` (ships) | 0.7546 |
| **`flanking_copy` — the model-free floor** | **0.7765** |
| `oracle` | 0.9808 |

| arm | section_2 | section_4 | section_6 | **median** | spread |
|---|---|---|---|---|---|
| 1b — SHIPPABLE, governs | +0.5371 | +0.1863 | +0.6603 | **+0.5371** | 0.4740 |
| 1a — ORACLE input, upper bound only | +0.5402 | +0.1513 | +0.6663 | **+0.5402** | 0.5150 |

| arm | cells placed / ground truth |
|---|---|
| 1b — SHIPPABLE, governs | section_2: 4165/4187, section_4: 4276/4102, section_6: 4289/4162 |
| 1a — ORACLE input, upper bound only | section_2: 4187/4187, section_4: 4102/4102, section_6: 4162/4162 |

🚩 **The ground-truth-count arm is an ORACLE input.** It is an upper bound and may not be
quoted as the method's score; §1 fixes that the flanking-density arm governs every band.

## **REFUTED**

+0.5371 is below the shipped layout's 0.7546. The count was not the defect. **No configuration generates a meaningful oblique section, and claim 2 cannot be made at all** -- the framing loses its novelty claim and the paper returns to the negative result of reports/diagnostic_programme_closed.md.

⚠️ **One seed, and no across-seed spread exists for this metric on the field arms** —
the r11 arms are one fit and one seed (`envelope_correction.md` §3). Bands are read on raw
deficits and **no multiple-of-envelope may be quoted**, exactly as the A4 row was
corrected to do.
