# `marker_field_r` — boundary stratification, arm `resample-grid`

> ⚠️ **REGENERATED 2026-09-08, and it supersedes a verdict rather than a measurement.**
> The version this replaces returned **BOUNDARY ELIMINATED** on a gap of **0.69x** an envelope of
> **0.0335** — a constant hard-coded in the generating script, being R10's **pooled** spread over
> six metrics on the **synthetic fixture**, applied to `bench3.evaluate_paper` numbers. Wrong on
> three axes (`specs/10` §4.2a, §4.2a-i). `--envelope` is now required and has no default.
>
> **Every measured value in the JSON beside this file is byte-identical to the superseded one** —
> `deficits`, `gap`, `floor_per_section`, `arm_per_section`, `density_*`, `pooled_median` and both
> uninformative conditions. Only `envelope` (0.0335 → `null`) and `outcome`
> (`boundary_eliminated` → `not_readable`) changed, and neither is a measurement. Derivation:
> `reports/envelope_correction.md` §3.

Deficit below **each section's own** `flanking_copy` floor, on the density-matched scores. Criteria pre-registered in `progress/t09_inference_and_calibration.md` (2026-09-07) before this script existed.

| section | regime | `flanking_copy` | arm | deficit |
|---|---|---|---|---|
| `section_2` | **boundary** (one-sided evidence) | 0.8470 | 0.6742 | **0.1729** |
| `section_4` | interior | 0.8857 | 0.6980 | **0.1877** |
| `section_6` | interior | 0.8873 | 0.6830 | **0.2043** |

Boundary deficit **0.1729**, interior mean **0.1960**, gap **-0.0231**. 🚩 **No envelope is quoted, and none is substituted** — see the verdict below.

## Verdict: **NOT READABLE**

NO VERDICT. Every branch of the pre-registered criterion compares the gap against an envelope, and none has been measured for this metric on this arm, dataset and instrument - so the criterion cannot fail and must not be scored as passed (`specs/10` §4.2j). The raw deficits are reported and are what any reading must rest on. See `reports/envelope_correction.md` §3.

⚠️ **This changes no verdict and makes `marker_field_r` no less a weakness.** Three sections, one seed, one dataset; `claim_min_seeds = 3` is about seeds and this has one. It is a pointer for whoever picks the metric up.

## Uninformative conditions

* **(b) NOT EVALUABLE** — no prior median is recorded for `resample` on the `grid` sampler (`reports/t10_rescore_exp.json` predates it). That is not the same as passing: nothing confirms these are the scores an earlier run produced, because there is no earlier run on this sampler.
* 🚨 **(c) FIRES as pre-registered** — matched density is not achieved at `section_2`, `section_6` (emitted below the ground-truth count, and subsampling cannot add cells). Per-section ratios: `section_2` 0.97x, `section_4` 1.02x, `section_6` 0.99x.
  * ⚠️ **Magnitude, POST-HOC**: the across-section density spread is **1.04x**, below the 2.0x cut — small enough that it cannot carry a difference between sections. This cut is a judgement made after seeing the numbers; the pre-registered condition was binary and fires above regardless.

