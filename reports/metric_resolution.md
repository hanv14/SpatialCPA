# What `paper_celltype_localization` can and cannot resolve

*Volunteered. It qualifies **every** localisation number this campaign has produced, not the oblique
rows, and a reviewer computing it themselves would be the worst outcome. Derived from the
evaluator's own constants, which are pinned and content-hashed.*

## The arithmetic

`bench3/evaluate_paper.py::celltype_localization` normalises coordinates by the tissue radius, then
scores each type by a debiased Sinkhorn divergence against a within-tissue null:

```python
centre = gt_xy.mean(axis=0)
radius = float(np.sqrt(((gt_xy - centre) ** 2).sum(axis=1)).mean())
G_all  = (gt_xy - centre) / radius
ref    = G_all[rng.choice(len(G_all), min(len(G_all), 400), replace=False)]
scale  = float(np.median(cdist(ref, ref, metric="sqeuclidean"))) or 1.0
...
K = np.exp(-C / eps) + 1e-300          #  C = sqeuclidean / scale,  eps = 0.05
```

So the transport kernel is `exp(−d²/(eps·scale))` in normalised units — a Gaussian whose
length-scale in micrometres is

```
    blur  =  radius · √(eps · scale)          (σ = blur / √2)
```

Everything on the right is the evaluator's: `eps = 0.05` is its default, `scale` is the median
squared inter-cell distance it computes, `radius` the mean distance from the centroid it computes.
**There is no free parameter here and nothing of ours.**

## What it comes to on real tissue

On `merfish_thick_hypothalamus`, `scale ≈ 1.4` and `radius ≈ 400–450 µm`, giving

> ### blur ≈ **105–120 µm**, against a tissue radius of ≈ 400 µm

**The statistic is blind to spatial structure below roughly a quarter of the tissue radius.** It
measures whether a cell type is in the right *region*. It does not measure whether it is in the
right place within that region.

## What follows, stated plainly

**1. It applies to every localisation number in this campaign.** The `flanking_copy` floor of
0.7765, v25's 0.5371, the `both_oracle` ceiling of 0.8375, every arm of the `test1b` split, every
per-section row — all of them were computed under a ~110 µm blur. None of them is evidence about
placement finer than that.

**2. It weakens a comparison we have leaned on.** `test1b` reported a 31× gap in across-section
spread between model positions and copy positions, and we read that as the model's layout being
wildly section-dependent. That reading survives — it is about *gross* placement, which is what the
metric sees — but it must be stated as such, and "the model places cells badly" must not be read as
a statement about fine structure. It is not one and this metric cannot make one.

**3. It is why the copy floor is so hard to beat.** A copy reproduces gross regional structure
exactly, which is precisely and only what the statistic rewards. The architecture ceiling we
measured (A1b) is a ceiling on reproducing structure at ≳ 110 µm.

**4. It cuts the other way on the comb.** `reports/the_comb_limit.md` shows an oblique evaluation
set is a comb. Convolving that comb with this kernel — closed form, `2|sin(πf)|/(πf)` times
`exp(−2π²σ²/p²)` — leaves a residual modulation of **1.7 × 10⁻⁴ at 30°, falling to 3 × 10⁻¹² at
60°** (it falls with angle because the strata crowd together). So the comb, which is a real limit on
the field, does **not** damage this particular statistic, and the pre-registration's F1 rule, which
would have excluded a continuous-fill arm on those grounds, is withdrawn as unnecessary. The comb
limit and the resolution limit are both real and they do not compound: one bounds what the data can
contain, the other what the statistic can distinguish, and here the second is the looser.

**5. It would bite a grid metric hard.** `paper_marker_field_r` and `paper_marker_field_ssim` bin on
a `FIELD_GRID = 20` square grid; `specs/10` §9 already warns that an oblique strip makes those
degenerate. A comb the Sinkhorn kernel cannot see is a comb a 20-bin grid resolves directly. Any
oblique result on a grid-based metric needs this analysis redone, not inherited.

## The honest summary for the paper

Two bounds, both on the field rather than on any method, both stated with their arithmetic:

| bound | what it limits | value here |
|---|---|---|
| **the comb limit** — `fill = t·cos θ / s` | what an oblique ground truth from serial sections *can contain* | 0.43 at 30°, 0.25 at 60°, **0** at 90° |
| **the metric's resolution** — `radius·√(eps·scale)` | what the statistic can *distinguish* | ≈ **110 µm**, a quarter of the tissue radius |

They are independent, they are both computable from published constants and a coordinate file, and
we have not found either stated in the literature. Reporting a localisation score without them is
reporting a number whose resolution is unknown.
