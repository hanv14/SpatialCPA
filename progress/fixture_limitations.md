# Limitations of the synthetic fixture — what it cannot tell us

**Every acceptance number from T05 to T09 was measured on `tests/fixtures/synthetic`.** That was the
right call — Convention 7 requires tests that run without a GPU, real data or network, and the
fixture is the only dataset with a known generative law, which is what makes an *achievable
ceiling* measurable at all. But the T10 pilot ran the same criteria against real STARmap and one of
them inverts. This file records where the fixture is known not to stand in for real data, so a
future task does not re-derive it the hard way.

Kept as its own file rather than folded into a task log because it is a property of the *fixture*,
cited by several tasks, and it will grow.

---

## 1. The emission regime: a 28-gene near-dense panel (found at T10)

**T06's `test_mean_variance_relation` passes on the fixture at 0.84 % and fails on real STARmap at
28.4 %, against the same < 15 % threshold.**

| | log-log mean-variance slope | relative error | verdict at T06's < 15 % |
|---|---|---|---|
| fixture, generated vs real | 1.7556 vs 1.7410 | **0.84 %** | pass (`progress/t06_expression_head.md`) |
| **real STARmap `section_2`, generated vs real** | **2.120 vs 1.738** | **22.0 %** | **fail** |

And the consequence is not a marginal metric — it is where the pilot's spatial-structure collapse
happens. The generation chain holds structure through the prior (Moran's I 0.9714), the flow
(0.9015) and the decoded `mu` (0.8607), then loses it at the count draw (0.1297). Real tissue
retains **62 %** across the same latent→counts step; the model retains **14 %**
(`reports/pilot.md` §8).

**Why the fixture cannot see this.** Its panel is 200 genes with deliberate sparse and dense
classes, drawn from a law the decoder family matches. STARmap's paper panel is **28 curated genes
with a median per-gene detection rate of 0.9999** — almost no zeros, and a per-gene mean in the
thousands. The ZINB's two failure modes on such a panel (zero-inflation with nothing to inflate,
and a dispersion that has to carry the whole spread) are both invisible on the fixture, because the
fixture's genes have real zeros and modest means.

**What this invalidates, and what it does not.** It does not invalidate T06's implementation or its
test — the test measures what it says and passes on the data it was given. It invalidates the
**inference** that a criterion passing on the fixture will pass on real data *in the emission
regime*. Three specific consequences:

* **T09's `calibrate_detection` has never been exercised where it has headroom.** It ships fitted
  but **not applied** because on the fixture the model error (0.0217) was below the real
  between-section variation (0.0397) — no headroom (SPEC_QUESTIONS C28). On real STARmap the same
  quantity has 22 % of headroom. A calibrator's default was set from a dataset that could not test
  it.
* **T06's detection-rate band is unusually tight here.** `assert_detection_rate` compares against a
  reference of ~1.0 on this panel, so a failure on STARmap says more about the training budget than
  about the decoder.
* **Any future "passes on the fixture" claim about the emission path needs a real-data check**
  before it is relied on. The distributional statistics — detection rate, mean-variance relation,
  dispersion — are the ones at risk. The *spatial* statistics transferred fine: the GRF prior and
  the flow both hold up on real data (0.97 and 0.90).

## 2. The layout regime (found at T10, R11)

The fixture ranked `layout_mode` **inside the reproducibility envelope** — `resample` won on rank
(3.0) against `hybrid` (4.2), `field`'s best cell ranked 7.0, and the margin (0.0344) sat against a
0.0335 envelope, so `hybrid` shipped on a within-noise tie-break. On real STARmap the same
preference is far outside the envelope: `celltype_localization` **0.4252** for `field` against
**0.7008** for `resample`, where the model-free `flanking_copy` floor is **0.7765**.

The fixture signal was not wrong — it was **underpowered**, and it pointed the right way. The
lesson is the same as §1's: a gate decided inside the envelope on the fixture is undecided, not
decided, and it should be revisited when real-data evidence exists rather than inherited.

## 3. Geometry the fixture does not have (found at T10)

* **Coincident coordinates.** bench3 flattens each multi-plane slab to its centre z, so 0.49 % of
  cells share an `(x, y)` with a section-mate. The fixture's hard-core point process cannot produce
  this, so T01's duplicate-coordinate check had never met it (`specs/10` §0).
* **Panel width.** `Config.expr_pca_dim = 32` exceeds STARmap's 28 genes and the fit is refused. No
  fixture configuration has fewer genes than that default.
* **Thin slabs.** The fixture is 1000 µm square with 9 sections; STARmap is ~1545 x 1545 x 77 µm.
  Re-sectioning orthogonally is degenerate on real data and impossible to notice on the fixture
  (`specs/10` §9).
