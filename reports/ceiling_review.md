# The emission-free ceiling, and a transform mismatch the ceiling exposed

Sources: `reports/a1_tier1.md`, `reports/a1_deep.md` (stage 4p added). Prediction:
`n5_and_m3_review.md` §7, committed before the run. **No code was changed.**

---

## 0. Headline

The ceiling **bounds the whole emission-repair programme, and it misses on both datasets** — over on
tier-1, under on `deep_starmap`:

| | model now | **emission-free ceiling** | tissue | what an emission repair can do |
|---|---|---|---|---|
| tier-1 | +0.5134 (1.11x) | **+0.8358 (1.80x)** | +0.4635 | **moves it further away** — distance 0.0499 -> 0.3723 |
| `deep_starmap` | +0.1154 (0.37x) | **+0.2594 (0.83x)** | +0.3123 | closes **73.1 %** of the deficit — distance 0.1969 -> 0.0529 |

My prediction was **+0.7775 / +0.4116**. Tier-1 was close and the conclusion is unchanged. **Deep was
wrong, and wrong in the direction that changes the conclusion**: I predicted an overshoot at 1.32x
and it undershoots at 0.83x.

And the tier-1 run exposed an instrument defect by contradicting itself: **stage 4p (+0.8358) exceeds
stage 3 (+0.7920)**. Adding independent noise to a field cannot raise its Moran's I. The two stages
are not on the same transform — every mean-field and latent stage is **raw**, every count stage is
**rank-normalised** — so every retention ratio in the campaign that divides a count stage by a `mu`
or latent stage is not a like-for-like ratio.

---

## 1. The pre-registered reading, applied

§7 fixed both branches in advance:

> *"If it comes back at or above those, no emission repair can reach the tissue without the latent
> being fixed too... If it comes back at or below the tissue on deep, the transfer is invalid and the
> emission repair stands on its own."*

**Tier-1: the first branch.** +0.8358 is above the prediction and far above the tissue. **The coupled
reading of `chain_shipped_review.md` §4a is established rather than inferred** — an emission repair
alone cannot land tier-1 on the tissue, because the emission's noise is what holds the over-smooth
latent down to 1.11x. Remove it and the model goes to 1.80x.

**Deep: the second branch.** +0.2594 is *below* the tissue, so the transfer was invalid and the
emission does not overshoot there. It closes **73.1 %** of the deficit and leaves **17 %** — the
remaining gap belongs to the mean field, not the emission.

Both branches were written before the run and both fired, one per dataset.

---

## 2. 🚩 The transform mismatch, detected by the data before the code

`4p` is `Poisson(mu)` — the same mean field, with independent noise added. Independent noise dilutes
autocorrelation; it cannot create it. So `I(4p) <= I(3)` must hold. On tier-1 it does not:
**+0.8358 against +0.7920**. That is a self-contradiction in one table, and it is what sent me to the
code rather than the other way round.

Confirmed there:

| stage | transform |
|---|---|
| 1 prior `h0`, 2 latent `h`, 3 decoded `mu`, 3c calibrated `mu` | **raw** |
| `REF real latent h1`, `A1a'` (`mu` from `h1`), `A1b'` (`mu_oracle`) | **raw** |
| 4 sampled counts, 4p, 4c, `REF real counts`, every A1 drawn arm | **rank-normalised** |

Rank-normalising a heavy-tailed field generally **raises** its Moran's I, by removing the leverage of
a few extreme cells. `mu` is `exp(shape) * s` — log-normal, heavy-tailed — so the raw figure is the
smaller one, and:

* **every `counts / mu` or `counts / latent` "retention" in the campaign is overstated**, by an
  amount nobody has measured. That includes the "three numbers" table's retention column (64.1 % and
  14.8 % here, and the tissue's 74.1 % and 98.5 %), and the `A1c / A1b'` style ratios in
  `a1_escalation_review.md` §5.
* **`4p > 3` is the visible symptom** and only shows where sparsity is negligible; on deep the drop
  is so large (0.7451 -> 0.2594) that the mismatch is hidden inside it.

**What it does NOT invalidate**, and this is most of the record: every conclusion that compares a
count stage with another count stage. All of these are rank-normalised on both sides —

* the ceiling itself (4p vs `REF real counts`, and 4p vs stage 4);
* every A1 arm and every `R`, which are anchored on `I(model counts)` and `I(real counts)`;
* N5's shares, which are differences among drawn arms;
* the deep 2x2 in §3;
* the cancelling-defects table, whose latent ratio is raw-vs-raw and whose `sd(log mu)` figures are
  not Moran's I at all.

So the headline results stand and the descriptive retention ratios do not.

**The rule this earns**, and it is a new one: *a ratio is only a ratio if both sides were computed
under the same transform.* The stage names even disclosed it — the ranked rows say
"(rank-normalised)" and the raw rows say nothing — but a label is not a guard, and nothing stopped a
number from one row being divided by a number from the other. **The invariant that catches it is
`I(4p) <= I(3)`**, which is free, and it should be asserted at runtime rather than noticed by a
reader.

---

## 3. The deep 2x2, now fully measured — every cell count-vs-count

Tissue: **+0.3123**.

| | model's `mu` (narrow) | `mu_oracle` (tissue-like) |
|---|---|---|
| **with the model's emission** | +0.1154 — 0.37x | +0.1722 — 0.55x |
| **Poisson only** | **+0.2594 — 0.83x** | +0.5007 — 1.60x |

* fix the **emission** alone: 0.37x -> **0.83x** (closes 73.1 %)
* fix **`mu`** alone: 0.37x -> 0.55x (closes 29 %)
* fix **both**: -> 1.60x, an overshoot of 60 %

**The emission is the larger lever on deep, and the two together over-correct.** The `mu_oracle`
column is optimistic — a kNN mean creates autocorrelation, so `I(mu_oracle)` = +0.9063 is inflated —
which makes the overshoot partly an artifact and the left column the trustworthy one.

---

## 4. The narrow-`mu` mechanism, quantified end-to-end for the first time

Compare the two mean fields **through the same Poisson draw**, count-vs-count, no decoder on either
side:

| | model's field | tissue's field | ratio |
|---|---|---|---|
| tier-1 | +0.8358 | +0.9114 | **0.917** |
| `deep_starmap` | +0.2594 | +0.5007 | **0.518** |

**On tier-1 the model's mean field is 92 % as good as the tissue's at surviving the draw; on deep it
is 52 %.** And that tracks `sd(log mu)` against the tissue's model-free bracket exactly:

| | model `sd(log mu)` | tissue bracket | where it sits |
|---|---|---|---|
| tier-1 | 0.6728 | [0.7165, 0.9438] | just below the lower edge |
| `deep_starmap` | 0.6725 | [1.0994, 1.3699] | **far below** |

This is the strongest support the campaign has for the narrow-`mu` mechanism, and unlike §8.3's
withdrawn gate **neither side passes through the decoder**. Caveat: the two mean fields differ in
spatial *pattern* as well as spread (+0.7451 against +0.9063, both raw), so the 0.518 mixes the two;
it is an upper bound on what spread alone costs.

---

## 5. What the ceiling does to §10, and to M3

**It bounds the programme without any redesign.** Even a *perfect* emission repair — every scrap of
`theta` and `pi` noise removed — delivers:

* **tier-1: a regression.** 1.11x -> 1.80x the tissue. A method paper cannot ship that silently.
* **deep: 83 % of the tissue**, a large improvement but not a fix.

**§10 stays suspended**, and the ceiling does not lift it — a bound is not the readable gate answer
the suspension waits on. But it **re-scopes** what a lifted §10 could deliver, and that belongs
beside the costing: *at best 83 % on deep and a regression on tier-1.*

**M3's value has gone UP, not down.** Its question is whether bounding `theta` makes the likelihood
move variance **into `mu`**. §4 now shows the narrow `mu` costs 48 % of the achievable `I` on deep
even with a perfect emission — so an intervention that fixes the emission **and** widens `mu` attacks
both measured defects at once, and M3 is the test of whether one lever does both. It was worth an
hour before; it is worth more now.

**And M3's verdict is unaffected by §2's defect.** `m3_verdict` reads `sd(log mu)` (not Moran's I),
stage 4, `REF real counts`, the distance between them, the `CV²` share and the `paper_*` metrics —
every one of them either count-vs-count or not an `I` ratio at all. The transform fix is not blocking.

---

## 6. What to do next

| # | action | cost | why |
|---|---|---|---|
| **Q1** | **Fix the transform mismatch.** Report every stage under **both** transforms; make the retention table like-for-like; assert `I(4p) <= I(3)` at runtime; flag every `counts / mu` retention already in the record as overstated by an unmeasured amount | free | §2. A ratio across two transforms is not a ratio, and the invariant that catches it costs nothing |
| **Q2** | **Run M3** at floor 2.599, `--role mechanism`, as amended | 1 fit, ~1 h | §5. Unaffected by Q1, and now testing whether one lever fixes both defects |
| **Q3** | **B1** — `ell` refit on `deep_starmap`, alarms escalated to a hard stop | ~3.3 h | deep is where the emission repair helps (73 %), and its fit is the one whose alarm fired at step 2399 |
| **Q4** | **M3 on `deep_starmap`, `--role benefit`**, on B1's checkpoint | 1 fit, ~3.3 h | the only place the benefit question can be asked |

**Q1 before Q2** only because it is free and Q2's report would otherwise print contaminated retention
rows beside a clean verdict. **Q3 before Q4** necessarily.

I would not add anything to §10's redesign on the strength of the ceiling. It is a bound, and the
bound is the result.

---

## 7. Where I was wrong

**The deep prediction, and it is instructive rather than embarrassing.** I transferred `A1c`'s Poisson
retention at `mu_oracle` (0.5525) onto `mu_gen` and predicted +0.4116. The measured retention on the
model's own field is **0.3481**. Retention is **not transferable across mean fields of different
spread** — `mu_gen`'s `sd(log mu)` is 0.6725 against `mu_oracle`'s 1.0994, so the Poisson floor bites
far harder on the model's field. That is the narrow-`mu` mechanism, and **the failed prediction
measured it more directly than a correct one would have**: had the transfer held, §4's 0.518 would
never have been computed.

Part of the error was also §2's: the transfer divided a rank-normalised count stage by a raw `mu`
stage, so the retention factor I carried across was not the quantity I thought it was.

**And the pre-registration was right to carry both branches.** A one-branch prediction would have had
nothing to say when the number came back on the other side.
