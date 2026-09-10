# The architecture's ceiling — a measured defect with a size, not a repair route

**The repair programme is closed.** This file is what replaces it: the size of the defect, stated
once, so that nothing downstream reads it as a direction of work.

## 0. The ceiling on the published scale, and where the remainder sits

The chain reproduces bench3's own tier-1 numbers — the copy at **0.9836** against a published
0.9836, v25 at **0.5600** against a published 0.5574 — so the tier-1 ladder is on the scale the
paper reports (`reports/scale_anchoring.md`). Medians over sections 2/4/6:

| rung | tier-1 (published scale) | deep, all 1017 genes |
|---|---|---|
| `flanking_copy` | **0.9836** | 0.9869 |
| `A1c` — Poisson from a perfect mean field | 0.8468 | 0.9699 |
| **`A1b` — perfect mean field, the model's own emission: THE CEILING** | **0.8369** | **0.8242** |
| `A1a` — the encoder's posterior on the truth | 0.6985 | 0.6732 |
| **v25** | **0.5600** | 0.7306 |

**Grant the model a perfect mean field and it reaches 0.837 against a copy at 0.984.** That is the
ceiling, on the scale the paper reports, and it loses. No arrangement of the prior, the flow,
`ell`, SEFL, θ or π gets past it, because `A1b` already grants all of them a perfect result.

### Tier-1 splits the 0.424 gap into near-equal thirds

| | | |
|---|---|---|
| the **latent** | `4 → A1a` | **0.139** |
| the **mean field** | `A1a → A1b` | **0.138** |
| the **irreducible remainder** | `A1b → copy` | **0.147** |

Fixing the first two *perfectly* reaches 0.837 and still loses by 0.147. That is the closure in one
table.

### The two datasets put the irreducible remainder in different places

This is not a caveat; it is the mechanism, and it differs by dataset:

- **deep** — `A1c` (0.9699) sits essentially at the copy, and `A1b` is **0.146** below `A1c`. The
  remainder is the **emission**: a Poisson draw from a perfect mean field reaches the floor and the
  model's own θ/π is what loses it.
- **tier-1** — `A1c` (0.8468) ≈ `A1b` (0.8369), and *both* sit ~0.14 below the copy. Removing the
  emission entirely gains nothing. The remainder is the **kNN mean field's smoothing and the draw
  itself**: a copy carries real cell-level count structure that no draw from a smoothed field
  reproduces.

So "what cannot be reached" has two different causes on the two datasets, and neither is a repair
route: on deep because Q1.5 measured that removing the emission moves the datasets in opposite
directions, and on tier-1 because the thing that cannot be reached is not part of the model.

## 1. The ceiling, per section on deep

`deep_starmap` section_4, all 1017 genes, `paper_morans_pearson` reconstructed by bench3's own
construction. The one scope the corrected null check leaves readable.

| rung | r | what it grants the model |
|---|---|---|
| `A1c` — Poisson(`mu_oracle`) | **+0.9699** | a perfect mean field **and** a perfect emission. Model-free. |
| `A1b` — `mu_oracle`, model θ/π | **+0.8242** | **a perfect mean field, the model's own emission — the architecture's ceiling** |
| stage 4 — where we are | +0.7306 | |
| `A1a` — encoder's posterior latent, full decode | +0.6732 | a perfect *latent*, through this encoder and decoder |
| `A1n` — permutation null | +0.0594 | |

Copy floor `flanking_copy` = 0.9836 on tier-1's 28 genes; its value on this scope is being measured
separately (`reports/flanking_copy_preregistration.md` §2b) and is not assumed here.

**Grant the model a perfect mean field and it reaches 0.8242.** That is the ceiling, and it loses
to a copy. There is no arrangement of the prior, the flow, `ell`, SEFL, θ or π that gets past it,
because `A1b` already grants all of them a perfect result.

## 2. The autoencoder is the largest single term, and it is still inside the ceiling

`A1b - A1a` = **0.151**: the same emission and the same real-derived information, differing only in
whether the mean field is routed through `encoder -> latent -> decoder` or taken from the kNN mean
directly. Its supporting number: `sd(log mu)` decoded from `h1` is **0.4797** against the tissue's
model-free bracket [1.0994, 1.3699] — 2.3–2.9× too narrow, on the **encoder's own posterior on the
truth**.

This is the biggest measured defect in the chain. **It is not a repair route**, for a reason that
does not depend on how good a fix might be: a *perfect* encoder-decoder is `A1b`, and `A1b` is
**0.8242** against a copy floor near 0.98. Fixing it completely still loses to copying. The size is
recorded so the paper can state it; the work is not scheduled.

## 3. A better latent — worth ~0 on deep and ~0.14 on tier-1 ⚠️ REVERSAL

**This section previously read "a better latent is worth nothing", stated generally. It was written
from deep and it is false on tier-1.** The reversal is recorded in
`reports/latent_claim_reversal.md`; the corrected reading is below.

`A1a` sits **below** stage 4 on deep (+0.6732 against +0.7306, resolved on 2 of 3 sections) and
**above** it on tier-1 (+0.6985 against +0.5600; `4 − A1a` = −0.1851, −0.2055, −0.1017, resolved on
**all three**). The claim was written from the first and stated as though it held everywhere.

**What holds.** On deep, a better latent buys nothing: `A1a` is the *low* rung and is drawn at the
real section's own cells, which `ladder_preregistration.md` §2a establishes as an advantage stage 4
does not have. A low rung with the advantage on the wrong side is conclusive there.

**What does not.** On tier-1 the latent is worth **~0.14 on the published scale** — a third of the
0.424 gap (§0). §2a would call `A1a` the *high* rung there and therefore permissive, and I am not
using that escape: these same runs bound the positional advantage it rests on. `flanking_copy` sits
at an **entirely different section's** cells and scores 0.95–0.99, so being at the wrong cells costs
a prediction 0.02–0.05, which cannot account for 0.10–0.21.

**Why the datasets differ.** `A1a`'s median `I` is 0.41 / 0.32 / 0.25 on tier-1 against a tissue at
0.46 / 0.39 / 0.41, but **0.043 / 0.043 / 0.022** on deep against a tissue at 0.31 / 0.28 / 0.32.
The encode→decode round trip is catastrophic on deep and mild on tier-1, so on deep the latent's
quality is invisible behind a decoder that destroys it either way. "The latent is worth nothing" was
a statement about deep's decoder, not about latents.

**One amendment to the pre-registration's wording, against my own reading.** §3c called `A1a` "what
a perfect latent buys". It is really "what the encoder's posterior on the truth buys through this
decoder", and §2 above shows that decoder is 2.3–2.9× too narrow. On deep the rung is still
conclusive in the direction it fires; on tier-1 it fires the other way.

**Closeability is unchanged, and I checked rather than asserted it.** Even granting tier-1 a perfect
latent *and* a perfect mean field, `A1b` = **0.8369** against a copy at **0.9836**. §3 of
`diagnostic_programme_closed.md` requires `A1a` to be *below* stage 4 **and** the low rung to
reopen; on tier-1 it is above. Nothing here meets it.

## 4. What may be said, and what may not

**May be said.** Within this architecture the reachable ceiling on `paper_morans_pearson` is
**0.8242** with a perfect mean field, against a model-free copy near 0.98. The model is at 0.7306
of that ceiling. The remaining 0.09 to its own ceiling is the encoder-decoder pair (0.151 gross,
partly offset by the flow's latent being smoother than the encoder's). None of it is in the prior,
the flow, `ell`, SEFL or θ.

**May not be said.** That any of this is a plan. Each number here is the size of something already
measured, and the three free items and the `flanking_copy` test are the only work in flight.

**Also may not be said**: that these figures transfer to tier-1. Its ladder fired the null check as
written and stays unreadable, its 28 genes may leave it permanently underpowered
(`null_band_preregistration.md` §3), and the two datasets already differ on exactly this axis
(`R1` = 0.68 on deep against 0.29 on tier-1).
