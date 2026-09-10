# The architecture's ceiling — a measured defect with a size, not a repair route

**The repair programme is closed.** This file is what replaces it: the size of the defect, stated
once, so that nothing downstream reads it as a direction of work.

## 1. The ceiling

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

## 3. A better latent is worth nothing — and this is not a bound, it is a measurement

`A1a` = **+0.6732**, *below* stage 4's **+0.7306**. `A1a` is drawn at the real section's own cells,
which `ladder_preregistration.md` §2a establishes as an advantage stage 4 does not have. A low rung
with the advantage on the wrong side is conclusive in the pre-registration's own terms.

So the flow's sampled latent scores **above** the encoder's posterior on the truth. The generative
mechanism this project is about — GRF prior, flow matching, SEFL — is not where the scored metric
is lost.

**One amendment to the pre-registration's wording, against my own reading.** §3c called `A1a`
"what a perfect latent buys". It is really "what the encoder's posterior on the truth buys through
this decoder", and §2 above shows that decoder is 2.3–2.9× too narrow. The rung is still conclusive
in the direction it fires — a better latent through *this* decoder buys nothing — but it bounds a
narrower claim than the one registered, and the difference is §2's finding rather than a caveat
that weakens it.

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
