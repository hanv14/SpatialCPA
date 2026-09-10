# Reversal — "a better latent is worth nothing" was written from one dataset

## What I wrote

`reports/diagnostic_programme_closed.md` §4, and `architecture_ceiling.md` §3:

> the loss is **not** in the prior, the flow, `ell` or SEFL — the flow's latent scores *above* the
> encoder's posterior on the truth

Stated generally, with no dataset attached.

## What is true

It holds on `deep_starmap` and is **false on `starmap_visual_cortex`**, by a margin that is
resolved on every section:

| `4 − A1a` | section_2 | section_4 | section_6 |
|---|---|---|---|
| **tier-1** | **−0.1851** [−0.3546, −0.0733] | **−0.2055** [−0.3891, −0.0642] | **−0.1017** [−0.2334, −0.0074] |
| deep | +0.0578 | +0.0574 | +0.0087 (contains zero) |

On tier-1 stage 4 sits **below** `A1a`, resolved 3/3. A better latent is worth **~0.14 on the
published scale** there — one third of the 0.424 gap.

## Why I am not taking the escape that was available

`ladder_preregistration.md` §2a says a *high* `A1a` rung is **permissive**: it is drawn at the real
section's own cells and stage 4 is not, so its advantage could account for the difference. That
would let the general sentence stand unamended.

It does not survive these runs' own numbers. `flanking_copy` is scored at an **entirely different
section's** cells — a stronger positional mismatch than stage 4's — and still reaches **0.95–0.99**.
If the wrong cell set costs a prediction 0.02–0.05, it cannot explain 0.10–0.21. The rule is sound
in general and does not bind here, and saying so is better than invoking it.

## Why the two datasets disagree

`A1a`'s median `I` against its own tissue:

| | `A1a` | tissue |
|---|---|---|
| tier-1 | 0.41 / 0.32 / 0.25 | 0.46 / 0.39 / 0.41 |
| deep | **0.043 / 0.043 / 0.022** | 0.31 / 0.28 / 0.32 |

The encode→decode round trip is catastrophic on deep and mild on tier-1. On deep the latent's
quality cannot show through a decoder that destroys it either way, so `A1a` lands low and stage 4's
smoother latent scores above it. **"The latent is worth nothing" was a statement about deep's
decoder.**

## What does not change

Closeability. Even granting tier-1 a perfect latent and a perfect mean field, `A1b` = **0.8369**
against a copy at **0.9836**. `diagnostic_programme_closed.md` §3 requires `A1a` to be *below*
stage 4 **and** the low rung before the question reopens; on tier-1 it is above. Checked, not
assumed.

## The pattern, which is the reason this file exists

**This is the third general claim in this campaign written from one dataset.**

1. tier-1's `1.11×` read as fidelity when it was two defects cancelling — corrected once tier-1's
   own `A1a` and 4p arms existed.
2. `chain_shipped_review.md` §6's narrow-`mu` mechanism, generalised from deep, where N2's
   model-free bracket now reads *supported* on deep (2 of 3 sections) and **untested still** on all
   three tier-1 sections.
3. this one.

The common shape: a mechanism is localised on the dataset where it is *visible*, and the sentence
is written without the dataset in it. The defence is not more caution in prose — it is that **a
claim about the model states the dataset it was measured on, and a claim without one is a claim
that both were checked.** Added to `specs/10` §4.2 as a reading rule for the record itself rather
than for an instrument.
