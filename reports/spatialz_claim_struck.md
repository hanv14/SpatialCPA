# STRUCK — "v20 beats SpatialZ 5 of 6"

> **The reason for the strike is §3: under the copy-based framing the claim is CIRCULAR.** It is not
> merely unsupported. Even if the comparator run returned exactly this number, quoting it as the
> method's competitiveness would be claiming the copy floor as a result of the method — which is the
> move `reports/architecture_ceiling.md` exists to forbid. §4's restatement is **adopted** as the
> claim the paper makes instead.

**The claim is withdrawn in full, not carried with a caveat.** It has been quoted since early in the
campaign and it cannot be located in the record.

## 1. It is not in the repository

Searched: `PROGRESS.md`, `progress/`, `reports/`, `specs/`, every `reports/*.json`. **No "5 of 6"
result against SpatialZ exists.** The nearest real figures, and what each actually is:

| what the record holds | what it is |
|---|---|
| tier-1: v20 **0.8804**, v21 **0.8881** vs SpatialZ **0.8522** | **one metric** (`marker_field_r`), not five of six |
| wide regime: v20 wins **7 of 7** | a different holdout design, whose numbers `specs/10` §1 says are not comparable to tier-1's |
| per dataset: **9–9** | the corrected count, after the pooled version was withdrawn |

If "5 of 6" exists only in `per_section_metrics.csv`, that file is the **cross-dataset average**
`specs/10` §4.2a forbids by name, and which this record already withdrew once
(`advisor_report.md`, `progress/numbers.md`, `t09_closeout.md` §493, all 2026-09-07).

## 2. Three conditions make even the real figures unusable, and they are already on the record

`reports/v20_v21_evidence_audit.md` §3 established all three before this question was asked:

1. **No floor, no ceiling.** §13 quotes neither `flanking_copy` nor `oracle`. By this project's own
   §2 rule a score means nothing until it sits between them, so *"v20 0.9704 beats SpatialZ
   0.9267"* is **unreadable**, not merely uncorroborated.
2. **Cross-instrument.** §13's numbers come from old evaluator revisions; today's probes come from
   the pinned one. Putting them side by side is the comparison §13.1 forbids. Only a re-score makes
   it legitimate, and `results_rescored/` is not in this repository.
3. **v20 is mechanically a copier.** R13 established that `cross-mix` under `resample` **is** a copy
   — matching a model-free "copy the nearest other section" to **0.001** on `deep_starmap`.

## 3. Why (3) matters most under the new framing, and it is not a detail

The reframing says: *expression comes from cross-mix, because the measurement says copying is
better.* Claim 1 then says: *and reconstruction is competitive, because cross-mix beats SpatialZ.*

**Those two together are circular.** The competitiveness would be the copy's, and this paper's own
ceiling analysis is the argument that a copy is near the achievable maximum. Quoting it as the
method's competitiveness claims the copy floor as a result of the method — the exact move the
ceiling analysis exists to forbid.

## 4. The restatement — ADOPTED 2026-09-11 as the paper's claim 1

> Reconstruction quality is **at the copy level**, and this paper shows the copy level is near the
> achievable maximum on this protocol. The contribution is not exceeding it — it is reaching it at
> **arbitrary orientations**, with **coherent volumes**, which copying cannot do.

**Why it is stronger and not a retreat.** The struck version competed on an axis where a copy
already wins, using a copy's own score. The restatement competes on an axis **no copy can reach at
all** — a copy has no definition at an arbitrary orientation — and it uses the ceiling result as
support rather than being embarrassed by it. It concedes the comparison and moves the claim onto the
axis where the method is actually different. It also makes the comparator run (`specs/10` step 5) a **corroboration** rather than a
load-bearing dependency: it would confirm that everyone is near the copy floor, which is the
methodological finding, rather than being needed to show we beat anyone.

## 5. What is struck, and where

Everywhere: any statement that v20, v21 or v25 **beats SpatialZ** on a count of metrics. The two
comparisons that may still be quoted, with their qualifiers attached and never as a headline:

- tier-1 `marker_field_r`, one metric, **no floor column** — so quotable only as "unreadable pending
  a re-score against `flanking_copy`";
- the wide-regime 7 of 7, **different design**, not comparable to tier-1.

Nothing in the campaign's conclusions depends on this claim. The ceiling result, the copy-floor
result and the panel-regime result are all measured against **model-free probes on the pinned
instrument**, and none cites SpatialZ.
