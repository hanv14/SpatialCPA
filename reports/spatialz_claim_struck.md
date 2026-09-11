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

---

## 6. AMENDMENT 2026-09-11 — the comparator run happened; one of the two objections is resolved

`evaluate_all --force` was run into `benchmark-pbya-v3/results_rescored/` for
`starmap_visual_cortex/paper_2_4_6`, **all six methods, 0 failures, on the pinned evaluator**
(`evaluate_paper.py`, sha256 `7362669…538992`). The tier-1 six-metric table now exists. §1 of this
file — *"it cannot be located in the record"* — **is resolved and is hereby marked superseded.**
`reports/pilot.md` §44 and `specs/10` step 5 were true when written and are stale on this point.

**§2's conditions 1 and 2 are also resolved:** the run is on the pinned instrument (not
cross-instrument), and `flanking_copy` / `oracle` from the probes tree supply the floor and ceiling
the published rows lacked.

### §3 is NOT resolved, and it was always the reason for the strike

The run supports the arithmetic. On the five metrics that have a probe, `spatialcpav20_gen` exceeds
SpatialZ on four and loses `celltype_localization` — the shape the struck claim asserted.

**And v20 sits on the copy floor**, which is what §3 said made the claim circular:

| metric | v20 | `flanking_copy` | v20 − floor |
|---|---|---|---|
| `celltype_localization` | 0.7766 | 0.7765 | **+0.0001** |
| `morans_pearson` | 0.9811 | 0.9836 | −0.0025 |
| `gearys_pearson` | 0.9815 | 0.9840 | −0.0025 |
| `marker_field_r` | 0.8707 | 0.8857 | −0.0150 |
| `marker_depth_r` | 0.8963 | 0.9794 | −0.0831 |

**One part in 7 765 on the metric the layout head exists to win.** R13's finding — `cross-mix` under
`resample` *is* a copy, matching a model-free nearest-section copier to 0.001 on `deep_starmap` — now
holds on a second dataset, on the pinned instrument, to four decimals.

**So the claim stays struck, on §3 alone.** Quoting v20's win as a method's competitiveness would be
quoting a model-free probe's score as a result of the model, which `architecture_ceiling.md` exists
to forbid. §4's restatement stands unchanged and is what the paper says.

**What the run licenses instead**, and what §8.2 of the paper states: *on four of the five readable
tier-1 metrics, **no** method in the table — SpatialZ, FEAST, isoST, v18, v20, v21 or v25 — reaches
a model-free copy of the flanking sections.* That is a statement about the benchmark, it generalises
§6.1 past our own method, and it needs no claim about who beats whom.

**Getting a number does not unstrike a claim.** The evidence objection is gone and the reasoning
objection is untouched, and it was the reasoning objection that was load-bearing. This is recorded
here because the natural move on receiving the long-awaited table was to reinstate the sentence it
appears to support.
