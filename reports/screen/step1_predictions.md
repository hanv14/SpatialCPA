# Step 1 — comparator predictions inventory

Generated 2026-09-08T19:15:04+00:00

## Evaluator pin

```
measured 7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992
pinned   7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992
```
**MATCH** (764 lines)

## `prediction.h5` found

| results root | method | dataset | holdout | size |
|---|---|---|---|---|
| — | — | — | — | **none** |

**total: 0**

## Quarantined `.h5.degraded` — NOT re-scorable


**total: 0**

## `metrics.json` with no `prediction.h5` beside it — re-scoring impossible


**total: 0**

## Verdict

Fork **A2** — no predictions. All four comparators re-run from scratch. Their runtime is
unmeasured in this repo: run 4 methods x the chosen datasets, ONE seed each, timed, and
re-derive the line item before committing the rest.
