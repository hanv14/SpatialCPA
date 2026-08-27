"""What the per-gene decomposition establishes *besides* the mechanism it was built to test.

`t09_depth_mechanism.py` asks one question — does the gain concentrate on genes whose text
neighbours carry strong depth gradients — and answers it with a partial correlation. But it
writes out every per-gene term, and those terms answer three further questions that the two-fold
margin cannot reach and that the diagnostic's own table does not display:

1. **Is the advantage broad or carried by a few genes?** A sign test and a Wilcoxon signed-rank
   over the per-gene gains. Caveat stated in the output: per-gene gains within one fold share a
   model and a section, so they are not independent and these p-values are optimistic. They are
   reported because 32 dependent genes still constrain far more than 2 folds, not because they
   are exact.
2. **Is *which* gene benefits reproducible?** Correlating the per-gene gain between folds. This
   is the question the fold-spread column raised and could not answer: a margin that is small
   relative to its own fold spread might still rest on a stable per-gene pattern, or might not.
3. **Do the two gates help the same genes?** Reported **with** the reason it is inflated: the two
   comparisons share their winning arm (``zinb-flow`` under ``prior_mode=correlated`` and
   ``medcpt`` under ``expr_mode=zinb-flow`` are the *same fitted config*), so both gains contain
   the same ``+r(shared arm)`` term and correlate by construction. The number is printed as a
   caution, not as a finding.

Reads the diagnostic's ``.json``. No model, no data, no refit — seconds, not minutes.

Usage::

    python scripts/t09_depth_mechanism_summary.py reports/t09_depth_mechanism_deep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, pearsonr, spearmanr, wilcoxon


def per_gene(row) -> dict:
    """``{gene: gain}`` for one row of the diagnostic's JSON."""
    return {g["gene"]: float(g["gain"]) for g in row.get("per_gene", [])}


def breadth(row) -> dict:
    """Is the gain broad across genes, or carried by a few? Sign test + signed-rank."""
    gain = np.array([g["gain"] for g in row["per_gene"]], dtype=np.float64)
    won = np.array([g["r_won"] for g in row["per_gene"]], dtype=np.float64)
    lost = np.array([g["r_lost"] for g in row["per_gene"]], dtype=np.float64)
    up = int((gain > 0).sum())
    return {
        "gate": row["gate"],
        "fold": row["fold"],
        "n": int(gain.size),
        "mean_gain": float(gain.mean()),
        "median_gain": float(np.median(gain)),
        "n_improved": up,
        "sign_p": float(binomtest(up, int(gain.size), 0.5).pvalue),
        "wilcoxon_p": float(wilcoxon(gain).pvalue) if np.any(gain != 0) else float("nan"),
        "median_r_won": float(np.median(won)),
        "median_r_lost": float(np.median(lost)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("json_path", help="the .json written by scripts/t09_depth_mechanism.py")
    ap.add_argument("--out", default=None, help="append the markdown here (default: stdout only)")
    args = ap.parse_args(argv)

    rows = json.loads(Path(args.json_path).read_text())
    folds = [r for r in rows if not str(r["fold"]).startswith("pooled") and r.get("per_gene")]
    if not folds:
        raise SystemExit(f"{args.json_path} carries no per-fold rows with per_gene terms")
    dataset = folds[0].get("dataset")
    gates: list[str] = []
    for r in folds:
        if r["gate"] not in gates:
            gates.append(r["gate"])

    lines = [
        f"## Per-gene decomposition — what it establishes on `{dataset}`",
        "",
        "The diagnostic's own table answers the mechanism question. These are the three things "
        "its per-gene terms answer that the 2-fold margin cannot.",
        "",
        "### 1. Is the advantage broad, or carried by a few genes?",
        "",
        "| gate | fold | genes | mean gain | median gain | improved | sign p | Wilcoxon p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in folds:
        b = breadth(r)
        lines.append(
            f"| `{b['gate']}` | `{b['fold']}` | {b['n']} | {b['mean_gain']:+.4f} | "
            f"{b['median_gain']:+.4f} | {b['n_improved']}/{b['n']} | {b['sign_p']:.1e} | "
            f"{b['wilcoxon_p']:.1e} |"
        )
    lines += [
        "",
        "**Per-gene gains within one fold are not independent** — one fitted model, one section, "
        "one marker set — so these p-values are optimistic and are not a substitute for a "
        "repeated-seed run. What they do establish is that the advantage is not carried by a "
        "handful of genes, which a mean over 2 folds cannot show either way.",
        "",
        "### 2. Is *which* gene benefits reproducible across folds?",
        "",
        "| gate | shared genes | Pearson | p | Spearman | p | improved in both |",
        "|---|---|---|---|---|---|---|",
    ]
    repro = {}
    for gate in gates:
        pair = [r for r in folds if r["gate"] == gate]
        if len(pair) != 2:
            continue
        a, b = (per_gene(r) for r in pair)
        shared = sorted(set(a) & set(b))
        xa = np.array([a[k] for k in shared])
        xb = np.array([b[k] for k in shared])
        pr, pp = pearsonr(xa, xb)
        sr = spearmanr(xa, xb)
        both = int(((xa > 0) & (xb > 0)).sum())
        repro[gate] = (float(pr), float(pp))
        lines.append(
            f"| `{gate}` | {len(shared)} | {pr:+.3f} | {pp:.3f} | {sr.statistic:+.3f} | "
            f"{sr.pvalue:.3f} | {both}/{len(shared)} |"
        )
    lines += [
        "",
        "This is the question the within-arm fold-spread column raised and could not answer. The "
        "two folds are held-out sections of one volume scored against the *same* pair of fitted "
        "models, so a correlation here says the per-gene pattern is a stable property of those "
        "models on unseen sections — **not** that it survives a new seed.",
        "",
        "### 3. Do the two gates help the same genes? (read with the caveat)",
        "",
        "| fold | shared genes | Pearson | p |",
        "|---|---|---|---|",
    ]
    if len(gates) == 2:
        for fold_id in [r["fold"] for r in folds if r["gate"] == gates[0]]:
            pair = [r for r in folds if r["fold"] == fold_id]
            if len(pair) != 2:
                continue
            a, b = (per_gene(r) for r in pair)
            shared = sorted(set(a) & set(b))
            xa = np.array([a[k] for k in shared])
            xb = np.array([b[k] for k in shared])
            pr, pp = pearsonr(xa, xb)
            lines.append(f"| `{fold_id}` | {len(shared)} | {pr:+.3f} | {pp:.4f} |")
    lines += [
        "",
        "**Inflated by construction, so not a finding.** The two comparisons share their winning "
        "arm — the same fitted config serves as `zinb-flow` for one gate and `medcpt` for the "
        "other — so both gains carry the same `+r(shared arm)` term and would correlate even if "
        "the two losing arms were unrelated. Printed as a caution against reading the two gates "
        "as independent corroboration of each other.",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        path = Path(args.out)
        existing = path.read_text().rstrip("\n") + "\n\n" if path.exists() else ""
        path.write_text(existing + text + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
