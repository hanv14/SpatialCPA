"""Find every persisted selection artifact and read the metric-aware weights out of it.

**The question this settles.** The record says `w_autocorr = w_profile = w_distribution = 0.5`
"ship". `Config` declares them **0.0**, `scripts/_starmap_run.py::base_config` never overrides them,
and `t09_ship_starmap.py --w-metric-aware` defaults to unset — so every real-data fit whose artifact
is in the repository ran at **0.0** except A9's deliberately-overridden `05` arm
(`reports/advisor_report.md` §6a). Two readings remain, and they are different failures:

**(a) A `selected.yaml` carrying 0.5 exists on the campaign machine.** Then 0.5 is a real persisted
selection that the campaign runs simply never resolved — `specs/10` §10.1's `--require-config` path
would have read it, and nothing used it. A wiring gap.

**(b) No such file exists anywhere.** Then 0.5 entered the record from a **selection report** —
`write_selection_report`'s table, or the printed rank — and was written into `Config`'s docstrings,
`specs/10`, the close-out and this project's own `--w-metric-aware` help text as "shipped" **without
ever being persisted or applied**. That is a value that was chosen, recorded as shipped, and never
ran: a provenance failure mode distinct from every one in §8, because nothing is missing and nothing
is mislabelled — the number is simply not connected to anything that executes.

This script cannot decide which; it can only look. It reads, and never writes.

`specs/10` §10.1 gives the layout it searches::

    $SPATIALCPAV25_SELECT_DIR/        (default: runs/select/)
      <dataset_id>/
        selected.yaml          the chosen Config + provenance
        scores.csv             the ScoreCache
        selection_report.md    write_selection_report's table

Usage::

    python scripts/t09_find_selection.py                      # the default root, plus the CWD
    python scripts/t09_find_selection.py --root /data/han/projects/Spatial3D --root ~/runs
    python scripts/t09_find_selection.py --self-check         # exercise the readers, no filesystem

Report every root you searched when reporting a negative: `specs/10` §4.2* — *a negative result is
only as broad as the corpus it searched*, and "not in these roots" and "does not exist" are
different claims.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

WEIGHTS = ("w_autocorr", "w_profile", "w_distribution")
DEFAULT_ROOTS = ("runs/select", "runs", ".")
NAMES = ("selected.yaml", "selection_report.md", "scores.csv")

# ``w_autocorr: 0.5`` / ``w_autocorr = 0.5`` / ``"w_autocorr": 0.5`` — the fallback when PyYAML is
# absent, and the reader for the markdown report and the CSV header alike.
FIELD = re.compile(
    r"""["']?(?P<key>w_autocorr|w_profile|w_distribution)["']?\s*[:=]\s*["']?(?P<val>-?\d+(?:\.\d+)?)""",
    re.IGNORECASE,
)


class SelectionScanError(RuntimeError):
    """A root was named that cannot be searched."""


def read_yaml(text: str) -> dict[str, Any]:
    """Parse ``selected.yaml``, falling back to a regex when PyYAML is unavailable.

    The fallback exists because this script's whole job is to run somewhere else — on the campaign
    machine, possibly outside the project env — and refusing to look because an import failed would
    be the wrong answer to "does this file exist".
    """
    try:
        import yaml
    except ImportError:
        return {m["key"].lower(): float(m["val"]) for m in FIELD.finditer(text)}
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        return {}
    flat: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk(value)
                else:
                    flat.setdefault(str(key), value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(loaded)
    return flat


def weights_in(text: str) -> dict[str, float]:
    """Return whichever of the three weights the text names, by any of the three spellings."""
    return {m["key"].lower(): float(m["val"]) for m in FIELD.finditer(text)}


def describe_file(path: Path) -> dict[str, Any]:
    """Return what one selection artifact says about the three weights and the arm."""
    out: dict[str, Any] = {"path": str(path), "kind": path.name, "weights": {}, "gates": {}}
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:  # unreadable is not absent, and the difference matters
        out["error"] = f"unreadable: {exc}"
        return out

    if path.name == "selected.yaml":
        flat = read_yaml(text)
        out["weights"] = {k: float(flat[k]) for k in WEIGHTS if isinstance(flat.get(k), (int, float))}
        if not out["weights"]:
            out["weights"] = weights_in(text)
        for gate in ("dataset_id", "text_emb_mode", "expr_pca_dim", "layout_mode", "train_steps",
                     "content_hash", "config_hash", "seed"):
            if gate in flat:
                out["gates"][gate] = flat[gate]
    elif path.name == "scores.csv":
        try:
            rows = list(csv.DictReader(text.splitlines()))
        except csv.Error as exc:
            out["error"] = f"unparseable csv: {exc}"
            return out
        out["gates"]["n_rows"] = len(rows)
        seen: dict[str, set[str]] = {w: set() for w in WEIGHTS}
        for row in rows:
            for w in WEIGHTS:
                if row.get(w) not in (None, ""):
                    seen[w].add(str(row[w]))
        out["weights_seen"] = {w: sorted(v) for w, v in seen.items() if v}
        if not out["weights_seen"]:
            out["weights_seen"] = {"(header carries no weight column)": []}
    else:  # selection_report.md
        out["weights"] = weights_in(text)
        out["gates"]["mentions_0.5"] = bool(re.search(r"\b0\.5\b", text))
    return out


def scan(roots: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (findings, roots actually searched). A root that does not exist is reported, not skipped silently."""
    findings: list[dict[str, Any]] = []
    searched: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            searched.append(f"{root}  (does not exist)")
            continue
        searched.append(str(root.resolve()))
        for name in NAMES:
            for hit in root.rglob(name):
                resolved = hit.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                findings.append(describe_file(hit))
    return findings, searched


def render(findings: list[dict[str, Any]], searched: list[str]) -> str:
    lines = ["# Persisted selection artifacts — do any carry the metric-aware weights at 0.5?", ""]
    lines.append("## Roots searched")
    lines += [f"  {r}" for r in searched]
    lines.append("")

    if not findings:
        lines += [
            "## Result: **NONE FOUND**",
            "",
            "No `selected.yaml`, `selection_report.md` or `scores.csv` under any root above.",
            "",
            "⚠️ **This is branch (b) only if the roots above are the right ones.** A negative result",
            "is as broad as the corpus it searched (`specs/10` §4.2*). Re-run with `--root` pointing",
            "at the campaign machine's `$SPATIALCPAV25_SELECT_DIR` and at any archived run tree",
            "before concluding the file does not exist — the same mistake the reflog episode cost a",
            "round trip on (`reports/advisor_report.md` §8c).",
            "",
            "If those roots are complete, the reading is: **0.5 was chosen, written into the record",
            "as shipped, and never persisted or applied** — a value that never ran.",
        ]
        return "\n".join(lines)

    lines.append(f"## Result: **{len(findings)} artifact(s) found**")
    lines.append("")
    carriers = []
    for f in findings:
        lines.append(f"### `{f['path']}`")
        if "error" in f:
            lines.append(f"  ⚠️ {f['error']}")
            lines.append("")
            continue
        weights = f.get("weights") or {}
        if weights:
            shown = ", ".join(f"{k}={v}" for k, v in sorted(weights.items()))
            lines.append(f"  weights: **{shown}**")
            if any(abs(float(v) - 0.5) < 1e-9 for v in weights.values()):
                carriers.append(f["path"])
        if "weights_seen" in f:
            lines.append(f"  weight values scored: {f['weights_seen']}")
        if f.get("gates"):
            lines.append(f"  other fields: {f['gates']}")
        if not weights and "weights_seen" not in f:
            lines.append("  (names none of the three weights)")
        lines.append("")

    lines.append("## Verdict")
    if carriers:
        lines += [
            "🚨 **Branch (a): a persisted selection carrying 0.5 EXISTS.**",
            "",
            *[f"  - `{c}`" for c in carriers],
            "",
            "So 0.5 is a real persisted selection that the campaign runs never resolved — A9's",
            "provenance reads `source: \"defaults\"`, `selection_path: null`. That is a **wiring**",
            "gap: `specs/10` §10.1's `--require-config` path exists and nothing used it. The record",
            "should say the weights were selected at 0.5, persisted, and **not applied by any run in",
            "the corpus** — which is a different sentence from either the one it carries now or the",
            "one branch (b) would license.",
        ]
    else:
        lines += [
            "**Branch (b): artifacts exist, and none of them carries 0.5.**",
            "",
            "Then the 0.5 in the record did not come from a persisted selection either. Report it as",
            "a value chosen in a report, recorded as shipped, and never persisted or applied.",
        ]
    return "\n".join(lines)


SELF_CHECK = {
    "selected.yaml": "dataset_id: starmap_visual_cortex\nw_autocorr: 0.5\nw_profile: 0.5\n"
                     "w_distribution: 0.5\ntext_emb_mode: medcpt\nexpr_pca_dim: 28\n",
    "selected_off.yaml": "dataset_id: x\nw_autocorr: 0.0\nw_profile: 0.0\nw_distribution: 0.0\n",
    "report": "| weights | 0.5 / 0.5 / 0.5 |\nw_autocorr = 0.5\n",
}


def self_check() -> int:
    failures: list[str] = []

    on = read_yaml(SELF_CHECK["selected.yaml"])
    if on.get("w_autocorr") != 0.5 or on.get("text_emb_mode") != "medcpt":
        failures.append(f"selected.yaml not parsed: {on}")
    off = read_yaml(SELF_CHECK["selected_off.yaml"])
    if off.get("w_autocorr") != 0.0:
        failures.append(f"zero weights not parsed: {off}")
    rep = weights_in(SELF_CHECK["report"])
    if rep.get("w_autocorr") != 0.5:
        failures.append(f"report scan failed: {rep}")

    # the regex fallback must agree with the yaml reader on the same text
    hidden = sys.modules.pop("yaml", None)
    sys.modules["yaml"] = None  # type: ignore[assignment]
    try:
        fallback = read_yaml(SELF_CHECK["selected.yaml"])
    finally:
        del sys.modules["yaml"]
        if hidden is not None:
            sys.modules["yaml"] = hidden
    if fallback.get("w_autocorr") != 0.5:
        failures.append(f"regex fallback disagrees with yaml: {fallback}")

    empty = render([], ["/nowhere  (does not exist)"])
    if "NONE FOUND" not in empty or "as broad as the corpus it searched" not in empty:
        failures.append("empty render lost its scope caveat")
    found = render([describe_or_stub()], ["/somewhere"])
    if "Branch (a)" not in found:
        failures.append("a 0.5 carrier did not trigger branch (a)")

    for f in failures:
        print(f"SELF-CHECK FAIL: {f}")
    if not failures:
        print("self-check OK — yaml reader, regex fallback, report scan, and both verdict branches")
    return 1 if failures else 0


def describe_or_stub() -> dict[str, Any]:
    return {"path": "/somewhere/selected.yaml", "kind": "selected.yaml",
            "weights": {"w_autocorr": 0.5}, "gates": {"dataset_id": "starmap_visual_cortex"}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", action="append", default=[], help="extra directory to search (repeatable)")
    ap.add_argument("--out", default=None, help="also write the report here")
    ap.add_argument("--self-check", action="store_true", help="exercise the readers, no filesystem")
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check()

    roots = [Path(r).expanduser() for r in args.root]
    env = os.environ.get("SPATIALCPAV25_SELECT_DIR")
    if env:
        roots.append(Path(env).expanduser())
    roots += [Path(r) for r in DEFAULT_ROOTS]

    findings, searched = scan(roots)
    text = render(findings, searched)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
