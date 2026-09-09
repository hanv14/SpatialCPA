"""Find every persisted selection artifact and read the metric-aware weights out of it.

**The question this settles.** The record says `w_autocorr = w_profile = w_distribution = 0.5`
"ship". `Config` declares them **0.0**, `scripts/_starmap_run.py::base_config` never overrides them,
and `t09_ship_starmap.py --w-metric-aware` defaults to unset — so every real-data fit whose artifact
is in the repository ran at **0.0** except A9's deliberately-overridden `05` arm
(`reports/advisor_report.md` §6a). Does *any* persisted selection carry 0.5?

**Answered 2026-09-09: no, and the one that exists is not a shipped selection.**
`runs/select/starmap_visual_cortex/selected.yaml` was added at `3d57725` and deleted at `5cd1fd6`.
It records the three weights at **0.0**, and `train_steps: 20`, `decoder_mu_link: softplus`,
`expr_pca_dim: 32` — a smoke artifact from the halted pilot, predating the link fix, the clamp rule
and R11. So the 0.5 has **no machine-readable source anywhere in the project**: not `Config`, not any
fit's recorded config, not the one persisted selection that exists. See
`reports/advisor_report.md` §6b.

🚨 **And the first version of this script reported NONE FOUND.** It walked filesystem roots, and the
file is tracked in the very repository it was run from — present in history, absent from the tree.
Its scope caveat warned about *roots*, i.e. space, while the file was missing in *time*. That is
`specs/10` §4.2j — an instrument reporting a check it did not perform — committed by an instrument
that quoted §8c's reflog lesson in its own output. `scan_history` is the fix, and it searches
``--all --reflog`` so a reset-away commit is covered too.

This script reads, and never writes.

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
import subprocess
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
    """Return what one on-disk selection artifact says about the three weights and the arm."""
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:  # unreadable is not absent, and the difference matters
        return {"path": str(path), "kind": path.name, "weights": {}, "gates": {},
                "error": f"unreadable: {exc}"}
    out = describe_text(text, path.name)
    out["path"] = str(path)
    return out


def describe_text(text: str, kind: str) -> dict[str, Any]:
    """Return what one artifact's *contents* say — shared by the tree and history readers."""
    out: dict[str, Any] = {"path": "<text>", "kind": kind, "weights": {}, "gates": {}}
    path = Path(kind)

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


def _git(args: list[str], repo: Path) -> str | None:
    """Run a read-only git command in ``repo``; ``None`` if git or the repo is unavailable."""
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def scan_history(repo: Path) -> tuple[list[dict[str, Any]], str]:
    """Find selection artifacts that ever existed in ``repo``'s history, including deleted ones.

    🚨 **This function exists because its absence produced a false negative on the one question
    this script was built for.** The first version walked filesystem roots only, and reported
    NONE FOUND for `runs/select/starmap_visual_cortex/selected.yaml` — a file **tracked in the very
    repository the script was run from**, added at `3d57725` and deleted at `5cd1fd6`. Its scope
    caveat warned about *roots*, i.e. space, and the file was missing in *time*. An instrument that
    reports a check it did not perform is `specs/10` §4.2j, and this one quoted §8c's reflog lesson
    in its own output while committing it.

    Commits come from ``git rev-list --all --reflog``, so branches, tags **and reflog-only commits**
    are covered — §8c rule 3: a commit that was reset away is still reachable there, and that is
    exactly how the thirteen artifacts were recovered.
    """
    if _git(["rev-parse", "--git-dir"], repo) is None:
        return [], f"{repo}  (not a git repository, or git unavailable — HISTORY NOT SEARCHED)"

    out = _git(["log", "--all", "--reflog", "--pretty=format:%H", "--name-only",
                "--diff-filter=AMR"], repo)
    if out is None:
        return [], f"{repo}  (git history unreadable — HISTORY NOT SEARCHED)"

    # newest-first, so the first commit naming a path is the latest version of it
    latest: dict[str, str] = {}
    commit = ""
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
            commit = line
        elif Path(line).name in NAMES:
            latest.setdefault(line, commit)

    findings: list[dict[str, Any]] = []
    for path, sha in sorted(latest.items()):
        blob = _git(["show", f"{sha}:{path}"], repo)
        if blob is None:
            continue
        item = describe_text(blob, Path(path).name)
        in_tree = (repo / path).exists()
        item["path"] = f"{path}  @{sha[:7]}" + ("" if in_tree else "  [DELETED from the tree]")
        item["from_history"] = True
        item["commit"] = sha
        item["deleted"] = not in_tree
        findings.append(item)
    return findings, f"{repo}  (git history: --all --reflog, {len(latest)} artifact path(s) ever seen)"


def scan(roots: list[Path], repo: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (findings, corpora actually searched) over the working tree **and** git history.

    A root that does not exist is reported rather than skipped silently, and the git corpus is
    reported separately — because "not in the tree" and "never existed" are different claims and
    conflating them is what this script got wrong the first time.
    """
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

    hist, note = scan_history(repo if repo is not None else Path.cwd())
    searched.append(note)
    findings += hist
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
            "🚨 **A persisted selection carrying 0.5 EXISTS.**",
            "",
            *[f"  - `{c}`" for c in carriers],
            "",
            "So 0.5 is a real persisted selection. Check whether any run resolved it — A9's",
            "provenance reads `source: \"defaults\"`, `selection_path: null` — and report a **wiring**",
            "gap if none did.",
        ]
    else:
        lines += [
            "**Selection artifacts exist, and NONE of them carries 0.5.**",
            "",
            "Every one found above records the three weights at some other value. So the 0.5 in the",
            "record came from neither `Config`, nor any fit's recorded config, nor any persisted",
            "selection: **it has no machine-readable source in this project at all.**",
            "",
            "⚠️ **Before quoting that, read the gates printed above and judge whether each artifact is",
            "a selection anyone would ship.** A file that records `train_steps` far below the",
            "selected budget, or a superseded `decoder_mu_link`, is a smoke or halted-pilot artifact:",
            "it is evidence about what was persisted, not about what was chosen. Say which it is.",
        ]
        smoke = [f for f in findings if isinstance(f.get("gates", {}).get("train_steps"), int)
                 and f["gates"]["train_steps"] < 100]
        if smoke:
            lines += [
                "",
                "⚠️ **Flagged as likely smoke artifacts by `train_steps` alone** (an observation, not a",
                "threshold — the gates are printed above so the call is visible):",
                *[f"  - `{f['path']}` — train_steps={f['gates']['train_steps']}" for f in smoke],
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
    if "carrying 0.5 EXISTS" not in found:
        failures.append("a 0.5 carrier did not trigger the exists-verdict")
    zeroed = describe_or_stub()
    zeroed["weights"] = {"w_autocorr": 0.0}
    zeroed["gates"] = {"train_steps": 20}
    other = render([zeroed], ["/somewhere"])
    if "no machine-readable source" not in other:
        failures.append("a non-0.5 artifact did not trigger the no-source verdict")
    if "likely smoke artifacts" not in other:
        failures.append("train_steps=20 was not flagged as a smoke artifact")

    # the history reader must find the file this script originally missed
    hist, note = scan_history(Path(__file__).resolve().parent.parent)
    hit = [f for f in hist if f["kind"] == "selected.yaml"]
    if not hit:
        failures.append(f"history scan found no selected.yaml ({note}) — the original false negative")
    else:
        w = hit[0].get("weights", {})
        if w.get("w_autocorr") != 0.0:
            failures.append(f"history scan misread the weights: {w}")
        if not hit[0].get("deleted"):
            failures.append("history scan did not mark the deleted file as deleted")

    for f in failures:
        print(f"SELF-CHECK FAIL: {f}")
    if not failures:
        print("self-check OK — readers, all three verdict states, the smoke-artifact flag, and a\n  regression on the false negative: the history scan finds the deleted selected.yaml")
    return 1 if failures else 0


def describe_or_stub() -> dict[str, Any]:
    return {"path": "/somewhere/selected.yaml", "kind": "selected.yaml",
            "weights": {"w_autocorr": 0.5}, "gates": {"dataset_id": "starmap_visual_cortex"}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", action="append", default=[], help="extra directory to search (repeatable)")
    ap.add_argument("--repo", default=".", help="git repository whose history to search (default: CWD)")
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

    findings, searched = scan(roots, repo=Path(args.repo).expanduser())
    text = render(findings, searched)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
