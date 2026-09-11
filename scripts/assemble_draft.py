"""Assemble `paper/DRAFT.md` from the numbered section files, and place the figures.

The draft had drifted: it still carried `## 4.4 Two specimens, measured` and `## 4.7 The sweep`
after both had been rewritten in `paper/04_angle_budget.md`. A draft maintained by hand beside its
sources is a second copy of the paper, and the second copy is always the stale one.

Figure placement lives here rather than in the section files so that a section stays readable on its
own and a figure can move without editing prose. Each entry names the heading it goes **after**; a
figure whose anchor is missing is a hard error, not a silent omission (Convention 6).

    python scripts/assemble_draft.py            # write paper/DRAFT.md
    python scripts/assemble_draft.py --check    # fail if DRAFT.md is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"

SECTIONS = (
    "00_abstract.md",
    "01_introduction.md",
    "02_method.md",
    "03_two_bounds.md",
    "04_angle_budget.md",
    "05_oblique_demonstration.md",
    "06_what_it_does_not_do.md",
    "07_limitations_and_retractions.md",
    "08_related_work.md",
)

TITLE = (
    "# SpatialCPA-v25-Gen: section generation at an arbitrary plane, "
    "and the limits of evaluating it\n\n"
    "*Full draft. Every number is measured and sourced; the provenance of each is in `reports/`.*\n"
    "*Figures are in `paper/figures/`, drawn from committed JSON by "
    "`scripts/make_figures.py`; `paper/FIGURES.md` is the figure list, including the one "
    "deliberate omission.*\n"
)

# (file, heading to place after, figure path, caption). The caption is the figure's own, not a
# restatement of the text around it.
FIGURES: tuple[tuple[str, str, str, str], ...] = (
    (
        "03_two_bounds.md",
        "## 3.1 The comb limit — what an oblique ground truth can contain",
        "figures/F1_comb.svg",
        "**Figure 1 — An oblique ground truth is a comb, not a section.** One panel per scored "
        "angle, drawn to scale from the measured stratum period and gap. `fill = t·cos θ / s` is "
        "the fraction of the plane real cells can occupy; at 90° it is exactly 0. This bounds any "
        "method, not ours.",
    ),
    (
        "03_two_bounds.md",
        "## 3.2 The resolution limit — what the standard metric can distinguish",
        "figures/F2_bounds.svg",
        "**Figure 2 — Two bounds, and only one of them moves with angle.** *A*: what an oblique "
        "ground truth can contain collapses with angle. *B*: what the statistic can distinguish "
        "does not move — `blur / radius = √(eps·scale)` is a dimensionless constant of "
        "`celltype_localization`, ≈ 0.26–0.30 on every angle measured, and therefore the same on "
        "any dataset at any magnification. The flat line is the surprising half.",
    ),
    (
        "05_oblique_demonstration.md",
        "## 5.3 The footprint: the baseline is not a section at these angles",
        "figures/F4_footprint.svg",
        "**Figure 4 — The previous method's off-axis output is not a section of the plane.** "
        "In-plane extent along the comb axis against the plane's own footprint (shaded). "
        "`copy-nearest-z` spans 4.2–4.9× the plane with 71–75% of its cells outside it; "
        "`resample-pd` spans 1.00× with 22–35% outside. Per §5.3 this is a claim about what can be "
        "scored **as a section of this plane**, not a general claim about the previous method's "
        "output.",
    ),
    (
        "05_oblique_demonstration.md",
        "## 5.5 The result",
        "figures/F6_result.svg",
        "**Figure 6 — Not one readable difference reaches a single standard error.** The bars are "
        "a leave-one-scorable-type-out jackknife combined in quadrature: an **upper bound on "
        "precision, not a confidence interval**, and not narrowed (§5.4). 60° is shown greyed "
        "rather than omitted because its null control fails P2.",
    ),
    (
        "05_oblique_demonstration.md",
        "## 5.6 Why it could not be resolved: four measurements",
        "figures/F5_null_floor.svg",
        "**Figure 5 — A section with randomised cell types scores 0.03–0.24, and it does not fall "
        "with cell count.** *A*: the ground truth's own types permuted among its own cells and "
        "scored against itself — no method, no donor, no arm. *B*: the two arms on the same axis. "
        "The arm-to-arm differences are the same size as the band's own width, which is why §5.5 "
        "could not resolve them.",
    ),
)


def build() -> str:
    out = [TITLE]
    placed: set[str] = set()
    for name in SECTIONS:
        body = (PAPER / name).read_text().rstrip("\n")
        for src, anchor, fig, cap in FIGURES:
            if src != name:
                continue
            if anchor not in body:
                raise SystemExit(
                    f"{name}: figure anchor not found: {anchor!r}\n"
                    f"  (a heading was renamed; update FIGURES in this script, "
                    f"do not drop the figure)"
                )
            if not (PAPER / fig).exists():
                raise SystemExit(
                    f"{fig} does not exist — run `python scripts/make_figures.py` first"
                )
            block = f"{anchor}\n\n![{cap.split('**')[1]}]({fig})\n\n{cap}\n"
            body = body.replace(anchor, block, 1)
            placed.add(fig)
        out.append("---\n\n" + body + "\n")
    missing = {f for _s, _a, f, _c in FIGURES} - placed
    if missing:
        raise SystemExit(f"figures declared but never placed: {sorted(missing)}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if paper/DRAFT.md differs from what the sections assemble to")
    args = ap.parse_args(argv)
    text = build()
    target = PAPER / "DRAFT.md"
    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != text:
            print("paper/DRAFT.md is STALE — re-run `python scripts/assemble_draft.py`")
            return 1
        print("paper/DRAFT.md is up to date")
        return 0
    target.write_text(text)
    figs = len(FIGURES)
    print(f"wrote {target} — {len(SECTIONS)} sections, {figs} figures placed, "
          f"{len(text.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
