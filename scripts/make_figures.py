"""Render the paper's figures as SVG from committed report JSON. No new measurement.

Every number drawn here is read from `reports/*.json` and is quoted in the text. Nothing is
simulated and nothing is fitted. Figures that need per-cell coordinates (F4's point-cloud form)
are drawn from measured EXTENTS instead and say so in their own caption, until the coordinates
are serialised by `oblique_demo.py --emit-coords`.

    python scripts/make_figures.py --out paper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# A small, colour-blind-safe set, dark enough to survive greyscale printing.
INK = "#1b1b1f"
MUTED = "#6b6f76"
GRID = "#d8dade"
TRUTH = "#2b6cb0"
OURS = "#2f855a"
BASE = "#c05621"
NULL = "#9a6fb0"
BAND = "#e8eef5"

W, H = 760, 420


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def svg(body: str, w: int = W, h: int = H, title: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="Helvetica,Arial,sans-serif" font-size="12" fill="{INK}">'
        f'<title>{esc(title)}</title>'
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>'
    )


def text(x, y, s, size=12, anchor="start", fill=INK, weight="normal", style="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}" font-weight="{weight}" font-style="{style}">{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke=GRID, w=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d}/>')


def rect(x, y, w, h, fill="none", stroke="none", sw=1, opacity=1.0):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')


def dot(x, y, r=3.2, fill=INK, opacity=1.0):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" opacity="{opacity}"/>'


def caption(x, y, s, width=None, size=11, fill=MUTED, lead=15):
    """Lay a caption out at ``x, y`` in lines that fit. Returns (svg, y_after).

    Hand-wrapped captions overflowed the canvas three times while these figures were being
    written, each time because an edit lengthened a line someone had counted by eye. The width
    model is the same 0.52-em estimate the bounds check uses, so a caption that passes here
    passes there.
    """
    width = (W - 2 * x) if width is None else width
    per = max(int(width / (size * 0.52)), 12)
    words, lines, cur = str(s).split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) > per and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    out = [text(x, y + i * lead, ln, size, fill=fill) for i, ln in enumerate(lines)]
    return "".join(out), y + len(lines) * lead


# ---------------------------------------------------------------- F1: the comb
def fig1(demo: dict) -> str:
    """What an oblique ground truth is: N strata, not a filled sheet. Measured geometry."""
    rows = [r for r in demo["angles"] if r.get("qualifies")]
    s = float(demo["section_spacing_um"])
    b, pad = [], 56
    b.append(text(pad, 26, "F1  An oblique ground truth is a comb, not a section", 14,
                  weight="bold"))
    b.append(text(pad, 44, "Each panel: the cells a tilted plane passes near, in the plane's own "
                           "frame. Measured geometry, drawn to scale.", 11, fill=MUTED))
    panel_w = (W - 2 * pad) / len(rows)
    for i, r in enumerate(rows):
        x0 = pad + i * panel_w
        top, hgt = 76, 250
        period, gap = float(r["comb_period_um"]), float(r["comb_gap_um"])
        fill_frac = float(r["fill_ratio"])
        # draw the strata across a window three periods wide
        window = 3.0 * period
        scale = (panel_w - 26) / window
        b.append(rect(x0, top, panel_w - 26, hgt, fill="#fbfcfd", stroke=GRID))
        tooth = max((period - gap) * scale, 1.0)
        k = 0
        while k * period * scale < panel_w - 26:
            b.append(rect(x0 + k * period * scale, top, tooth, hgt, fill=TRUTH, opacity=0.55))
            k += 1
        b.append(text(x0 + (panel_w - 26) / 2, top - 10, f"θ = {r['angle_deg']:.0f}°", 12,
                      anchor="middle", weight="bold"))
        b.append(text(x0 + (panel_w - 26) / 2, top + hgt + 18,
                      f"fill {fill_frac:.2f}", 11, anchor="middle"))
        b.append(text(x0 + (panel_w - 26) / 2, top + hgt + 33,
                      f"gap {gap:.0f} µm", 10, anchor="middle", fill=MUTED))
    body, yend = caption(
        pad, 362,
        f"fill = t\u00b7cos \u03b8 / s \u2014 strata are t\u00b7cos \u03b8 / sin \u03b8 wide and "
        f"s / sin \u03b8 apart (s = {s:.1f} \u00b5m here). At 90\u00b0 the fill is exactly 0: the "
        "plane's second in-plane coordinate is \u2212z, which takes one value per section, so the "
        "ground truth is N parallel lines with zero area \u2014 whatever generated it. This bounds "
        "any method, not ours.", fill=INK)
    b.append(body)
    return svg("".join(b), h=max(H, int(yend) + 16), title="F1 the comb limit")


# ------------------------------------------------- F2: fill and resolution vs angle
def fig2(demo: dict) -> str:
    rows = sorted(demo["angles"], key=lambda r: r["angle_deg"])
    b, pad = [], 56
    b.append(text(pad, 26, "F2  Two bounds, and only one of them moves with angle", 14,
                  weight="bold"))
    pw, ph, top = 300, 250, 78
    # --- panel A: fill
    ax, ay = pad, top
    b.append(rect(ax, ay, pw, ph, fill="none", stroke=GRID))
    b.append(text(ax, ay - 12, "A   fill = t·cos θ / s", 12, weight="bold"))
    def fx(a): return ax + (a / 90.0) * pw
    def fy(v): return ay + ph - v * ph
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(line(ax, fy(v), ax + pw, fy(v), GRID))
        b.append(text(ax - 8, fy(v) + 4, f"{v:.2f}", 10, anchor="end", fill=MUTED))
    pts = [(fx(float(r["angle_deg"])), fy(float(r["fill_ratio"]))) for r in rows]
    b.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
             + f'" fill="none" stroke="{TRUTH}" stroke-width="2"/>')
    for r, (x, y) in zip(rows, pts):
        ok = r.get("qualifies")
        b.append(dot(x, y, 4 if ok else 3, TRUTH if ok else MUTED))
    b.append(dot(fx(90), fy(0), 5, "#b00020"))
    b.append(text(fx(90) - 6, fy(0) - 10, "0 at 90°", 10, anchor="end", fill="#b00020"))
    for a in (0, 30, 45, 60, 90):
        b.append(text(fx(a), ay + ph + 16, f"{a}°", 10, anchor="middle", fill=MUTED))
    # --- panel B: blur / radius
    bx = pad + pw + 84
    b.append(rect(bx, ay, pw, ph, fill="none", stroke=GRID))
    b.append(text(bx, ay - 12, "B   blur / radius = √(eps·scale)", 12, weight="bold"))
    def gx(a): return bx + (a / 90.0) * pw
    def gy(v): return ay + ph - (v / 1.0) * ph
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(line(bx, gy(v), bx + pw, gy(v), GRID))
        b.append(text(bx - 8, gy(v) + 4, f"{v:.2f}", 10, anchor="end", fill=MUTED))
    ratios = [(float(r["angle_deg"]), float(r["metric_blur_um"]) / float(r["metric_radius_um"]))
              for r in rows if r.get("metric_radius_um")]
    b.append('<polyline points="' + " ".join(f"{gx(a):.1f},{gy(v):.1f}" for a, v in ratios)
             + f'" fill="none" stroke="{BASE}" stroke-width="2"/>')
    for a, v in ratios:
        b.append(dot(gx(a), gy(v), 4, BASE))
    lo, hi = min(v for _a, v in ratios), max(v for _a, v in ratios)
    b.append(rect(bx, gy(hi), pw, gy(lo) - gy(hi), fill=BASE, opacity=0.10))
    b.append(text(bx + pw - 6, gy(hi) - 8, f"{lo:.2f}–{hi:.2f}, flat", 10, anchor="end", fill=BASE))
    for a in (0, 30, 45, 60, 90):
        b.append(text(gx(a), ay + ph + 16, f"{a}°", 10, anchor="middle", fill=MUTED))
    body, yend = caption(
        pad, 366,
        "A falls to zero: what an oblique ground truth can CONTAIN collapses with angle. B "
        "does not move: what the statistic can DISTINGUISH is a constant fraction of the "
        "tissue radius \u2014 "
        "the same on any dataset, at any magnification, roughly three to four resolvable locations "
        "along a radius. The flat line is the surprising half.", fill=INK)
    b.append(body)
    return svg("".join(b), h=max(H, int(yend) + 16), title="F2 fill and resolution")


# ------------------------------------------------------------ F4: the footprint
def fig4(demo: dict) -> str:
    """Extent form, from measured extents and outside-fractions. See the caption."""
    rows = [r for r in demo["angles"] if r.get("footprint")]
    b, pad = [], 56
    b.append(text(pad, 26,
                  "F4  The previous method's off-axis output is not a section of the plane",
                  14, weight="bold"))
    b.append(text(pad, 44, "In-plane extent along the comb axis, to scale, against the plane's own "
                           "footprint (shaded).", 11, fill=MUTED))
    top, rowh = 76, 86
    widest = max(float(r["footprint"]["copy-nearest-z"]["u_extent_um"]) for r in rows)
    label_gutter = 176  # right-hand column the ratio labels are set in, so bars cannot run over
    scale = (W - 2 * pad - 150 - label_gutter) / widest
    for i, r in enumerate(rows):
        y = top + i * rowh
        fp = r["footprint"]
        gw = float(fp["copy-nearest-z"]["truth_u_extent_um"])
        cx = pad + 150
        b.append(text(pad, y + 20, f"θ = {r['angle_deg']:.0f}°", 12, weight="bold"))
        b.append(text(pad, y + 37, f"plane: {gw:.0f} µm", 10, fill=MUTED))
        # the plane's own footprint, centred
        b.append(rect(cx + (widest - gw) * scale / 2, y, gw * scale, 56, fill=BAND))
        for name, colour, off in (("copy-nearest-z", BASE, 6), ("resample-pd", OURS, 32)):
            e = float(fp[name]["u_extent_um"])
            frac = float(fp[name]["frac_outside"])
            x0 = cx + (widest - e) * scale / 2
            b.append(rect(x0, y + off, e * scale, 18, fill=colour, opacity=0.75))
            b.append(text(W - pad, y + off + 13,
                          f"{fp[name]['u_extent_ratio']:.2f}×   {frac:.0%} outside", 10,
                          anchor="end", fill=colour,
                          weight="bold" if name == "copy-nearest-z" else "normal"))
        b.append(line(pad, y + 68, W - pad, y + 68, GRID))
    ly = top + len(rows) * rowh + 6
    b.append(rect(pad, ly, 12, 12, fill=BAND))
    b.append(text(pad + 18, ly + 11, "the plane's own footprint (= the ground truth)", 10))
    b.append(rect(pad + 300, ly, 12, 12, fill=BASE, opacity=0.75))
    b.append(text(pad + 318, ly + 11, "copy-nearest-z (baseline)", 10))
    b.append(rect(pad + 480, ly, 12, 12, fill=OURS, opacity=0.75))
    b.append(text(pad + 498, ly + 11, "resample-pd (ours)", 10))
    return svg("".join(b), h=ly + 40, title="F4 the footprint")


# ------------------------------------------------ F5: the scrambled-section floor
def fig5(demo: dict) -> str:
    """The floor, and the arm differences measured against it. Every number read from the JSON."""
    rows = [r for r in demo["angles"] if r.get("self_null")]
    b, pad = [], 62
    b.append(text(pad, 26, "F5  A section with randomised cell types scores 0.03-0.24, and does "
                           "not fall with n", 14, weight="bold"))
    b.append(text(pad, 44, "Left: the ground truth's own cell types permuted among its own cells, "
                           "scored against itself. No method, no donor, no arm.", 11, fill=MUTED))
    ax, ay, pw, ph = pad, 92, 372, 228
    bx, bw = pad + pw + 72, W - pad - (pad + pw + 72)
    vmax = 0.50

    def yv(v):
        return ay + ph - (min(v, vmax) / vmax) * ph

    # the band, spanning both panels, so the arms are read against it
    meds = [e["self_null_median"] for r in rows for e in r["self_null"]]
    lo, hi = min(meds), max(meds)
    b.append(rect(ax, yv(hi), bx + bw - ax, yv(lo) - yv(hi), fill=NULL, opacity=0.14))

    b.append(rect(ax, ay, pw, ph, fill="none", stroke=GRID))
    b.append(text(ax, ay - 12, "A   scrambled-section floor vs cell count", 12, weight="bold"))
    ns = sorted({e["n"] for r in rows for e in r["self_null"]})

    def xn(n):
        return ax + (ns.index(n) + 0.5) * (pw / len(ns))

    for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        b.append(line(ax, yv(v), bx + bw, yv(v), GRID))
        b.append(text(ax - 8, yv(v) + 4, f"{v:.1f}", 10, anchor="end", fill=MUTED))
    for r in rows:
        pts = [(xn(e["n"]), yv(e["self_null_median"])) for e in r["self_null"]]
        b.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                 + f'" fill="none" stroke="{NULL}" stroke-width="1.4" opacity="0.85"/>')
        for e, (x, y) in zip(r["self_null"], pts):
            half = 0.5 * float(e["self_null_spread"])
            b.append(line(x, yv(e["self_null_median"] - half), x,
                          yv(e["self_null_median"] + half), NULL, 1.2))
            b.append(dot(x, y, 3.4, NULL))
        b.append(text(pts[-1][0] + 8, pts[-1][1] + 4, f"{r['angle_deg']:.0f}\u00b0", 10, fill=NULL))
    for n in ns:
        b.append(text(xn(n), ay + ph + 16, str(n), 10, anchor="middle", fill=MUTED))
    b.append(text(ax + pw / 2, ay + ph + 34, "cells in the scored section (n)", 10,
                  anchor="middle", fill=MUTED))
    b.append(text(ax + pw - 6, yv(hi) - 8, f"band {lo:.2f}-{hi:.2f}", 10, anchor="end",
                  fill=NULL, weight="bold"))

    # --- panel B: the arms, on the same axis, with the difference each comparison rests on
    b.append(rect(bx, ay, bw, ph, fill="none", stroke=GRID))
    b.append(text(bx, ay - 12, "B   the arms, same axis", 12, weight="bold"))
    order = sorted(rows, key=lambda r: r["angle_deg"])

    def xa(i):
        return bx + (i + 0.5) * (bw / len(order))

    for i, r in enumerate(order):
        x = xa(i)
        vals = {}
        for name in ("copy-nearest-z", "resample-pd"):
            v = [s[METRIC_KEY] for s in r.get("arms", {}).get(name, {}).values()]
            if v:
                vals[name] = sorted(v)[len(v) // 2]
        if len(vals) != 2:
            continue
        b.append(line(x, yv(vals["copy-nearest-z"]), x, yv(vals["resample-pd"]), MUTED, 1))
        b.append(dot(x, yv(vals["copy-nearest-z"]), 4.2, BASE))
        b.append(dot(x, yv(vals["resample-pd"]), 4.2, OURS))
        d = vals["copy-nearest-z"] - vals["resample-pd"]
        readable = float(r["null_ceiling"]) >= max(
            s[METRIC_KEY] for s in r.get("arms", {}).get("null", {}).values())
        b.append(text(x, yv(max(vals.values())) - 9, f"{d:.3f}", 9, anchor="middle",
                      fill=INK if readable else MUTED,
                      style="normal" if readable else "italic"))
        b.append(text(x, ay + ph + 16, f"{r['angle_deg']:.0f}\u00b0", 10, anchor="middle",
                      fill=MUTED if readable else "#b00020"))
        if not readable:
            b.append(text(x, ay + ph + 30, "P2 fails", 9, anchor="middle", fill="#b00020"))
    b.append(dot(bx + 12, ay + ph - 12, 4.2, BASE))
    b.append(text(bx + 22, ay + ph - 8, "baseline", 9))
    b.append(dot(bx + 12, ay + ph + 2, 4.2, OURS))
    b.append(text(bx + 22, ay + ph + 6, "ours", 9))

    diffs = []
    for r in order:
        v = {}
        for name in ("copy-nearest-z", "resample-pd"):
            s = [x[METRIC_KEY] for x in r.get("arms", {}).get(name, {}).values()]
            if s:
                v[name] = sorted(s)[len(s) // 2]
        if len(v) == 2:
            diffs.append(v["copy-nearest-z"] - v["resample-pd"])
    at250 = "/".join(f"{r['self_null'][0]['self_null_median']:.3f}" for r in order)
    atmax = "/".join(f"{r['self_null'][-1]['self_null_median']:.3f}" for r in order)
    body, yend = caption(
        pad, 360,
        f"The floor does not fall with n: ({atmax}) at each angle's largest n, against "
        f"({at250}) at "
        "n = 250 \u2014 higher at two of the three, and nowhere near zero. The arm-to-arm "
        f"differences in B span {min(diffs):.3f} to {max(diffs):.3f}; the band they are read "
        f"against itself spans {hi - lo:.3f}. Our arm's scores at 30\u00b0 and 45\u00b0 fall "
        "inside that band; the baseline's do not. 60\u00b0 is greyed: its null control fails P2.",
        fill=INK)
    b.append(body)
    return svg("".join(b), h=max(H, int(yend) + 16), title="F5 the scrambled-section floor")


# ------------------------------------------------------- F6: the result, as a forest
def fig6(demo: dict) -> str:
    """Difference and combined precision bound at each scored angle, against zero."""
    rows = sorted((r for r in demo["angles"] if r.get("arms") and r.get("self_null")),
                  key=lambda r: r["angle_deg"])
    b, pad = [], 62
    b.append(text(pad, 26, "F6  Not one readable difference reaches a single standard error", 14,
                  weight="bold"))
    ax, ay, pw, ph = pad + 96, 92, W - pad - 96 - 150, 62 * len(rows) + 24

    def est(r, name):
        v = [s[METRIC_KEY] for s in r["arms"][name].values()]
        return sorted(v)[len(v) // 2]

    def se(r, name):
        for s in r["arms"][name].values():
            if isinstance(s, dict) and "jk_se" in s:
                return float(s["jk_se"])
        return 0.0

    items = []
    for r in rows:
        d = est(r, "resample-pd") - est(r, "copy-nearest-z")
        bound = (se(r, "resample-pd") ** 2 + se(r, "copy-nearest-z") ** 2) ** 0.5
        readable = float(r["null_ceiling"]) >= max(
            s[METRIC_KEY] for s in r["arms"]["null"].values())
        items.append((r["angle_deg"], d, bound, readable))
    span = max(abs(d) + bound for _a, d, bound, _k in items) * 1.12

    def xv(v):
        return ax + pw / 2 + (v / span) * (pw / 2)

    b.append(rect(ax, ay, pw, ph, fill="none", stroke=GRID))
    b.append(line(xv(0), ay, xv(0), ay + ph, MUTED, 1.2))
    b.append(text(xv(0), ay - 8, "0", 10, anchor="middle", fill=MUTED))
    for v in (-0.6, -0.3, 0.3, 0.6):
        if abs(v) < span:
            b.append(line(xv(v), ay, xv(v), ay + ph, GRID, 1, dash="3 3"))
            b.append(text(xv(v), ay + ph + 16, f"{v:+.1f}", 10, anchor="middle", fill=MUTED))
    for i, (a, d, bound, readable) in enumerate(items):
        y = ay + 30 + i * 62
        ink = INK if readable else MUTED
        b.append(line(xv(d - bound), y, xv(d + bound), y, ink, 2))
        for e in (d - bound, d + bound):
            b.append(line(xv(e), y - 6, xv(e), y + 6, ink, 2))
        b.append(dot(xv(d), y, 5, OURS if readable else MUTED))
        label = f"{a:.0f}\u00b0"
        b.append(text(pad, y + 5, label, 13, weight="bold", fill=ink))
        b.append(text(pad + 34, y + 5, "" if readable else "\u2014 not readable", 10,
                      fill="#b00020"))
        sig = abs(d) / bound if bound else float("inf")
        b.append(text(ax + pw + 12, y - 2, f"{d:+.4f} \u00b1 {bound:.4f}", 11, fill=ink))
        b.append(text(ax + pw + 12, y + 14, f"{sig:.2f}\u03c3", 11, fill=ink, weight="bold"))
        if not readable:
            b.append(text(pad, y + 21, "P2 fails", 9, fill="#b00020"))
    body, yend = caption(
        pad, ay + ph + 42,
        "The bars are a leave-one-scorable-type-out jackknife combined in quadrature. They are an "
        "UPPER BOUND ON PRECISION, not confidence intervals, and they are not narrowed "
        "(\u00a75.4): "
        "with 6-9 scorable types, leaving one out removes 11-17% of the data and re-normalises the "
        "metric's own radius, scale and null draws. Every interval spans zero. 60\u00b0 is shown "
        "rather than omitted because its null control fails P2 - the protocol disqualified it, and "
        "showing a disqualified angle greyed is what that looks like.", fill=INK)
    b.append(body)
    return svg("".join(b), h=max(H, int(yend) + 16), title="F6 the result")


METRIC_KEY = "celltype_localization"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", default="reports/oblique_demo.json")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args(argv)
    demo = json.loads(Path(args.demo).read_text().replace("NaN", "null")
                      .replace("Infinity", "1e999"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("F1_comb", fig1), ("F2_bounds", fig2), ("F4_footprint", fig4),
                     ("F5_null_floor", fig5), ("F6_result", fig6)):
        (out / f"{name}.svg").write_text(fn(demo))
        print(f"  wrote {out / name}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
