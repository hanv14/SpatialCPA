"""Nature/Springer figure theme for benchmark-pbya-v3.

Journals constrain figures far more tightly than a screen does, and the
constraints are mechanical rather than aesthetic. This module encodes the
Nature-family ones in one place so a figure is submission-shaped by
construction:

* **Column widths are fixed.** 89 mm single, 120 mm 1.5-column, 183 mm double,
  and nothing may exceed 247 mm tall. A figure drawn at an arbitrary size gets
  rescaled by production, which rescales the type with it — so the width is
  chosen up front and the type sizes below are then final-size sizes.
* **Type is sans, 5-7 pt at final size.** 7 pt for labels, 6 pt for ticks,
  8 pt bold for panel letters. Nothing smaller than 5 pt survives printing.
* **Lines are hairlines.** 0.5 pt axes and ticks (Nature's floor is 0.25 pt).
* **Vector output with embedded, editable type.** ``fonttype=42`` embeds
  TrueType rather than converting glyphs to paths, which is what lets a
  production editor restyle text. A PNG is written alongside at 600 dpi for
  quick viewing, but the PDF is the deliverable.

Colour
------
``PALETTE`` is Okabe-Ito (the Color Universal Design set, the de-facto
colourblind-safe palette in Nature-family figures) with two changes, both
forced by validation against a white print surface:

* **Black is dropped.** As a neutral it fails the lightness-band and
  chroma-floor checks — it reads as chrome, not as a series.
* **The pale yellow ``#F0E442`` is replaced by a dark gold ``#8B6914``.**
  At 1.1:1 against white, a yellow bar fill is invisible in print.

The slot ordering is not cosmetic — it is the CVD-safety mechanism. All 5040
orderings of the seven hues were enumerated and scored; this one leads with
blue and clears every gate on the adjacent pairlist (bars, stacks, lines):
**worst adjacent CVD dE 15.8** against a target of 8, **worst adjacent
normal-vision dE 16.4** against a floor of 15.

Two caveats the palette carries, and how the figures discharge them:

* *All-pairs forms* (scatter, dot plots — where any two series can land side by
  side) only clear the floors for the **first four slots**. Past four,
  ``plot_cross_dataset`` adds secondary encoding: a distinct marker shape per
  method plus direct labels, so identity never rests on hue alone.
* ``#56B4E9`` (2.31:1) and ``#E69F00`` (2.25:1) sit below 3:1 on white. That is
  a relief obligation, not a dismissable warning: every figure writes the
  numbers it drew to a CSV beside itself, which is the table view.

Deliberate deviation from the house data-viz spec: bars are square-ended, not
4 px rounded. Rounded caps are a screen idiom; journal figures use square
marks, and the journal's convention wins inside this module.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

# ── Canvas ────────────────────────────────────────────────────────────────────
MM_PER_INCH = 25.4

WIDTH_SINGLE_MM = 89.0     # single column
WIDTH_HALF_MM = 120.0      # 1.5 column
WIDTH_DOUBLE_MM = 183.0    # double column (full page width)
MAX_HEIGHT_MM = 247.0      # hard ceiling — a taller figure is rejected


def mm(value_mm):
    """Millimetres -> inches, the unit matplotlib's ``figsize`` wants."""
    return float(value_mm) / MM_PER_INCH


def figsize(width_mm=WIDTH_DOUBLE_MM, height_mm=120.0):
    """``figsize`` in inches, with the page-height ceiling enforced.

    Exceeding 247 mm is not a soft problem — production will scale the whole
    figure down to fit, and the 7 pt type goes with it — so it warns loudly
    rather than silently producing something that cannot be submitted.
    """
    if height_mm > MAX_HEIGHT_MM:
        warnings.warn(
            f"figure height {height_mm:.0f} mm exceeds the {MAX_HEIGHT_MM:.0f} mm "
            f"page limit; it will be scaled down at production and the type with "
            f"it. Drop panels, or split the figure.",
            stacklevel=2)
    return (mm(width_mm), mm(height_mm))


# ── Ink ───────────────────────────────────────────────────────────────────────
INK = "#000000"          # primary text
INK_MUTED = "#4d4d4d"    # axis labels, secondary text
AXIS = "#000000"         # spines and ticks — black, print-standard
GRID = "#d9d9d9"         # hairline grid, when one is used at all
SURFACE = "#ffffff"      # the print surface every check above is measured against

# ── Categorical palette ───────────────────────────────────────────────────────
# See the module docstring: validated ordering, not a preference.
PALETTE = [
    "#0072B2",   # 1 blue
    "#D55E00",   # 2 vermillion
    "#CC79A7",   # 3 reddish purple
    "#E69F00",   # 4 orange
    "#8B6914",   # 5 dark gold
    "#56B4E9",   # 6 sky blue
    "#009E73",   # 7 bluish green
]

# Beyond four slots a scatter/dot form needs a second channel; these are the
# shapes it uses, in the same fixed order as the hues.
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

# Diverging ramp for agreement heatmaps: two poles with a neutral grey midpoint,
# because the metrics have a real zero (no agreement) and a negative cell means
# anti-correlated, not merely small. Blue is the positive pole.
DIVERGING = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#f7f7f7",
             "#92c5de", "#4393c3", "#2166ac", "#053061"]

# Single-hue sequential ramp, light -> dark, for magnitude without a zero.
SEQUENTIAL = ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6",
              "#2171b5", "#08519c", "#08306b"]

# Nature-family type is Helvetica; Arial is the accepted substitute. Everything
# after those is a fallback for machines that ship neither.
FONT_STACK = ["Helvetica", "Helvetica Neue", "Arial", "Nimbus Sans",
              "Liberation Sans", "FreeSans", "DejaVu Sans"]

# Final-size type sizes, in points.
FONT_BASE = 7
FONT_TICK = 6
FONT_PANEL_LETTER = 8
LINE_HAIRLINE = 0.5


def series_color(index):
    """Colour for categorical slot ``index`` (0-based), never cycled.

    Cycling would give two methods the same hue, which is worse than failing:
    the figure would still render and would simply be wrong. Seven is the
    validated depth of the palette, so an eighth series has to be handled by
    faceting or by dropping methods, and the caller is told so.
    """
    index = int(index)
    if index >= len(PALETTE):
        raise ValueError(
            f"categorical slot {index} is past the {len(PALETTE)}-hue palette. "
            f"Hues are never cycled — two series would share a colour. Plot at "
            f"most {len(PALETTE)} methods (--method-order), or split the figure.")
    return PALETTE[index]


def series_marker(index):
    """Marker shape for categorical slot ``index`` — the secondary channel."""
    return MARKERS[int(index) % len(MARKERS)]


def resolve_font():
    """First font in ``FONT_STACK`` that is actually installed.

    Falling back to DejaVu is not fatal but it is not Helvetica either — the
    metrics differ, so a layout tuned on one shifts on the other. Say so once
    rather than let a submission figure quietly use the wrong face.
    """
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_STACK:
        if name in installed:
            return name
    return FONT_STACK[-1]


def activate():
    """Apply the theme and return ``matplotlib.pyplot``.

    Uses the Agg backend: these figures are written to disk, never shown, and
    importing pyplot without a display otherwise picks a backend that may fail.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = resolve_font()
    if font in ("DejaVu Sans", "FreeSans"):
        warnings.warn(
            f"neither Helvetica nor Arial is installed; falling back to {font!r}. "
            f"The figure is correct but the type is not the journal's face — "
            f"install one of {FONT_STACK[:3]} before submitting.", stacklevel=2)

    plt.rcParams.update({
        # Type
        "font.family": "sans-serif",
        "font.sans-serif": [font] + FONT_STACK,
        "font.size": FONT_BASE,
        "axes.titlesize": FONT_BASE,
        "axes.labelsize": FONT_BASE,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_TICK,
        "figure.titlesize": FONT_PANEL_LETTER,
        # Embed editable TrueType rather than outlining glyphs — production
        # needs the text to still be text.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Surface
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        # Chrome: hairlines, outward ticks, no top/right spines, no grid.
        "axes.edgecolor": AXIS,
        "axes.linewidth": LINE_HAIRLINE,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "text.color": INK,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": LINE_HAIRLINE,
        "ytick.major.width": LINE_HAIRLINE,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.4,
        # Legends sit in the figure, not in a box.
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.0,
        # NOT "tight": a tight bounding box recrops to the drawn content, which
        # changes the output width. A figure declared at 183 mm would then be
        # saved at whatever the ink happened to span, production would rescale
        # it back, and the 7 pt type would go with it. The whole point of fixing
        # the column width is that the saved file *is* that width, so every
        # figure here reserves its own margins instead.
        "savefig.bbox": "standard",
        "figure.dpi": 200,
    })
    return plt


def panel_letter(ax, letter, dx=-0.06, dy=1.06):
    """Bold lowercase panel letter, the Nature convention (**a**, **b**, ...)."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=FONT_PANEL_LETTER,
            fontweight="bold", va="bottom", ha="right", color=INK)


# ── Layout arithmetic ─────────────────────────────────────────────────────────
# Without a tight bounding box, every margin has to be reserved deliberately,
# which means knowing roughly how much room a run of text needs. These are
# estimates in millimetres, deliberately a little generous: over-reserving costs
# whitespace, under-reserving overlaps two labels, and only one of those is
# recoverable by the reader.

_MM_PER_PT = 25.4 / 72.0
_MEAN_GLYPH_EM = 0.52      # mean advance width of a Helvetica-like sans


def text_width_mm(text, fontsize=FONT_TICK):
    """Approximate rendered width of a single line of text."""
    return len(str(text)) * _MEAN_GLYPH_EM * float(fontsize) * _MM_PER_PT


def text_height_mm(fontsize=FONT_TICK):
    """Approximate rendered height of a single line of text."""
    return 1.2 * float(fontsize) * _MM_PER_PT


def wrap_to_mm(text, available_mm, fontsize=FONT_BASE):
    """Wrap ``text`` onto as many lines as it takes to fit ``available_mm``.

    A centred title wider than its panel does not get clipped — it overhangs
    into whatever is beside it, which at small panel sizes means the next
    panel's label or letter. Wrapping keeps a long metric name inside the block
    it belongs to.
    """
    import textwrap

    per_char_mm = _MEAN_GLYPH_EM * float(fontsize) * _MM_PER_PT
    width = max(6, int(float(available_mm) / max(per_char_mm, 1e-6)))
    return "\n".join(textwrap.wrap(str(text), width)) or str(text)


def tick_layout(labels, available_mm, fontsize=FONT_TICK, rotation=45.0):
    """Choose a rotation for a row of tick labels, and say how tall it will be.

    Horizontal labels read best and are used whenever the widest one fits in the
    space one tick actually owns. When they do not, the choice is rotate or
    collide — so this rotates, and reports the height the rotated block needs so
    the caller can reserve it rather than discover the overlap in the PDF.

    Returns ``(rotation_deg, horizontal_alignment, height_mm)``.
    """
    labels = [str(x) for x in labels] or [""]
    widest = max(text_width_mm(x, fontsize) for x in labels)
    line_h = text_height_mm(fontsize)
    if widest <= float(available_mm):
        return 0.0, "center", line_h + 1.0
    theta = np.radians(float(rotation))
    height = widest * np.sin(theta) + line_h * np.cos(theta)
    return float(rotation), "right", height + 1.0


def value_ramp(values, force=None):
    """Pick the colour ramp that matches the data's job, and its limits.

    Sequential and diverging are not interchangeable: a diverging ramp spends
    half its range on negative values and asserts that zero is a meaningful
    midpoint. That is right for agreement metrics *when some are negative* — a
    method anti-correlated with the ground truth should not look merely weak.
    When every value is positive there is no polarity to show, and a diverging
    ramp would throw away half its resolution on an empty half-range, flattening
    the differences that are actually there.

    Returns ``(colormap, vmin, vmax)``.
    """
    from matplotlib.colors import LinearSegmentedColormap

    vals = np.asarray([v for v in np.ravel(values) if np.isfinite(v)], dtype=float)
    diverging = force == "diverging" or (
        force is None and (len(vals) > 0 and vals.min() < 0))
    if diverging:
        cmap = LinearSegmentedColormap.from_list("nature_div", DIVERGING)
        lo, hi = -1.0, 1.0
    else:
        cmap = LinearSegmentedColormap.from_list("nature_seq", SEQUENTIAL)
        lo, hi = 0.0, 1.0
    cmap = cmap.copy()
    cmap.set_bad(SURFACE)
    return cmap, lo, hi


def legend_row_major(handles, labels, ncol):
    """Reorder legend entries so a multi-column legend reads across, not down.

    Matplotlib fills a legend column by column, so with 7 entries in 4 columns
    the second entry lands underneath the first rather than beside it — and the
    legend order stops matching the bar order it is supposed to decode.
    """
    n = len(handles)
    ncol = max(1, int(ncol))
    nrow = int(np.ceil(n / ncol))
    order = [r * ncol + c for c in range(ncol) for r in range(nrow)]
    order = [i for i in order if i < n]
    return [handles[i] for i in order], [labels[i] for i in order]


def save(fig, path_stem, formats=("pdf", "png"), dpi=600):
    """Write one figure in each format; returns the paths written.

    The PDF is the submission artefact (vector, embedded type); the PNG at
    600 dpi is for looking at without a PDF viewer. 600 is the journal's
    minimum for a raster figure containing line art or text.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi)
        written.append(out)
    return written
