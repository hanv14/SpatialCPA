#!/usr/bin/env python
"""Build a v3 paper-protocol dataset: a volume -> N consecutive 2-D sections.

The protocol is the SpatialZ STARmap one (``config.DATASET_SPECS``):

1. Read a 3-D volume at single-cell resolution and put coordinates in µm.
2. Trim the noisy extremes of z.
3. Partition what remains into ``n_sections`` **consecutive** sections.
4. Flatten each section: cells keep their real (x, y) and take the section's
   centre z, because the protocol treats a partition as a 2-D section and a
   virtual slice has to be a plane. Provenance is kept in ``obs['z_plane']`` /
   ``obs['z_original_um']``; ``--no-flatten-z`` opts out.
5. Hold out the even-indexed sections (2, 4, 6) — that split lives in
   ``design.py``; this script only builds the dataset, which every method shares.

Steps 2 and 3 come in two flavours, because volumes come in two shapes.

``partition="planes"`` — discrete optical planes
    STARmap's z are integer plane indices (6..94). The trim is stated in plane
    numbers (drop 6-13 and 91-94) and the surviving *planes* are split into equal
    blocks, so 77 planes become 7 sections of exactly 11. Boundaries land where
    the paper puts them.

``partition="z_width"`` — continuous z
    ExSeq gives every cell its own float z, so there are no planes to count and
    ``np.unique(z)`` is just the cell list. Splitting *those* evenly would make
    sections of equal cell count and unequal thickness, which is not what
    "consecutive sections" means. Instead the z range is cut into equal-width
    slabs — the same construction, expressed in the only unit available.

``partition="sections"`` — real serial sections
    Each source slice already *is* a section (3-D IMC: 15 of them, 10 um apart),
    so they are used as-is and a consecutive window of ``n_sections`` is kept.

A note on trimming
------------------
Only STARmap's trim is protocol. Dropping z 6-13 and 91-94 comes from the paper's
own analysis of that volume, so it is applied always and ``--no-trim`` is refused
for it. The analogue datasets have no published trim and none is invented:
``z_width`` clips a very small quantile purely so that a few outlying z values
cannot stretch the equal-width bin edges, and ``sections`` picks a window only
because the design needs exactly ``n_sections`` of the available slices.
``--no-trim``, ``--z-trim-quantile`` and ``--section-trim`` make both optional.

Usage:
    python -m src.bench3.prepare_dataset --dataset exseq_visual_cortex
    python -m src.bench3.prepare_dataset --dataset starmap_visual_cortex --raw vol.h5ad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .config import (
    DATASET_NAME, RANDOM_SEED, dataset_path, held_out_indices, resolve_raw,
    section_label, spec,
)
from .sources import load_source


# ── Raw-volume readers ────────────────────────────────────────────────────────

def extract_xyz(adata, prefer="auto"):
    """Return ``(coords (n, 3), units, source)`` — units is 'voxels' or 'um'.

    Accepts a raw volume or a v1-*processed* file. Getting the units wrong is a
    silent error — re-applying a voxel calibration to micrometres would shrink x
    and y while leaving z alone, distorting every spatial metric — so units are
    determined alongside the coordinates, never assumed.

    Priority mirrors the v1 processor: ``obs['x','y','z']`` first (the STARmap
    distribution carries the z-planes there even when ``obsm['spatial']`` is only
    2-D, and v1 leaves those columns as original voxel indices), then a 3-D obsm,
    then 2-D obsm + an obs z column.

    ``prefer="obsm"`` skips the obs branch entirely. That is not a tidiness knob:
    the Allen atlases copy their whole metadata CSV into ``obs``, which carries
    *section-local* ``x``/``y`` next to the reconstructed CCF position in
    ``obsm['spatial']``. Taking the obs columns there would silently rebuild the
    volume out of unrelated per-section frames — a stack that looks healthy and
    is geometric nonsense — so those specs say which array to believe rather than
    relying on a priority order that happens to be right for STARmap.
    """
    if prefer not in ("auto", "obsm"):
        raise ValueError(f"unknown coords preference {prefer!r}; use 'auto' or 'obsm'")

    if prefer == "auto":
        for xc, yc, zc in (("x", "y", "z"), ("X", "Y", "Z")):
            if all(c in adata.obs.columns for c in (xc, yc, zc)):
                return np.column_stack([
                    adata.obs[xc].values.astype(np.float64),
                    adata.obs[yc].values.astype(np.float64),
                    adata.obs[zc].values.astype(np.float64),
                ]), "voxels", f"obs['{xc}','{yc}','{zc}']"

    declared = str((adata.uns.get("spatial_metadata") or {})
                   .get("coordinate_units", "")).lower()
    units = "um" if declared.startswith(("micro", "um", "µm")) else "voxels"
    for key in ("spatial3d", "spatial"):
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key], dtype=np.float64)
            if coords.shape[1] >= 3:
                return coords[:, :3], units, f"obsm['{key}']"
            if coords.shape[1] == 2:
                for zc in ("z", "Z"):
                    if zc in adata.obs.columns:
                        return (np.column_stack(
                            [coords, adata.obs[zc].values.astype(np.float64)]),
                            units, f"obsm['{key}'] + obs['{zc}']")
    raise ValueError(
        "Could not find 3-D coordinates: expected a 3-D obsm['spatial']/'spatial3d'"
        + ("" if prefer == "obsm" else ", or obs['x','y','z']")
        + f" (coords preference {prefer!r})."
    )


def check_sections_separate_in_z(z_um, sec_idx, dataset, coord_source):
    """For a serial-section dataset, assert z actually separates the sections.

    ``verify`` checks that section centres increase with section index — but for
    ``partition="sections"`` the index is *assigned* by sorting on those centres,
    so that check is satisfied by construction and cannot fail. It would pass on
    coordinates that have nothing to do with the stack.

    That is not hypothetical. A source whose ``obs`` carries section-*local*
    coordinates next to the reconstructed volume position (the Allen atlases) will
    build a perfectly healthy-looking stack out of the wrong array: right number
    of sections, right cell counts, monotone centres, and geometry that is noise.

    The property that distinguishes them is physical. In real serial sections
    every cell in a section shares that section's z, so the spread *within* a
    section is far smaller than the gap *between* consecutive sections. If the two
    are comparable, z is a per-cell quantity rather than a section coordinate, and
    the stack is not what it claims to be.
    """
    centres, spreads = [], []
    for i in np.unique(sec_idx):
        if i == 0:
            continue
        zi = z_um[sec_idx == i]
        centres.append(float(np.median(zi)))
        spreads.append(float(np.std(zi)))
    if len(centres) < 2:
        return
    centres = np.sort(np.asarray(centres))
    gaps = np.diff(centres)
    median_gap = float(np.median(gaps))
    median_spread = float(np.median(spreads))
    if median_gap > 0 and median_spread < 0.5 * median_gap:
        return
    raise ValueError(
        f"{dataset}: z does not separate the sections — median within-section "
        f"spread {median_spread:.3g} um vs median between-section gap "
        f"{median_gap:.3g} um.\n"
        f"  Coordinates were read from {coord_source}.\n"
        f"  For real serial sections every cell in a section shares that "
        f"section's z, so the spread must be small against the gap. Comparable "
        f"values mean these are per-cell coordinates inside each section, not a "
        f"stack coordinate — a common shape when a source carries section-local "
        f"x/y/z alongside a reconstructed volume position.\n"
        f"  Fix by pointing the dataset at the volume coordinates: set "
        f"\"coords\": \"obsm\" in config.DATASET_SPECS[{dataset!r}] if the "
        f"reconstructed position lives in obsm['spatial'].")


# ── Size caps ─────────────────────────────────────────────────────────────────
# Both are applied to the *built dataset*, so the ground truth and every method's
# input carry the same cells and the same genes. Capping only the method input
# would leave the evaluator scoring against a panel the method never saw.

def select_genes(adata, n_hvg, keep, verbose=True):
    """Reduce to ``n_hvg`` highly variable genes, force-keeping ``keep``.

    Whole-transcriptome datasets (Open-ST, ST, Visium) arrive with 20-35k genes
    against the 28-1000 of the targeted panels. That breaks the benchmark twice
    over: the per-gene autocorrelation families average over tens of thousands of
    near-empty genes, so the number that comes out is dominated by genes carrying
    no spatial signal at all; and any method that materializes a dense cell-by-gene
    matrix cannot load the input.

    ``keep`` — the dataset's own marker and layer genes — is force-included, so
    the ``paper_marker_*`` group can never be silenced by the selection.
    """
    if not n_hvg or adata.n_vars <= int(n_hvg):
        return adata, None

    import scanpy as sc

    n_hvg = int(n_hvg)
    keep_idx = {i for i, v in enumerate(adata.var_names)
                if _canon(v) in {_canon(k) for k in keep}}

    work = adata.copy()
    work.X = work.X.tocsr() if sp.issparse(work.X) else sp.csr_matrix(work.X)
    # HVG selection wants log-scale input; the built file keeps its own values.
    counts_like = bool(np.all(np.asarray(
        (work.X[:min(work.n_obs, 2000)]).todense()) % 1 == 0))
    if counts_like:
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
    try:
        sc.pp.highly_variable_genes(work, n_top_genes=n_hvg, flavor="seurat")
        chosen = set(np.where(work.var["highly_variable"].values)[0].tolist())
    except Exception:
        # Fall back to plain variance ranking — a dispersion fit can fail on a
        # panel with too few genes or degenerate means, and losing the cap is a
        # worse outcome than losing the flavour.
        X = work.X
        mean = np.asarray(X.mean(axis=0)).ravel()
        sq = np.asarray(X.multiply(X).mean(axis=0)).ravel() if sp.issparse(X) \
            else np.asarray((X ** 2).mean(axis=0)).ravel()
        var = np.maximum(sq - mean ** 2, 0.0)
        chosen = set(np.argsort(-var)[:n_hvg].tolist())

    chosen |= keep_idx
    order = np.sort(np.fromiter(chosen, dtype=int))
    out = adata[:, order].copy()
    note = {"n_hvg": n_hvg, "genes_before": int(adata.n_vars),
            "genes_after": int(out.n_vars),
            "markers_forced": sorted(str(adata.var_names[i]) for i in keep_idx)}
    if verbose:
        print(f"  genes: {adata.n_vars} -> {out.n_vars} "
              f"(top {n_hvg} highly variable"
              + (f" + {len(keep_idx)} marker/layer forced" if keep_idx else "")
              + ")")
    return out, note


def subsample_sections(adata, sec_idx, max_cells, seed, verbose=True):
    """Cap each section's cell count, sampling uniformly *in space*.

    A plain random draw thins dense regions and sparse ones by the same factor,
    which is what we want — but drawn independently per cell it also leaves the
    per-section spatial support ragged at the low-density edges. Sampling within
    a coarse spatial grid instead keeps the tissue outline and the density
    gradient, which the field and localization metrics both read.

    Returns ``(keep_mask, note)``.
    """
    keep = np.ones(adata.n_obs, dtype=bool)
    if not max_cells:
        return keep, None
    max_cells = int(max_cells)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    rng = np.random.default_rng(seed)

    before = {}
    for i in np.unique(sec_idx):
        rows = np.where(sec_idx == i)[0]
        before[int(i)] = int(len(rows))
        if len(rows) <= max_cells:
            continue
        xy = coords[rows][:, :2]
        # ~sqrt(max_cells) bins per axis puts a handful of cells in each, so the
        # draw is spread over the section rather than over the cell list.
        nb = max(1, int(np.sqrt(max_cells) / 2))
        gx = np.clip(((xy[:, 0] - xy[:, 0].min()) /
                      max(np.ptp(xy[:, 0]), 1e-9) * nb).astype(int), 0, nb - 1)
        gy = np.clip(((xy[:, 1] - xy[:, 1].min()) /
                      max(np.ptp(xy[:, 1]), 1e-9) * nb).astype(int), 0, nb - 1)
        cell = gx * nb + gy
        # Proportional allocation with the remainder given to the fullest bins,
        # so the sample keeps the density field rather than flattening it.
        order = rng.permutation(len(rows))
        ranks = np.empty(len(rows), dtype=int)
        for c in np.unique(cell):
            m = np.where(cell[order] == c)[0]
            ranks[order[m]] = np.arange(len(m))
        quota = np.ceil(np.bincount(cell, minlength=nb * nb) *
                        (max_cells / len(rows))).astype(int)
        take = ranks < quota[cell]
        chosen = np.where(take)[0]
        if len(chosen) > max_cells:            # ceil() overshoots; trim at random
            chosen = rng.choice(chosen, size=max_cells, replace=False)
        drop = np.setdiff1d(np.arange(len(rows)), chosen, assume_unique=False)
        keep[rows[drop]] = False

    after = {int(i): int(keep[sec_idx == i].sum()) for i in np.unique(sec_idx)}
    if not (~keep).any():
        return keep, None
    note = {"max_cells_per_section": max_cells, "seed": int(seed),
            "cells_before": before, "cells_after": after}
    if verbose:
        print(f"  cells: capped at {max_cells}/section — "
              f"{int((~keep).sum())} of {adata.n_obs} dropped "
              f"(per section {min(before.values())}-{max(before.values())} -> "
              f"{min(after.values())}-{max(after.values())})")
    return keep, note


def cell_type_series(adata):
    """Cell-type labels, matching the v1 processor's choice for comparability."""
    for col in ("cell_type", "leiden"):
        if col in adata.obs.columns:
            vals = adata.obs[col].astype(str).values
            if len(np.unique(vals)) > 1 or col == "cell_type":
                return vals, col
    return np.array(["unannotated"] * adata.n_obs), "none"


def _canon(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_markers(requested, panel):
    """Map requested marker names onto the panel's own spelling.

    Antibody channels are written a dozen ways — panCK / PanCK / pan-CK /
    Pan_Cytokeratin — and an exact-match intersection would silently return an
    empty marker set and drop a whole metric group without anyone noticing.
    Matching ignores case and punctuation and returns the panel's spelling, so
    what is recorded is what the data actually calls the channel.
    """
    lookup = {}
    for name in panel:
        lookup.setdefault(_canon(name), str(name))
    out = []
    for want in requested:
        # Exact match after canonicalization — deliberately NOT a prefix match,
        # which would make "CD3" silently select "CD31".
        hit = lookup.get(_canon(want))
        if hit is not None and hit not in out:
            out.append(hit)
    return out


def suggest_markers(missing, panel, n=4):
    """Panel names a missing marker was probably meant to be, for the warning."""
    import difflib
    canon = {_canon(v): str(v) for v in panel}
    hits = difflib.get_close_matches(_canon(missing), list(canon), n=n, cutoff=0.6)
    return [canon[h] for h in hits]


# ── Partition ─────────────────────────────────────────────────────────────────

def partition_planes(z_planes, n_sections, drop_low, drop_high):
    """Trim named planes, then split the survivors into equal consecutive blocks.

    Returns ``(kept_planes, {plane: 1-based section})``.
    """
    kept = np.sort(np.unique(np.asarray(z_planes)))
    if drop_low:
        kept = kept[~((kept >= drop_low[0]) & (kept <= drop_low[1]))]
    if drop_high:
        kept = kept[~((kept >= drop_high[0]) & (kept <= drop_high[1]))]
    if len(kept) < n_sections:
        raise ValueError(
            f"only {len(kept)} planes survive the trim; need >= {n_sections}")
    assignment = {}
    for i, block in enumerate(np.array_split(kept, n_sections), start=1):
        for plane in block:
            assignment[int(plane)] = i
    return kept, assignment


def partition_z_width(z, n_sections, trim_quantile=0.0):
    """Trim the z extremes by quantile, then cut z into equal-width slabs.

    Returns ``(keep_mask, section_index (1-based, 0 where trimmed), edges)``.
    """
    z = np.asarray(z, dtype=np.float64)
    lo, hi = (np.quantile(z, trim_quantile), np.quantile(z, 1.0 - trim_quantile)) \
        if trim_quantile and trim_quantile > 0 else (z.min(), z.max())
    keep = (z >= lo) & (z <= hi)
    if keep.sum() < n_sections:
        raise ValueError(f"only {int(keep.sum())} cells survive the z trim")

    edges = np.linspace(lo, hi, n_sections + 1)
    # digitize is right-open; the final edge must be inclusive so the last slab
    # keeps the cells sitting exactly on hi.
    idx = np.clip(np.digitize(z, edges[1:-1], right=False) + 1, 1, n_sections)
    sec = np.where(keep, idx, 0)
    empty = [i for i in range(1, n_sections + 1) if (sec == i).sum() == 0]
    if empty:
        raise ValueError(
            f"sections {empty} would be empty — the z distribution is too uneven "
            f"for {n_sections} equal-width slabs (try fewer sections)")
    return keep, sec, edges


MIN_CELLS_PER_SECTION = 50
"""Below this a section cannot carry the protocol.

It is not a metric threshold — it is what the *task* needs. A section that thin is
useless as a method input (SpatialZ's flanking-slice PCA needs more cells than
components; every interpolation method needs a neighbourhood), and useless as
ground truth (the binned field, the depth profile and the per-type OT all become
noise). Building such a dataset only defers the failure to a confusing crash
inside a method wrapper, so the build refuses instead.
"""


def suggest_partition(z, n_sections, min_cells, mode="z_width"):
    """(trim, n_sections) combinations that would give every section enough cells.

    Reported in the build error so the fix is a command, not a search.
    """
    z = np.asarray(z, dtype=np.float64)
    out = []
    for q in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        for n in sorted({n_sections, 5, 3}, reverse=True):
            if n < 3:
                continue
            try:
                keep, sec, _ = partition_z_width(z, n, q)
            except ValueError:
                continue
            counts = [int((sec == i).sum()) for i in range(1, n + 1)]
            if min(counts) >= min_cells:
                out.append((q, n, min(counts), max(counts)))
                break                      # the largest n that works at this trim
    return out


def partition_sections(labels_by_z, n_sections, trim="center"):
    """Keep a consecutive window of ``n_sections`` of the dataset's own sections.

    For physically cut serial sections each existing slice already *is* a section,
    so they must not be merged into slabs — merging two 10 um IMC sections would
    invent a 20 um "section" that no microtome produced, and would flatten two
    separately-stained, separately-registered slices onto one plane.

    The window mirrors the paper's trim of the uppermost and lowermost planes:
    ``center`` drops the extremes symmetrically (15 sections -> keep the middle 7),
    ``low`` / ``high`` keep the first / last ``n_sections``.
    """
    if len(labels_by_z) < n_sections:
        raise ValueError(
            f"dataset has {len(labels_by_z)} sections; need >= {n_sections}")
    drop = len(labels_by_z) - n_sections
    if trim == "low":
        start = 0
    elif trim == "high":
        start = drop
    else:
        start = drop // 2
    keep = list(labels_by_z[start:start + n_sections])
    return keep, {lab: i for i, lab in enumerate(keep, start=1)}


# ── Build ─────────────────────────────────────────────────────────────────────

def sanitize_frame(df, what, verbose=True):
    """Make ``obs``/``var`` writable to h5ad, in place.

    Readers that concatenate one file per section with ``join='outer'`` inherit
    every per-file annotation column, and a gene (or cell) missing from the file
    a column came from gets ``NaN``. The column then has object dtype holding a
    mix of real values and ``NaN``, and h5py's variable-length string writer
    rejects it outright:

        TypeError: Can't implicitly convert non-string objects to strings
        Error raised while writing key 'mt' of ... to /var

    Open-ST is the case that hit this — scanpy's boolean ``var['mt']`` QC flag,
    NaN for every gene outside the first section's panel. None of these columns
    is used by the benchmark (only ``var_names`` is), so the goal is simply to
    write something faithful rather than to fail the build: an all-empty column
    is dropped, a bool/NaN column becomes bool, a numeric one stays numeric, and
    anything else becomes a string with missing values as ``''``.
    """
    import pandas as pd

    dropped, coerced = [], []
    for col in list(df.columns):
        s = df[col]
        if s.isna().all():                        # nothing to preserve
            df.drop(columns=[col], inplace=True)
            dropped.append(str(col))
            continue
        if s.dtype != object:                     # numeric, bool, categorical: fine
            continue
        vals = s.dropna()
        if len(vals) == len(s) and vals.map(lambda v: isinstance(v, str)).all():
            continue                              # plain strings: already writable
        if vals.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            df[col] = np.array([bool(v) if pd.notna(v) else False for v in s],
                               dtype=bool)
        elif vals.map(lambda v: isinstance(v, (int, float, np.number))).all():
            df[col] = pd.to_numeric(s, errors="coerce")
        else:
            df[col] = s.fillna("").astype(str)
        coerced.append(str(col))

    if verbose and (dropped or coerced):
        bits = []
        if coerced:
            bits.append(f"coerced {coerced} (mixed types are not h5ad-writable)")
        if dropped:
            bits.append(f"dropped all-empty {dropped}")
        print(f"  {what}: " + "; ".join(bits))


def merge_colocated_sections(secs_all, by_z, z_um_all, tol):
    """Merge source sections whose depths coincide into one section each.

    Some sources label sections by something other than depth. Ortiz names two
    replicate sections at one anterior-posterior coordinate ``23A`` and ``23B``,
    and after registration into the Allen frame both sit at the same z — so the
    "stack" contains pairs of sections at identical depth, which is not a stack
    and which no interpolation method can be asked to order.

    Physically they *are* one plane sampled twice, so they are merged into one
    section rather than dropped: nothing is lost and the depth axis becomes
    strictly increasing, which is what the protocol requires.
    """
    z_of = {lab: float(np.median(z_um_all[secs_all == lab])) for lab in by_z}
    groups, current = [], [by_z[0]]
    for prev, lab in zip(by_z, by_z[1:]):
        if abs(z_of[lab] - z_of[prev]) <= tol:
            current.append(lab)
        else:
            groups.append(current)
            current = [lab]
    groups.append(current)

    remap, new_order = {}, []
    n_merged = 0
    for g in groups:
        name = g[0] if len(g) == 1 else "+".join(g)
        new_order.append(name)
        n_merged += len(g) - 1
        for lab in g:
            remap[lab] = name
    return np.array([remap[s] for s in secs_all]), new_order, n_merged


def drop_nulls(obj):
    """Recursively remove None values so the result is h5ad-writable everywhere.

    Only the *absence* of a key carries meaning here, never a null: every reader
    of ``paper_protocol`` uses ``.get()`` with a default.
    """
    if isinstance(obj, dict):
        return {k: drop_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [drop_nulls(v) for v in obj if v is not None]
    return obj


def build(dataset=DATASET_NAME, raw_path=None, output_path=None, flatten_z=True,
          n_sections=None, verbose=True, trim=True, z_trim_quantile=None,
          section_trim=None, min_cells=MIN_CELLS_PER_SECTION,
          allow_small=False, n_hvg=None, max_cells_per_section=None):
    """Build and write a dataset's paper-protocol view. Returns the AnnData.

    ``trim=False`` disables the dataset's trim where one is optional. STARmap's
    is not: dropping z 6-13 and 91-94 is part of the published protocol, so it is
    applied regardless and ``trim=False`` is refused for it.
    """
    s = spec(dataset)
    raw_path = Path(raw_path or resolve_raw(dataset))
    if output_path is None:
        output_path = dataset_path(dataset)
    n_sections = n_sections if n_sections is not None else s["n_sections"]
    if n_sections is not None:
        n_sections = int(n_sections)
    elif s["partition"] != "sections":
        raise ValueError(
            f"{dataset}: n_sections must be set for partition={s['partition']!r} "
            f"(only 'sections' can take it from the data)")
    if not trim and s["kind"] == "paper":
        raise ValueError(
            f"{dataset}: the trim is part of the published protocol (drop "
            f"z {s['drop_z_low'][0]}-{s['drop_z_low'][1]} and "
            f"{s['drop_z_high'][0]}-{s['drop_z_high'][1]}); refusing --no-trim. "
            f"Use a different dataset if you want an untrimmed volume.")
    q = s.get("z_trim_quantile", 0.0) if z_trim_quantile is None else z_trim_quantile
    if not trim:
        q = 0.0
    window = section_trim or s.get("section_trim", "center")

    if not raw_path.exists():
        tried = "\n".join(f"    {c}" for c in s["raw_candidates"])
        raise FileNotFoundError(
            f"{dataset}: source volume not found: {raw_path}\n"
            f"  Looked in:\n{tried}\n"
            f"  Fix by any of:\n"
            f"    python -m src.bench3.prepare_dataset --dataset {dataset} --raw /path/vol.h5ad\n"
            f"    export {s['env_var']}=/path/to/volume.h5ad\n"
            f"  A benchmark-pbya *processed* data.h5ad is an accepted source."
        )

    if verbose:
        print(f"Loading {raw_path} ...")
        print(f"  output -> {Path(output_path)}")
    adata = load_source(raw_path, s, verbose=verbose)
    if verbose:
        print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes "
              f"[{dataset}, partition={s['partition']}]")

    coords, units, source = extract_xyz(adata, prefer=s.get("coords", "auto"))
    # A spec may state the source units outright. v1's ExSeq processor writes
    # micrometres into obsm['spatial'] but records no ``coordinate_units``, so
    # detection alone would call it voxels; that happens to be harmless only
    # because its calibration is 1.0, which is not something to rely on.
    declared = s.get("source_units", "auto")
    if declared in ("um", "voxels") and declared != units:
        if verbose:
            print(f"  units: detected {units!r}, overridden to {declared!r} by the "
                  f"dataset spec")
        units = declared
    if units == "voxels":
        coords_um = np.column_stack([coords[:, :2] * s.get("voxel_xy_um", 1.0),
                                     coords[:, 2] * s.get("voxel_z_um", 1.0)])
    else:
        coords_um = coords
    if verbose:
        print(f"  coordinates: {source}, units={units}"
              f"{' -> converted to um' if units == 'voxels' else ' (already um)'}")

    z_um_all = coords_um[:, 2]

    # For real serial sections the count can come from the data itself.
    if s["partition"] == "sections" and n_sections is None:
        n_sections = int(len(np.unique(adata.obs["section"].values.astype(str))))
        # Taking the count from the data is convenient but it is also blind: a
        # reader that silently failed to place some sections produces a smaller
        # volume, and since the hold-out pattern is derived from the count
        # (``held_out: "alternate"``), nothing downstream can tell that apart
        # from a dataset that genuinely has fewer sections. Where the source is a
        # published block of known size, say so and check it.
        expected = s.get("expected_sections")
        if expected is not None and n_sections != int(expected):
            raise ValueError(
                f"{dataset}: the source is documented as {expected} sections but "
                f"only {n_sections} arrived from the reader "
                f"({sorted(np.unique(adata.obs['section'].values.astype(str)))}).\n"
                f"  This changes the hold-out split, so the build stops rather "
                f"than quietly producing a different experiment.\n"
                f"  Check the reader's output above for sections it could not "
                f"place, and re-download the raw archive if any are missing.\n"
                f"  To build the smaller volume deliberately, pass "
                f"--n-sections {n_sections}.")
    held_out = tuple(section_label(i) for i in held_out_indices(s, n_sections))

    if s["partition"] == "planes":
        planes_all = np.rint(z_um_all / s["voxel_z_um"]).astype(int)
        kept, assignment = partition_planes(
            planes_all, n_sections,
            s.get("drop_z_low") if trim else None,
            s.get("drop_z_high") if trim else None)
        keep_mask = np.isin(planes_all, kept)
        sec_idx_all = np.array([assignment.get(int(p), 0) for p in planes_all])
        # a plane-partitioned dataset need not have a published trim (only
        # STARmap does); None means "keep every plane"
        parts = [f"z {r[0]}-{r[1]}" for r in (s.get("drop_z_low"),
                                              s.get("drop_z_high")) if r]
        trim_desc = " and ".join(parts) if parts else "none (all planes kept)"
        extra = (f"; {len(kept)} planes remain (z {kept.min()}-{kept.max()})")
    elif s["partition"] == "z_width":
        keep_mask, sec_idx_all, edges = partition_z_width(z_um_all, n_sections, q)
        planes_all = np.rint(z_um_all).astype(int)
        trim_desc = (f"outermost {q:.2%} of cells by z at each end (outlier guard)"
                     if q else "none (full z range)")
        extra = (f"; z {edges[0]:.1f}-{edges[-1]:.1f} um cut into {n_sections} "
                 f"slabs of {np.diff(edges)[0]:.2f} um")
    elif s["partition"] == "sections":
        secs_all = adata.obs["section"].values.astype(str)
        by_z = sorted(np.unique(secs_all),
                      key=lambda lab: float(np.median(z_um_all[secs_all == lab])))
        tol = float(s.get("section_z_tolerance", 0.0))
        if tol > 0:
            secs_all, by_z, n_merged = merge_colocated_sections(
                secs_all, by_z, z_um_all, tol)
            if verbose and n_merged:
                print(f"  merged {n_merged} co-located source sections "
                      f"(within {tol:g} um) -> {len(by_z)} distinct depths")
            if n_sections is None:
                n_sections = len(by_z)
        keep_labels, assignment = partition_sections(by_z, n_sections, window)
        window_note = "" if len(keep_labels) == len(by_z) else f" ({window} window)"
        keep_mask = np.isin(secs_all, keep_labels)
        sec_idx_all = np.array([assignment.get(lab, 0) for lab in secs_all])
        planes_all = np.rint(z_um_all).astype(int)
        keep_labels = [str(l) for l in keep_labels]
        dropped_labels = [str(l) for l in by_z if str(l) not in keep_labels]
        trim_desc = (f"none — all {len(by_z)} source sections kept"
                     if not dropped_labels else
                     f"{len(dropped_labels)} of {len(by_z)} source sections"
                     f"{window_note}: {dropped_labels}")
        extra = f"; kept {keep_labels}"
    else:
        raise ValueError(f"unknown partition mode {s['partition']!r}")

    if verbose:
        n_drop = int((~keep_mask).sum())
        print(f"  trimmed {trim_desc}: dropped {n_drop} cells "
              f"({n_drop / adata.n_obs:.1%}){extra}")

    if s["partition"] == "sections":
        check_sections_separate_in_z(
            z_um_all[keep_mask], sec_idx_all[keep_mask], dataset, source)

    out = adata[keep_mask].copy()
    coords_um = coords_um[keep_mask]
    planes = planes_all[keep_mask]
    sec_idx = sec_idx_all[keep_mask]

    out.X = out.X.tocsr() if sp.issparse(out.X) else sp.csr_matrix(out.X)
    out.var_names_make_unique()

    # Size caps, applied once the sections are known so the cell cap can be
    # per-section. Both are recorded in uns['paper_protocol'] below — a result
    # measured on a capped panel has to say so.
    cap_cells = (s.get("max_cells_per_section")
                 if max_cells_per_section is None else max_cells_per_section)
    if cap_cells:
        out.obsm["spatial"] = np.column_stack([coords_um[:, :2], coords_um[:, 2]])
        sub_keep, subsample_note = subsample_sections(
            out, sec_idx, cap_cells, seed=RANDOM_SEED, verbose=verbose)
        if not sub_keep.all():
            out = out[sub_keep].copy()
            coords_um = coords_um[sub_keep]
            planes = planes[sub_keep]
            sec_idx = sec_idx[sub_keep]
    else:
        subsample_note = None

    cap_genes = s.get("n_hvg") if n_hvg is None else n_hvg
    out, hvg_note = select_genes(
        out, cap_genes,
        keep=tuple(s["marker_genes"]) + tuple(s["layer_superficial"])
             + tuple(s["layer_deep"]),
        verbose=verbose)

    z_um = coords_um[:, 2]

    sections = np.array([section_label(i) for i in sec_idx])

    if flatten_z:
        z_section = np.empty_like(z_um)
        for i in range(1, n_sections + 1):
            m = sec_idx == i
            z_section[m] = float(np.median(z_um[m]))
    else:
        z_section = z_um

    out.obsm["spatial"] = np.column_stack([coords_um[:, :2], z_section])
    out.obs["section"] = sections.astype(str)
    out.obs["section_index"] = sec_idx
    out.obs["z_plane"] = planes                       # provenance
    out.obs["z_original_um"] = z_um
    out.obs["role"] = np.where(np.isin(sections, list(held_out)),
                               "held_out", "input")

    cts, ct_source = cell_type_series(out)
    out.obs["cell_type"] = cts

    # Drop graphs/embeddings computed on the untrimmed volume — obsp is sized for
    # the old n_obs and the rest would be silently wrong for this subset.
    for key in list(out.obsp.keys()):
        del out.obsp[key]
    for key in ("neighbors", "spatial_neighbors", "umap", "moranI"):
        out.uns.pop(key, None)
    out.obsm.pop("X_umap", None)

    section_z = {section_label(i): float(np.median(z_section[sec_idx == i]))
                 for i in range(1, n_sections + 1)}
    z_ranges = {section_label(i): [float(z_um[sec_idx == i].min()),
                                   float(z_um[sec_idx == i].max())]
                for i in range(1, n_sections + 1)}
    plane_ranges = {section_label(i): [int(planes[sec_idx == i].min()),
                                       int(planes[sec_idx == i].max())]
                    for i in range(1, n_sections + 1)}

    panel = [str(v) for v in out.var_names]
    out.uns["expression_type"] = adata.uns.get("expression_type", "raw_counts")
    out.uns["dataset_name"] = dataset
    out.uns["spatial_metadata"] = {
        "technology": s["technology"], "species": s["species"],
        "tissue": s["tissue"], "n_sections": int(n_sections),
        "coordinate_units": "micrometers",
        "expression_type": out.uns["expression_type"],
        "source": s["source"],
    }
    # Self-describing: design.py and evaluate_paper read the protocol from here,
    # so a result can never be scored with another dataset's marker panel.
    out.uns["paper_protocol"] = {
        "paper": "SpatialZ (Lin et al. 2025, Nature Methods) — STARmap visual cortex",
        "kind": s["kind"],
        "protocol": s["protocol"],
        "dataset": dataset,
        "partition": s["partition"],
        "trim": trim_desc,
        "n_sections": int(n_sections),
        "planes_per_section": {k: int((sec_idx == i).sum() and
                                      len(np.unique(planes[sec_idx == i])))
                               for i, k in enumerate(map(section_label,
                                                         range(1, n_sections + 1)), start=1)},
        "plane_ranges": plane_ranges,
        "z_ranges_um": z_ranges,
        "section_z_um": section_z,
        "held_out_sections": list(held_out),
        "input_sections": [section_label(i) for i in range(1, n_sections + 1)
                           if section_label(i) not in held_out],
        "flattened_z": bool(flatten_z),
        "cell_type_source": ct_source,
        "marker_genes": resolve_markers(s["marker_genes"], panel),
        "marker_genes_requested": [str(g) for g in s["marker_genes"]],
        "layer_superficial": resolve_markers(s["layer_superficial"], panel),
        "layer_deep": resolve_markers(s["layer_deep"], panel),
        # A spot array is not the resolution this protocol was designed on, and a
        # capped panel is not the panel the assay produced. Both travel with the
        # data so a row can never be read as something it is not.
        "resolution": s.get("resolution", "single_cell"),
        "gene_selection": hvg_note,
        "cell_subsample": subsample_note,
    }
    # A None in uns is written as h5ad ``encoding_type='null'``, which older
    # anndata cannot read back — and the method conda envs pin older anndata than
    # the build environment. An uncapped dataset therefore produced a
    # train_registered.h5ad that every wrapper crashed on:
    #     IORegistryError: No read method registered for IOSpec(encoding_type='null')
    #     Error raised while reading key 'cell_subsample' of /uns/paper_protocol
    # Absence already means "no cap applied", so the keys are simply dropped
    # rather than written as null.
    out.uns["paper_protocol"] = drop_nulls(out.uns["paper_protocol"])

    sanitize_frame(out.var, "var", verbose=verbose)
    sanitize_frame(out.obs, "obs", verbose=verbose)

    hint = ""
    if s["partition"] == "z_width":
        # the *untrimmed* z: suggestions are (trim, n) pairs to apply from
        # scratch, so computing them on already-trimmed z would double the clip
        opts = suggest_partition(z_um_all, n_sections, min_cells)
        if opts:
            hint = ".\n  These would work: " + "; ".join(
                f"--z-trim-quantile {q:g} --n-sections {n} "
                f"(cells/section {lo}-{hi})" for q, n, lo, hi in opts[:3])
    verify(out, n_sections=n_sections, held_out=held_out, verbose=verbose,
           min_cells=min_cells, allow_small=allow_small, hint=hint)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  writing {output_path} ...")
        out.write_h5ad(str(output_path))
    return out


def verify(adata, n_sections, held_out, verbose=True,
           min_cells=MIN_CELLS_PER_SECTION, allow_small=False, hint=""):
    """Assert the built dataset satisfies the benchmark + protocol contract."""
    assert sp.issparse(adata.X) and adata.X.format == "csr", "X must be CSR sparse"
    assert "spatial" in adata.obsm, "missing obsm['spatial']"
    assert adata.obsm["spatial"].shape == (adata.n_obs, 3), "spatial must be (n, 3)"
    assert "section" in adata.obs.columns, "missing obs['section']"
    assert "cell_type" in adata.obs.columns, "missing obs['cell_type']"

    labels = adata.obs["section"].astype(str).values
    uniq = sorted(set(labels), key=lambda s: int(s.split("_")[1]))
    assert len(uniq) == n_sections, f"expected {n_sections} sections, got {len(uniq)}"

    z = adata.obsm["spatial"][:, 2]
    zs = [float(np.median(z[labels == s])) for s in uniq]
    ties = [(a, b, za) for a, b, za, zb in zip(uniq, uniq[1:], zs, zs[1:]) if za >= zb]
    if ties:
        raise ValueError(
            "section z must increase with index, but these consecutive sections "
            "do not:\n"
            + "\n".join(f"    {a} and {b} both at z = {za:.4g} um"
                         for a, b, za in ties)
            + "\n  Two sections at the same z are not a stack. The usual cause is a "
              "source whose section labels distinguish something other than depth — "
              "replicate sections at one anatomical coordinate (Ortiz names them "
              "'23A'/'23B'), or several animals registered into a shared frame.\n"
              "  Fix by setting \"section_z_tolerance\" in "
              "config.DATASET_SPECS[<dataset>] so labels within that many um of each "
              "other are merged into one section, which is what they physically are.")

    missing = [s for s in held_out if s not in uniq]
    assert not missing, f"held-out sections absent from the build: {missing}"

    counts = {s_: int((labels == s_).sum()) for s_ in uniq}
    thin = {s_: n for s_, n in counts.items() if n < min_cells}
    if thin and not allow_small:
        raise ValueError(
            f"sections too thin to run the protocol (< {min_cells} cells): "
            f"{thin}\n"
            f"  all sections: {counts}\n"
            f"  A section this small breaks methods (a flanking-slice PCA cannot "
            f"ask for more components than cells) and makes the field, depth and "
            f"localization metrics noise.\n"
            f"  Fix by trimming the sparse end of the volume or asking for fewer "
            f"sections{hint}\n"
            f"  or pass --allow-small-sections to build it anyway.")

    pp = adata.uns["paper_protocol"]
    if verbose:
        print(f"  verified: {adata.n_obs} cells x {adata.n_vars} genes, "
              f"{n_sections} sections [{pp['kind']}]")
        for s_, zc in zip(uniq, zs):
            n = int((labels == s_).sum())
            role = "HELD OUT" if s_ in held_out else "input   "
            zr = pp["z_ranges_um"][s_]
            print(f"    {s_:<12s} z {zr[0]:7.2f}-{zr[1]:<7.2f}  centre={zc:7.2f} um  "
                  f"n={n:5d}  {role}")
        req = [str(g) for g in pp.get("marker_genes_requested", [])]
        cts = [int((labels == s_).sum()) for s_ in uniq]
        print(f"  cells/section: min={min(cts)} median={int(np.median(cts))} "
              f"max={max(cts)}"
              + (f"   (imbalance {max(cts) / max(min(cts), 1):.1f}x — check the "
                 f"sparse end)" if max(cts) > 3 * max(min(cts), 1) else ""))
        print(f"  marker genes present: {pp['marker_genes']}"
              + (f"   (requested: {', '.join(req)})" if req else ""))
        matched = {_canon(g) for g in pp["marker_genes"]}
        for want in req:
            if _canon(want) not in matched:
                near = suggest_markers(want, list(adata.var_names))
                print(f"  WARNING: no channel matched {want!r}"
                      + (f" — closest panel names: {', '.join(near)}" if near
                         else " and nothing in the panel is close"))
        print(f"  laminar axis genes:   superficial={pp['layer_superficial']} "
              f"deep={pp['layer_deep']}")
        if not pp["marker_genes"]:
            print("  WARNING: the panel carries none of this dataset's marker "
                  "genes — paper_marker_* will be unavailable.")
        if pp["cell_type_source"] == "none":
            print("  WARNING: no cell-type labels — paper_celltype_* will be "
                  "unavailable.")


def main():
    ap = argparse.ArgumentParser(description="Build a v3 paper-protocol dataset")
    ap.add_argument("--dataset", default=DATASET_NAME,
                    help="dataset name from config.DATASET_SPECS")
    ap.add_argument("--raw", default=None, help="source h5ad (raw or v1-processed)")
    ap.add_argument("--output", default=None, help="output h5ad")
    ap.add_argument("--n-sections", type=int, default=None)
    ap.add_argument("--no-flatten-z", action="store_true",
                    help="keep each cell's original z instead of collapsing each "
                         "section onto its centre plane (not the paper protocol)")
    ap.add_argument("--no-trim", action="store_true",
                    help="skip the optional trim (analogue datasets only). "
                         "STARmap's trim is part of the published protocol and "
                         "cannot be skipped")
    ap.add_argument("--z-trim-quantile", type=float, default=None,
                    help="z_width datasets: fraction of cells clipped at each end "
                         "before equal-width binning (default: the dataset's)")
    ap.add_argument("--section-trim", choices=["low", "center", "high"], default=None,
                    help="sections datasets: which consecutive window of the "
                         "source sections to keep (default: the dataset's)")
    ap.add_argument("--allow-small-sections", action="store_true",
                    help="build even when a section has too few cells to run the "
                         "protocol (it will break methods and make metrics noise)")
    ap.add_argument("--min-cells-per-section", type=int,
                    default=MIN_CELLS_PER_SECTION)
    ap.add_argument("--n-hvg", type=int, default=None,
                    help="cap the panel at this many highly variable genes "
                         "(the dataset's marker/layer genes are always kept). "
                         "Default: the dataset's own; needed for the "
                         "whole-transcriptome datasets")
    ap.add_argument("--max-cells-per-section", type=int, default=None,
                    help="cap each section by spatially stratified subsample "
                         "(default: the dataset's own). Applied to the built "
                         "dataset, so ground truth and method input match")
    ap.add_argument("--print-protocol", action="store_true",
                    help="print the resolved protocol as JSON and exit")
    args = ap.parse_args()

    adata = build(dataset=args.dataset, raw_path=args.raw, output_path=args.output,
                  flatten_z=not args.no_flatten_z, n_sections=args.n_sections,
                  trim=not args.no_trim, z_trim_quantile=args.z_trim_quantile,
                  section_trim=args.section_trim,
                  min_cells=args.min_cells_per_section,
                  allow_small=args.allow_small_sections,
                  n_hvg=args.n_hvg,
                  max_cells_per_section=args.max_cells_per_section)
    if args.print_protocol:
        print(json.dumps(adata.uns["paper_protocol"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
