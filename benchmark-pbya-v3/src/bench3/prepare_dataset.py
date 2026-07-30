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
    slabs — the same construction, expressed in the only unit available. The trim
    is a quantile of cells at each end rather than named planes, since no
    published trim exists for these volumes.

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
    DATASET_NAME, dataset_path, resolve_raw, section_label, spec,
)
from .sources import load_source


# ── Raw-volume readers ────────────────────────────────────────────────────────

def extract_xyz(adata):
    """Return ``(coords (n, 3), units, source)`` — units is 'voxels' or 'um'.

    Accepts a raw volume or a v1-*processed* file. Getting the units wrong is a
    silent error — re-applying a voxel calibration to micrometres would shrink x
    and y while leaving z alone, distorting every spatial metric — so units are
    determined alongside the coordinates, never assumed.

    Priority mirrors the v1 processor: ``obs['x','y','z']`` first (the STARmap
    distribution carries the z-planes there even when ``obsm['spatial']`` is only
    2-D, and v1 leaves those columns as original voxel indices), then a 3-D obsm,
    then 2-D obsm + an obs z column.
    """
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
        "Could not find 3-D coordinates: expected obs['x','y','z'] or a 3-D "
        "obsm['spatial']/'spatial3d'."
    )


def cell_type_series(adata):
    """Cell-type labels, matching the v1 processor's choice for comparability."""
    for col in ("cell_type", "leiden"):
        if col in adata.obs.columns:
            vals = adata.obs[col].astype(str).values
            if len(np.unique(vals)) > 1 or col == "cell_type":
                return vals, col
    return np.array(["unannotated"] * adata.n_obs), "none"


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


# ── Build ─────────────────────────────────────────────────────────────────────

def build(dataset=DATASET_NAME, raw_path=None, output_path=None, flatten_z=True,
          n_sections=None, verbose=True):
    """Build and write a dataset's paper-protocol view. Returns the AnnData."""
    s = spec(dataset)
    raw_path = Path(raw_path or resolve_raw(dataset))
    if output_path is None:
        output_path = dataset_path(dataset)
    n_sections = int(n_sections or s["n_sections"])
    held_out = tuple(section_label(i) for i in s["held_out"])

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

    coords, units, source = extract_xyz(adata)
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
        coords_um = np.column_stack([coords[:, :2] * s["voxel_xy_um"],
                                     coords[:, 2] * s["voxel_z_um"]])
    else:
        coords_um = coords
    if verbose:
        print(f"  coordinates: {source}, units={units}"
              f"{' -> converted to um' if units == 'voxels' else ' (already um)'}")

    z_um_all = coords_um[:, 2]

    if s["partition"] == "planes":
        planes_all = np.rint(z_um_all / s["voxel_z_um"]).astype(int)
        kept, assignment = partition_planes(planes_all, n_sections,
                                            s.get("drop_z_low"), s.get("drop_z_high"))
        keep_mask = np.isin(planes_all, kept)
        sec_idx_all = np.array([assignment.get(int(p), 0) for p in planes_all])
        trim_desc = (f"z {s['drop_z_low'][0]}-{s['drop_z_low'][1]} and "
                     f"{s['drop_z_high'][0]}-{s['drop_z_high'][1]}")
        extra = (f"; {len(kept)} planes remain (z {kept.min()}-{kept.max()})")
    elif s["partition"] == "z_width":
        keep_mask, sec_idx_all, edges = partition_z_width(
            z_um_all, n_sections, s.get("z_trim_quantile", 0.0))
        planes_all = np.rint(z_um_all).astype(int)
        q = s.get("z_trim_quantile", 0.0)
        trim_desc = f"outermost {q:.1%} of cells by z at each end"
        extra = (f"; z {edges[0]:.1f}-{edges[-1]:.1f} um cut into {n_sections} "
                 f"slabs of {np.diff(edges)[0]:.2f} um")
    else:
        raise ValueError(f"unknown partition mode {s['partition']!r}")

    if verbose:
        n_drop = int((~keep_mask).sum())
        print(f"  trimmed {trim_desc}: dropped {n_drop} cells "
              f"({n_drop / adata.n_obs:.1%}){extra}")

    out = adata[keep_mask].copy()
    coords_um = coords_um[keep_mask]
    planes = planes_all[keep_mask]
    sec_idx = sec_idx_all[keep_mask]
    z_um = coords_um[:, 2]

    out.X = out.X.tocsr() if sp.issparse(out.X) else sp.csr_matrix(out.X)
    out.var_names_make_unique()

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

    panel = set(map(str, out.var_names))
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
        "marker_genes": [g for g in s["marker_genes"] if g in panel],
        "layer_superficial": [g for g in s["layer_superficial"] if g in panel],
        "layer_deep": [g for g in s["layer_deep"] if g in panel],
    }

    verify(out, n_sections=n_sections, held_out=held_out, verbose=verbose)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  writing {output_path} ...")
        out.write_h5ad(str(output_path))
    return out


def verify(adata, n_sections, held_out, verbose=True):
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
    assert all(a < b for a, b in zip(zs, zs[1:])), "section z must increase with index"

    missing = [s for s in held_out if s not in uniq]
    assert not missing, f"held-out sections absent from the build: {missing}"

    pp = adata.uns["paper_protocol"]
    if verbose:
        print(f"  verified: {adata.n_obs} cells x {adata.n_vars} genes, "
              f"{n_sections} sections [{pp['kind']}]")
        for s_, zc in zip(uniq, zs):
            n = int((labels == s_).sum())
            role = "HELD OUT" if s_ in held_out else "input   "
            zr = pp["z_ranges_um"][s_]
            print(f"    {s_}  z {zr[0]:7.2f}-{zr[1]:<7.2f}  centre={zc:7.2f} um  "
                  f"n={n:5d}  {role}")
        print(f"  marker genes present: {pp['marker_genes']}")
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
    ap.add_argument("--print-protocol", action="store_true",
                    help="print the resolved protocol as JSON and exit")
    args = ap.parse_args()

    adata = build(dataset=args.dataset, raw_path=args.raw, output_path=args.output,
                  flatten_z=not args.no_flatten_z, n_sections=args.n_sections)
    if args.print_protocol:
        print(json.dumps(adata.uns["paper_protocol"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
