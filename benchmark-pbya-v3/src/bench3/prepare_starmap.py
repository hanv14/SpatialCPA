#!/usr/bin/env python
"""Build the SpatialZ-paper STARmap visual-cortex dataset (7 consecutive sections).

Protocol (SpatialZ, Lin et al. 2025 — STARmap visual cortex benchmark)
----------------------------------------------------------------------
1. Start from the raw 3-D STARmap volume (Wang et al. 2018): single-cell
   resolution, 89 consecutive z-planes (z = 6..94).
2. Drop the uppermost planes (z = 6-13) and the lowermost planes (z = 91-94) to
   reduce technical noise. 77 planes remain (z = 14..90).
3. Partition the remaining tissue into **seven consecutive 2-D sections**. 77
   planes divide into exactly 11 planes per section, so the partition is even
   and needs no fudging:

       section_1: z 14-24    section_5: z 58-68
       section_2: z 25-35    section_6: z 69-79
       section_3: z 36-46    section_7: z 80-90
       section_4: z 47-57

4. Sections 2, 4 and 6 are held out (simulated missing slices); 1, 3, 5 and 7
   are the input. That split lives in ``design.py`` — this script only builds the
   dataset, which is identical for every method.

Flattening
----------
Each partition is an 11-plane slab of the original volume, but the protocol
treats it as a *2-D section* — the thing a microtome would produce and the thing
a virtual slice has to be. So by default each slab's cells keep their real
(x, y) and are assigned the slab's centre z (``--no-flatten-z`` disables this).
The original plane index survives in ``obs['z_plane']``, so nothing is lost.

Output matches the standard benchmark contract used by v1/v2:
CSR ``X``, 3-D ``obsm['spatial']`` (micrometres), ``obs['section']``,
``obs['cell_type']``.

Usage:
    python -m src.bench3.prepare_starmap                    # default output
    python -m src.bench3.prepare_starmap --raw /path/raw.h5ad --output out.h5ad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .config import (
    DATASET_NAME, DATASET_PATH, DROP_Z_HIGH, DROP_Z_LOW, HELD_OUT_SECTIONS,
    INPUT_SECTIONS, MARKER_GENES, N_SECTIONS, RAW_STARMAP,
    RAW_STARMAP_CANDIDATES, VOXEL_XY_UM, VOXEL_Z_UM, section_label,
)


# ── Raw-volume readers ────────────────────────────────────────────────────────

def _extract_xyz(adata):
    """Return ``(coords (n, 3), units, source)`` — units is 'voxels' or 'um'.

    Accepts either the raw STARmap volume or the v1-*processed* file, which is
    the same cells with coordinates already in micrometres. Getting the units
    wrong is a silent error — re-applying the voxel calibration to micrometres
    would shrink x/y by 0.859 while leaving z alone, distorting every spatial
    metric — so units are determined alongside the coordinates, never assumed.

    Priority mirrors the v1 processor: ``obs['x','y','z']`` first (the STARmap
    distribution carries the 89 z-planes there even when ``obsm['spatial']`` is
    only 2-D, and v1 leaves those columns as the original voxel indices), then a
    3-D obsm, then 2-D obsm + an obs z column.
    """
    for xc, yc, zc in (("x", "y", "z"), ("X", "Y", "Z")):
        if all(c in adata.obs.columns for c in (xc, yc, zc)):
            return np.column_stack([
                adata.obs[xc].values.astype(np.float64),
                adata.obs[yc].values.astype(np.float64),
                adata.obs[zc].values.astype(np.float64),
            ]), "voxels", f"obs['{xc}','{yc}','{zc}']"

    # No native voxel columns: fall back to obsm, and trust the file's own
    # declaration of units (v1 writes 'micrometers' into spatial_metadata).
    declared = str((adata.uns.get("spatial_metadata") or {})
                   .get("coordinate_units", "")).lower()
    units = "um" if declared.startswith(("micro", "um", "µm")) else "voxels"
    for key in ("spatial3d", "spatial"):
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key], dtype=np.float64)
            if coords.shape[1] == 3:
                return coords, units, f"obsm['{key}']"
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


def _cell_type_series(adata):
    """Cell-type labels, matching the v1 processor's choice for comparability."""
    if "cell_type" in adata.obs.columns:
        return adata.obs["cell_type"].astype(str).values, "cell_type"
    if "leiden" in adata.obs.columns:
        return adata.obs["leiden"].astype(str).values, "leiden"
    return np.array(["unannotated"] * adata.n_obs), "none"


# ── Paper partition ───────────────────────────────────────────────────────────

def partition_planes(z_planes, n_sections=N_SECTIONS,
                     drop_low=DROP_Z_LOW, drop_high=DROP_Z_HIGH):
    """Apply the paper's trim + even partition to the sorted unique z-planes.

    Returns
    -------
    kept : sorted unique plane indices that survive the trim
    assignment : dict plane -> 1-based section index
    """
    z_planes = np.asarray(z_planes)
    kept = np.sort(np.unique(z_planes))
    kept = kept[~((kept >= drop_low[0]) & (kept <= drop_low[1]))]
    kept = kept[~((kept >= drop_high[0]) & (kept <= drop_high[1]))]
    if len(kept) < n_sections:
        raise ValueError(
            f"only {len(kept)} planes survive the trim; need >= {n_sections}")

    # np.array_split gives consecutive, near-equal blocks; for the STARmap volume
    # (77 planes / 7 sections) every block is exactly 11 planes.
    blocks = np.array_split(kept, n_sections)
    assignment = {}
    for i, block in enumerate(blocks, start=1):
        for plane in block:
            assignment[int(plane)] = i
    return kept, assignment


# ── Build ─────────────────────────────────────────────────────────────────────

def build(raw_path=RAW_STARMAP, output_path=DATASET_PATH, flatten_z=True,
          n_sections=N_SECTIONS, verbose=True):
    """Build and write the 7-section paper dataset. Returns the AnnData."""
    import anndata as ad

    raw_path = Path(raw_path)
    if not raw_path.exists():
        tried = "\n".join(f"    {c}" for c in RAW_STARMAP_CANDIDATES)
        raise FileNotFoundError(
            f"STARmap volume not found: {raw_path}\n"
            f"  Looked in:\n{tried}\n"
            f"  Fix by any of:\n"
            f"    python -m src.bench3.prepare_starmap --raw /path/to/volume.h5ad\n"
            f"    export BENCH_V3_RAW_STARMAP=/path/to/volume.h5ad\n"
            f"    python ../benchmark-pbya/src/data/download/download_starmap_visual_cortex.py\n"
            f"  Either the raw volume or the v1-processed data.h5ad is accepted."
        )

    if verbose:
        print(f"Loading {raw_path} ...")
    adata = ad.read_h5ad(str(raw_path))
    if verbose:
        print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes")

    coords, units, source = _extract_xyz(adata)
    # Normalize to micrometres once, here, so everything downstream is in one
    # unit regardless of whether the input was raw voxels or already processed.
    if units == "voxels":
        coords_um = np.column_stack([coords[:, :2] * VOXEL_XY_UM,
                                     coords[:, 2] * VOXEL_Z_UM])
    else:
        coords_um = coords
    planes = np.rint(coords_um[:, 2] / VOXEL_Z_UM).astype(int)
    if verbose:
        print(f"  coordinates: {source}, units={units}"
              f"{' -> converted to um' if units == 'voxels' else ' (already um)'}")

    kept, assignment = partition_planes(planes, n_sections=n_sections)
    keep_mask = np.isin(planes, kept)
    if verbose:
        n_drop = int((~keep_mask).sum())
        print(f"  trimmed z {DROP_Z_LOW[0]}-{DROP_Z_LOW[1]} and "
              f"{DROP_Z_HIGH[0]}-{DROP_Z_HIGH[1]}: dropped {n_drop} cells "
              f"({n_drop / adata.n_obs:.1%}); {len(kept)} planes remain "
              f"(z {kept.min()}-{kept.max()})")

    out = adata[keep_mask].copy()
    coords_um = coords_um[keep_mask]
    planes = planes[keep_mask]

    # X: CSR, unmodified counts.
    out.X = out.X.tocsr() if sp.issparse(out.X) else sp.csr_matrix(out.X)
    out.var_names_make_unique()

    # Section assignment (1-based, ordered by z).
    sec_idx = np.array([assignment[int(p)] for p in planes], dtype=int)
    sections = np.array([section_label(i) for i in sec_idx])

    xy_um = coords_um[:, :2]
    z_um = coords_um[:, 2]

    # Flatten each slab onto its centre plane: the protocol treats a partition as
    # a 2-D section, and a virtual slice is a plane.
    if flatten_z:
        z_section = np.empty_like(z_um)
        for i in range(1, n_sections + 1):
            m = sec_idx == i
            z_section[m] = float(np.median(z_um[m]))
    else:
        z_section = z_um

    out.obsm["spatial"] = np.column_stack([xy_um, z_section])
    out.obs["section"] = sections
    out.obs["section_index"] = sec_idx
    out.obs["z_plane"] = planes                       # provenance: original plane
    out.obs["z_original_um"] = z_um
    out.obs["role"] = np.where(np.isin(sections, list(HELD_OUT_SECTIONS)),
                               "held_out", "input")

    cts, ct_source = _cell_type_series(out)
    out.obs["cell_type"] = cts
    out.obs["section"] = out.obs["section"].astype(str)

    # Drop stale graphs/embeddings computed on the untrimmed volume — they would
    # be silently wrong for this subset (and obsp is sized for the old n_obs).
    for key in list(out.obsp.keys()):
        del out.obsp[key]
    for key in ("neighbors", "spatial_neighbors", "umap", "moranI"):
        out.uns.pop(key, None)
    out.obsm.pop("X_umap", None)

    section_z = {section_label(i): float(np.median(z_section[sec_idx == i]))
                 for i in range(1, n_sections + 1)}
    plane_ranges = {section_label(i): [int(min(p for p, s in assignment.items() if s == i)),
                                       int(max(p for p, s in assignment.items() if s == i))]
                    for i in range(1, n_sections + 1)}

    out.uns["expression_type"] = "raw_counts"
    out.uns["dataset_name"] = DATASET_NAME
    out.uns["spatial_metadata"] = {
        "technology": "STARmap",
        "species": "mouse",
        "tissue": "visual cortex",
        "n_sections": int(n_sections),
        "coordinate_units": "micrometers",
        "expression_type": "raw_counts",
        "source": "Wang et al. 2018 Science; starmapresources.org",
    }
    out.uns["paper_protocol"] = {
        "paper": "SpatialZ (Lin et al. 2025, Nature Methods) — STARmap visual cortex",
        "dropped_z_planes": [list(DROP_Z_LOW), list(DROP_Z_HIGH)],
        "n_sections": int(n_sections),
        "planes_per_section": {k: v[1] - v[0] + 1 for k, v in plane_ranges.items()},
        "plane_ranges": plane_ranges,
        "section_z_um": section_z,
        "held_out_sections": list(HELD_OUT_SECTIONS),
        "input_sections": list(INPUT_SECTIONS),
        "flattened_z": bool(flatten_z),
        "cell_type_source": ct_source,
        "marker_genes": [g for g in MARKER_GENES if g in out.var_names],
    }

    verify(out, n_sections=n_sections, verbose=verbose)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  writing {output_path} ...")
        out.write_h5ad(str(output_path))
    return out


def verify(adata, n_sections=N_SECTIONS, verbose=True):
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

    missing = [s for s in HELD_OUT_SECTIONS if s not in uniq]
    assert not missing, f"held-out sections absent from the build: {missing}"

    if verbose:
        print(f"  verified: {adata.n_obs} cells x {adata.n_vars} genes, "
              f"{n_sections} sections")
        pr = adata.uns["paper_protocol"]["plane_ranges"]
        for s, zc in zip(uniq, zs):
            n = int((labels == s).sum())
            role = "HELD OUT" if s in HELD_OUT_SECTIONS else "input   "
            print(f"    {s}  planes {pr[s][0]:>3d}-{pr[s][1]:<3d}  "
                  f"z={zc:7.2f} um  n={n:5d}  {role}")
        present = [g for g in MARKER_GENES if g in adata.var_names]
        print(f"  marker genes present: {present}")


def main():
    ap = argparse.ArgumentParser(
        description="Build the SpatialZ-paper STARmap 7-section dataset")
    ap.add_argument("--raw", default=str(RAW_STARMAP), help="raw STARmap h5ad")
    ap.add_argument("--output", default=str(DATASET_PATH), help="output h5ad")
    ap.add_argument("--n-sections", type=int, default=N_SECTIONS)
    ap.add_argument("--no-flatten-z", action="store_true",
                    help="keep each cell's original z instead of collapsing each "
                         "section onto its centre plane (not the paper protocol)")
    ap.add_argument("--print-protocol", action="store_true",
                    help="print the resolved protocol as JSON and exit")
    args = ap.parse_args()

    adata = build(raw_path=args.raw, output_path=args.output,
                  flatten_z=not args.no_flatten_z, n_sections=args.n_sections)
    if args.print_protocol:
        print(json.dumps(adata.uns["paper_protocol"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
