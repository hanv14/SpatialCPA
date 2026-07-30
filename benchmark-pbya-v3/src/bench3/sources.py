"""Read a dataset's source volume into an AnnData v3 can partition.

Most sources are already an ``.h5ad`` — a raw distribution or one of
``benchmark-pbya``'s processed files — and need no help. ExSeq is not: its raw
distribution is a cell-by-gene CSV plus a separate annotation file, and v1's
processed h5ad (which v3 also accepts) is produced from it by
``benchmark-pbya/src/data/process/process_exseq_visual_cortex.py``.

``read_exseq_csv`` reads that raw form directly, mirroring what the v1 processor
does, so v3 can build from ``data/raw/`` without requiring v1's pipeline to have
been run first. It is a re-implementation, not a call into v1 — v3 does not modify
or import anything outside itself — and the two are kept equivalent on the points
that matter: micrometre coordinates straight from ``x_um/y_um/z_um``, CSR counts,
one ``visual_cortex`` section, and cell types taken from the SpaceTx EDV results
when they are present and row-aligned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

EXSEQ_CSV = "spacejam2_cellxgene.csv"
EXSEQ_EDV = "results_adata.h5ad"
EXSEQ_EDV_COLUMN = "edv_predictions_|_merged_cluster_smFISH"
EXSEQ_COORD_COLS = ("x_um", "y_um", "z_um")


def read_exseq_csv(path, verbose=True):
    """Read the raw ExSeq visual-cortex distribution (spacejam2) as AnnData.

    ``path`` may be the raw directory or the CSV itself.
    """
    import anndata as ad
    import pandas as pd

    path = Path(path)
    csv_path = path / EXSEQ_CSV if path.is_dir() else path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ExSeq source not found: {csv_path}\n"
            f"  Expected {EXSEQ_CSV} in the raw directory, e.g.\n"
            f"    benchmark-pbya/data/raw/exseq_visual_cortex/{EXSEQ_CSV}")

    if verbose:
        print(f"  reading {csv_path.name} ...")
    df = pd.read_csv(csv_path, index_col=0)
    missing = [c for c in EXSEQ_COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing coordinate column(s) {missing}; "
            f"found {list(df.columns)[:8]}...")

    gene_cols = [c for c in df.columns if c not in EXSEQ_COORD_COLS]
    if not gene_cols:
        raise ValueError(f"{csv_path.name} carries no gene columns")

    X = sp.csr_matrix(df[gene_cols].values.astype(np.float32))
    coords = np.column_stack([df[c].values.astype(np.float64)
                              for c in EXSEQ_COORD_COLS])

    # Cell types live in the SpaceTx EDV results, row-aligned with the CSV. v1
    # verified that alignment by coordinate matching; the length check plus the
    # column check is the same guard, and anything else stays "unknown" rather
    # than being invented.
    cell_types = np.array(["unknown"] * len(df), dtype=object)
    edv_path = (csv_path.parent / EXSEQ_EDV)
    if edv_path.exists():
        try:
            edv = ad.read_h5ad(str(edv_path))
            if len(edv) == len(df) and EXSEQ_EDV_COLUMN in edv.obs.columns:
                cell_types = edv.obs[EXSEQ_EDV_COLUMN].values.astype(str)
                if verbose:
                    n_typed = int((cell_types != "unknown").sum())
                    print(f"  cell types from {EXSEQ_EDV}: {n_typed}/{len(df)} "
                          f"cells, {len(np.unique(cell_types))} types")
            elif verbose:
                print(f"  WARNING: {EXSEQ_EDV} is not row-aligned with the CSV "
                      f"({len(edv)} vs {len(df)}) — cell types left unknown")
        except Exception as e:                       # annotation is optional
            if verbose:
                print(f"  WARNING: could not read {EXSEQ_EDV} ({e}); "
                      f"cell types left unknown")
    elif verbose:
        print(f"  note: no {EXSEQ_EDV} beside the CSV — cell types will be "
              f"'unknown', so paper_celltype_* will be unavailable")

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"section": "visual_cortex",
                          "cell_type": cell_types.astype(str)},
                         index=df.index.astype(str)),
        var=pd.DataFrame(index=pd.Index(gene_cols, name=None)))
    adata.obsm["spatial"] = coords
    adata.uns["expression_type"] = "raw_counts"
    adata.uns["dataset_name"] = "exseq_visual_cortex"
    adata.uns["spatial_metadata"] = {
        "technology": "ExSeq", "species": "mouse", "tissue": "visual cortex",
        "coordinate_units": "micrometers", "expression_type": "raw_counts",
        "source": "Boyden Lab / spacejam2 (raw, read by bench3.sources)",
    }
    return adata


READERS = {"exseq_csv": read_exseq_csv}


def load_source(path, spec, verbose=True):
    """Load a dataset's source volume, dispatching on the spec's ``reader``.

    ``reader="auto"`` (the default) reads ``.h5ad`` directly; anything else is
    handed to the named reader. A spec that names a reader still gets the h5ad
    path when it points at one, so a raw directory and v1's processed file are
    both valid sources for the same dataset.
    """
    import anndata as ad

    path = Path(path)
    if path.suffix == ".h5ad" and path.is_file():
        return ad.read_h5ad(str(path))

    reader = spec.get("reader", "auto")
    if reader in READERS:
        return READERS[reader](path, verbose=verbose)

    raise ValueError(
        f"don't know how to read {path} for this dataset "
        f"(reader={reader!r}); expected an .h5ad, or a raw directory for a "
        f"dataset whose spec names a reader in bench3.sources.READERS")
