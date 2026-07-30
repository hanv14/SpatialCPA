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

IMC_Z_STEP_UM = 10.0        # fallback when the filename does not name the step


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


def read_imc_zstack(path, verbose=True):
    """Read the 3-D IMC breast-cancer raw distribution: one h5ad per z-section.

    Kuett et al. ship ``MainHer2BreastCancerModel_zstep10_<i>.h5ad``, one file per
    serial section at 10 um spacing. Mirrors v1's processor: z from the file index
    times the step, ``section = z<i>``, cell types from the file's own column (or
    leiden, else unannotated), CSR intensities.

    ``path`` may be the raw directory or one of the files in it.
    """
    import anndata as ad
    import re

    path = Path(path)
    raw_dir = path if path.is_dir() else path.parent
    files = sorted(raw_dir.glob("*zstep*_*.h5ad"),
                   key=lambda p: int(re.search(r"_(\d+)\.h5ad$", p.name).group(1))
                   if re.search(r"_(\d+)\.h5ad$", p.name) else 0)
    if not files:
        raise FileNotFoundError(
            f"no IMC z-stack files under {raw_dir}\n"
            f"  expected e.g. MainHer2BreastCancerModel_zstep10_0.h5ad ... _14.h5ad")

    step = IMC_Z_STEP_UM
    m = re.search(r"zstep(\d+)", files[0].name)
    if m:
        step = float(m.group(1))          # the step is named in the file itself
    if verbose:
        print(f"  reading {len(files)} IMC sections from {raw_dir.name} "
              f"(z step {step:g} um)")

    slices = []
    for i, f in enumerate(files):
        a = ad.read_h5ad(str(f))
        xy = None
        for key in ("spatial", "spatial3d"):
            if key in a.obsm:
                xy = np.asarray(a.obsm[key], dtype=np.float64)[:, :2]
                break
        if xy is None:
            for xc, yc in (("x", "y"), ("X", "Y")):
                if xc in a.obs.columns and yc in a.obs.columns:
                    xy = np.column_stack([a.obs[xc].values.astype(np.float64),
                                          a.obs[yc].values.astype(np.float64)])
                    break
        if xy is None:
            raise ValueError(f"{f.name}: no spatial coordinates in obsm or obs")

        a.obsm["spatial"] = np.column_stack([xy, np.full(len(xy), i * step)])
        a.obs["section"] = f"z{i}"
        if "cell_type" not in a.obs.columns:
            a.obs["cell_type"] = (a.obs["leiden"].astype(str)
                                  if "leiden" in a.obs.columns else "unannotated")
        a.X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
        a.var_names_make_unique()
        a.obs_names = [f"z{i}_{n}" for n in a.obs_names]
        slices.append(a)

    adata = ad.concat(slices, join="outer", merge="first")
    adata.obs_names_make_unique()
    adata.X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    adata.obs["section"] = adata.obs["section"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.uns["expression_type"] = "fluorescence_intensity"
    adata.uns["dataset_name"] = "imc_breast_cancer"
    adata.uns["spatial_metadata"] = {
        "technology": "3D IMC", "species": "human",
        "tissue": "breast cancer (HER2+)", "coordinate_units": "micrometers",
        "expression_type": "fluorescence_intensity",
        "source": "Zenodo 10.5281/zenodo.4752030 (raw, read by bench3.sources)",
    }
    if verbose:
        print(f"  {adata.n_obs} cells x {adata.n_vars} channels, "
              f"{adata.obs['section'].nunique()} sections")
    return adata


DEEP_STARMAP_FILES = ("Brain_Deep_STARmap_expression_matrix.csv",
                      "Brain_Deep_STARmap_spatial.csv")
DEEP_STARMAP_VOXEL_XY_UM = 0.32
DEEP_STARMAP_VOXEL_Z_UM = 0.70


def read_deep_starmap_csv(path, verbose=True):
    """Read the raw Deep-STARmap distribution (Sui et al. 2025) as AnnData.

    Expression and spatial CSVs, positionally aligned. Voxel indices become
    micrometres with the paper's calibration (0.32 x 0.32 x 0.70 um), cell types
    come from the spatial file's FUSEmap annotation, and the z value doubles as
    the section label — exactly what v1's processor does.
    """
    import anndata as ad
    import pandas as pd

    path = Path(path)
    raw_dir = path if path.is_dir() else path.parent
    expr_path, spatial_path = (raw_dir / f for f in DEEP_STARMAP_FILES)
    missing = [f for f in DEEP_STARMAP_FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Deep-STARmap source incomplete under {raw_dir}: missing {missing}")

    if verbose:
        print(f"  reading {expr_path.name} + {spatial_path.name} ...")
    expr = pd.read_csv(expr_path, index_col=0)
    spatial = pd.read_csv(spatial_path, index_col=0)
    if len(expr) != len(spatial):
        raise ValueError(
            f"Deep-STARmap files disagree on cell count: expression {len(expr)} "
            f"vs spatial {len(spatial)} (they are positionally aligned)")

    coords = np.column_stack([
        spatial["x"].values.astype(np.float64) * DEEP_STARMAP_VOXEL_XY_UM,
        spatial["y"].values.astype(np.float64) * DEEP_STARMAP_VOXEL_XY_UM,
        spatial["z"].values.astype(np.float64) * DEEP_STARMAP_VOXEL_Z_UM,
    ])
    ct_col = next((c for c in ("FUSEmap_sub_level", "Harmony_labels",
                               "FUSEmap_main_level", "cell_type")
                   if c in spatial.columns), None)
    cell_types = (spatial[ct_col].values.astype(str) if ct_col
                  else np.array(["unannotated"] * len(spatial)))
    if verbose:
        print(f"  cell types from {ct_col or 'nothing (unannotated)'}: "
              f"{len(np.unique(cell_types))} types")

    adata = ad.AnnData(
        X=sp.csr_matrix(expr.values.astype(np.float32)),
        obs=pd.DataFrame({"section": spatial["z"].values.astype(str),
                          "cell_type": cell_types},
                         index=[f"cell_{i}" for i in range(len(expr))]),
        var=pd.DataFrame(index=pd.Index(expr.columns, name=None)))
    adata.obsm["spatial"] = coords
    adata.uns["expression_type"] = "raw_counts"
    adata.uns["dataset_name"] = "deep_starmap"
    adata.uns["spatial_metadata"] = {
        "technology": "Deep-STARmap", "species": "mouse",
        "tissue": "brain (thick blocks)", "coordinate_units": "micrometers",
        "expression_type": "raw_counts",
        "source": "Zenodo 10.5281/zenodo.16783354 (raw, read by bench3.sources)",
    }
    return adata


READERS = {"exseq_csv": read_exseq_csv, "imc_zstack": read_imc_zstack,
           "deep_starmap_csv": read_deep_starmap_csv}


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
