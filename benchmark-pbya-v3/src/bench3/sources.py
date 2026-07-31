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


# Coordinate columns as the various raw distributions spell them. Matched
# case-insensitively, in order, so the plainest name wins when a file has more
# than one candidate.
COORD_ALIASES = {
    "x": ("x", "x_um", "x_micron", "x_microns", "x_position", "x_centroid",
          "center_x", "centroid_x", "global_x", "spatial_x", "col", "column"),
    "y": ("y", "y_um", "y_micron", "y_microns", "y_position", "y_centroid",
          "center_y", "centroid_y", "global_y", "spatial_y", "row"),
    "z": ("z", "z_um", "z_micron", "z_microns", "z_position", "z_centroid",
          "center_z", "centroid_z", "global_z", "spatial_z", "slice", "plane",
          "section", "z_slice", "zslice"),
}


def pick_coord_column(df, axis, where):
    """The column of ``df`` holding ``axis``, whatever the file happens to call it.

    Raw distributions are not consistent about coordinate column names, and the
    failure mode of assuming one is a bare ``KeyError: 'x'`` that says nothing
    about what the file *does* carry. This resolves the usual spellings and,
    failing that, reports the real header so the fix is a one-line alias.
    """
    lower = {}
    for c in df.columns:                 # first spelling wins on a case collision
        lower.setdefault(str(c).strip().lower(), c)
    for cand in COORD_ALIASES[axis]:
        if cand in lower:
            return lower[cand]
    raise ValueError(
        f"{where}: no {axis} coordinate column. Tried {list(COORD_ALIASES[axis])};\n"
        f"  the file carries {list(df.columns)[:12]}"
        f"{' ...' if len(df.columns) > 12 else ''}\n"
        f"  Add the right spelling to COORD_ALIASES[{axis!r}] in src/bench3/sources.py."
    )


def read_csv_auto_index(path, **kwargs):
    """``read_csv`` that only consumes a first column when it really is an index.

    The Deep-STARmap CSVs carry no index: ``x`` is the spatial file's first data
    column and a gene is the expression file's. Reading them with ``index_col=0``
    silently eats both — the spatial file then fails with ``KeyError: 'x'``, and
    the expression file would have quietly lost its first gene. pandas names a
    genuinely unnamed first column ``Unnamed: 0``, which is the only case where
    consuming it is right.
    """
    import pandas as pd

    df = pd.read_csv(path, **kwargs)
    first = str(df.columns[0])
    if first.startswith("Unnamed:") or first == "":
        df = df.set_index(df.columns[0])
        df.index.name = None
    return df


def coord_is_um(column_name):
    """Whether a resolved coordinate column is already in micrometres.

    Names such as ``x_um`` state their unit; multiplying those by a voxel
    calibration would scale the volume a second time.
    """
    n = str(column_name).strip().lower()
    return n.endswith(("_um", "_micron", "_microns")) or "micron" in n


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

    Expression and spatial CSVs, positionally aligned and carrying **no index
    column** — the spatial file's first column is ``x`` and the expression file's
    is a gene. Voxel indices become micrometres with the paper's calibration
    (0.32 x 0.32 x 0.70 um), cell types come from the spatial file's FUSEmap
    annotation, and the z value doubles as the section label — exactly what v1's
    processor does.
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
    expr = read_csv_auto_index(expr_path)
    spatial = read_csv_auto_index(spatial_path)

    # Only numeric columns are genes. Nothing in the published distribution is
    # non-numeric here, but a re-export carrying a cell-id column would otherwise
    # be cast to float and crash — or worse, be counted as a gene.
    non_numeric = [c for c in expr.columns
                   if not pd.api.types.is_numeric_dtype(expr[c])]
    if non_numeric:
        if verbose:
            print(f"  dropping {len(non_numeric)} non-numeric expression "
                  f"column(s): {non_numeric[:5]}")
        expr = expr.drop(columns=non_numeric)

    if len(expr) != len(spatial):
        raise ValueError(
            f"Deep-STARmap files disagree on cell count: expression {len(expr)} "
            f"vs spatial {len(spatial)} (they are positionally aligned)")

    xc, yc, zc = (pick_coord_column(spatial, a, spatial_path.name)
                  for a in ("x", "y", "z"))
    # Voxel indices unless the column names say micrometres — scaling an
    # already-calibrated coordinate would shrink the volume by 3x in z.
    sx, sy, sz = ((1.0, 1.0, 1.0) if all(map(coord_is_um, (xc, yc, zc)))
                  else (DEEP_STARMAP_VOXEL_XY_UM, DEEP_STARMAP_VOXEL_XY_UM,
                        DEEP_STARMAP_VOXEL_Z_UM))
    if verbose:
        print(f"  coordinates from columns ({xc}, {yc}, {zc}), "
              + ("already in um" if sx == 1.0 else
                 f"voxels x ({sx}, {sy}, {sz}) um"))

    coords = np.column_stack([
        spatial[xc].values.astype(np.float64) * sx,
        spatial[yc].values.astype(np.float64) * sy,
        spatial[zc].values.astype(np.float64) * sz,
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
        obs=pd.DataFrame({"section": spatial[zc].values.astype(str),
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


COSMX_PREPROCESSED_ZIP = "preprocessed_objects.zip"
COSMX_FLAT_ZIP = "cosmx_flat_files.zip"
COSMX_SECTIONS = (4, 10, 16, 22, 28, 34)      # every 6th 5-um cryosection
COSMX_SECTION_RUN = {4: "Run1129", 10: "Run1129", 16: "Run1129",
                     22: "Run1129", 28: "Run1129", 34: "Run1129"}
COSMX_PIXEL_UM = 0.18                          # Nanostring SMI, 20x
COSMX_SECTION_SPACING_UM = 30.0


def read_cosmx_raw(path, verbose=True):
    """Read the raw CosMx NSCLC 3-D distribution (two zips) as AnnData.

    The shipped h5ad carries expression and cell types but STIM-registered
    coordinates in arbitrary [0, 1000] units, which are useless for a benchmark
    measured in micrometres. The physical positions live in the per-section flat
    files, keyed by (section, fov, cell_ID) — the same key encoded in the h5ad's
    obs_names — so the two have to be joined. Mirrors v1's processor: pixels to um
    at 0.18 um/px, each section shifted to its own origin (the stage offset is
    meaningless and differs per section), z from section order at 30 um.
    """
    import tempfile
    import zipfile
    import re

    import anndata as ad
    import pandas as pd

    path = Path(path)
    raw_dir = path if path.is_dir() else path.parent
    pre_zip, flat_zip = raw_dir / COSMX_PREPROCESSED_ZIP, raw_dir / COSMX_FLAT_ZIP
    missing = [z.name for z in (pre_zip, flat_zip) if not z.exists()]
    if missing:
        raise FileNotFoundError(
            f"CosMx source incomplete under {raw_dir}: missing {missing}")

    # ── expression + cell types ──
    with zipfile.ZipFile(pre_zip) as zf:
        members = [n for n in zf.namelist() if n.endswith(".h5ad")]
        if not members:
            raise ValueError(f"no .h5ad inside {pre_zip.name}")
        target = next((m for m in members if m.endswith("cosmx.h5ad")), members[0])
        if verbose:
            print(f"  extracting {target} ...")
        with tempfile.TemporaryDirectory() as tmp:
            zf.extract(target, path=tmp)
            adata = ad.read_h5ad(Path(tmp) / target)
    if verbose:
        print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    if "celltypes" in adata.obs.columns:
        adata.obs = adata.obs.rename(columns={"celltypes": "cell_type"})
    elif "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "unknown"

    # ── physical coordinates from the flat files ──
    coords = {}
    with zipfile.ZipFile(flat_zip) as zf:
        for sec in COSMX_SECTIONS:
            run = COSMX_SECTION_RUN[sec]
            member = (f"cosmx_flat_files/Section_{sec:02d}/"
                      f"{run}_Section_{sec:02d}_metadata_file.csv")
            try:
                with zf.open(member) as f:
                    df = pd.read_csv(f, usecols=["fov", "cell_ID",
                                                 "CenterX_global_px",
                                                 "CenterY_global_px"])
            except KeyError:
                if verbose:
                    print(f"  WARNING: {member} not in {flat_zip.name}; "
                          f"section {sec} will have no coordinates")
                continue
            for fov, cid, cx, cy in zip(df["fov"].astype(int),
                                        df["cell_ID"].astype(int),
                                        df["CenterX_global_px"],
                                        df["CenterY_global_px"]):
                coords[(sec, fov, cid)] = (cx, cy)
    if verbose:
        print(f"  {len(coords)} flat-file coordinates")

    key_re = re.compile(r"section_(\d+)_fov_(\d+)_ID_(\d+)")
    sec_to_z = {sec: float(i) * COSMX_SECTION_SPACING_UM
                for i, sec in enumerate(COSMX_SECTIONS)}

    spatial = np.zeros((adata.n_obs, 3), dtype=np.float64)
    per_section, sections = {}, np.array(["" for _ in range(adata.n_obs)], dtype=object)
    for idx, name in enumerate(adata.obs_names):
        m = key_re.match(str(name))
        if not m:
            continue
        sec, fov, cid = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hit = coords.get((sec, fov, cid))
        if hit is None:
            continue
        per_section.setdefault(sec, []).append((idx, hit[0], hit[1]))
        sections[idx] = f"section_{sec}"

    matched = sum(len(v) for v in per_section.values())
    if matched == 0:
        raise ValueError(
            "no cell matched the flat files — obs_names were expected to look "
            "like 'section_4_fov_1_ID_5'")
    if verbose:
        print(f"  matched {matched}/{adata.n_obs} cells to physical coordinates")

    for sec, cells in per_section.items():
        idxs = [c[0] for c in cells]
        xs = np.array([c[1] for c in cells], dtype=np.float64)
        ys = np.array([c[2] for c in cells], dtype=np.float64)
        # each section has its own arbitrary stage offset; shift to a common origin
        spatial[idxs, 0] = (xs - xs.min()) * COSMX_PIXEL_UM
        spatial[idxs, 1] = (ys - ys.min()) * COSMX_PIXEL_UM
        spatial[idxs, 2] = sec_to_z[sec]
        if verbose:
            print(f"    section_{sec:02d}: {len(cells)} cells, extent "
                  f"{(xs.max()-xs.min())*COSMX_PIXEL_UM:.0f} x "
                  f"{(ys.max()-ys.min())*COSMX_PIXEL_UM:.0f} um, "
                  f"z={sec_to_z[sec]:.0f} um")

    keep = sections != ""
    adata = adata[keep].copy()
    adata.obsm["spatial"] = spatial[keep]
    adata.obs["section"] = sections[keep].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    adata.uns["expression_type"] = "raw_counts"
    adata.uns["dataset_name"] = "cosmx_nsclc_3d"
    adata.uns["spatial_metadata"] = {
        "technology": "CosMx SMI", "species": "human",
        "tissue": "lung cancer (NSCLC)", "coordinate_units": "micrometers",
        "expression_type": "raw_counts",
        "source": "Zenodo 15240431 (raw, read by bench3.sources)",
    }
    return adata


MOFFITT_CSV = "Moffitt2018_MERFISH_hypothalamus_full.csv"
MOFFITT_META_COLS = ("Cell_ID", "Animal_ID", "Animal_sex", "Behavior",
                     "Bregma", "Centroid_X", "Centroid_Y",
                     "Cell_class", "Neuron_cluster_ID")
MOFFITT_DEFAULT_ANIMAL = 1      # animals 1, 2, 7 have the full 12-section coverage


def read_merfish_hypothalamus_csv(path, animal_id=MOFFITT_DEFAULT_ANIMAL,
                                  verbose=True):
    """Read one animal out of the Moffitt 2018 MERFISH hypothalamus CSV.

    The distribution is a single ~1 GB table of 1M cells across 36 animals in
    four behavioural conditions. A benchmark volume has to be *one* animal — the
    sections of different animals are different tissue — so one naive animal is
    selected and its Bregma values become the sections. Animals 1, 2 and 7 carry
    the full 12-section anterior-posterior coverage; the rest have 5-6.

    Mirrors v1's processor: Centroid_X/Y are already micrometres, z is Bregma in
    mm scaled to um, cell types are ``Cell_class``, and the Blank control and
    all-NaN gene columns are dropped.
    """
    import anndata as ad
    import pandas as pd

    path = Path(path)
    csv_path = path / MOFFITT_CSV if path.is_dir() else path
    if not csv_path.exists():
        raise FileNotFoundError(f"MERFISH hypothalamus CSV not found: {csv_path}")

    if verbose:
        print(f"  reading {csv_path.name} (~1 GB, all animals) ...")
    df = pd.read_csv(csv_path)
    if "Animal_ID" not in df.columns:
        raise ValueError(f"{csv_path.name} has no Animal_ID column")

    sel = df[df["Animal_ID"] == int(animal_id)].copy()
    if sel.empty:
        raise ValueError(
            f"animal {animal_id} not in {csv_path.name}; "
            f"available: {sorted(df['Animal_ID'].unique())[:12]}")
    sel = sel.sort_values("Bregma")

    genes = [c for c in df.columns
             if c not in MOFFITT_META_COLS and not str(c).startswith("Blank")]
    genes = [g for g in genes if not sel[g].isna().all()]
    if not genes:
        raise ValueError("no usable gene columns after dropping Blank/all-NaN")

    bregmas = sorted(sel["Bregma"].unique())
    if verbose:
        print(f"  animal {animal_id}: {len(sel)} cells, {len(genes)} genes, "
              f"{len(bregmas)} sections (Bregma "
              f"{bregmas[0]:+.2f}..{bregmas[-1]:+.2f} mm)")

    X = sp.csr_matrix(np.nan_to_num(sel[genes].values.astype(np.float32)))
    coords = np.column_stack([
        sel["Centroid_X"].values.astype(np.float64),
        sel["Centroid_Y"].values.astype(np.float64),
        sel["Bregma"].values.astype(np.float64) * 1000.0,       # mm -> um
    ])
    cell_types = (sel["Cell_class"].astype(str).values if "Cell_class" in sel
                  else np.array(["unannotated"] * len(sel)))

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"section": [f"bregma_{b:+.2f}" for b in sel["Bregma"].values],
             "cell_type": cell_types},
            index=[str(v) for v in (sel["Cell_ID"] if "Cell_ID" in sel
                                    else range(len(sel)))]),
        var=pd.DataFrame(index=pd.Index(genes, name=None)))
    adata.obs_names_make_unique()
    adata.obsm["spatial"] = coords
    adata.uns["expression_type"] = "normalized"
    adata.uns["dataset_name"] = "merfish_hypothalamus"
    adata.uns["spatial_metadata"] = {
        "technology": "MERFISH", "species": "mouse", "tissue": "hypothalamus",
        "coordinate_units": "micrometers", "expression_type": "normalized",
        "animal_id": int(animal_id),
        "source": "Moffitt et al. 2018 Science; Dryad (raw, read by bench3.sources)",
    }
    return adata


OPENST_PIXEL_UM = 0.345          # Keyence BZ-X810 20x, Open-ST docs
OPENST_SECTION_UM = 10.0         # 10 um cryosections (Schott et al. 2024)


def read_openst_lymph_node(path, verbose=True):
    """Read the Open-ST metastatic lymph-node serial sections.

    One ``.h5ad(.gz)`` per 10 um cryosection, named ``..._S<n>.h5ad``; the section
    number sets z. Coordinates are pixels at 0.345 um. Only the metastatic
    lymph-node sections are used — the GEO series also carries unrelated
    specimens, and mixing them would not be one volume.
    """
    import gzip
    import re
    import shutil
    import tempfile

    import anndata as ad
    import pandas as pd

    path = Path(path)
    raw_dir = path if path.is_dir() else path.parent
    files = [f for f in sorted(raw_dir.glob("*metastatic_lymph_node_S*.h5ad*"))
             if f.suffix in (".h5ad", ".gz")]
    if not files:
        raise FileNotFoundError(
            f"no Open-ST lymph-node sections under {raw_dir}\n"
            f"  expected e.g. GSM7990100_metastatic_lymph_node_S2.h5ad(.gz); "
            f"extract GSE251926_RAW.tar first")

    def _sec_num(f):
        m = re.search(r"_S(\d+)\.h5ad", f.name)
        return int(m.group(1)) if m else -1

    files = sorted({_sec_num(f): f for f in files}.items())
    if verbose:
        print(f"  reading {len(files)} Open-ST sections from {raw_dir.name}")

    slices = []
    with tempfile.TemporaryDirectory() as tmp:
        for sec_num, f in files:
            if f.suffix == ".gz":
                plain = Path(tmp) / f.with_suffix("").name
                with gzip.open(f, "rb") as fi, open(plain, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                f = plain
            a = ad.read_h5ad(str(f))

            xy = None
            if "spatial" in a.obsm:
                xy = np.asarray(a.obsm["spatial"], dtype=np.float64)[:, :2]
            else:
                for xc, yc in (("x", "y"), ("X", "Y"), ("centroid_x", "centroid_y")):
                    if xc in a.obs.columns and yc in a.obs.columns:
                        xy = np.column_stack([a.obs[xc].values.astype(np.float64),
                                              a.obs[yc].values.astype(np.float64)])
                        break
            if xy is None:
                raise ValueError(f"{f.name}: no spatial coordinates")

            xy = xy * OPENST_PIXEL_UM
            a.obsm["spatial"] = np.column_stack(
                [xy, np.full(len(xy), sec_num * OPENST_SECTION_UM)])
            a.obs["section"] = f"S{sec_num}"
            ct = next((c for c in ("cell_type", "celltype", "CellType", "cluster",
                                   "leiden", "louvain", "annotation")
                       if c in a.obs.columns), None)
            a.obs["cell_type"] = (a.obs[ct].astype(str) if ct else "unannotated")
            a.obs = a.obs[["section", "cell_type"]]
            a.X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
            a.var_names_make_unique()
            a.obs_names = [f"S{sec_num}_{n}" for n in a.obs_names]
            slices.append(a)

    adata = ad.concat(slices, join="outer", merge="first")
    adata.obs_names_make_unique()
    adata.X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    adata.obs["section"] = adata.obs["section"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.uns["expression_type"] = "raw_counts"
    adata.uns["dataset_name"] = "openst_lymph_node"
    adata.uns["spatial_metadata"] = {
        "technology": "Open-ST", "species": "human",
        "tissue": "metastatic lymph node", "coordinate_units": "micrometers",
        "expression_type": "raw_counts",
        "source": "GEO GSE251926 (raw, read by bench3.sources)",
    }
    if verbose:
        print(f"  {adata.n_obs} cells x {adata.n_vars} genes, "
              f"{adata.obs['section'].nunique()} sections")
    return adata


READERS = {"exseq_csv": read_exseq_csv, "imc_zstack": read_imc_zstack,
           "deep_starmap_csv": read_deep_starmap_csv,
           "cosmx_raw": read_cosmx_raw,
           "merfish_hypothalamus_csv": read_merfish_hypothalamus_csv,
           "openst_lymph_node": read_openst_lymph_node}


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
