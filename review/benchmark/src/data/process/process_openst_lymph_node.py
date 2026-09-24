#!/usr/bin/env python
"""Process OpenST lymph node to standardized h5ad."""
import gzip
import re
import shutil
import sys
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "openst_lymph_node"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_PATH = PROJECT_ROOT / "data" / "processed" / f"{DATASET_NAME}.h5ad"

METADATA = {
    "technology": "OpenST",
    "species": "human",
    "tissue": "metastatic lymph node",
    "n_sections": 19,
    "expression_type": "raw_counts",
    "section_thickness_um": 10.0,
    "source": "GEO GSE251926",
}


def _fetch_ensembl_symbols(ensembl_ids: list) -> pd.DataFrame:
    """Fetch gene symbols from Ensembl BioMart. Caches to TSV."""
    cache = RAW_DIR / "ensembl_to_symbol.tsv"
    if cache.exists():
        return pd.read_csv(cache, sep="\t", index_col=0)

    import urllib.request
    import urllib.parse
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="hgnc_symbol"/>
    <Attribute name="gene_biotype"/>
  </Dataset>
</Query>'''
    url = "https://ensembl.org/biomart/martservice?query=" + urllib.parse.quote(xml)
    try:
        print("  Fetching gene symbols from Ensembl BioMart...")
        with urllib.request.urlopen(url, timeout=120) as resp:
            df = pd.read_csv(resp, sep="\t")
        df.columns = ["ensembl_id", "symbol", "biotype"]
        df = df.dropna(subset=["ensembl_id"])
        df = df.drop_duplicates(subset="ensembl_id", keep="first")
        df = df.set_index("ensembl_id")
        df.to_csv(cache, sep="\t")
        print(f"  BioMart: cached {len(df)} mappings to {cache.name}")
        return df
    except Exception as e:
        print(f"  WARNING: BioMart fetch failed: {e}")
        return pd.DataFrame(columns=["symbol", "biotype"])


def ensure_sparse_csr(X):
    """Convert expression matrix to CSR sparse format."""
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def build_spatial_3d(coords_2d, z=0.0):
    """Add z column to 2D spatial coordinates."""
    n = coords_2d.shape[0]
    z_col = np.full((n, 1), z, dtype=np.float64)
    return np.hstack([np.array(coords_2d, dtype=np.float64), z_col])


def verify(adata):
    """Verify output meets standardized format requirements."""
    assert sp.issparse(adata.X) and adata.X.format == "csr"
    assert "spatial" in adata.obsm and adata.obsm["spatial"].shape == (adata.n_obs, 3)
    assert "section" in adata.obs.columns and "cell_type" in adata.obs.columns
    print(f"  Verified: {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"  Spatial: x=[{adata.obsm['spatial'][:,0].min():.1f}, {adata.obsm['spatial'][:,0].max():.1f}], "
        f"y=[{adata.obsm['spatial'][:,1].min():.1f}, {adata.obsm['spatial'][:,1].max():.1f}], "
        f"z=[{adata.obsm['spatial'][:,2].min():.1f}, {adata.obsm['spatial'][:,2].max():.1f}]"
    )
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")


SECTION_THICKNESS_UM = 10.0  # 10 µm cryosections (Schott et al., Cell 2024)
PIXEL_SIZE_UM = 0.345  # µm per pixel (Keyence BZ-X810, 20x; Open-ST docs)


def _extract_section_number(filename):
    """Extract cryostat section number from filename like 'GSM7990100_metastatic_lymph_node_S2.h5ad.gz'."""
    m = re.search(r'_S(\d+)\.h5ad', filename)
    if m:
        return int(m.group(1))
    return None


def extract_tar_if_needed():
    """Extract GSE251926_RAW.tar if not already extracted."""
    tar_path = RAW_DIR / "GSE251926_RAW.tar"
    if tar_path.exists():
        h5ad_gz_files = list(RAW_DIR.glob("*.h5ad.gz"))
        if not h5ad_gz_files:
            print(f"  Extracting {tar_path.name}...")
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(RAW_DIR)
            print("  Extraction complete.")


def decompress_h5ad_gz(gz_path):
    """Decompress a .h5ad.gz file to .h5ad, return path to decompressed file."""
    h5ad_path = gz_path.with_suffix("")  # Remove .gz
    if h5ad_path.exists():
        return h5ad_path

    print(f"    Decompressing {gz_path.name}...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(h5ad_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return h5ad_path


def extract_spatial_coords(adata):
    """Extract 2D spatial coordinates from an AnnData object."""
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.shape[1] >= 2:
            return coords[:, :2]

    # Try common column name patterns
    for x_col, y_col in [
        ("x", "y"),
        ("X", "Y"),
        ("x_coord", "y_coord"),
        ("centroid_x", "centroid_y"),
        ("spatial_x", "spatial_y"),
        ("array_row", "array_col"),
    ]:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            return np.column_stack(
                [adata.obs[x_col].values.astype(np.float64),
                 adata.obs[y_col].values.astype(np.float64)]
            )

    # Check for spatial coordinates in other obsm keys
    for key in adata.obsm:
        arr = np.asarray(adata.obsm[key])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]

    return None


def extract_cell_type(adata):
    """Extract cell type annotations from AnnData."""
    for col in ["cell_type", "celltype", "CellType", "cluster", "leiden", "louvain", "annotation"]:
        if col in adata.obs.columns:
            return adata.obs[col].values.astype(str)
    return np.full(adata.n_obs, "unknown", dtype=object)


def process():
    """Main processing pipeline. Merges all sections into a single h5ad."""
    print(f"Processing {DATASET_NAME}...")

    extract_tar_if_needed()

    # Find metastatic lymph node section files only (filter out mouse head, hippocampus, HNSCC, healthy LN)
    # Also exclude the 3D annotated file (GSE251926_metastatic_lymph_node_3d.h5ad)
    h5ad_gz_files = sorted(f for f in RAW_DIR.glob("*metastatic_lymph_node_S*.h5ad.gz"))
    h5ad_files = sorted(f for f in RAW_DIR.glob("*metastatic_lymph_node_S*.h5ad"))
    all_section_files = h5ad_gz_files if h5ad_gz_files else h5ad_files

    if not all_section_files:
        print(f"ERROR: No metastatic lymph node h5ad files found in {RAW_DIR}")
        sys.exit(1)

    print(f"  Found {len(all_section_files)} metastatic lymph node section files.")

    adatas = []
    spatials = []
    expr_types = set()

    for section_idx, section_file in enumerate(all_section_files):
        section_num = _extract_section_number(section_file.name)
        z_um = section_num * SECTION_THICKNESS_UM if section_num is not None else float(section_idx) * SECTION_THICKNESS_UM
        print(f"\n  Processing section {section_idx} (S{section_num}, z={z_um:.0f} µm): {section_file.name}")

        if section_file.suffix == ".gz":
            h5ad_path = decompress_h5ad_gz(section_file)
        else:
            h5ad_path = section_file

        try:
            adata = ad.read_h5ad(h5ad_path)
        except Exception as e:
            print(f"    WARNING: Failed to load {h5ad_path.name}: {e}")
            continue

        print(f"    Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
        adata.X = ensure_sparse_csr(adata.X)

        # Filter very sparse capture spots
        genes_per_cell = (adata.X > 0).sum(axis=1).A1
        median_genes = np.median(genes_per_cell)
        if median_genes < 10:
            min_counts = 50
            min_genes = 10
            counts_per_cell = adata.X.sum(axis=1).A1
            keep = (counts_per_cell >= min_counts) & (genes_per_cell >= min_genes)
            n_before = adata.n_obs
            adata = adata[keep].copy()
            adata.X = ensure_sparse_csr(adata.X)
            print(f"    Filtered: {n_before} -> {adata.n_obs} spots")

        if adata.n_obs == 0:
            print(f"    WARNING: No cells after filtering, skipping.")
            continue

        # Spatial coordinates — convert pixels to µm, z from physical cryostat section number
        coords_2d = extract_spatial_coords(adata)
        if coords_2d is not None:
            coords_2d = coords_2d * PIXEL_SIZE_UM  # pixels → µm
            spatial_3d = build_spatial_3d(coords_2d, z=z_um)
        else:
            print(f"    WARNING: No spatial coordinates found, using zeros.")
            spatial_3d = np.zeros((adata.n_obs, 3), dtype=np.float64)
            spatial_3d[:, 2] = z_um

        # Deduplicate observation names
        if adata.obs_names.duplicated().any():
            adata.obs_names_make_unique()
        # Prefix with section index to ensure global uniqueness
        adata.obs_names = [f"s{section_idx:03d}_{n}" for n in adata.obs_names]

        section_name = section_file.stem.replace(".h5ad", "")
        adata.obs["section"] = section_name
        adata.obs["cell_type"] = extract_cell_type(adata)

        # Determine expression type
        sample_vals = adata.X.data[:min(10000, len(adata.X.data))]
        is_integer = np.allclose(sample_vals, np.round(sample_vals))
        expr_type = "raw_counts" if is_integer else "log1p_normalized"
        expr_types.add(expr_type)

        adatas.append(adata)
        spatials.append(spatial_3d)

    if not adatas:
        print("ERROR: No sections loaded.")
        sys.exit(1)

    # Concatenate
    print(f"\n  Concatenating {len(adatas)} sections...")
    adata_combined = ad.concat(adatas, join="outer", fill_value=0)
    adata_combined.X = ensure_sparse_csr(adata_combined.X)
    adata_combined.obsm["spatial"] = np.vstack(spatials)

    # Transfer cell type annotations from 3D annotated file if available
    import re as _re
    annot_path = RAW_DIR / "GSE251926_metastatic_lymph_node_3d.h5ad"
    if annot_path.exists():
        import h5py
        print("  Loading cell type annotations from 3D file...")
        with h5py.File(annot_path, "r") as f3d:
            cats = [c.decode() if isinstance(c, bytes) else c for c in f3d["obs"]["annotation"]["categories"][:]]
            codes = f3d["obs"]["annotation"]["codes"][:]
            n_secs = f3d["obs"]["n_section"][:]
            cid_masks = f3d["obs"]["cell_ID_mask"][:]
            lookup = {}
            for i in range(len(codes)):
                if codes[i] >= 0:
                    lookup[(int(n_secs[i]), int(cid_masks[i]))] = cats[codes[i]]
        n_matched = 0
        in_3d_file = []
        new_ct = []
        for idx in range(adata_combined.n_obs):
            sec_name = adata_combined.obs["section"].iloc[idx]
            m = _re.search(r"_S(\d+)$", sec_name)
            if m:
                key = (int(m.group(1)), int(adata_combined.obs["cell_ID_mask"].iloc[idx]))
                ct = lookup.get(key, None)
                if ct is not None:
                    in_3d_file.append(True)
                    new_ct.append(ct)
                    if ct != "unknown":
                        n_matched += 1
                else:
                    in_3d_file.append(False)
                    new_ct.append("unknown")
            else:
                in_3d_file.append(False)
                new_ct.append("unknown")
        adata_combined.obs["cell_type"] = new_ct
        n_in_3d = sum(in_3d_file)
        n_excluded = adata_combined.n_obs - n_in_3d
        print(f"  Annotations: {n_matched}/{adata_combined.n_obs} cells with cell type, {n_in_3d} in 3D file total")
        # Remove cells not in the 3D file (failed authors' QC)
        if n_excluded > 0:
            keep_mask = np.array(in_3d_file)
            adata_combined = adata_combined[keep_mask].copy()
            adata_combined.obsm["spatial"] = np.vstack(spatials)[keep_mask]
            print(f"  Removed {n_excluded} cells not in authors' 3D file (failed QC) → {adata_combined.n_obs} remaining")

    # Use mixed expression type if sections differ
    if len(expr_types) == 1:
        final_expr_type = expr_types.pop()
    else:
        final_expr_type = "mixed"

    # Map remaining Ensembl IDs to gene symbols
    import re
    ens_mask = [bool(re.match(r'^ENSG\d{11}$', n)) for n in adata_combined.var_names]
    n_ens = sum(ens_mask)
    if n_ens > 0:
        ens_ids = [n for n, m in zip(adata_combined.var_names, ens_mask) if m]
        biomart = _fetch_ensembl_symbols(ens_ids)
        print(f"  BioMart: {len(biomart)} mappings loaded")
        new_names = list(adata_combined.var_names)
        adata_combined.var["ensembl_id"] = ""
        for i, (name, is_ens) in enumerate(zip(adata_combined.var_names, ens_mask)):
            if is_ens:
                adata_combined.var.iloc[i, adata_combined.var.columns.get_loc("ensembl_id")] = name
                if name in biomart.index and pd.notna(biomart.loc[name, "symbol"]) and biomart.loc[name, "symbol"] != "":
                    new_names[i] = biomart.loc[name, "symbol"]
        adata_combined.var.index = new_names
        adata_combined.var_names_make_unique()
        n_mapped = sum(1 for n, m in zip(adata_combined.var_names, ens_mask) if m and not n.startswith("ENSG"))
        print(f"  Gene symbols: mapped {n_mapped}/{n_ens} Ensembl IDs to symbols")

    adata_combined.uns['spatial_metadata'] = METADATA.copy()
    adata_combined.uns["expression_type"] = final_expr_type
    adata_combined.uns["dataset_name"] = DATASET_NAME

    verify(adata_combined)

    # Remove old per-section output directory
    old_dir = OUT_PATH.parent / DATASET_NAME
    if old_dir.is_dir():
        shutil.rmtree(old_dir)
        print(f"  Removed old per-section output directory: {old_dir}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to {OUT_PATH}...")
    adata_combined.write_h5ad(OUT_PATH)
    print(f"  Done. File size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    process()
