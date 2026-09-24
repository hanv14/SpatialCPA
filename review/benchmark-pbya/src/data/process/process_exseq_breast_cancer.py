#!/usr/bin/env python
"""Process ExSeq breast cancer dataset to standardized h5ad.

Data source: Zenodo (Alon et al. 2021, Science)
Raw files:
  - cell_expression-1222.txt       : Tab-separated count matrix (genes x cells, no headers)
  - GeneNames-1222.txt             : 297 gene names (one per line)
  - CellsIDs-1222.txt              : 3107 cell IDs (one per line)
  - SeuratCellsIDs_2Dseg_20191222.csv : Cell ID -> Seurat cell type mapping (2395 entries)
  - HTAPP-transcriptobjects-SeuratClassified-1222.mat : MATLAB v5 file with transcript-level
    3D positions (globalpos field) and cell assignments (cell_id field)
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "exseq_breast_cancer"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_PATH = PROJECT_ROOT / "data" / "processed" / f"{DATASET_NAME}.h5ad"

METADATA = {
    "technology": "ExSeq",
    "species": "human",
    "tissue": "breast cancer",
    "expression_type": "raw_counts",
    "section_spacing_um": None,
    "coordinate_units": "micrometers",
    "source": "Zenodo — Alon et al. 2021, Science",
    "spatial_metadata": {
        "technology": "ExSeq",
        "is_3d": True,
    },
}

# Pixel sizes from Alon et al. 2021 supplementary: Zyla sCMOS 4.2 MP, Nikon 40X/1.15 NA
# Post-expansion pixel sizes (4x expansion factor applied):
PIXEL_XY_UM = 0.17   # µm per pixel in x,y (post-expansion)
PIXEL_Z_UM = 0.4     # µm per z-step (post-expansion)

REQUIRED_FILES = [
    "cell_expression-1222.txt",
    "GeneNames-1222.txt",
    "CellsIDs-1222.txt",
    "SeuratCellsIDs_2Dseg_20191222.csv",
    "HTAPP-transcriptobjects-SeuratClassified-1222.mat",
]


def check_raw_data():
    """Verify all required raw files are present."""
    missing = [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing files in {RAW_DIR}:")
        for f in missing:
            print(f"  - {f}")
        print(f"Run: python src/data/download/download_{DATASET_NAME}.py")
        sys.exit(1)
    print(f"  All required files present in {RAW_DIR}")


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
    assert sp.issparse(adata.X) and adata.X.format == "csr", \
        "X must be CSR sparse"
    assert "spatial" in adata.obsm and adata.obsm["spatial"].shape == (adata.n_obs, 3), \
        "obsm['spatial'] must be (n_obs, 3)"
    assert "section" in adata.obs.columns, "obs must have 'section' column"
    assert "cell_type" in adata.obs.columns, "obs must have 'cell_type' column"
    print(f"  Verified: {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"  Spatial: x=[{adata.obsm['spatial'][:, 0].min():.1f}, "
        f"{adata.obsm['spatial'][:, 0].max():.1f}], "
        f"y=[{adata.obsm['spatial'][:, 1].min():.1f}, "
        f"{adata.obsm['spatial'][:, 1].max():.1f}], "
        f"z=[{adata.obsm['spatial'][:, 2].min():.1f}, "
        f"{adata.obsm['spatial'][:, 2].max():.1f}]"
    )
    print(f"  Sparsity: {1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]):.1%}")


def load_count_matrix():
    """Load count matrix and return (X, gene_names, cell_ids).

    cell_expression-1222.txt is genes x cells (297 rows, 3107 columns),
    tab-separated, no headers.  We transpose to cells x genes.
    """
    expr_path = RAW_DIR / "cell_expression-1222.txt"
    gene_path = RAW_DIR / "GeneNames-1222.txt"
    cell_path = RAW_DIR / "CellsIDs-1222.txt"

    print("  Loading gene names...")
    gene_names = pd.read_csv(gene_path, header=None)[0].tolist()
    print(f"    {len(gene_names)} genes")

    print("  Loading cell IDs...")
    cell_ids = pd.read_csv(cell_path, header=None)[0].tolist()
    print(f"    {len(cell_ids)} cells")

    print("  Loading count matrix (genes x cells)...")
    mat = pd.read_csv(expr_path, sep="\t", header=None).values  # (297, 3107)
    print(f"    Raw matrix shape: {mat.shape}")

    if mat.shape[0] != len(gene_names):
        raise ValueError(
            f"Row count mismatch: matrix has {mat.shape[0]} rows "
            f"but {len(gene_names)} gene names"
        )
    if mat.shape[1] != len(cell_ids):
        raise ValueError(
            f"Column count mismatch: matrix has {mat.shape[1]} columns "
            f"but {len(cell_ids)} cell IDs"
        )

    # Transpose: cells x genes
    X = mat.T.astype(np.float32)
    print(f"    Transposed to cells x genes: {X.shape}")
    return X, gene_names, cell_ids


def load_cell_types():
    """Load Seurat cell type annotations.

    Returns a dict mapping cell_id (int) -> cell_type (str).
    """
    csv_path = RAW_DIR / "SeuratCellsIDs_2Dseg_20191222.csv"
    print("  Loading Seurat cell type annotations...")
    df = pd.read_csv(csv_path)
    # Columns: "" (row index), "cellids", "seuratCellType"
    # Find the cell ID and cell type columns
    cellid_col = None
    celltype_col = None
    for col in df.columns:
        if col.lower() in ("cellids", "cell_id", "cell_ids"):
            cellid_col = col
        elif col.lower() in ("seuratcelltype", "cell_type", "celltype"):
            celltype_col = col

    if cellid_col is None or celltype_col is None:
        # Fall back to positional: first non-index numeric col = cell IDs,
        # second = cell types
        cols = df.columns.tolist()
        # The CSV has an unnamed index column first
        if cols[0] == "" or cols[0].startswith("Unnamed"):
            cellid_col = cols[1]
            celltype_col = cols[2]
        else:
            cellid_col = cols[0]
            celltype_col = cols[1]

    # Cell IDs in the CSV may have an "X" prefix (e.g. "X2" -> 2)
    raw_ids = df[cellid_col].astype(str).str.lstrip("X")
    mapping = dict(zip(raw_ids.astype(int), df[celltype_col].astype(str)))
    print(f"    {len(mapping)} cell type annotations loaded")
    return mapping


def load_spatial_centroids(cell_ids_in_matrix):
    """Extract 3D centroids per cell from the MATLAB transcript-objects file.

    The .mat file contains one structured element per transcript.  Each element
    has a `globalpos` field (shape [[x, y, z]]) and a `cell_id` field
    (shape [[int]]).  We group transcripts by cell_id and compute the centroid.

    Returns a dict mapping cell_id (int) -> np.ndarray([x, y, z]).
    """
    mat_path = RAW_DIR / "HTAPP-transcriptobjects-SeuratClassified-1222.mat"
    print(f"  Loading .mat file: {mat_path.name}")
    mat = scipy.io.loadmat(str(mat_path))

    obj = mat["transcript_objects_classes"]  # shape (N, 1)
    n_transcripts = obj.shape[0]
    print(f"    {n_transcripts} transcript objects found")

    # Accumulate positions per cell
    cell_pos_sum = {}   # cell_id -> [sum_x, sum_y, sum_z]
    cell_pos_count = {}  # cell_id -> count

    # Target cell IDs set for fast membership testing
    target_ids = set(int(c) for c in cell_ids_in_matrix)

    print("    Extracting 3D positions by cell...")
    BATCH = 50_000
    for i in range(n_transcripts):
        if i > 0 and i % BATCH == 0:
            print(f"      {i}/{n_transcripts} transcripts processed...")

        elem = obj[i, 0]

        # cell_id: nested object array [[array([[int]])]] -> int
        cid_raw = elem["cell_id"]
        while isinstance(cid_raw, np.ndarray) and cid_raw.dtype == object:
            cid_raw = cid_raw.flat[0]
        cid = int(np.asarray(cid_raw).flat[0])

        # Only keep transcripts assigned to cells in our count matrix
        # (cell_id == 0 typically means unassigned)
        if cid == 0 or cid not in target_ids:
            continue

        # globalpos: nested object array [[array([[x, y, z]])]] -> (3,)
        pos_raw = elem["globalpos"]
        while isinstance(pos_raw, np.ndarray) and pos_raw.dtype == object:
            pos_raw = pos_raw.flat[0]
        pos = np.asarray(pos_raw, dtype=np.float64).flatten()
        if pos.shape[0] < 3:
            continue

        if cid not in cell_pos_sum:
            cell_pos_sum[cid] = np.zeros(3, dtype=np.float64)
            cell_pos_count[cid] = 0

        cell_pos_sum[cid] += pos[:3]
        cell_pos_count[cid] += 1

    print(f"    Centroid computed for {len(cell_pos_sum)} cells")

    centroids = {}
    for cid, total in cell_pos_sum.items():
        centroids[cid] = total / cell_pos_count[cid]

    return centroids


def process():
    """Main processing pipeline."""
    print(f"Processing {DATASET_NAME}...")
    check_raw_data()

    # --- 1. Load count matrix ---
    X_dense, gene_names, cell_ids_raw = load_count_matrix()
    # cell_ids_raw is a list of integers (read from CellsIDs-1222.txt)
    cell_ids = [int(c) for c in cell_ids_raw]

    # --- 2. Load cell type annotations ---
    cell_type_map = load_cell_types()

    # --- 3. Load 3D spatial centroids from .mat file ---
    centroids = load_spatial_centroids(cell_ids)

    # --- 4. Build spatial array (n_cells, 3) ---
    print("  Building spatial coordinates array...")
    cell_types = [cell_type_map.get(c, "unknown") for c in cell_ids]
    n_cells = len(cell_ids)
    spatial = np.zeros((n_cells, 3), dtype=np.float64)
    n_with_coords = 0
    for i, cid in enumerate(cell_ids):
        if cid in centroids:
            spatial[i] = centroids[cid]
            n_with_coords += 1

    # Convert pixel coordinates to physical µm (post-expansion)
    spatial[:, 0] *= PIXEL_XY_UM  # x: pixels → µm
    spatial[:, 1] *= PIXEL_XY_UM  # y: pixels → µm
    spatial[:, 2] *= PIXEL_Z_UM   # z: z-steps → µm

    n_without_coords = n_cells - n_with_coords
    print(f"    {n_with_coords}/{n_cells} cells have 3D coordinates from .mat file")
    if n_without_coords > 0:
        print(f"    Dropping {n_without_coords} cells without spatial coordinates")
        has_coords = np.array([cid in centroids for cid in cell_ids])
        spatial = spatial[has_coords]
        X_dense = X_dense[has_coords]
        cell_ids = [c for c, h in zip(cell_ids, has_coords) if h]
        cell_types = [ct for ct, h in zip(cell_types, has_coords) if h]
        n_cells = len(cell_ids)

    # --- 5. Build obs (cell metadata) ---
    print("  Building cell metadata...")
    obs_index = [str(c) for c in cell_ids]
    obs = pd.DataFrame(
        {
            "section": "breast_cancer",
            "cell_type": cell_types,
            "cell_id_original": cell_ids,
        },
        index=obs_index,
    )
    obs.index.name = None

    # --- 6. Assemble AnnData ---
    print("  Assembling AnnData...")
    var = pd.DataFrame(index=gene_names)
    var.index.name = None

    X_sparse = ensure_sparse_csr(X_dense)

    adata = ad.AnnData(X=X_sparse, obs=obs, var=var)
    adata.obsm["spatial"] = spatial

    # Store metadata
    adata.uns.update(METADATA)
    adata.uns["dataset_name"] = DATASET_NAME

    # Annotation summary
    n_typed = sum(1 for ct in cell_types if ct != "unknown")
    print(f"  Cell types: {n_typed}/{n_cells} cells annotated")
    ct_counts = pd.Series(cell_types).value_counts()
    for ct, cnt in ct_counts.items():
        print(f"    {ct}: {cnt}")

    # --- 7. Verify and write ---
    verify(adata)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUT_PATH)
    print(f"Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


def main():
    process()


if __name__ == "__main__":
    main()
