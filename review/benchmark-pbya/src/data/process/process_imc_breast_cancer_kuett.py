#!/usr/bin/env python
"""Process 3D IMC breast cancer dataset (Kuett et al. 2022) to standardized h5ad.

Raw data: data/raw/imc_breast_cancer_kuett/
  - MainHer2BreastCancerModel.zip: 152 serial sections, HER2+ invasive ductal carcinoma
  - SecondHer2BreastCancerModel.zip: 92 serial sections, HER2+ invasive ductal carcinoma
  - LVIBloodBreastCancerModel.zip: lymphovascular invasion (blood vessel)
  - LVILymphBreastCancerModel.zip: lymphovascular invasion (lymph vessel), 16 sections

Each ZIP contains:
  - *_mean_intensities.csv: single-cell mean marker intensities
  - *_labels_area.csv: cell label IDs and areas (pixels)
  - *_panel.csv: antibody panel metadata
  - final_3D_stack_order*.csv: section-to-z-index mapping
  - measured_mask_*segmentation*.tif: 3D segmentation mask (z, y, x) with cell labels
  - compensationMatrix.csv: signal compensation matrix

Spatial coordinates:
  - x, y: cell centroid from segmentation mask, in pixels (1 um/pixel, IMC standard)
  - z: section index * 2 um (serial 2-um-thick sections)

Output: data/processed/imc_breast_cancer_kuett/<model>/data.h5ad
  Models: MainHer2, SecondHer2, LVIBlood, LVILymph
"""
import sys
import tempfile
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "imc_breast_cancer_kuett"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUT_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME

# Section thickness: 2 um (Kuett et al. 2022, Methods)
SECTION_THICKNESS_UM = 2.0

# IMC lateral resolution: ~1 um/pixel (standard for Hyperion imaging system)
PIXEL_SIZE_UM = 1.0

# Models to process with their ZIP and file naming conventions
MODELS = {
    "MainHer2": {
        "zip": "MainHer2BreastCancerModel.zip",
        "prefix": "MainHer2BreastCancerModel",
        "intensities": "model201710_mean_intensities.csv",
        "labels_area": "model201710_labels_area.csv",
        "panel": "model201710_panel.csv",
        "stack_order": "final_3D_stack_order_model201710.csv",
        "mask_pattern": "measured_mask_final_segmentation_hwatershed_500.00_90%.tif",
    },
    "SecondHer2": {
        "zip": "SecondHer2BreastCancerModel.zip",
        "prefix": "SecondHer2BreastCancerModel",
        "intensities": None,  # will be detected
        "labels_area": None,
        "panel": None,
        "stack_order": None,
        "mask_pattern": None,
    },
    "LVIBlood": {
        "zip": "LVIBloodBreastCancerModel.zip",
        "prefix": "LVIBloodBreastCancerModel",
        "intensities": None,
        "labels_area": None,
        "panel": None,
        "stack_order": None,
        "mask_pattern": None,
    },
    "LVILymph": {
        "zip": "LVILymphBreastCancerModel.zip",
        "prefix": "LVILymphBreastCancerModel",
        "intensities": "LVI_lymph_mean_intensities.csv",
        "labels_area": "LVI_lymph_labels_area.csv",
        "panel": "LVI_lymph_panel.csv",
        "stack_order": "final_3D_stack_order.csv",
        "mask_pattern": "measured_mask_final_segmentation_hwatershed_bg500_90%.tif",
    },
}


def check_raw_data():
    """Verify that required ZIP archives are present."""
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data directory not found: {RAW_DIR}")
        print(f"  Run: python src/data/download/download_{DATASET_NAME}.py")
        sys.exit(1)

    for model_name, cfg in MODELS.items():
        zpath = RAW_DIR / cfg["zip"]
        if not zpath.exists():
            print(f"ERROR: Required archive not found: {zpath}")
            sys.exit(1)
        print(f"  Found: {cfg['zip']} ({zpath.stat().st_size / 1e6:.0f} MB)")


def ensure_sparse_csr(X):
    """Convert X to CSR sparse matrix if not already."""
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def build_spatial_3d(coords_2d, z=0.0):
    """Append a z column to 2-D coordinate array."""
    n = coords_2d.shape[0]
    z_col = np.full((n, 1), z, dtype=np.float64)
    return np.hstack([np.array(coords_2d, dtype=np.float64), z_col])


def detect_files_in_zip(zf, prefix):
    """Auto-detect CSV and mask filenames in a ZIP archive."""
    names = [n for n in zf.namelist() if n.startswith(prefix + "/")]
    result = {}

    for n in names:
        basename = n.split("/")[-1]
        if basename.endswith("_mean_intensities.csv") or "mean_intensities" in basename:
            result["intensities"] = basename
        elif basename.endswith("_labels_area.csv") or "labels_area" in basename:
            result["labels_area"] = basename
        elif basename.endswith("_panel.csv") or "panel" in basename:
            result["panel"] = basename
        elif "stack_order" in basename.lower() and basename.endswith(".csv"):
            result["stack_order"] = basename
        elif "measured_mask" in basename and basename.endswith(".tif"):
            result["mask_pattern"] = basename

    return result


def process_model(model_name, cfg):
    """Process a single 3D IMC model from its ZIP archive."""
    zpath = RAW_DIR / cfg["zip"]
    prefix = cfg["prefix"]

    print(f"\n  Opening: {cfg['zip']}")
    zf = zipfile.ZipFile(zpath, "r")

    # Auto-detect files if not specified
    detected = detect_files_in_zip(zf, prefix)
    intensities_name = cfg["intensities"] or detected.get("intensities")
    labels_area_name = cfg["labels_area"] or detected.get("labels_area")
    panel_name = cfg["panel"] or detected.get("panel")
    stack_order_name = cfg["stack_order"] or detected.get("stack_order")
    mask_name = cfg["mask_pattern"] or detected.get("mask_pattern")

    if not intensities_name or not labels_area_name or not mask_name:
        print(f"  ERROR: Could not find required files in {cfg['zip']}")
        print(f"    intensities: {intensities_name}")
        print(f"    labels_area: {labels_area_name}")
        print(f"    mask: {mask_name}")
        return None

    print(f"    intensities: {intensities_name}")
    print(f"    labels_area: {labels_area_name}")
    print(f"    panel: {panel_name}")
    print(f"    stack_order: {stack_order_name}")
    print(f"    mask: {mask_name}")

    # --- Read mean intensities ---
    print(f"  Reading mean intensities...")
    with zf.open(f"{prefix}/{intensities_name}") as f:
        intensities = pd.read_csv(f)
    print(f"    {intensities.shape[0]} cells x {intensities.shape[1]} markers")

    # --- Read labels + area ---
    print(f"  Reading labels and areas...")
    with zf.open(f"{prefix}/{labels_area_name}") as f:
        labels_area = pd.read_csv(f)
    print(f"    {labels_area.shape[0]} cells")

    assert intensities.shape[0] == labels_area.shape[0], (
        f"Row count mismatch: intensities={intensities.shape[0]}, labels_area={labels_area.shape[0]}"
    )

    # --- Read panel ---
    panel = None
    if panel_name:
        print(f"  Reading panel...")
        with zf.open(f"{prefix}/{panel_name}") as f:
            panel = pd.read_csv(f)
        # Filter to markers in MeasureMask (these are in the mean_intensities)
        # Column name varies: "MeasureMask" or "Measure Mask"
        mask_col = None
        for col in ["MeasureMask", "Measure Mask"]:
            if col in panel.columns:
                mask_col = col
                break
        if mask_col:
            measured = panel[panel[mask_col] == 1]
            print(f"    {len(measured)} measured markers in panel")
        else:
            print(f"    No MeasureMask column found in panel; using all rows")

    # --- Read stack order ---
    if stack_order_name:
        print(f"  Reading stack order...")
        with zf.open(f"{prefix}/{stack_order_name}") as f:
            stack_lines = f.read().decode().strip().split("\n")
        n_sections = len(stack_lines)
        print(f"    {n_sections} sections in stack")
    else:
        n_sections = None

    # --- Read segmentation mask and compute centroids ---
    print(f"  Reading segmentation mask (this may take a moment)...")
    import tifffile
    with zf.open(f"{prefix}/{mask_name}") as f:
        # tifffile needs seekable file, extract to temp
        with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
            tmp.write(f.read())
            tmp.flush()
            mask = tifffile.imread(tmp.name)

    print(f"    Mask shape: {mask.shape} (z, y, x)")
    print(f"    Mask dtype: {mask.dtype}")

    # Get unique cell labels (excluding background=0)
    mask_labels = np.unique(mask)
    mask_labels = mask_labels[mask_labels > 0]
    print(f"    {len(mask_labels)} unique cell labels in mask")

    # Match labels from mask with labels_area
    area_labels = labels_area["label"].values
    common = np.intersect1d(mask_labels, area_labels)
    print(f"    {len(common)} labels in common between mask and labels_area")

    if len(common) < len(area_labels) * 0.9:
        print(f"  WARNING: Only {len(common)}/{len(area_labels)} labels found in mask")

    # Compute centroids for all labels in the mask that match the CSV data
    # Use the CSV label order to maintain alignment with intensities
    print(f"  Computing cell centroids from segmentation mask...")
    label_to_idx = {lab: i for i, lab in enumerate(area_labels)}

    # ndimage.center_of_mass with specific labels
    centroids = ndimage.center_of_mass(mask > 0, mask, area_labels)
    centroids = np.array(centroids)  # shape: (n_cells, 3) in (z, y, x) order
    print(f"    Centroids computed: {centroids.shape}")

    # Convert to physical coordinates (um)
    # x, y in pixels * PIXEL_SIZE_UM; z in section index * SECTION_THICKNESS_UM
    x_um = centroids[:, 2] * PIXEL_SIZE_UM
    y_um = centroids[:, 1] * PIXEL_SIZE_UM
    z_um = centroids[:, 0] * SECTION_THICKNESS_UM

    spatial = np.column_stack([x_um, y_um, z_um])
    print(f"    Spatial range: x=[{x_um.min():.1f}, {x_um.max():.1f}], "
          f"y=[{y_um.min():.1f}, {y_um.max():.1f}], "
          f"z=[{z_um.min():.1f}, {z_um.max():.1f}] um")

    # --- Determine section assignment per cell ---
    section_idx = np.round(centroids[:, 0]).astype(int)

    # --- Build AnnData ---
    print(f"  Building AnnData object...")

    # Expression matrix (mean intensities)
    marker_names = list(intensities.columns)
    X = ensure_sparse_csr(intensities.values.astype(np.float32))

    # Obs metadata
    obs = pd.DataFrame({
        "cell_label": area_labels,
        "area_pixels": labels_area["area"].values,
        "section": section_idx,
    })
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    # Var metadata
    var = pd.DataFrame(index=marker_names)
    if panel is not None:
        # Try to add metal tag info
        clean_to_metal = {}
        for _, row in panel.iterrows():
            ct = row.get("clean_target", "")
            mt = row.get("Metal Tag", "")
            if pd.notna(ct) and pd.notna(mt):
                clean_to_metal[ct] = mt
        var["metal_tag"] = [clean_to_metal.get(m, "") for m in marker_names]

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=var,
    )
    adata.obsm["spatial"] = spatial.astype(np.float64)

    # uns metadata
    adata.uns["spatial_metadata"] = {
        "coord_system": "physical_um",
        "x_unit": "um",
        "y_unit": "um",
        "z_unit": "um",
        "pixel_size_um": PIXEL_SIZE_UM,
        "section_thickness_um": SECTION_THICKNESS_UM,
        "n_sections": int(mask.shape[0]),
        "source_model": model_name,
    }
    adata.uns["expression_type"] = "mean_intensities"
    adata.uns["dataset_name"] = DATASET_NAME

    return adata


def verify(adata, model_name):
    """Run verification checks on processed AnnData."""
    print(f"\n  Verifying {model_name}...")
    ok = True

    # Shape
    print(f"    Shape: {adata.shape}")
    if adata.shape[0] < 100:
        print(f"    WARNING: Very few cells ({adata.shape[0]})")
        ok = False

    # Sparse matrix
    if not sp.issparse(adata.X):
        print(f"    WARNING: X is not sparse")
        ok = False

    # Spatial coordinates
    if "spatial" not in adata.obsm:
        print(f"    ERROR: No spatial coordinates")
        ok = False
    else:
        coords = adata.obsm["spatial"]
        if coords.shape[1] != 3:
            print(f"    ERROR: spatial has {coords.shape[1]} columns, expected 3")
            ok = False
        z_vals = coords[:, 2]
        n_sections = len(np.unique(np.round(z_vals / SECTION_THICKNESS_UM).astype(int)))
        print(f"    Spatial: {coords.shape}, z range: [{z_vals.min():.1f}, {z_vals.max():.1f}] um, "
              f"~{n_sections} sections")

    # Expression
    if adata.uns.get("expression_type") != "mean_intensities":
        print(f"    WARNING: expression_type = {adata.uns.get('expression_type')}")

    # NaN/Inf
    X_dense = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    if np.any(np.isnan(X_dense)):
        print(f"    WARNING: NaN values in expression matrix")
        ok = False
    if np.any(np.isinf(X_dense)):
        print(f"    WARNING: Inf values in expression matrix")
        ok = False

    if ok:
        print(f"    {model_name}: ALL CHECKS PASSED")
    else:
        print(f"    {model_name}: SOME CHECKS FAILED")
    return ok


def process():
    """Process all models."""
    check_raw_data()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")

        try:
            adata = process_model(model_name, cfg)
            if adata is None:
                print(f"  FAILED: {model_name}")
                continue

            verify(adata, model_name)

            out_path = OUT_DIR / model_name / "data.h5ad"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"\n  Writing: {out_path}")
            adata.write_h5ad(out_path)
            print(f"  Written: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
            results[model_name] = adata.shape

        except Exception as e:
            print(f"  ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Processing complete: {len(results)}/{len(MODELS)} models")
    for name, shape in results.items():
        print(f"  {name}: {shape[0]} cells x {shape[1]} markers")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print(f"Processing: {DATASET_NAME}")
    print(f"Source: Kuett et al. 2022, Nature Cancer")
    print(f"Raw data: {RAW_DIR}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)
    process()


if __name__ == "__main__":
    main()
