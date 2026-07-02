#!/usr/bin/env python
"""Generate a QC report for all processed spatial transcriptomics h5ad files.

Discovers all h5ad files in data/processed/, computes quality metrics,
outputs a CSV report, and prints a summary table to stdout.

Usage:
    python src/data/process/report_all_datasets.py
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "qc_report.csv"


def discover_h5ad_files(base_dir: Path) -> list[Path]:
    """Find all h5ad files in base_dir, both direct and in subdirectories."""
    files = sorted(base_dir.rglob("*.h5ad"))
    # Exclude the report itself if it somehow ends up as h5ad
    return [f for f in files if f.name != "qc_report.h5ad"]


def derive_dataset_name(filepath: Path, base_dir: Path) -> str:
    """Derive a dataset name from the file path relative to base_dir.

    Handles both old and new directory structures:
      Old: {name}.h5ad -> name
      Old: {name}/{specimen}.h5ad -> name/specimen
      New: {name}/data.h5ad -> name
      New: {name}/{specimen}/data.h5ad -> name/specimen
    """
    rel = filepath.relative_to(base_dir)
    parts = rel.parts
    if filepath.name == "data.h5ad":
        # New structure: strip trailing data.h5ad, use remaining path
        return str(Path(*parts[:-1]))
    if len(parts) == 1:
        # Direct file in processed/
        return filepath.stem
    else:
        # File in subdirectory: use subdir/stem
        return str(rel.with_suffix(""))


def compute_metrics(filepath: Path, base_dir: Path) -> dict:
    """Load an h5ad file and compute QC metrics."""
    dataset = derive_dataset_name(filepath, base_dir)
    record = {"dataset": dataset}
    issues = []

    try:
        adata = ad.read_h5ad(filepath, backed="r")
    except Exception as e:
        record["validation_issues"] = f"Failed to load: {e}"
        return record

    n_obs, n_vars = adata.shape
    record["n_obs"] = n_obs
    record["n_vars"] = n_vars

    # --- Expression stats (requires loading X into memory) ---
    try:
        X = adata.X[:]
        if sparse.issparse(X):
            X_sparse = X.tocsr()
            is_sparse = True
        else:
            issues.append("Dense X (not sparse)")
            X_sparse = sparse.csr_matrix(X)
            is_sparse = True  # converted for stats
    except Exception:
        X_sparse = None
        is_sparse = False

    if X_sparse is not None:
        total_elements = n_obs * n_vars
        nnz_total = X_sparse.nnz
        record["sparsity_pct"] = round(
            100.0 * (1.0 - nnz_total / total_elements), 2
        ) if total_elements > 0 else 0.0

        # nnz per cell
        nnz_per_cell = np.diff(X_sparse.indptr)
        record["mean_nnz_per_cell"] = round(float(np.mean(nnz_per_cell)), 2)
        record["median_nnz_per_cell"] = round(float(np.median(nnz_per_cell)), 2)
        record["min_nnz_per_cell"] = int(np.min(nnz_per_cell))
        record["max_nnz_per_cell"] = int(np.max(nnz_per_cell))

        # nnz per gene
        X_csc = X_sparse.tocsc()
        nnz_per_gene = np.diff(X_csc.indptr)
        record["mean_nnz_per_gene"] = round(float(np.mean(nnz_per_gene)), 2)
    else:
        record["sparsity_pct"] = None
        record["mean_nnz_per_cell"] = None
        record["median_nnz_per_cell"] = None
        record["min_nnz_per_cell"] = None
        record["max_nnz_per_cell"] = None
        record["mean_nnz_per_gene"] = None

    # --- Spatial coordinates ---
    if "spatial" in adata.obsm:
        spatial = np.array(adata.obsm["spatial"])
        if spatial.ndim == 2 and spatial.shape[1] == 3:
            record["x_min"] = round(float(np.nanmin(spatial[:, 0])), 4)
            record["x_max"] = round(float(np.nanmax(spatial[:, 0])), 4)
            record["y_min"] = round(float(np.nanmin(spatial[:, 1])), 4)
            record["y_max"] = round(float(np.nanmax(spatial[:, 1])), 4)
            record["z_min"] = round(float(np.nanmin(spatial[:, 2])), 4)
            record["z_max"] = round(float(np.nanmax(spatial[:, 2])), 4)
        else:
            issues.append(
                f"Wrong spatial shape: {spatial.shape} (expected n,3)"
            )
            record.update({
                "x_min": None, "x_max": None,
                "y_min": None, "y_max": None,
                "z_min": None, "z_max": None,
            })
    else:
        issues.append("Missing obsm['spatial']")
        record.update({
            "x_min": None, "x_max": None,
            "y_min": None, "y_max": None,
            "z_min": None, "z_max": None,
        })

    # --- Section count and unique coordinate counts ---
    if "section" in adata.obs.columns:
        record["n_sections"] = int(adata.obs["section"].nunique())
    else:
        issues.append("Missing 'section' column in obs")
        record["n_sections"] = None

    if "spatial" in adata.obsm:
        spatial = np.array(adata.obsm["spatial"])
        if spatial.ndim == 2 and spatial.shape[1] >= 3:
            x_vals = spatial[:, 0]
            y_vals = spatial[:, 1]
            z_vals = spatial[:, 2]
            record["n_unique_x"] = len(np.unique(x_vals[~np.isnan(x_vals)]))
            record["n_unique_y"] = len(np.unique(y_vals[~np.isnan(y_vals)]))
            n_unique_z = len(np.unique(z_vals[~np.isnan(z_vals)]))
            record["n_unique_z"] = n_unique_z

            # Flag section/z mismatch (only for section-based datasets,
            # not volumetric ones where n_unique_z >> n_sections)
            n_sec = record.get("n_sections")
            if n_sec is not None and n_unique_z != n_sec:
                if n_unique_z < n_sec:
                    issues.append(
                        f"Fewer z-planes ({n_unique_z}) than sections ({n_sec})"
                    )
                elif n_unique_z <= n_sec * 2:
                    # Small mismatch — likely a few sections sharing z
                    issues.append(
                        f"z-planes ({n_unique_z}) != sections ({n_sec})"
                    )
                # else: volumetric data (per-cell z), mismatch is expected
        else:
            record["n_unique_x"] = None
            record["n_unique_y"] = None
            record["n_unique_z"] = None
    else:
        record["n_unique_x"] = None
        record["n_unique_y"] = None
        record["n_unique_z"] = None

    # --- Annotation coverage ---
    if "cell_type" in adata.obs.columns:
        ct = adata.obs["cell_type"].astype(str)
        unannotated_labels = {"unannotated", "unknown", "nan", ""}
        n_annotated = int((~ct.isin(unannotated_labels)).sum())
        record["annotation_coverage_pct"] = round(
            100.0 * n_annotated / n_obs, 2
        ) if n_obs > 0 else 0.0
    else:
        issues.append("Missing 'cell_type' column in obs")
        record["annotation_coverage_pct"] = None

    # --- uns metadata ---
    # Check multiple possible locations for technology/species/tissue
    def _find_metadata(key):
        """Look for a metadata key in uns['spatial_metadata'], uns directly, or uns['dataset']."""
        if "spatial_metadata" in adata.uns:
            val = adata.uns["spatial_metadata"].get(key)
            if val is not None:
                return val
        if key in adata.uns:
            val = adata.uns[key]
            if val is not None:
                return val
        if "dataset" in adata.uns and isinstance(adata.uns["dataset"], dict):
            val = adata.uns["dataset"].get(key)
            if val is not None:
                return val
        return None

    record["technology"] = _find_metadata("technology")
    record["species"] = _find_metadata("species")
    record["tissue"] = _find_metadata("tissue")

    if "spatial_metadata" not in adata.uns:
        issues.append("Missing uns['spatial_metadata']")

    if "expression_type" in adata.uns:
        record["expression_type"] = adata.uns["expression_type"]
    else:
        record["expression_type"] = None

    # --- obs column inventory ---
    obs_cols = list(adata.obs.columns)
    record["n_obs_columns"] = len(obs_cols)
    record["obs_columns"] = "; ".join(obs_cols)

    # --- Section details: unique values with cell counts ---
    if "section" in adata.obs.columns:
        sec_counts = adata.obs["section"].value_counts()
        # Sort by section name for consistent ordering
        sec_counts = sec_counts.sort_index()
        parts = [f"{name} ({count})" for name, count in sec_counts.items()]
        record["section_details"] = "; ".join(parts)
        record["min_cells_per_section"] = int(sec_counts.min())
        record["max_cells_per_section"] = int(sec_counts.max())
    else:
        record["section_details"] = ""
        record["min_cells_per_section"] = None
        record["max_cells_per_section"] = None

    record["validation_issues"] = "; ".join(issues) if issues else ""

    adata.file.close()
    return record


def format_table(df: pd.DataFrame) -> str:
    """Format a summary table for stdout."""
    try:
        from tabulate import tabulate
        return tabulate(df, headers="keys", tablefmt="grid", showindex=False)
    except ImportError:
        pass

    # Manual formatting fallback
    cols = list(df.columns)
    col_widths = {c: max(len(str(c)), df[c].astype(str).str.len().max()) for c in cols}
    header = " | ".join(str(c).ljust(col_widths[c]) for c in cols)
    sep = "-+-".join("-" * col_widths[c] for c in cols)
    lines = [header, sep]
    for _, row in df.iterrows():
        line = " | ".join(str(row[c]).ljust(col_widths[c]) for c in cols)
        lines.append(line)
    return "\n".join(lines)


def main():
    if not PROCESSED_DIR.exists():
        print(f"Processed data directory not found: {PROCESSED_DIR}")
        sys.exit(1)

    h5ad_files = discover_h5ad_files(PROCESSED_DIR)
    if not h5ad_files:
        print(f"No h5ad files found in {PROCESSED_DIR}")
        sys.exit(1)

    print(f"Found {len(h5ad_files)} h5ad file(s) in {PROCESSED_DIR}\n")

    records = []
    for filepath in h5ad_files:
        print(f"  Processing: {filepath.relative_to(PROCESSED_DIR)} ... ", end="", flush=True)
        record = compute_metrics(filepath, PROCESSED_DIR)
        records.append(record)
        issues = record.get("validation_issues", "")
        if issues:
            print(f"done (issues: {issues})")
        else:
            print("done")

    # Build DataFrame with ordered columns
    column_order = [
        "dataset", "n_obs", "n_vars", "sparsity_pct",
        "mean_nnz_per_cell", "median_nnz_per_cell",
        "min_nnz_per_cell", "max_nnz_per_cell",
        "mean_nnz_per_gene",
        "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
        "n_sections", "n_unique_x", "n_unique_y", "n_unique_z",
        "min_cells_per_section", "max_cells_per_section",
        "section_details",
        "annotation_coverage_pct",
        "technology", "species", "tissue", "expression_type",
        "n_obs_columns", "obs_columns",
        "validation_issues",
    ]
    df = pd.DataFrame(records)
    # Reorder columns, keeping only those that exist
    present_cols = [c for c in column_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in column_order]
    df = df[present_cols + extra_cols]

    # Save CSV
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_PATH, index=False)
    print(f"\nQC report saved to: {REPORT_PATH}")

    # Print summary table (subset of columns for readability)
    summary_cols = [
        "dataset", "n_obs", "n_vars", "sparsity_pct",
        "mean_nnz_per_cell", "n_sections", "n_unique_z",
        "annotation_coverage_pct",
        "technology", "validation_issues",
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    print(f"\n{'='*80}")
    print("  QC SUMMARY")
    print(f"{'='*80}\n")
    print(format_table(df[summary_cols]))

    # Flag datasets with validation issues
    with_issues = df[df["validation_issues"].astype(str).str.len() > 0]
    if not with_issues.empty:
        print(f"\n  WARNING: {len(with_issues)} dataset(s) have validation issues:")
        for _, row in with_issues.iterrows():
            print(f"    - {row['dataset']}: {row['validation_issues']}")
    else:
        print(f"\n  All {len(df)} datasets passed validation checks.")


if __name__ == "__main__":
    main()
