"""FEAST method wrapper for virtual slice interpolation.

FEAST: From features to slice — parameter-cloud modeling for 3D interpolation.
Paper: Wang et al. (2025), bioRxiv.
Install: pip install FEAST-py

Pipeline per held-out section:
  1. Identify flanking observed sections (by z-coordinate)
  2. Prepare 2D slice AnnData (raw counts in .layers['counts'], normalized in .X)
  3. Compute PASTE2 partial alignment between flanking slices
  4. FEAST.interpolate_slices(slice_below, slice_above, alignment, t=alpha)
  5. Collect output → prediction.h5

Usage:
    python src/benchmark/methods/run_feast.py \
        --input data/processed/cosmx_nsclc_3d/data.h5ad \
        --holdout-sections section_3 section_4 \
        --output results/feast/cosmx_nsclc_3d/loo_section_3/prediction.h5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def check_environment():
    """Verify FEAST and PASTE2 are importable."""
    ok = True
    try:
        import FEAST
        print(f"FEAST {FEAST.__version__} imported")
    except ImportError:
        print("ERROR: FEAST not installed. Run: pip install FEAST-py", file=sys.stderr)
        ok = False
    try:
        from paste2.PASTE2 import partial_pairwise_align
        print("PASTE2 alignment available")
    except ImportError:
        print("ERROR: paste2 not installed", file=sys.stderr)
        ok = False
    return ok


def prepare_input(adata, holdout_sections):
    """Remove holdout sections, return training adata and target z-values."""
    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, holdout_sections)

    train_adata = adata[~holdout_mask].copy()

    target_z = {}
    for sec in holdout_sections:
        sec_mask = sections == sec
        target_z[sec] = float(np.median(adata.obsm["spatial"][sec_mask, 2]))

    return train_adata, target_z


def _extract_2d_slice(adata, section_label, expression_type):
    """Extract a single section as 2D AnnData ready for FEAST.

    - obsm['spatial'] = (n, 2) x,y coordinates
    - .layers['counts'] = raw counts (or original expression)
    - .X = log1p-normalized (for query slice averaging)
    """
    sections = adata.obs["section"].values.astype(str)
    mask = sections == section_label
    slc = adata[mask].copy()

    # Set 2D spatial
    slc.obsm["spatial"] = slc.obsm["spatial"][:, :2].copy()

    # Store counts layer: FEAST uses layers['counts'] for parameter-cloud estimation
    # and expects raw (non-log) values there.
    if expression_type == "log2_normalized":
        # Reverse log2(x+1) to get approximate raw counts
        X_dense = slc.X.toarray() if sp.issparse(slc.X) else slc.X.copy()
        slc.layers["counts"] = sp.csr_matrix(np.clip(np.power(2, X_dense) - 1, 0, None))
    elif expression_type == "log1p_normalized":
        # Reverse log1p to get approximate raw counts
        X_dense = slc.X.toarray() if sp.issparse(slc.X) else slc.X.copy()
        slc.layers["counts"] = sp.csr_matrix(np.clip(np.expm1(X_dense), 0, None))
    else:
        slc.layers["counts"] = slc.X.copy()

    # Normalize .X for FEAST's query averaging
    if expression_type == "raw_counts":
        sc.pp.normalize_total(slc, target_sum=1e4)
        sc.pp.log1p(slc)
    elif expression_type in ("log1p_normalized", "log2_normalized", "normalized"):
        pass  # already normalized
    else:
        # fluorescence_intensity, etc. — normalize anyway
        sc.pp.normalize_total(slc, target_sum=1e4)
        sc.pp.log1p(slc)

    return slc


def _compute_alignment(slice1, slice2, s=0.7, alpha=0.1, max_cells=3000):
    """Compute alignment between two slices.

    Tries PASTE2 partial alignment first. Falls back to PASTE1 pairwise_align
    if PASTE2's EMD solver fails (common with large or unbalanced slices).
    Subsamples to max_cells for scalability.

    Returns
    -------
    alignment_matrix : (n1, n2) transport plan
    """
    # Subsample if slices are too large
    s1, s2 = slice1, slice2
    if slice1.n_obs > max_cells or slice2.n_obs > max_cells:
        n1 = min(max_cells, slice1.n_obs)
        n2 = min(max_cells, slice2.n_obs)
        idx1 = np.random.choice(slice1.n_obs, n1, replace=False)
        idx2 = np.random.choice(slice2.n_obs, n2, replace=False)
        s1 = slice1[idx1].copy()
        s2 = slice2[idx2].copy()
        print(f"      Subsampled for alignment: {slice1.n_obs}->{n1}, {slice2.n_obs}->{n2}")

    # Try PASTE2 partial alignment first
    try:
        from paste2.PASTE2 import partial_pairwise_align
        pi = partial_pairwise_align(s1, s2, s=s, alpha=alpha, verbose=False)
        return pi
    except (ValueError, Exception) as e:
        print(f"      PASTE2 failed ({e}), falling back to PASTE1")

    # Fallback: PASTE1 balanced OT (more robust)
    try:
        from paste.PASTE import pairwise_align
        pi = pairwise_align(s1, s2, alpha=alpha, verbose=False)
        return pi
    except Exception as e2:
        print(f"      PASTE1 also failed ({e2}), using uniform alignment")

    # Last resort: uniform transport plan
    n1, n2 = s1.n_obs, s2.n_obs
    pi = np.ones((n1, n2)) / (n1 * n2)
    return pi


def _find_flanking_sections(sorted_sections, section_z, target_z):
    """Find the two sections that flank the target z coordinate."""
    z_values = [section_z[s] for s in sorted_sections]
    for i in range(len(sorted_sections) - 1):
        if z_values[i] <= target_z <= z_values[i + 1]:
            return sorted_sections[i], sorted_sections[i + 1]
    # If target is at exact boundary of a section
    for i, s in enumerate(sorted_sections):
        if abs(z_values[i] - target_z) < 1e-6:
            if i > 0 and i < len(sorted_sections) - 1:
                return sorted_sections[i - 1], sorted_sections[i + 1]
    return None


def run_method(train_adata, target_z, seed=42, paste2_s=0.7, paste2_alpha=0.1):
    """Execute FEAST interpolation for each held-out section.

    Parameters
    ----------
    train_adata : AnnData
        Training data (holdout sections removed), 3D coords in obsm['spatial'].
    target_z : dict
        section_label -> z coordinate for each section to predict.
    seed : int
    paste2_s : float
        PASTE2 mass fraction parameter.
    paste2_alpha : float
        PASTE2 expression/spatial balance.

    Returns
    -------
    results : dict mapping section_label -> dict with X, coords, cell_type
    """
    from FEAST import interpolate_slices, InterpolationConfig

    np.random.seed(seed)

    expression_type = train_adata.uns.get("expression_type", "raw_counts")

    # Build section z-map
    train_sections = train_adata.obs["section"].values.astype(str)
    unique_sections = np.unique(train_sections)
    section_z = {}
    for sec in unique_sections:
        mask = train_sections == sec
        section_z[sec] = float(np.median(train_adata.obsm["spatial"][mask, 2]))
    sorted_sections = sorted(unique_sections, key=lambda s: section_z[s])

    # Cache extracted 2D slices and alignments
    slice_cache = {}
    alignment_cache = {}

    results = {}
    for target_sec, tz in sorted(target_z.items(), key=lambda kv: kv[1]):
        flanking = _find_flanking_sections(sorted_sections, section_z, tz)
        if flanking is None:
            print(f"  WARNING: no flanking sections for {target_sec} z={tz}, skipping")
            continue

        sec_below, sec_above = flanking
        z_below, z_above = section_z[sec_below], section_z[sec_above]

        # Interpolation parameter t
        t = (tz - z_below) / (z_above - z_below) if z_above != z_below else 0.5

        print(f"  {target_sec}: z={tz:.1f}, between {sec_below}(z={z_below:.1f}) "
              f"and {sec_above}(z={z_above:.1f}), t={t:.3f}")

        # Get or create 2D slices
        if sec_below not in slice_cache:
            slice_cache[sec_below] = _extract_2d_slice(
                train_adata, sec_below, expression_type)
        if sec_above not in slice_cache:
            slice_cache[sec_above] = _extract_2d_slice(
                train_adata, sec_above, expression_type)

        sl_below = slice_cache[sec_below]
        sl_above = slice_cache[sec_above]

        # Get or compute alignment
        pair_key = (sec_below, sec_above)
        if pair_key not in alignment_cache:
            print(f"    Computing PASTE2 alignment ({sl_below.n_obs} x {sl_above.n_obs})...")
            alignment_cache[pair_key] = _compute_alignment(
                sl_below, sl_above, s=paste2_s, alpha=paste2_alpha)
        alignment = alignment_cache[pair_key]

        # Run FEAST interpolation
        try:
            config = InterpolationConfig(
                t=t,
                random_seed=seed,
                use_normalized=True,
                verbose=False,
                boundary_multiplier=1.1,
                sigma=0,  # paper repo: sigma=0 (interpolation_sim_pipeline.py)
                feature_weights={"mean": 1.0, "variance": 1.0, "zero_prop": 1.0},
            )
            interpolated = interpolate_slices(sl_below, sl_above, alignment, config)

            X = interpolated.X
            if not sp.issparse(X):
                X = sp.csr_matrix(X)

            # Build 3D coordinates
            if "spatial" in interpolated.obsm:
                xy = interpolated.obsm["spatial"]
                if xy.shape[1] >= 2:
                    coords = np.zeros((xy.shape[0], 3))
                    coords[:, :2] = xy[:, :2]
                    coords[:, 2] = tz
                else:
                    coords = np.zeros((X.shape[0], 3))
                    coords[:, 2] = tz
            else:
                coords = np.zeros((X.shape[0], 3))
                coords[:, 2] = tz

            cell_types = np.array(["NA"] * X.shape[0])
            if "cell_type" in interpolated.obs:
                cell_types = interpolated.obs["cell_type"].values.astype(str)

            results[target_sec] = {
                "X": sp.csr_matrix(X),
                "coords": coords,
                "cell_type": cell_types,
            }
            print(f"    -> {X.shape[0]} cells predicted")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def format_output(results, gene_names, holdout_sections, method_params,
                  wall_time, output_path):
    """Write prediction.h5 in standardized format."""
    if not results:
        print("No results to write!")
        return

    all_X, all_ids, all_x, all_y, all_z = [], [], [], [], []
    all_section, all_ct = [], []

    cell_counter = 0
    for sec in holdout_sections:
        if sec not in results:
            continue
        r = results[sec]
        n = r["X"].shape[0]
        all_X.append(r["X"])
        all_ids.extend([f"pred_{cell_counter + i}" for i in range(n)])
        all_x.append(r["coords"][:, 0])
        all_y.append(r["coords"][:, 1])
        all_z.append(r["coords"][:, 2])
        all_section.extend([sec] * n)
        all_ct.extend(r["cell_type"].tolist())
        cell_counter += n

    if cell_counter == 0:
        print("No cells predicted!")
        return

    X = sp.vstack(all_X, format="csr")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        g = f.create_group("X")
        g.create_dataset("data", data=X.data)
        g.create_dataset("indices", data=X.indices)
        g.create_dataset("indptr", data=X.indptr)
        g.create_dataset("shape", data=np.array(X.shape))

        obs = f.create_group("obs")
        obs.create_dataset("cell_id", data=np.array(all_ids, dtype="S"))
        obs.create_dataset("x", data=np.concatenate(all_x))
        obs.create_dataset("y", data=np.concatenate(all_y))
        obs.create_dataset("z", data=np.concatenate(all_z))
        obs.create_dataset("section", data=np.array(all_section, dtype="S"))
        obs.create_dataset("cell_type", data=np.array(all_ct, dtype="S"))

        var = f.create_group("var")
        var.create_dataset("gene_name", data=np.array(gene_names, dtype="S"))

        uns = f.create_group("uns")
        uns.create_dataset("method_name", data="feast")
        uns.create_dataset("holdout_sections", data=json.dumps(holdout_sections))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)

    print(f"Wrote {cell_counter} predicted cells to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FEAST virtual slice interpolation")
    parser.add_argument("--input", required=True, help="Path to data.h5ad")
    parser.add_argument("--holdout-sections", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Output prediction.h5 path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paste2-s", type=float, default=0.7)
    parser.add_argument("--paste2-alpha", type=float, default=0.1)
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)
    gene_names = adata.var_names.tolist()

    print(f"Preparing input (holdout: {args.holdout_sections})...")
    train_adata, target_z = prepare_input(adata, args.holdout_sections)
    del adata

    print(f"Running FEAST interpolation...")
    t0 = time.time()
    results = run_method(train_adata, target_z, seed=args.seed,
                         paste2_s=args.paste2_s, paste2_alpha=args.paste2_alpha)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed,
        "paste2_s": args.paste2_s,
        "paste2_alpha": args.paste2_alpha,
    }
    format_output(results, gene_names, args.holdout_sections,
                  method_params, wall_time, args.output)


if __name__ == "__main__":
    main()
