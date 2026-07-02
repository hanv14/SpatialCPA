"""SpatialZ method wrapper for virtual slice interpolation.

SpatialZ: Bridging the Dimensional Gap from Planar ST to 3D Cell Atlases.
Paper: Lin et al. (2025), Nature Methods.
Code: Zenodo 10.5281/zenodo.17416727

API: Generate_spatialz(adata1, adata2, alpha, cell_type_key, ...)
  - Takes two 2D slices (obsm['spatial'] 2D, obs['cell_type'], X = expression)
  - Returns AnnData with synthesized virtual slice

Usage:
    python src/benchmark/methods/run_spatialz.py \
        --input data/processed/cosmx_nsclc_3d/data.h5ad \
        --holdout-sections section_3 \
        --output results/spatialz/cosmx_nsclc_3d/loo_section_3/prediction.h5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp


TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "spatialz" / "SpatialZ_code"


def check_environment():
    """Verify SpatialZ and its dependencies are available."""
    if not TOOLS_DIR.exists():
        print(f"ERROR: SpatialZ not found at {TOOLS_DIR}", file=sys.stderr)
        print("Download from Zenodo: https://doi.org/10.5281/zenodo.17416727", file=sys.stderr)
        return False

    sys.path.insert(0, str(TOOLS_DIR))
    try:
        from SpatialZ import Generate_spatialz
        print("SpatialZ.Generate_spatialz imported")
    except ImportError as e:
        print(f"ERROR importing SpatialZ: {e}", file=sys.stderr)
        return False

    try:
        import MENDER
        print("MENDER available")
    except ImportError:
        print("ERROR: MENDER not installed. Run: pip install MENDER", file=sys.stderr)
        return False

    return True


def prepare_input(adata, holdout_sections):
    """Remove holdout sections, return training data and targets."""
    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, holdout_sections)
    train_adata = adata[~holdout_mask].copy()

    target_z = {}
    for sec in holdout_sections:
        sec_mask = sections == sec
        target_z[sec] = float(np.median(adata.obsm["spatial"][sec_mask, 2]))

    return train_adata, target_z


def _extract_2d_slice(adata, section_label):
    """Extract a single section as 2D AnnData for SpatialZ.

    SpatialZ writes directly to X[i] during gene expression synthesis,
    so X must be dense. It also requires cell_type annotations for the
    kNN-based gene expression transfer.
    """
    sections = adata.obs["section"].values.astype(str)
    mask = sections == section_label
    slc = adata[mask].copy()

    # SpatialZ needs 2D spatial coordinates
    slc.obsm["spatial"] = slc.obsm["spatial"][:, :2].copy()

    # Densify X — SpatialZ assigns to X[i] directly
    if sp.issparse(slc.X):
        slc.X = slc.X.toarray()

    # Normalize expression (demo notebook uses pre-normalized floats)
    import scanpy as sc
    expr_type = adata.uns.get("expression_type", "raw_counts") if hasattr(adata, 'uns') else "raw_counts"
    if expr_type == "raw_counts":
        sc.pp.normalize_total(slc, target_sum=1e4)
        sc.pp.log1p(slc)
    elif expr_type in ("log1p_normalized", "log2_normalized", "normalized"):
        pass  # already normalized
    else:
        # fluorescence_intensity, etc. — normalize
        sc.pp.normalize_total(slc, target_sum=1e4)
        sc.pp.log1p(slc)

    # Ensure cell_type exists with meaningful labels
    if "cell_type" not in slc.obs or slc.obs["cell_type"].nunique() <= 1:
        # If no cell types, create proxy clusters via Leiden
        import scanpy as sc
        temp = slc.copy()
        if sp.issparse(temp.X):
            temp.X = temp.X.toarray()
        sc.pp.normalize_total(temp, target_sum=1e4)
        sc.pp.log1p(temp)
        sc.pp.pca(temp, n_comps=min(20, temp.n_vars - 1))
        sc.pp.neighbors(temp, n_pcs=min(20, temp.n_vars - 1))
        sc.tl.leiden(temp, resolution=0.5)
        slc.obs["cell_type"] = temp.obs["leiden"].values
        del temp
    slc.obs["cell_type"] = slc.obs["cell_type"].astype("category")

    return slc


def _find_flanking_sections(sorted_sections, section_z, target_z):
    """Find the two sections that flank the target z coordinate."""
    z_values = [section_z[s] for s in sorted_sections]
    for i in range(len(sorted_sections) - 1):
        if z_values[i] <= target_z <= z_values[i + 1]:
            return sorted_sections[i], sorted_sections[i + 1]
    for i, s in enumerate(sorted_sections):
        if abs(z_values[i] - target_z) < 1e-6:
            if 0 < i < len(sorted_sections) - 1:
                return sorted_sections[i - 1], sorted_sections[i + 1]
    return None


def run_method(train_adata, target_z, seed=42, syn_mode="default", device="auto"):
    """Execute SpatialZ virtual slice generation.

    Parameters
    ----------
    train_adata : AnnData
        Training data with 3D coords in obsm['spatial'].
    target_z : dict
        section_label -> z coordinate for each section to predict.
    seed : int
    syn_mode : str
        'default' (detailed microenvironment) or 'fast'.
    device : str
        'auto', 'cpu', or 'cuda:0'.
    """
    sys.path.insert(0, str(TOOLS_DIR))
    from SpatialZ import Generate_spatialz

    np.random.seed(seed)

    # Build section z-map
    train_sections = train_adata.obs["section"].values.astype(str)
    unique_sections = np.unique(train_sections)
    section_z = {}
    for sec in unique_sections:
        mask = train_sections == sec
        section_z[sec] = float(np.median(train_adata.obsm["spatial"][mask, 2]))
    sorted_sections = sorted(unique_sections, key=lambda s: section_z[s])

    slice_cache = {}
    results = {}

    for target_sec, tz in sorted(target_z.items(), key=lambda kv: kv[1]):
        flanking = _find_flanking_sections(sorted_sections, section_z, tz)
        if flanking is None:
            print(f"  WARNING: no flanking sections for {target_sec} z={tz}, skipping")
            continue

        sec_below, sec_above = flanking
        z_below, z_above = section_z[sec_below], section_z[sec_above]
        alpha = (tz - z_below) / (z_above - z_below) if z_above != z_below else 0.5

        print(f"  {target_sec}: z={tz:.1f}, between {sec_below}(z={z_below:.1f}) "
              f"and {sec_above}(z={z_above:.1f}), alpha={alpha:.3f}")

        # Get 2D slices
        if sec_below not in slice_cache:
            slice_cache[sec_below] = _extract_2d_slice(train_adata, sec_below)
        if sec_above not in slice_cache:
            slice_cache[sec_above] = _extract_2d_slice(train_adata, sec_above)

        # SpatialZ: adata1=above slice, adata2=below slice, alpha = factor toward adata1
        # When alpha=0 → all adata2 (below), alpha=1 → all adata1 (above)
        # Our alpha: 0=below, 1=above → matches if adata1=above, adata2=below
        sl_above = slice_cache[sec_above].copy()
        sl_below = slice_cache[sec_below].copy()

        try:
            synthesized = Generate_spatialz(
                adata1=sl_above,
                adata2=sl_below,
                adata1_id=sec_above,
                adata2_id=sec_below,
                alpha=alpha,
                device=device,
                seed=seed,
                lr=1e-5,          # demo notebook default (Generate_multiple_slices)
                nb_iter_max=1000,  # demo notebook uses 1000
                k_sam=50,          # demo notebook uses 50
                syn_mode=syn_mode,
                cell_type_key="cell_type",
                verbose=True,
            )

            X = synthesized.X
            if not sp.issparse(X):
                X = sp.csr_matrix(X)

            # Build 3D coords
            if "spatial" in synthesized.obsm:
                xy = np.array(synthesized.obsm["spatial"])
            else:
                xy = np.zeros((X.shape[0], 2))

            coords = np.zeros((X.shape[0], 3))
            coords[:, :2] = xy[:, :2] if xy.shape[1] >= 2 else xy
            coords[:, 2] = tz

            cell_types = np.array(["NA"] * X.shape[0])
            if "cell_type" in synthesized.obs:
                cell_types = synthesized.obs["cell_type"].values.astype(str)

            results[target_sec] = {
                "X": sp.csr_matrix(X),
                "coords": coords,
                "cell_type": cell_types,
            }
            print(f"    -> {X.shape[0]} cells synthesized")

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
        uns.create_dataset("method_name", data="spatialz")
        uns.create_dataset("holdout_sections", data=json.dumps(holdout_sections))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)

    print(f"Wrote {cell_counter} predicted cells to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SpatialZ virtual slice generation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--holdout-sections", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--syn-mode", default="default", choices=["default", "fast"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)
    gene_names = adata.var_names.tolist()

    print(f"Preparing input (holdout: {args.holdout_sections})...")
    train_adata, target_z = prepare_input(adata, args.holdout_sections)
    del adata

    print(f"Running SpatialZ...")
    t0 = time.time()
    results = run_method(train_adata, target_z, seed=args.seed,
                         syn_mode=args.syn_mode, device=args.device)
    wall_time = time.time() - t0

    method_params = {"seed": args.seed, "syn_mode": args.syn_mode, "device": args.device}
    format_output(results, gene_names, args.holdout_sections,
                  method_params, wall_time, args.output)


if __name__ == "__main__":
    main()
