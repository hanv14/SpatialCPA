"""SVGP method wrapper for virtual slice interpolation.

Uses GPyTorch's Stochastic Variational Gaussian Process (SVGP) to learn a
continuous mapping from 3D coordinates to gene expression per gene, then
predict expression at held-out section coordinates.

Matches the algorithm used by Spateo (Qiu et al. 2024, Cell) but calls
GPyTorch directly to avoid Spateo's logging deadlock and import issues.

Parameters match Spateo defaults: SVGP, training_iter=50, batch_size=1024,
inducing_num=512, RBF kernel, Gaussian likelihood, Adam lr=0.01.

Usage:
    conda run -n bench_spateo python src/benchmark/methods/run_spateo_gp.py \
        --input data/processed/imc_breast_cancer/data.h5ad \
        --holdout-sections z7 \
        --output results/spateo_gp/imc_breast_cancer/loo_z7/prediction.h5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Cap CPU threads before numpy/torch are imported so this method cannot
# saturate every core (tune via BENCH_NUM_THREADS). Shared benchmark helper.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../src
from benchmark._cpu import limit_cpu_threads
limit_cpu_threads()

import anndata as ad
import gpytorch
import h5py
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset


# ── SVGP model (matches Spateo's Approx_GPModel) ─────────────────────────────

class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _train_svgp_one_gene(train_x, train_y, inducing_points, training_iter,
                          batch_size, device):
    """Train SVGP for one gene. Returns (model, likelihood) in eval mode."""
    model = SVGPModel(inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=0.01)

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(training_iter):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def _predict_svgp(model, likelihood, test_x, chunk_size=5000):
    """Predict with SVGP, chunked to avoid OOM."""
    preds = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, test_x.size(0), chunk_size):
            chunk = test_x[i:i + chunk_size]
            pred = likelihood(model(chunk)).mean
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


# ── Benchmark interface ───────────────────────────────────────────────────────

def check_environment():
    """Verify GPyTorch is available."""
    print(f"GPyTorch {gpytorch.__version__}, PyTorch {torch.__version__}, "
          f"CUDA: {torch.cuda.is_available()}")
    return True


def prepare_input(adata, holdout_sections):
    """Remove holdout sections, return training data and targets."""
    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, holdout_sections)

    train_adata = adata[~holdout_mask].copy()
    holdout_adata = adata[holdout_mask].copy()

    target_z = {}
    for sec in holdout_sections:
        sec_mask = sections == sec
        target_z[sec] = float(np.median(adata.obsm["spatial"][sec_mask, 2]))

    return train_adata, holdout_adata, target_z


def run_method(train_adata, holdout_adata, target_z, seed=42,
               training_iter=50, n_genes_max=2000, device="0"):
    """Execute SVGP interpolation for held-out sections.

    Parameters
    ----------
    train_adata : AnnData
        Training data with 3D coords in obsm['spatial'].
    holdout_adata : AnnData
        Held-out data (used only for target coordinates).
    target_z : dict
        section_label -> z coordinate.
    seed : int
    training_iter : int
        GP training iterations per gene.
    n_genes_max : int
        Max genes to interpolate.
    device : str
        'cpu' or GPU index like '0'.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_cuda = device != "cpu" and torch.cuda.is_available()
    torch_device = torch.device(f"cuda:{device}" if use_cuda else "cpu")
    print(f"  Device: {torch_device}")

    results = {}
    sections = holdout_adata.obs["section"].values.astype(str)

    # Normalize expression (Gaussian likelihood assumes ~Gaussian targets)
    import scanpy as sc
    expr_type = train_adata.uns.get("expression_type", "raw_counts")
    if expr_type == "raw_counts":
        sc.pp.normalize_total(train_adata, target_sum=1e4)
        sc.pp.log1p(train_adata)
    elif expr_type in ("log1p_normalized", "log2_normalized"):
        pass
    elif expr_type in ("fluorescence_intensity", "mean_intensity", "normalized"):
        sc.pp.log1p(train_adata)

    # Select genes
    n_genes = train_adata.n_vars
    if n_genes > n_genes_max:
        X = train_adata.X
        if sp.issparse(X):
            X = X.toarray()
        gene_var = np.var(X, axis=0)
        top_idx = np.argsort(gene_var)[-n_genes_max:]
        gene_names = train_adata.var_names[top_idx].tolist()
        print(f"  Selected top {n_genes_max} variable genes (of {n_genes})")
    else:
        gene_names = train_adata.var_names.tolist()

    # Prepare training spatial coords (normalize to zero-mean, unit-variance)
    train_coords = train_adata.obsm["spatial"].astype(np.float32)
    coord_mean = train_coords.mean(axis=0)
    coord_std = np.sqrt(np.sum((train_coords - coord_mean) ** 2) / len(train_coords))
    if coord_std == 0:
        coord_std = 1.0
    train_coords_norm = (train_coords - coord_mean) / coord_std
    train_x = torch.from_numpy(train_coords_norm).to(torch_device)

    # Inducing points
    inducing_num = min(512, train_adata.n_obs // 2)
    inducing_idx = np.random.choice(len(train_coords_norm), inducing_num, replace=False)
    inducing_points = train_x[inducing_idx].clone()

    # Get expression matrix
    X_full = train_adata.X
    if sp.issparse(X_full):
        X_full = X_full.toarray()

    # Gene name to index mapping
    var_names = train_adata.var_names.tolist()
    gene_idx_map = {g: var_names.index(g) for g in gene_names}

    batch_size = min(1024, train_adata.n_obs)

    for target_sec, tz in sorted(target_z.items(), key=lambda kv: kv[1]):
        sec_mask = sections == target_sec
        target_coords = holdout_adata.obsm["spatial"][sec_mask].astype(np.float32)
        n_target = target_coords.shape[0]

        target_coords_norm = (target_coords - coord_mean) / coord_std
        test_x = torch.from_numpy(target_coords_norm).to(torch_device)

        print(f"  {target_sec}: {n_target} target cells, {len(gene_names)} genes")

        pred_matrix = np.zeros((n_target, len(gene_names)), dtype=np.float32)

        for gi, gene in enumerate(gene_names):
            col_idx = gene_idx_map[gene]
            train_y = torch.from_numpy(
                X_full[:, col_idx].astype(np.float32)).to(torch_device)

            try:
                model, likelihood = _train_svgp_one_gene(
                    train_x, train_y, inducing_points.clone(),
                    training_iter, batch_size, torch_device)
                pred_matrix[:, gi] = _predict_svgp(model, likelihood, test_x)
            except Exception as e:
                print(f"      Gene {gene} failed: {e}")
                pred_matrix[:, gi] = 0.0

            if (gi + 1) % 100 == 0 or gi == len(gene_names) - 1:
                print(f"      {gi + 1}/{len(gene_names)} genes done", flush=True)

            # Free GPU memory
            del model, likelihood
            if use_cuda:
                torch.cuda.empty_cache()

        coords = target_coords.copy()
        coords[:, 2] = tz

        results[target_sec] = {
            "X": sp.csr_matrix(pred_matrix),
            "coords": coords,
            "cell_type": np.array(["NA"] * n_target),
            "gene_names": gene_names,
        }
        print(f"    -> {n_target} cells predicted")

    return results


def format_output(results, gene_names_full, holdout_sections, method_params,
                  wall_time, output_path):
    """Write prediction.h5 in standardized format."""
    if not results:
        print("No results to write!")
        return

    first_result = next(iter(results.values()))
    gene_names = first_result.get("gene_names", gene_names_full)

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
        uns.create_dataset("method_name", data="spateo_gp")
        uns.create_dataset("holdout_sections", data=json.dumps(holdout_sections))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)

    print(f"Wrote {cell_counter} predicted cells to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SVGP virtual slice interpolation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--holdout-sections", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-iter", type=int, default=50)
    parser.add_argument("--n-genes-max", type=int, default=2000)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)
    gene_names = adata.var_names.tolist()

    print(f"Preparing input (holdout: {args.holdout_sections})...")
    train_adata, holdout_adata, target_z = prepare_input(adata, args.holdout_sections)
    del adata

    print(f"Running SVGP interpolation...")
    t0 = time.time()
    results = run_method(train_adata, holdout_adata, target_z, seed=args.seed,
                         training_iter=args.training_iter,
                         n_genes_max=args.n_genes_max, device=args.device)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed,
        "training_iter": args.training_iter,
        "n_genes_max": args.n_genes_max,
    }
    format_output(results, gene_names, args.holdout_sections,
                  method_params, wall_time, args.output)


if __name__ == "__main__":
    main()
