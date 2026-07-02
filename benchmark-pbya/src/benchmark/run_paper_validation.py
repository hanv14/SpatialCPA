"""Run paper-matching validation experiments.

Reproduces the exact dataset/holdout configurations used in each method's
original paper to validate our implementations against published results.

Usage:
    python -m src.benchmark.run_paper_validation --experiment feast
    python -m src.benchmark.run_paper_validation --experiment isost
    python -m src.benchmark.run_paper_validation --experiment spatialz
    python -m src.benchmark.run_paper_validation --experiment stvgp
    python -m src.benchmark.run_paper_validation --experiment all
    python -m src.benchmark.run_paper_validation --list
"""

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from .config import DATASETS, METHODS, PROJECT_ROOT, RESULTS_DIR, RANDOM_SEED
from .holdout import get_sorted_sections
from .run_benchmark import run_single


# ── Subset h5ad creation ─────────────────────────────────────────────────────

SUBSET_DIR = RESULTS_DIR / "paper_validation_subsets"


def _create_subset_h5ad(source_path, section_names, output_path):
    """Create a subset h5ad containing only the specified sections.

    Skips creation if the file already exists with the correct sections.
    """
    output_path = Path(output_path)
    if output_path.exists():
        # Quick check: already has the right sections?
        try:
            a = ad.read_h5ad(str(output_path), backed="r")
            existing = set(a.obs["section"].unique())
            a.file.close()
            if existing == set(section_names):
                print(f"  Subset already exists: {output_path}")
                return str(output_path)
        except Exception:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Creating subset h5ad: {len(section_names)} sections -> {output_path}")
    adata = ad.read_h5ad(str(source_path))
    mask = adata.obs["section"].isin(section_names)
    subset = adata[mask].copy()
    subset.write_h5ad(str(output_path))
    print(f"    {subset.shape[0]} cells, {subset.shape[1]} genes")
    del adata, subset
    return str(output_path)


# ── Paper-matching experiment definitions ────────────────────────────────────

def _feast_experiment():
    """FEAST paper: Zhuang-ABCA-1, slices 5-9.

    Train on sections .005 and .009, hold out .006-.008.
    Paper claims gene_mean_pearson >0.9.
    """
    dataset = "allen_zhuang_merfish/Zhuang-ABCA-1"
    h5ad_path = str(DATASETS[dataset]["path"])
    adata = ad.read_h5ad(h5ad_path, backed="r")

    sections, z_by_section = get_sorted_sections(adata)
    adata.file.close()

    # FEAST uses section numbers 5-9 (Zhuang-ABCA-1.005 through .009)
    feast_sections = [s for s in sections if any(
        s.endswith(f".{n:03d}") for n in range(5, 10))]
    feast_sections = sorted(feast_sections, key=lambda s: z_by_section[s])

    if len(feast_sections) != 5:
        print(f"WARNING: Expected 5 FEAST sections, found {len(feast_sections)}: {feast_sections}")
        return []

    # Create subset h5ad with just these 5 sections (avoids loading 2.8M cells)
    subset_path = _create_subset_h5ad(
        h5ad_path, feast_sections,
        SUBSET_DIR / "feast_abca1_slices5to9.h5ad")

    # Hold out middle 3, train on first and last
    holdout = feast_sections[1:4]  # .006, .007, .008
    remaining = [feast_sections[0], feast_sections[4]]  # .005, .009

    holdout_config = {
        "holdout_id": "feast_paper_slices5to9",
        "holdout_sections": holdout,
        "remaining_sections": remaining,
        "holdout_z": {s: z_by_section[s] for s in holdout},
    }

    return [{
        "method": "feast",
        "dataset": dataset,
        "holdout_config": holdout_config,
        "extra_args": [],
        "input_override": subset_path,
        "paper_expectation": "gene_mean_pearson >0.9",
        "paper_reference": "Wang et al. 2025, FEAST",
    }]


def _isost_experiment():
    """isoST paper: Zhuang-ABCA-2, alternating holdout.

    Paper used 54 sections (we have 66), alternating train/test.
    Paper reports MSE and Dice metrics.
    """
    dataset = "allen_zhuang_merfish/Zhuang-ABCA-2"
    h5ad_path = str(DATASETS[dataset]["path"])
    adata = ad.read_h5ad(h5ad_path, backed="r")

    sections, z_by_section = get_sorted_sections(adata)
    adata.file.close()

    # Alternating holdout: odd-indexed sections held out
    holdout = [sections[i] for i in range(1, len(sections) - 1, 2)]
    remaining = [sections[i] for i in range(0, len(sections), 2)]
    # Add last section to remaining if it was odd-indexed
    if (len(sections) - 1) % 2 == 1:
        remaining.append(sections[-1])

    holdout_config = {
        "holdout_id": f"isost_paper_alternating_{len(holdout)}sec",
        "holdout_sections": holdout,
        "remaining_sections": remaining,
        "holdout_z": {s: z_by_section[s] for s in holdout},
    }

    return [{
        "method": "isost",
        "dataset": dataset,
        "holdout_config": holdout_config,
        "extra_args": ["--batch-num", "5", "--epochs", "100", "100", "100"],
        "paper_expectation": "competitive MSE, Dice >0.5",
        "paper_reference": "Li et al. 2025, isoST",
    }]


def _spatialz_experiment():
    """SpatialZ paper: MERFISH hypothalamus, 5 sections.

    Paper used bregma 0.04, 0.09, 0.14, 0.19, 0.24 (fractional bregma coords).
    Our sections: bregma_-0.29 to bregma_+0.26, spaced 0.05mm apart.
    The SpatialZ demo uses 5 sections, holds out middle 3.
    """
    dataset = "merfish_hypothalamus/animal_1"
    h5ad_path = str(DATASETS[dataset]["path"])
    adata = ad.read_h5ad(h5ad_path, backed="r")

    sections, z_by_section = get_sorted_sections(adata)
    adata.file.close()

    # SpatialZ demo: 5 consecutive sections from the middle
    # Our 12 sections span bregma -0.29 to +0.26
    # Select 5 consecutive interior sections for a clean test
    # Use sections 4-8 (0-indexed) which gives us 5 interior sections
    if len(sections) >= 7:
        subset = sections[3:8]  # 5 consecutive sections with flanking on both sides
    else:
        subset = sections[1:-1][:5]

    # Create subset h5ad with just these 5 sections
    subset_path = _create_subset_h5ad(
        h5ad_path, subset,
        SUBSET_DIR / "spatialz_merfish_hypo_5sec.h5ad")

    holdout = subset[1:4]  # middle 3
    remaining = [subset[0], subset[4]]  # anchor 2

    holdout_config = {
        "holdout_id": "spatialz_paper_5sec",
        "holdout_sections": holdout,
        "remaining_sections": remaining,
        "holdout_z": {s: z_by_section[s] for s in holdout},
    }

    return [{
        "method": "spatialz",
        "dataset": dataset,
        "holdout_config": holdout_config,
        "extra_args": [],
        "input_override": subset_path,
        "paper_expectation": "positive Moran's I improvement",
        "paper_reference": "Lin et al. 2025, SpatialZ",
    }]


def _stvgp_experiment():
    """stVGP paper: st_mouse_brain_ortiz (GSE147747).

    Paper used ADMB dataset. LOO on a subset of interior sections.
    """
    dataset = "st_mouse_brain_ortiz"
    h5ad_path = str(DATASETS[dataset]["path"])
    adata = ad.read_h5ad(h5ad_path, backed="r")

    sections, z_by_section = get_sorted_sections(adata)
    adata.file.close()

    # LOO on every 5th interior section to keep runtime manageable
    # (75 sections total, ~15 holdout experiments)
    experiments = []
    for i in range(2, len(sections) - 2, 5):
        sec = sections[i]
        remaining = [s for s in sections if s != sec]
        holdout_config = {
            "holdout_id": f"stvgp_paper_loo_{sec}",
            "holdout_sections": [sec],
            "remaining_sections": remaining,
            "holdout_z": {sec: z_by_section[sec]},
        }
        experiments.append({
            "method": "stvgp",
            "dataset": dataset,
            "holdout_config": holdout_config,
            "extra_args": [],
            "paper_expectation": "Pearson r >0.1",
            "paper_reference": "Wang et al. 2026, stVGP",
        })

    return experiments


# Also run all other methods on the paper-matching datasets for cross-comparison
def _cross_comparison_experiments():
    """Run all methods on each paper-matching dataset for fair comparison."""
    experiments = []

    # All methods on Zhuang-ABCA-1 (FEAST's dataset, 5 sections)
    feast_exp = _feast_experiment()
    if feast_exp:
        holdout_config = feast_exp[0]["holdout_config"]
        dataset = feast_exp[0]["dataset"]
        input_override = feast_exp[0].get("input_override")
        for method in METHODS:
            if method == "feast":
                continue  # already in feast_exp
            experiments.append({
                "method": method,
                "dataset": dataset,
                "holdout_config": holdout_config,
                "extra_args": [],
                "input_override": input_override,
                "paper_expectation": "cross-comparison",
                "paper_reference": f"{method} on FEAST's dataset",
            })

    return experiments


EXPERIMENTS = {
    "feast": _feast_experiment,
    "isost": _isost_experiment,
    "spatialz": _spatialz_experiment,
    "stvgp": _stvgp_experiment,
    "cross": _cross_comparison_experiments,
}


def list_experiments():
    """Print available experiments."""
    print("Available paper-validation experiments:")
    print()
    for name, func in EXPERIMENTS.items():
        exps = func()
        print(f"  {name}: {len(exps)} run(s)")
        for e in exps[:3]:
            print(f"    - {e['method']} on {e['dataset']} ({e['holdout_config']['holdout_id']})")
            print(f"      Expected: {e['paper_expectation']}")
        if len(exps) > 3:
            print(f"    ... and {len(exps) - 3} more")
    print()
    print("  all: run feast + isost + spatialz + stvgp")


def main():
    parser = argparse.ArgumentParser(
        description="Run paper-matching validation experiments")
    parser.add_argument("--experiment", required=False,
                        help="Experiment name or 'all'")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--include-cross", action="store_true",
                        help="Also run cross-comparison (all methods on each paper dataset)")
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --list)")

    # Collect experiments to run
    if args.experiment == "all":
        names = ["feast", "isost", "spatialz", "stvgp"]
    elif args.experiment in EXPERIMENTS:
        names = [args.experiment]
    else:
        print(f"Unknown experiment: {args.experiment}")
        list_experiments()
        sys.exit(1)

    all_experiments = []
    for name in names:
        all_experiments.extend(EXPERIMENTS[name]())

    if args.include_cross:
        all_experiments.extend(_cross_comparison_experiments())

    print(f"Running {len(all_experiments)} paper-validation experiment(s)...")
    print()

    results_summary = []
    for i, exp in enumerate(all_experiments):
        print(f"{'='*60}")
        print(f"[{i+1}/{len(all_experiments)}] {exp['method']} on {exp['dataset']}")
        print(f"  Holdout: {exp['holdout_config']['holdout_id']}")
        print(f"  Paper: {exp['paper_reference']}")
        print(f"  Expected: {exp['paper_expectation']}")
        print()

        result = run_single(
            exp["method"],
            exp["dataset"],
            exp["holdout_config"],
            extra_args=exp.get("extra_args"),
            dry_run=args.dry_run,
            input_override=exp.get("input_override"),
        )

        result["experiment"] = exp["paper_reference"]
        result["expected"] = exp["paper_expectation"]
        results_summary.append(result)

        if result.get("success"):
            # Load and print key metrics
            metrics_path = result.get("metrics_path")
            if metrics_path and Path(metrics_path).exists():
                metrics = json.loads(Path(metrics_path).read_text())
                key_metrics = {k: v for k, v in metrics.items()
                               if k in ("gene_mean_pearson", "pearson_mean",
                                        "dice_density", "morans_i_median",
                                        "rmse_median")}
                print(f"  Key metrics: {json.dumps(key_metrics, indent=4)}")
        print()

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for r in results_summary:
        status = "PASS" if r.get("success") else "FAIL"
        print(f"  [{status}] {r.get('experiment', '?')}: {r.get('expected', '?')}")
        if r.get("metrics_path") and Path(r["metrics_path"]).exists():
            m = json.loads(Path(r["metrics_path"]).read_text())
            if "gene_mean_pearson" in m:
                print(f"         gene_mean_pearson = {m['gene_mean_pearson']:.4f}")
            if "pearson_mean" in m:
                print(f"         pearson_mean = {m['pearson_mean']:.4f}")
            if "dice_density" in m:
                print(f"         dice_density = {m['dice_density']:.4f}")

    # Save summary
    summary_path = RESULTS_DIR / "paper_validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
