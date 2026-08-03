"""Validate every registered dataset's build without needing the real volumes.

``selftest.py`` proves the *evaluator* orders known-quality reconstructions
correctly. This proves the other half: that each entry in
``config.DATASET_SPECS`` actually builds — right partition mode, right section
count, right hold-out pattern, markers that survive any gene cap — against a
synthetic source shaped like that dataset's real one.

It is not a substitute for building from the real files, which is the only thing
that can catch a wrong column name or a wrong unit in a reader. What it does
catch is the class of mistake that a spec makes on its own: a partition mode that
cannot express the volume, an ``n_sections`` that leaves sections too thin, a
hold-out rule that produces an unbracketed section, a marker panel that resolves
empty, and a gene cap that silences the marker group it was supposed to preserve.

The one negative control is the important part of the file. The Allen atlases
carry section-local ``x``/``y``/``z`` in ``obs`` next to the reconstructed volume
position in ``obsm``, and building from the wrong one produces a dataset that
passes every structural check while being geometric nonsense. The control builds
that dataset with the guard disabled and asserts the build is *refused*.

Usage:
    python -m src.bench3.selftest_datasets
    python -m src.bench3.selftest_datasets --only allen_zhuang_abca1
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from .config import DATASET_SPECS, spec
from .design import build_designs
from .prepare_dataset import build

# Panels the synthetic sources carry, so ``resolve_markers`` has something to
# find. Real panels are larger; these are the genes the specs actually name.
PANELS = {
    "cortex": ("Flt1", "Pcp4", "Cux2", "Rasgrf2", "Nov", "Ctgf", "Sema3e"),
    # Zhuang-shaped: Gad2 and Sox10 present, Gad1 and Mbp absent — the
    # panel that exposed the original marker pick as unresolvable.
    "brain": ("Slc17a7", "Gad2", "Sox10", "Plp1", "Gja1"),
    "hypothalamus": ("Gad1", "Slc17a6", "Nts", "Gad2", "Slc32a1"),
    "tumour": ("EPCAM", "PTPRC", "COL1A1", "KRT8", "KRT18"),
    "imc": ("panCK", "CD3", "CD31"),
}

# One case per registered dataset: the shape its real source has.
#   panel        which marker panel the synthetic var_names carry
#   z            the z positions present in the source
#   per_z        cells at each z
#   extra_genes  filler genes, to exercise a gene cap where one is configured
#   decoy_obs    plant section-local obs x/y/z (the Allen shape)
#   expect       (n_sections, holdout_id) the build must produce
CASES = {
    "starmap_visual_cortex": dict(
        panel="cortex", z=list(range(6, 95)), per_z=60, obs_xyz=True,
        expect=(7, "paper_2_4_6")),
    "exseq_visual_cortex": dict(
        panel="cortex", z=list(np.linspace(0, 76, 60)), per_z=25,
        expect=(7, "paper_2_4_6")),
    "imc_breast_cancer": dict(
        panel="imc", z=list(np.arange(15) * 10.0), per_z=200,
        expect=(15, "paper_alt7of15")),
    "cosmx_nsclc_3d": dict(
        panel="tumour", z=list(np.arange(6) * 30.0), per_z=300,
        expect=(6, "paper_2_4")),
    "deep_starmap": dict(
        panel="cortex", z=list(np.arange(60) * 0.7), per_z=40,
        expect=(7, "paper_2_4_6")),
    "merfish_hypothalamus": dict(
        panel="hypothalamus", z=list(np.arange(12) * 50.0), per_z=300,
        expect=(12, "paper_alt5of12")),
    "openst_lymph_node": dict(
        panel="tumour", z=list(np.arange(19) * 10.0), per_z=200,
        expect=(19, "paper_alt9of19")),
    "allen_merfish_brain": dict(
        panel="brain", z=list(np.arange(59) * 200.0), per_z=120,
        extra_genes=60, decoy_obs=True, expect=(59, "paper_alt29of59")),
    "allen_zhuang_abca1": dict(
        panel="brain", z=list(np.arange(40) * 200.0), per_z=120,
        extra_genes=60, decoy_obs=True, expect=(15, "paper_alt7of15")),
    "allen_zhuang_abca2": dict(
        panel="brain", z=list(np.arange(40) * 200.0), per_z=120,
        extra_genes=60, decoy_obs=True, expect=(15, "paper_alt7of15")),
    "merfish_thick_cortex": dict(
        panel="brain", z=list(np.linspace(0, 100, 50)), per_z=20,
        extra_genes=40, expect=(7, "paper_2_4_6")),
    "merfish_thick_hypothalamus": dict(
        panel="hypothalamus", z=list(np.linspace(0, 200, 50)), per_z=20,
        extra_genes=40, expect=(7, "paper_2_4_6")),
    "easi_fish_lha1": dict(
        panel="hypothalamus", z=list(np.linspace(0, 150, 60)), per_z=15,
        expect=(7, "paper_2_4_6")),
    "easi_fish_lha2": dict(
        panel="hypothalamus", z=list(np.linspace(0, 150, 60)), per_z=15,
        expect=(7, "paper_2_4_6")),
    "easi_fish_lha3": dict(
        panel="hypothalamus", z=list(np.linspace(0, 150, 60)), per_z=15,
        expect=(7, "paper_2_4_6")),
    "exseq_breast_cancer": dict(
        panel="tumour", z=list(np.linspace(0, 80, 40)), per_z=20,
        extra_genes=60, expect=(5, "paper_2_4")),
    "st_mouse_brain_ortiz": dict(
        panel="brain", z=list(np.arange(75) * 200.0), per_z=200, spot=True,
        extra_genes=8000, expect=(15, "paper_alt7of15")),
    "visium_mouse_brain_c2l": dict(
        panel="brain", z=[0.0, 210.0, 420.0], per_z=900, spot=True,
        extra_genes=8000, expect=(3, "paper_2")),
}


def synth_source(case, seed=0):
    """A source volume in ``benchmark-pbya``'s processed form."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    genes = list(PANELS[case["panel"]]) + [
        f"G{i}" for i in range(case.get("extra_genes", 0))]

    z_vals = list(case["z"])
    zz = np.repeat(np.asarray(z_vals, dtype=float), case["per_z"])
    n = len(zz)
    span = 2000.0 if case.get("spot") else 400.0
    xy = rng.uniform(0, span, size=(n, 2))

    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    obs["section"] = np.repeat([f"S{i}" for i in range(len(z_vals))],
                               case["per_z"])
    obs["cell_type"] = rng.choice(["A", "B", "C"], size=n)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, size=(n, len(genes)))
                                   .astype(np.float32)),
                   obs=obs, var=pd.DataFrame(index=genes))
    a.obsm["spatial"] = np.column_stack([xy, zz])
    if case.get("decoy_obs"):
        # Section-local coordinates, as the Allen metadata CSV supplies them.
        a.obs["x"] = rng.uniform(0, 10, size=n)
        a.obs["y"] = rng.uniform(0, 10, size=n)
        a.obs["z"] = rng.uniform(0, 1, size=n)
    a.uns["expression_type"] = "raw_counts"
    a.uns["spatial_metadata"] = {"coordinate_units": "micrometers"}
    if case.get("obs_xyz"):
        # STARmap's distribution keeps the z-planes as voxel indices in obs, and
        # its spec reads them from there; the trim is stated in plane numbers.
        a.obs["x"], a.obs["y"], a.obs["z"] = xy[:, 0], xy[:, 1], zz
    return a


def check_one(name, case, tmp, verbose=True):
    """Build one dataset from a synthetic source. Returns a list of failures."""
    fails = []
    src = synth_source(case)
    src_path = Path(tmp) / f"{name}_src.h5ad"
    src.write_h5ad(str(src_path))

    try:
        built = build(dataset=name, raw_path=src_path,
                      output_path=Path(tmp) / name / "data.h5ad", verbose=verbose)
    except Exception as e:
        return [f"{name}: build raised {type(e).__name__}: {e}"]

    pp = built.uns["paper_protocol"]
    want_n, want_id = case["expect"]
    got_id = build_designs(Path(tmp) / name / "data.h5ad", design="paper")[0]["holdout_id"]

    if pp["n_sections"] != want_n:
        fails.append(f"{name}: n_sections={pp['n_sections']} want {want_n}")
    if got_id != want_id:
        fails.append(f"{name}: holdout_id={got_id!r} want {want_id!r}")
    if pp.get("resolution") != spec(name).get("resolution", "single_cell"):
        fails.append(f"{name}: resolution={pp.get('resolution')!r}")
    if not pp["marker_genes"]:
        fails.append(f"{name}: no marker genes resolved from the panel")
    # A gene cap must never remove a marker or layer gene.
    if pp.get("gene_selection"):
        panel = {str(v).lower() for v in built.var_names}
        lost = [g for g in (list(pp["marker_genes"]) + list(pp["layer_superficial"])
                            + list(pp["layer_deep"])) if str(g).lower() not in panel]
        if lost:
            fails.append(f"{name}: gene cap dropped marker/layer genes {lost}")
    # Every held-out section must be bracketed by input sections on both sides.
    idx = sorted(int(s.rsplit("_", 1)[1]) for s in pp["held_out_sections"])
    if idx and (min(idx) < 2 or max(idx) >= pp["n_sections"]):
        fails.append(f"{name}: held-out sections {idx} are not all bracketed "
                     f"within 1..{pp['n_sections']}")
    return fails


def check_coordinate_guard(tmp, verbose=True):
    """The negative control — see this module's docstring."""
    name = "allen_zhuang_abca1"
    case = CASES[name]
    src = synth_source(case)
    src_path = Path(tmp) / "guard_src.h5ad"
    src.write_h5ad(str(src_path))

    saved = DATASET_SPECS[name].get("coords")
    DATASET_SPECS[name]["coords"] = "auto"     # believe the decoy obs x/y/z
    try:
        build(dataset=name, raw_path=src_path,
              output_path=Path(tmp) / "guard" / "data.h5ad", verbose=False)
    except Exception as e:
        if verbose:
            print(f"  refused, as it must: {type(e).__name__}: {str(e)[:90]}...")
        return []
    else:
        return ["coordinate guard is inert: a build from section-local obs "
                "x/y/z succeeded, so a wrong coordinate array would be silent"]
    finally:
        DATASET_SPECS[name]["coords"] = saved


def main():
    ap = argparse.ArgumentParser(
        description="Validate every registered dataset's build on synthetic sources")
    ap.add_argument("--only", nargs="*", help="limit to these dataset names")
    ap.add_argument("--verbose", action="store_true",
                    help="show each build's full output")
    args = ap.parse_args()

    missing = sorted(set(DATASET_SPECS) - set(CASES))
    if missing:
        print(f"FAIL: registered datasets with no selftest case: {missing}\n"
              f"  Add one to CASES — a dataset that is not exercised here is a "
              f"dataset whose spec has never been run.", file=sys.stderr)
        return 1

    names = [n for n in CASES if not args.only or n in set(args.only)]
    if not names:
        print(f"no such dataset; known: {sorted(CASES)}", file=sys.stderr)
        return 1

    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in names:
                print(f"\n{'=' * 72}\n  {name}\n{'=' * 72}")
                f = check_one(name, CASES[name], tmp, verbose=args.verbose)
                fails += f
                print("  FAIL " + "\n  FAIL ".join(f) if f
                      else f"  ok — {CASES[name]['expect'][0]} sections, "
                           f"holdout {CASES[name]['expect'][1]}")
            if not args.only:
                print(f"\n{'=' * 72}\n  negative control: coordinate guard\n{'=' * 72}")
                fails += check_coordinate_guard(tmp, verbose=True)

    print(f"\n{'=' * 72}")
    if fails:
        print(f"{len(fails)} FAILURE(S):")
        for f in fails:
            print(f"  {f}")
        return 1
    print(f"All {len(names)} dataset builds passed"
          + ("" if args.only else ", coordinate guard active"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
