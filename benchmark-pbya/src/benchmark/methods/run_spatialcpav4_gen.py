"""SpatialCPA-v4 in de-novo *generation* mode (benchmark method `spatialcpav4_gen`).

This is a thin variant of ``run_spatialcpav4.py`` that never sees the held-out
cells' real (x, y) coordinates. For a virtual-slice-generation benchmark, that
is the honest regime: the method must *synthesize* the slice — its cell count
and positions are emergent — rather than being handed the answer's geometry.

It simply forces two flags before delegating to the shared implementation:

  --generate-mode                 (grid over the flanking training slices' XY
                                   bbox at the target z; keep occupancy>threshold)
  --method-name spatialcpav4_gen  (so prediction.h5 / results are labelled apart)

Everything else — CLI, output format, leakage safeguards — is identical to
``run_spatialcpav4.py``, so it runs under ``run_benchmark.py`` unchanged. Extra
tuning flags (``--grid-points``, ``--grid-type``, ``--occupancy-threshold``, the
model/training knobs) can still be passed through and are respected.

Usage:
    conda run -n bench_spatialcpa python \
        src/benchmark/methods/run_spatialcpav4_gen.py \
        --input data/processed/cosmx_nsclc_3d/data.h5ad \
        --holdout-sections section_10 \
        --output results/spatialcpav4_gen/cosmx_nsclc_3d/loo_section_10/prediction.h5 \
        --seed 42
"""

import sys
from pathlib import Path

# Ensure the sibling module is importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_spatialcpav4  # noqa: E402

METHOD_NAME = "spatialcpav4_gen"


def main():
    argv = sys.argv[1:]
    # Force generation mode unless the caller already asked for it.
    if "--generate-mode" not in argv:
        sys.argv.append("--generate-mode")
    # Label the run distinctly unless explicitly overridden.
    if "--method-name" not in argv:
        sys.argv += ["--method-name", METHOD_NAME]
    run_spatialcpav4.main()


if __name__ == "__main__":
    main()
