"""Shared I/O and guards for benchmark-pbya-v2 generation-only wrappers.

Every v2 method wrapper shares the same contract:

    --input <train_registered.h5ad>   training-only, re-registered (NO holdout)
    --target-section S1 [S2 ...]       held-out section label(s), for output tags
    --target-z Z1 [Z2 ...]             target z per section (parallel to labels)
    --output <prediction.h5>
    --seed N

The wrapper synthesizes a virtual slice at each target z from the training
slices and writes ``prediction.h5`` in the standard benchmark format (so the v2
evaluator, which extends the v1 one, reads it unchanged).  Crucially the input
file already excludes the held-out section(s); :func:`guard_no_holdout` re-checks
this so a wrapper cannot silently consume leaked cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import scipy.sparse as sp


def add_v2_args(parser):
    """Add the shared generation-only CLI arguments to an ArgumentParser."""
    parser.add_argument("--input", required=True,
                        help="training-only, re-registered h5ad (no held-out cells)")
    parser.add_argument("--target-section", nargs="+", required=True,
                        help="held-out section label(s), for output tagging / eval join")
    parser.add_argument("--target-z", nargs="+", type=float, required=True,
                        help="target z per section (parallel to --target-section)")
    parser.add_argument("--output", required=True, help="output prediction.h5 path")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_targets(args) -> List[Tuple[str, float]]:
    """Return [(section_label, target_z), ...] from parsed args."""
    if len(args.target_section) != len(args.target_z):
        raise ValueError("--target-section and --target-z must have equal length")
    return [(str(s), float(z)) for s, z in zip(args.target_section, args.target_z)]


def guard_no_holdout(adata, target_sections) -> None:
    """Hard-fail if any target (held-out) section is present in the method input."""
    present = (set(adata.obs["section"].values.astype(str))
               & {str(s) for s in target_sections})
    if present:
        raise AssertionError(
            f"LEAKAGE: target sections {sorted(present)} present in the method "
            f"input — the input must be training-only."
        )


def write_prediction_h5(results: Dict[str, dict], gene_names, target_sections,
                        method_params: dict, wall_time: float, output_path: str,
                        method_name: str) -> None:
    """Write prediction.h5 in the standardized benchmark format.

    ``results`` : section_label -> {X (csr), coords (n,3), cell_type (n,)}.
    ``uns/holdout_sections`` is set to ``target_sections`` so the evaluator joins
    predictions to the correct ground-truth section.
    """
    all_X, all_ids, all_x, all_y, all_z = [], [], [], [], []
    all_section, all_ct = [], []
    counter = 0
    for sec in target_sections:
        if sec not in results:
            continue
        r = results[sec]
        n = r["X"].shape[0]
        if n == 0:
            continue
        all_X.append(r["X"])
        all_ids.extend([f"pred_{counter + i}" for i in range(n)])
        all_x.append(r["coords"][:, 0])
        all_y.append(r["coords"][:, 1])
        all_z.append(r["coords"][:, 2])
        all_section.extend([sec] * n)
        all_ct.extend([str(c) for c in r["cell_type"]])
        counter += n

    if counter == 0:
        print("No cells produced — writing nothing.")
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
        uns.create_dataset("method_name", data=method_name)
        uns.create_dataset("holdout_sections", data=json.dumps([str(s) for s in target_sections]))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)

    print(f"Wrote {counter} synthesized cells to {output_path}")
