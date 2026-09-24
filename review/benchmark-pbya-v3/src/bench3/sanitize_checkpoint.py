"""Make a torch checkpoint loadable under torch >= 2.6's ``weights_only=True``.

Runs **inside the method's conda env** (it needs that env's torch), invoked by
``assets.ensure_safe_checkpoint``. Standalone on purpose: no bench3 imports, so
``conda run -n <env> python sanitize_checkpoint.py`` is all it takes.

Why this exists
---------------
torch 2.6 flipped ``torch.load(weights_only=)`` to ``True``. A *training*
checkpoint — OmiCLIP ships one as ``checkpoint.pt`` — pickles numpy scalars
(epoch/step bookkeeping) and optimizer state next to the tensors, and the new
default refuses to unpickle those globals::

    UnpicklingError: Weights only load failed ...
    Unsupported global: GLOBAL numpy.core.multiarray.scalar

open_clip loads a checkpoint through ``torch.load`` with that default, so the
teacher never builds. Nothing is wrong with the checkpoint; only the extra
bookkeeping in it is unloadable. Writing the tensors alone to a new file makes
the very same weights load cleanly, with no change to the method package, no
``weights_only=False``, and no pickle of anything executable.

Prints one machine-readable line for the caller to parse::

    SANITIZE_RESULT=passthrough                     # already safe, use the original
    SANITIZE_RESULT=converted tensors=<n> dropped=<n>
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="checkpoint to check/convert")
    ap.add_argument("--output", required=True, help="where to write a converted copy")
    args = ap.parse_args()

    import torch

    # 1. Already loadable under the safe default? Then leave it alone.
    try:
        torch.load(args.input, map_location="cpu", weights_only=True)
        print("SANITIZE_RESULT=passthrough")
        return 0
    except TypeError:
        # torch too old to have weights_only: nothing to protect against.
        print("SANITIZE_RESULT=passthrough")
        return 0
    except Exception as first:
        reason = str(first).splitlines()[0]

    # 2. Full load of the operator's own asset, then keep only the tensors.
    ck = torch.load(args.input, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    if not isinstance(sd, dict):
        print(f"ERROR: {args.input} holds a {type(sd).__name__}, not a state dict",
              file=sys.stderr)
        return 1

    tensors, dropped = {}, []
    for k, v in sd.items():
        if torch.is_tensor(v):
            key = str(k)
            tensors[key[len("module."):] if key.startswith("module.") else key] = v
        else:
            dropped.append(str(k))
    if not tensors:
        print(f"ERROR: no tensors in {args.input}; top-level keys: "
              f"{sorted(map(str, sd))[:12]}", file=sys.stderr)
        return 1

    torch.save(tensors, args.output)
    torch.load(args.output, map_location="cpu", weights_only=True)   # prove it
    print(f"  original failed weights_only load: {reason}")
    if dropped:
        print(f"  dropped {len(dropped)} non-tensor entries: {sorted(dropped)[:8]}")
    print(f"SANITIZE_RESULT=converted tensors={len(tensors)} dropped={len(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
