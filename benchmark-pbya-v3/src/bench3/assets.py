"""Method assets v3 prepares before handing them to a wrapper.

Right now that means one thing: a pretrained-weights file a method loads with
``torch.load``. torch >= 2.6 defaults to ``weights_only=True`` and refuses the
numpy scalars in a *training* checkpoint (OmiCLIP's ``checkpoint.pt`` is one), so
the method's teacher fails to build. That failure is silent in the worst way — v11
catches it and degrades to its OT-morph fallback, which still writes a plausible
prediction the harness would happily score.

v3 fixes it on its own side of the boundary: before the wrapper runs, the
checkpoint is checked and, if needed, rewritten as a tensors-only copy under
``results/_assets/``. The rewritten file holds exactly the same weights, so the
method is unchanged in every way that matters — v3 just hands it an asset the
current torch will load. Neither ``benchmark-pbya-v2`` nor the method packages are
touched.

The check itself needs the *method's* torch, not the orchestrator's, so it runs in
that conda env via ``sanitize_checkpoint.py``. Results are cached per (path, size,
mtime): a checkpoint that is already safe is recorded as pass-through and costs
one subprocess, once.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .config import ASSETS_CACHE

_SANITIZER = Path(__file__).resolve().parent / "sanitize_checkpoint.py"


def _fingerprint(path: Path):
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def ensure_safe_checkpoint(weights, conda_env, cache_dir=None, verbose=True):
    """Return a path to ``weights`` that torch >= 2.6 loads with weights_only=True.

    Returns the original path when it already loads (or when it cannot be checked
    — a missing file or a broken env is the wrapper's error to report, not ours).
    """
    src = Path(str(weights))
    if not src.is_file():
        return str(weights)

    cache_dir = Path(cache_dir or ASSETS_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _fingerprint(src)
    converted = cache_dir / f"{src.stem}.safe.{tag}.pt"
    record = cache_dir / f"{src.stem}.safe.{tag}.json"

    if record.exists():
        meta = json.loads(record.read_text())
        if meta.get("result") == "passthrough":
            return str(src)
        if converted.exists():
            if verbose:
                print(f"  weights: reusing sanitized copy {converted}")
            return str(converted)

    cmd = ["conda", "run", "-n", conda_env, "--no-capture-output",
           "python", str(_SANITIZER), "--input", str(src), "--output", str(converted)]
    if verbose:
        print(f"  weights: checking {src} loads under this torch...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in proc.stdout.splitlines()
                 if l.startswith("SANITIZE_RESULT=")), None)

    if proc.returncode != 0 or line is None:
        # Leave the original in place and let the method report the real problem.
        if verbose:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            print(f"  weights: could not check the checkpoint "
                  f"({detail[-1] if detail else f'exit {proc.returncode}'}); "
                  f"passing it through unchanged")
        return str(src)

    result = line[len("SANITIZE_RESULT="):].strip()
    if result.startswith("passthrough"):
        record.write_text(json.dumps({"source": str(src), "result": "passthrough"},
                                     indent=2))
        if verbose:
            print("  weights: loads as-is, using the original")
        return str(src)

    record.write_text(json.dumps(
        {"source": str(src), "result": "converted", "path": str(converted),
         "detail": result}, indent=2))
    if verbose:
        for l in proc.stdout.splitlines():
            if l.startswith("  "):
                print(f"  weights:{l}")
        print(f"  weights: {result} -> {converted}")
    return str(converted)


def sanitize_weight_args(cmd, arg_names, conda_env, verbose=True):
    """Rewrite ``arg_names`` values in ``cmd`` to sanitized checkpoint copies.

    Handles both ``--flag value`` and ``--flag=value``. Returns a new list; the
    original command is left untouched.
    """
    if not arg_names:
        return list(cmd)
    out = list(cmd)
    for i, tok in enumerate(out):
        name, sep, value = tok.partition("=")
        if sep and name in arg_names:
            safe = ensure_safe_checkpoint(value, conda_env, verbose=verbose)
            if safe != value:
                out[i] = f"{name}={safe}"
        elif tok in arg_names and i + 1 < len(out):
            value = out[i + 1]
            safe = ensure_safe_checkpoint(value, conda_env, verbose=verbose)
            if safe != value:
                out[i + 1] = safe
    return out
