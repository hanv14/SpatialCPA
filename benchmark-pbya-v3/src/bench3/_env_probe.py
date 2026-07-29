"""Standalone environment probe, executed inside a method's own conda env.

Run as::

    python _env_probe.py <wrapper.py> [<package_name>]

It loads the method wrapper **as a module** — which runs the wrapper's own
package-discovery code — then calls the wrapper's own ``check_environment()``.
Nothing here re-implements discovery, so the probe can never disagree with what
the real run would do.

Emits one machine-readable line on stdout::

    __BENCH3_PROBE__ {"ok": true, "package_root": "...", "package_file": "...", ...}

This file must stay dependency-free (stdlib only) and must NOT import ``bench3``:
it runs in bench_spatialz / bench_feast / bench_isost / bench_spatialcpa, none of
which have the harness on their path.
"""

import importlib
import importlib.util
import json
import os
import sys
import traceback

SENTINEL = "__BENCH3_PROBE__"


def main():
    if len(sys.argv) < 2:
        print(f"{SENTINEL} " + json.dumps(
            {"ok": False, "error": "usage: _env_probe.py <wrapper.py> [package]"}))
        return 2

    wrapper = os.path.abspath(sys.argv[1])
    package = sys.argv[2] if len(sys.argv) > 2 else None
    out = {"ok": False, "wrapper": wrapper, "package": package,
           "package_root": None, "package_file": None, "version": None,
           "python": sys.version.split()[0], "error": None}

    try:
        spec = importlib.util.spec_from_file_location("_bench3_wrapper", wrapper)
        module = importlib.util.module_from_spec(spec)
        # Executing the module runs its module-level discovery (the `_ROOT = ...`
        # line) and defines check_environment; `main()` stays guarded behind
        # __name__ == "__main__", so nothing is benchmarked here.
        spec.loader.exec_module(module)
    except Exception as e:
        out["error"] = f"wrapper import failed: {type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        print(f"{SENTINEL} " + json.dumps(out))
        return 1

    # The wrapper records where it found its package under one of these names.
    for attr in ("_ROOT", "_SPATIALCPAV8_ROOT", "_SPATIALCPA_ROOT"):
        if getattr(module, attr, None):
            out["package_root"] = str(getattr(module, attr))
            break

    try:
        out["ok"] = bool(module.check_environment())
    except Exception as e:
        out["error"] = f"check_environment raised: {type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()

    # Resolve the actual file that got imported — this is what makes a stale or
    # shadowed copy visible, since a package root alone can look plausible.
    if package:
        mod = sys.modules.get(package)
        if mod is None:
            try:
                mod = importlib.import_module(package)
            except Exception:
                mod = None
        if mod is not None:
            out["package_file"] = getattr(mod, "__file__", None)
            out["version"] = str(getattr(mod, "__version__", "") or "") or None

    print(f"{SENTINEL} " + json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
