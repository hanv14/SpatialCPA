"""Check every method's environment *before* launching a campaign.

Four of the seven methods import their package through a per-wrapper discovery
rule — ``$SPATIALCPAV8_ROOT`` / ``$SPATIALCPAV11_ROOT`` / … first, then a walk up
the wrapper's own parent directories. That is flexible, and it is exactly how a
run silently ends up executing a *stale copy* of a method: the wrapper asks for a
configuration the installed package doesn't support and the run dies three
sections in, or worse, an older package quietly produces a plausible-but-wrong
number.

This module answers, per method and in that method's own conda env:

* does the wrapper import at all (are its dependencies installed)?
* which package directory did it resolve, and which file was actually loaded?
* does the wrapper's own ``check_environment()`` pass?
* for the SpatialCPA variants, is the resolved package **the copy in this
  checkout** — or a different one somewhere else on the machine?

Discovery is never re-implemented here: ``_env_probe.py`` loads the wrapper as a
module (running the wrapper's own resolution code) and calls the wrapper's own
``check_environment()``, so the probe cannot disagree with what a real run does.

Usage:
    python -m src.bench3.preflight                     # all methods
    python -m src.bench3.preflight --methods spatialcpav8_gen
    python -m src.bench3.preflight --no-conda          # current interpreter
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import METHOD_ORDER, METHODS, REPO_ROOT

PROBE = Path(__file__).resolve().parent / "_env_probe.py"
SENTINEL = "__BENCH3_PROBE__"


def resolve_methods(requested=None):
    if not requested:
        return [m for m in METHOD_ORDER if METHODS[m].get("available", False)]
    unknown = [m for m in requested if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s): {unknown}; known: {sorted(METHODS)}")
    return list(requested)


def probe_method(method, use_conda=True, timeout=300):
    """Run the probe for one method; return a result dict (never raises)."""
    info = METHODS[method]
    wrapper = Path(info["wrapper"])
    out = {"method": method, "env": info.get("conda_env"), "ok": False,
           "package": info.get("package"), "package_root": None,
           "package_file": None, "version": None, "error": None,
           "in_checkout": None}

    if not wrapper.exists():
        out["error"] = f"wrapper missing: {wrapper}"
        return out

    # Under conda, `python` resolves inside the activated env — that is the whole
    # point. Without it, use *this* interpreter rather than whatever `python`
    # happens to be on PATH, so --no-conda really means "the current interpreter".
    cmd = [("python" if use_conda else sys.executable), str(PROBE), str(wrapper)]
    if info.get("package"):
        cmd.append(info["package"])
    if use_conda:
        cmd = ["conda", "run", "-n", info["conda_env"], "--no-capture-output"] + cmd

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(wrapper.resolve().parents[3]))
    except FileNotFoundError as e:
        out["error"] = f"could not launch probe ({e}); is conda on PATH?"
        return out
    except subprocess.TimeoutExpired:
        out["error"] = f"probe timed out after {timeout}s"
        return out

    payload = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith(SENTINEL):
            try:
                payload = json.loads(line[len(SENTINEL):].strip())
            except json.JSONDecodeError:
                pass
    if payload is None:
        tail = ((proc.stderr or proc.stdout or "").strip().splitlines() or ["no output"])[-1]
        out["error"] = f"probe produced no result (exit {proc.returncode}): {tail}"
        return out

    out.update({k: payload.get(k) for k in
                ("ok", "package_root", "package_file", "version", "error")})

    # A wrapper's check_environment() reports failure by *printing* and returning
    # False, not by raising — so without this the table would say FAIL with no
    # reason, which is the one thing a preflight must never do.
    if not out["ok"] and not out["error"]:
        lines = [ln.strip() for ln in
                 ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()
                 if ln.strip() and not ln.startswith(SENTINEL)]
        hits = [ln for ln in lines if "ERROR" in ln or "not installed" in ln
                or "not found" in ln]
        out["error"] = " | ".join((hits or lines)[-2:]) or "check_environment() returned False"

    # Is the resolved package the copy in THIS checkout? A package that resolves
    # somewhere else is not automatically wrong — but it is the thing to look at
    # first when a method misbehaves, so surface it rather than assuming.
    if out["package"] and out["package_file"]:
        expected = (REPO_ROOT / out["package"]).resolve()
        try:
            actual = Path(out["package_file"]).resolve().parent
            out["in_checkout"] = (actual == expected)
            out["expected_package_dir"] = str(expected)
            out["actual_package_dir"] = str(actual)
        except Exception:
            pass
    return out


def _short(path, keep=3):
    if not path:
        return "-"
    parts = Path(path).parts
    return ("…/" + "/".join(parts[-keep:])) if len(parts) > keep else str(path)


def report(results, verbose=False):
    """Print the table; return (n_ok, n_failed, n_warned)."""
    print(f"\n{'method':<20} {'env':<18} {'status':<8} {'version':<8} package")
    print("-" * 100)
    n_ok = n_fail = n_warn = 0
    for r in results:
        if r["ok"] and r.get("in_checkout") is False:
            status, n_warn = "WARN", n_warn + 1
        elif r["ok"]:
            status, n_ok = "OK", n_ok + 1
        else:
            status, n_fail = "FAIL", n_fail + 1
        print(f"{r['method']:<20} {str(r['env']):<18} {status:<8} "
              f"{str(r.get('version') or '-'):<8} "
              f"{_short(r.get('package_file') or r.get('package_root'), 4)}")
        if r.get("error"):
            print(f"{'':<20} └─ {r['error']}")
        if r.get("in_checkout") is False:
            print(f"{'':<20} └─ resolved OUTSIDE this checkout")
            print(f"{'':<20}    loaded:   {r.get('actual_package_dir')}")
            print(f"{'':<20}    checkout: {r.get('expected_package_dir')}")
            env_var = f"{r['package'].upper()}_ROOT"
            print(f"{'':<20}    if unintended: unset {env_var}, or sync that copy "
                  f"from the checkout")

    print("-" * 100)
    print(f"{n_ok} ok, {n_warn} resolved outside the checkout, {n_fail} failed")
    if n_warn:
        print("\nWARN is not automatically an error — but a method loaded from "
              "outside\nthe checkout can be an older build, which either crashes "
              "mid-run or\nquietly produces a number for a different "
              "configuration. Verify before\ntrusting the results.")
    return n_ok, n_fail, n_warn


def main():
    ap = argparse.ArgumentParser(
        description="Check each method's environment before a campaign")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--no-conda", action="store_true",
                    help="probe in the current interpreter instead of `conda run` "
                         "(for debugging; the real runs always use conda)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", help="also write the raw results here")
    args = ap.parse_args()

    methods = resolve_methods(args.methods)
    print(f"Preflight: {len(methods)} method(s) — {', '.join(methods)}")
    if args.no_conda:
        print("  (probing in the current interpreter; --no-conda)")

    results = []
    for m in methods:
        print(f"  probing {m} ...", flush=True)
        results.append(probe_method(m, use_conda=not args.no_conda,
                                    timeout=args.timeout))

    _, n_fail, _ = report(results)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
