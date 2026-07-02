"""Cap CPU thread usage so a single method cannot saturate the server.

Every method here leans on libraries that, by default, spin up one thread per
CPU core: PyTorch (intra-op parallelism), and the BLAS/OpenMP backends behind
numpy / scipy / scikit-learn (OpenBLAS, MKL, ...). Running even a single method
therefore pins every core to ~100%, which is what overloads a shared server.

This module caps all of those thread pools to one value. The cap is read from
the ``BENCH_NUM_THREADS`` environment variable; if unset it defaults to half of
the visible cores, leaving headroom for the rest of the machine.

IMPORTANT: the BLAS/OpenMP backends read their thread-count environment
variables only once, when the native library is first imported. So
``limit_cpu_threads()`` (or at least :func:`set_thread_env`) must run *before*
``import numpy`` / ``torch`` / ``scanpy``. Call it as the very first thing in an
entry point.

Usage
-----
    from benchmark._cpu import limit_cpu_threads
    limit_cpu_threads()          # then import numpy / torch / scanpy

Override the cap without touching code::

    BENCH_NUM_THREADS=1 python -m benchmark.run_all ...
"""

import os

# BLAS / OpenMP backends whose thread counts are controlled by env vars.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def resolve_thread_count():
    """Number of threads each method is allowed to use.

    Precedence: ``BENCH_NUM_THREADS`` (if a positive integer) else half of the
    visible cores (at least 1).
    """
    env = os.environ.get("BENCH_NUM_THREADS")
    if env:
        try:
            n = int(env)
            if n > 0:
                return n
        except ValueError:
            pass
    n_cpu = os.cpu_count() or 1
    return max(1, n_cpu // 2)


def set_thread_env(n=None):
    """Set the BLAS/OpenMP thread-count env vars (does not import numpy/torch).

    Uses ``setdefault`` so an explicit ``OMP_NUM_THREADS=...`` in the
    environment is always honoured. Returns the resolved thread count.
    """
    n = n if n is not None else resolve_thread_count()
    s = str(n)
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, s)
    return n


def thread_env(n=None):
    """Return an ``os.environ`` copy with the thread caps applied.

    Handy for handing a capped environment to a child process
    (``subprocess.Popen(..., env=thread_env())``).
    """
    n = n if n is not None else resolve_thread_count()
    env = os.environ.copy()
    s = str(n)
    for var in _THREAD_ENV_VARS:
        env.setdefault(var, s)
    return env


def limit_cpu_threads(n=None, verbose=True):
    """Cap BLAS/OpenMP env vars *and* PyTorch's thread pools.

    Call before importing numpy / torch / scanpy. Safe to call more than once.
    Returns the resolved thread count.
    """
    n = set_thread_env(n)

    # Cap PyTorch too (it does not read the OMP env var for its own intra-op
    # pool). Import is deferred so this module stays import-cheap.
    try:
        import torch

        torch.set_num_threads(n)
        try:
            # interop pool can only be set once, before any parallel work runs
            torch.set_num_interop_threads(min(n, 2))
        except Exception:
            pass
    except Exception:
        pass

    if verbose:
        print(f"[cpu] limiting to {n} thread(s) per method "
              f"(set BENCH_NUM_THREADS to change)")
    return n
