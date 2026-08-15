"""A deterministic stand-in for the frozen text encoder.

MedCPT cannot be downloaded in the test environment (Convention 7: no network), so every
test that needs text vectors uses this instead: a fixed pseudo-random map from descriptor
string to vector, derived from ``sha256(descriptor)``. It is deliberately *not* importable
from the package - a hash-embedding fallback living inside `spatialcpav25_gen` would be
exactly the kind of silent fallback Convention 6 forbids.
"""

from __future__ import annotations

import hashlib

import numpy as np
import numpy.typing as npt
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data import text as text_mod

__all__ = ["HashTextBackend", "RaisingBackend", "fake_text_vecs", "install_fake_backend"]


class HashTextBackend:
    """Maps each descriptor to a fixed vector drawn from a seed derived from its sha256."""

    def __init__(self, cfg: Config) -> None:
        self.dim = cfg.text_dim_in
        self.calls = 0

    def encode_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self.calls += 1
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            gen = np.random.default_rng(int.from_bytes(digest, "big"))
            out[i] = gen.standard_normal(self.dim, dtype=np.float32)
        return out


class RaisingBackend:
    """A backend whose construction fails: proves the cached path never loads a model."""

    def __init__(self, cfg: Config) -> None:
        raise AssertionError("the text model must not be loaded when every vector is cached")

    def encode_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        raise AssertionError("unreachable")


def install_fake_backend(monkeypatch, cfg: Config) -> HashTextBackend:
    """Point `load_transformer_backend` at a `HashTextBackend` and return it."""
    backend = HashTextBackend(cfg)
    monkeypatch.setattr(text_mod, "load_transformer_backend", lambda _cfg: backend)
    return backend


def install_raising_backend(monkeypatch) -> None:
    """Point `load_transformer_backend` at something that raises when it is called."""

    def _raise(_cfg: Config) -> None:
        raise AssertionError("the text model must not be loaded when every vector is cached")

    monkeypatch.setattr(text_mod, "load_transformer_backend", _raise)


def fake_text_vecs(n: int, dim: int, seed: int) -> npt.NDArray[np.float32]:
    """``(n, dim)`` float32 L2-normalised pseudo-MedCPT vectors."""
    gen = np.random.default_rng(seed)
    vecs = gen.standard_normal((n, dim)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
