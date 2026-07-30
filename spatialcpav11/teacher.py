"""
Frozen multimodal foundation-model *teacher* for Stage-1 layout distillation.

The layout generator is distilled from a frozen multimodal FM (OmiCLIP / Path2Space)
trained on paired H&E + ST. Even with **no images** in our serial data, such a
teacher's ST/expression tower provides *layout-related* supervision: a per-spot
**embedding** the student layout code is aligned to, and a **pseudo-layout** (spatial-
domain clustering) the student's type/region field is distilled toward.

Every teacher exposes the same interface::

    teacher.embed(expr, xy)   -> (n, d) float32 per-spot embedding
    teacher.domains(expr, xy) -> (n,)  int  spatial-domain pseudo-labels

Concrete teachers
-----------------
* :class:`OmiCLIPTeacher` (``--teacher omiclip``) — the **real** OmiCLIP mechanism:
  each spot's expression is turned into a "sentence" of its top-N expressed gene
  symbols and encoded by OmiCLIP's CLIP/CoCa **text tower** (via ``open_clip``), the
  same representation OmiCLIP/Loki use to embed ST without images. Needs
  ``open_clip_torch`` and the OmiCLIP checkpoint (``--teacher-weights``).
* :class:`GeneEmbeddingTeacher` (``--teacher path2space`` / generic) — projects
  expression through a **pretrained gene-embedding matrix** ``cell = X_norm @ W``
  (scGPT / Geneformer / Gene2vec, or a Path2Space-derived gene-program matrix). Needs
  a ``--gene-embedding`` ``.npz``/``.npy``.
* :class:`ProxyTeacher` — data-derived stand-in when no FM asset is available, so
  training always runs; a documented approximation, not the real FM.

Register additional real teachers with :func:`register_teacher`.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy.spatial import cKDTree

# builder(cfg, stack, gene_names) -> teacher
TEACHER_REGISTRY: Dict[str, Callable] = {}


def register_teacher(name: str, builder: Callable) -> None:
    TEACHER_REGISTRY[name] = builder


def _spatial_smooth(expr, xy, k=12):
    n = expr.shape[0]
    if n < k + 1:
        return expr.copy()
    _, nn = cKDTree(xy).query(xy, k=min(k + 1, n))
    return expr[nn].mean(axis=1)


def _kmeans_domains(E, n_domains):
    try:
        from sklearn.cluster import KMeans
        k = min(n_domains, max(2, E.shape[0] // 20))
        return KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(
            np.ascontiguousarray(E, np.float32)).astype(int)
    except Exception:
        return np.zeros(E.shape[0], dtype=int)


def _teacher_device(cfg):
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# Gene-symbol resolution (OmiCLIP keys on gene symbols)                         #
# --------------------------------------------------------------------------- #
def _looks_ensembl(genes, frac=0.3):
    hits = sum(str(g).upper().startswith(("ENSG", "ENSMUSG", "ENST", "ENSMUST",
                                          "ENSRNOG", "ENSMMUG")) for g in genes)
    return hits >= max(1, int(frac * len(genes)))


def _load_symbol_map(path):
    """Load an id->symbol map from a .npz (ids/symbols) or a 2-column TSV/CSV."""
    import os
    if not path or not os.path.exists(path):
        return {}
    try:
        if path.endswith(".npz"):
            d = np.load(path, allow_pickle=True)
            ids = [str(x) for x in d["ids"]]
            syms = [str(x) for x in d["symbols"]]
            return dict(zip(ids, syms))
        m = {}
        for line in open(path):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) >= 2:
                m[parts[0].strip()] = parts[1].strip()
        return m
    except Exception:
        return {}


def resolve_symbols(gene_names, gene_symbols=None, symbol_map_path=None):
    """Resolve gene SYMBOLS for the OmiCLIP sentence, in priority order:

    1. explicit ``gene_symbols`` from the caller (e.g. an ``adata.var`` symbol column);
    2. an id->symbol mapping file (``symbol_map_path``) applied to the panel names;
    3. the panel names as-is (with a warning if they look like Ensembl IDs, since the
       OmiCLIP vocabulary is symbol-based and those tokens won't match).

    Strips Ensembl version suffixes (``ENSG…​.3`` -> ``ENSG…``) before mapping.
    """
    names = [str(g) for g in gene_names]
    if gene_symbols is not None:
        syms = [str(s) if s not in (None, "", "nan") else names[i]
                for i, s in enumerate(gene_symbols)]
        return syms
    m = _load_symbol_map(symbol_map_path)
    if m:
        def _map(g):
            return m.get(g) or m.get(g.split(".")[0]) or g
        syms = [_map(g) for g in names]
        n_mapped = sum(1 for g, s in zip(names, syms) if s != g)
        print(f"[spatialcpav11] symbol map applied: {n_mapped}/{len(names)} genes translated.")
        return syms
    if _looks_ensembl(names):
        print("[spatialcpav11] WARNING: panel looks like Ensembl IDs but no symbol map "
              "was given — OmiCLIP's text tower keys on gene symbols, so distillation "
              "will be weak. Pass an adata.var symbol column or --teacher-symbol-map.")
    return names


# --------------------------------------------------------------------------- #
# Real teacher 1: OmiCLIP (gene-sentence -> CLIP/CoCa text tower)               #
# --------------------------------------------------------------------------- #
class OmiCLIPTeacher:
    """Real OmiCLIP teacher via its expression/text tower (no images needed).

    Each spot -> a sentence of its top-N expressed gene symbols -> OmiCLIP's CLIP
    (CoCa) text encoder -> a normalized embedding. Mirrors how OmiCLIP / Loki embed
    spatial transcriptomics.
    """

    def __init__(self, cfg, gene_names, gene_symbols=None):
        import torch
        import open_clip
        self.cfg = cfg
        syms = resolve_symbols(gene_names, gene_symbols, cfg.symbol_map_path)
        self.genes = [str(g).upper() for g in syms]
        self.dev = _teacher_device(cfg)
        model, _, _ = open_clip.create_model_and_transforms(
            cfg.model_arch, pretrained=cfg.weights_path)
        self.model = model.to(self.dev).eval()
        self.tokenizer = open_clip.get_tokenizer(cfg.model_arch)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._torch = torch

    def _sentences(self, expr):
        X = np.asarray(expr, np.float64)
        n = X.shape[0]
        top = self.cfg.top_genes
        order = np.argsort(-X, axis=1)[:, :top]
        return [" ".join(self.genes[j] for j in order[i] if X[i, j] > 0) or "cell"
                for i in range(n)]

    def embed(self, expr, xy):
        torch = self._torch
        sents = self._sentences(expr)
        out = []
        with torch.no_grad():
            for i in range(0, len(sents), self.cfg.encode_batch):
                toks = self.tokenizer(sents[i:i + self.cfg.encode_batch]).to(self.dev)
                feat = self.model.encode_text(toks)
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
                out.append(feat.float().cpu().numpy())
        return np.ascontiguousarray(np.concatenate(out, 0), np.float32)

    def domains(self, expr, xy):
        return _kmeans_domains(self.embed(expr, xy), self.cfg.n_pseudo_domains)


def _build_omiclip(cfg, stack, gene_names, gene_symbols=None):
    return OmiCLIPTeacher(cfg, gene_names, gene_symbols=gene_symbols)


# --------------------------------------------------------------------------- #
# Real teacher 2: pretrained gene-embedding projection (path2space / scGPT / …) #
# --------------------------------------------------------------------------- #
class GeneEmbeddingTeacher:
    """Project expression through a pretrained gene-embedding matrix: cell = X_norm @ W.

    ``W`` is a ``(G_panel, d)`` matrix aligned to the panel (scGPT / Geneformer /
    Gene2vec token embeddings, or a Path2Space-derived gene-program matrix).
    """

    def __init__(self, cfg, gene_names, W, fit_stack=None):
        self.cfg = cfg
        self.W = np.asarray(W, np.float32)
        if fit_stack is not None:
            X = fit_stack.union_expression()
            self._mean = X.mean(0, keepdims=True)
            self._std = X.std(0, keepdims=True); self._std[self._std == 0] = 1.0
        else:
            self._mean, self._std = 0.0, 1.0

    def embed(self, expr, xy):
        Z = ((np.asarray(expr, np.float64) - self._mean) / self._std) @ self.W
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        return np.ascontiguousarray(Z, np.float32)

    def domains(self, expr, xy):
        return _kmeans_domains(self.embed(expr, xy), self.cfg.n_pseudo_domains)


def _load_gene_matrix(path, gene_names):
    import os
    if not path or not os.path.exists(path):
        return None
    genes = [str(g) for g in gene_names]
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        vocab = [str(g) for g in d["genes"]]; emb = np.asarray(d["embedding"], np.float32)
    else:
        emb = np.asarray(np.load(path), np.float32)
        if emb.shape[0] != len(genes):
            return None
        vocab = genes
    lut = {g: i for i, g in enumerate(vocab)}
    W = np.zeros((len(genes), emb.shape[1]), np.float32)
    for j, g in enumerate(genes):
        i = lut.get(g)
        if i is not None:
            W[j] = emb[i]
    return W if np.any(W) else None


def _build_gene_embedding(cfg, stack, gene_names, gene_symbols=None):
    W = _load_gene_matrix(cfg.gene_embedding_path or cfg.weights_path, gene_names)
    if W is None:
        raise RuntimeError("no usable gene-embedding matrix (set --gene-embedding)")
    return GeneEmbeddingTeacher(cfg, gene_names, W, fit_stack=stack)


register_teacher("omiclip", _build_omiclip)
register_teacher("path2space", _build_gene_embedding)
register_teacher("gene_embedding", _build_gene_embedding)


# --------------------------------------------------------------------------- #
# Data-derived stand-in                                                        #
# --------------------------------------------------------------------------- #
class ProxyTeacher:
    """Data-derived stand-in for OmiCLIP / Path2Space (documented approximation)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, stack):
        F = np.concatenate([_spatial_smooth(np.asarray(s.expression, np.float64),
                                            np.asarray(s.coords_xy, np.float64))
                            for s in stack.slices], axis=0)
        self._mean = F.mean(0, keepdims=True); self._std = F.std(0, keepdims=True)
        self._std[self._std == 0] = 1.0
        Fz = (F - self._mean) / self._std
        d = int(min(self.cfg.embed_dim, min(Fz.shape) - 1)) if min(Fz.shape) > 1 else 1
        _, _, Vt = np.linalg.svd(Fz, full_matrices=False)
        self._comps = Vt[:max(d, 1)]
        return self

    def embed(self, expr, xy):
        Fs = _spatial_smooth(np.asarray(expr, np.float64), np.asarray(xy, np.float64))
        E = ((Fs - self._mean) / self._std) @ self._comps.T
        return np.ascontiguousarray(E, np.float32)

    def domains(self, expr, xy):
        return _kmeans_domains(self.embed(expr, xy), self.cfg.n_pseudo_domains)


# --------------------------------------------------------------------------- #
# Dispatch                                                                     #
# --------------------------------------------------------------------------- #
def build_teacher(cfg, stack, gene_names, gene_symbols=None):
    """Instantiate the teacher (real FM if requested/available, else the proxy)."""
    want = None
    if cfg.kind in ("omiclip", "path2space", "gene_embedding"):
        want = cfg.kind
    elif cfg.kind == "auto" and (cfg.weights_path or cfg.gene_embedding_path):
        want = cfg.name if cfg.name in TEACHER_REGISTRY else "omiclip"
    if want is not None and want in TEACHER_REGISTRY:
        try:
            t = TEACHER_REGISTRY[want](cfg, stack, gene_names, gene_symbols)
            print(f"[spatialcpav11] teacher: real {want} "
                  f"(weights={cfg.weights_path or cfg.gene_embedding_path}).")
            return t
        except Exception as e:
            if cfg.kind != "auto":
                print(f"[spatialcpav11] real teacher '{want}' failed ({e}).")
                if cfg.kind == want:  # explicitly requested -> do not silently downgrade
                    raise
            else:
                print(f"[spatialcpav11] real teacher '{want}' unavailable ({e}); using proxy.")
    print("[spatialcpav11] teacher: data-derived proxy stand-in "
          "(supply OmiCLIP weights via --teacher omiclip --teacher-weights for the real FM).")
    return ProxyTeacher(cfg).fit(stack)
