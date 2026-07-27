"""Phase 1.1-1.3 — QC, the continuous type embedding ``e_c``, and the hierarchy.

This is the *widened bottleneck* of the whole method.  Everything downstream
(the structure field in Phase 2, the expression conditioning in Phase 3) sees
types only through ``e_c``; a bare integer label carries no similarity structure
and is what makes a generator collapse to type-means.

The proposal's recipe is followed literally:

* **1.2.0** branch on what the labels *are* (numbered clusters / text names /
  mixed) before doing anything else;
* **1.2.A/B** identify markers per type with **signed effect sizes**, and for
  text labels compose ``"{name}; markers: {...}"``;
* **1.2.1** encode that description **order-invariantly** (it is a *set*) into a
  continuous vector;
* **1.2.2** use a **data-driven backbone** (batch-corrected pseudobulk -> PCA),
  keep the encoder vector as an **external prior**, report their agreement
  (Mantel + kNN overlap), and blend them **per type** with a sample-size
  shrinkage weight so rare types borrow from the prior and well-sampled types do
  not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import HierarchyConfig, QCConfig, TypeConfig

_NUMBERED = re.compile(r"^\s*(?:leiden|cluster|clust|louvain|type|class|group|c)?"
                       r"[\s_\-]*\d+\s*$", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*")


# ── Phase 1.1 ─────────────────────────────────────────────────────────────────

def section_qc(stack, cfg: QCConfig) -> Tuple[object, Dict]:
    """Per-section QC.  Returns the (possibly filtered) stack and a report.

    Deliberately per-section, so batch structure stays visible rather than being
    averaged away (Phase 1.1).
    """
    from .data import SliceStack

    report = {"sections": [], "n_dropped": 0, "low_quality_sections": []}
    kept = []
    for s in stack.slices:
        totals = s.expression.sum(axis=1)
        keep = totals > cfg.min_total_expression
        n_drop = int((~keep).sum())
        report["n_dropped"] += n_drop
        s2 = s.subset(keep) if n_drop else s
        det = (s2.expression > 0).sum(axis=1) if s2.n_spots else np.zeros(0)
        report["sections"].append({
            "section": s.section_id,
            "z": s.z,
            "n_cells": s2.n_spots,
            "median_total": float(np.median(totals[keep])) if keep.any() else 0.0,
            "median_genes_detected": float(np.median(det)) if s2.n_spots else 0.0,
            "n_dropped": n_drop,
        })
        if s2.n_spots < cfg.min_cells_per_section:
            report["low_quality_sections"].append(s.section_id)
        kept.append(s2)
    return SliceStack(kept), report


# ── Phase 1.2.0 — what *are* the labels? ──────────────────────────────────────

def classify_label_vocabulary(names: List[str]) -> str:
    """Return ``"numbered"``, ``"text"`` or ``"mixed"`` (Phase 1.2.0)."""
    if not names:
        return "numbered"
    numbered = [bool(_NUMBERED.match(str(n))) for n in names]
    if all(numbered):
        return "numbered"
    if not any(numbered):
        return "text"
    return "mixed"


# ── Phase 1.2.A/B — markers with signed effect size ──────────────────────────

@dataclass
class MarkerSet:
    """The marker signature of one type: gene indices + *signed* effect sizes."""

    gene_idx: np.ndarray
    effect: np.ndarray          # Cohen's d, signed (direction is kept)

    def as_description(self, gene_names: List[str], label: Optional[str] = None) -> str:
        """The Phase 1.2.B textual description (order is deterministic: by gene
        name, not by rank, so the description is a *set*, not a ranking)."""
        order = np.argsort([gene_names[i] for i in self.gene_idx])
        parts = []
        for o in order:
            g = gene_names[self.gene_idx[o]]
            parts.append(f"{'+' if self.effect[o] >= 0 else '-'}{g}")
        body = "markers: " + ", ".join(parts)
        return f"{label}; {body}" if label else body


def identify_markers(X: np.ndarray, types: np.ndarray, n_types: int,
                     cfg: TypeConfig) -> List[MarkerSet]:
    """Signed effect size of each gene in each type vs. the rest (Phase 1.2.A.1).

    Cohen's d with a pooled standard deviation — an effect size with a direction,
    not a rank position, exactly as the proposal insists.
    """
    X = np.asarray(X, dtype=np.float64)
    n, G = X.shape
    out: List[MarkerSet] = []
    k = min(cfg.top_k_markers, G)
    for c in range(n_types):
        m = types == c
        n_c = int(m.sum())
        if n_c < cfg.min_cells_for_markers or n_c == n:
            out.append(MarkerSet(np.zeros(0, dtype=int), np.zeros(0)))
            continue
        a, b = X[m], X[~m]
        ma, mb = a.mean(axis=0), b.mean(axis=0)
        va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
        pooled = np.sqrt(((n_c - 1) * va + (b.shape[0] - 1) * vb)
                         / max(n - 2, 1)) + 1e-8
        d = (ma - mb) / pooled
        idx = np.argsort(-np.abs(d))[:k]
        out.append(MarkerSet(idx.astype(int), d[idx]))
    return out


# ── Phase 1.2.1 — order-invariant encoder -> e_prior ─────────────────────────

def _hashed_symbol_features(symbols: List[str], dim: int, seed: int = 0) -> np.ndarray:
    """A deterministic, data-independent embedding of gene/label *symbols*.

    Character 3-grams (plus the whole token) are hashed into ``dim`` buckets with
    signed hashing, then L2-normalized.  Symbols from the same family (``Gad1`` /
    ``Gad2``, ``Sst`` / ``Sstr2``) land close; unrelated ones are near-orthogonal.

    This is the **external prior** of Phase 1.2.2: it is computed from names
    only, so it knows nothing about this specimen's expression — which is the
    whole point of using it to shore up under-sampled types.  Supplying
    ``TypeConfig.fm_gene_embedding_path`` swaps in a pretrained gene-aware
    encoder instead; nothing downstream changes.
    """
    F = np.zeros((len(symbols), dim), dtype=np.float64)
    for i, sym in enumerate(symbols):
        s = str(sym).lower()
        grams = [s] + [s[j:j + 3] for j in range(max(len(s) - 2, 0))]
        for g in grams:
            h = 0
            for ch in g:                     # FNV-1a style, seeded, deterministic
                h = ((h ^ ord(ch)) * 16777619 + seed) & 0xFFFFFFFF
            F[i, h % dim] += 1.0 if (h >> 16) & 1 else -1.0
        nrm = np.linalg.norm(F[i])
        if nrm > 0:
            F[i] /= nrm
    return F


def load_gene_encoder(gene_names: List[str], cfg: TypeConfig) -> Tuple[np.ndarray, str]:
    """Per-gene encoder vectors ``S`` (G, d_sym) and the source that produced them."""
    path = cfg.fm_gene_embedding_path
    if path:
        try:
            if str(path).endswith(".npz"):
                z = np.load(path, allow_pickle=True)
                genes = [str(g) for g in z["genes"]]
                emb = np.asarray(z["embedding"], dtype=np.float64)
                lut = {g: i for i, g in enumerate(genes)}
                S = np.zeros((len(gene_names), emb.shape[1]))
                hit = 0
                for i, g in enumerate(gene_names):
                    j = lut.get(g, lut.get(g.upper(), lut.get(g.capitalize())))
                    if j is not None:
                        S[i] = emb[j]
                        hit += 1
                if hit >= max(3, 0.2 * len(gene_names)):
                    return S, f"pretrained:{path} ({hit}/{len(gene_names)} genes)"
            else:
                S = np.asarray(np.load(path), dtype=np.float64)
                if S.shape[0] == len(gene_names):
                    return S, f"pretrained:{path} (panel-aligned)"
        except Exception:
            pass
    return (_hashed_symbol_features(gene_names, cfg.symbol_embedding_dim),
            "symbol-hash encoder (no pretrained embedding supplied)")


def encode_types_prior(markers: List[MarkerSet], gene_S: np.ndarray,
                       type_names: List[str], label_kind: str,
                       cfg: TypeConfig) -> np.ndarray:
    """Phase 1.2.1 — encode each type's description to a continuous vector.

    The marker component is pooled over the marker *set* with signed,
    effect-size weights, so the result cannot depend on ranking noise.  For text
    labels the name's own tokens are pooled into the same space and added, which
    is the "name + markers" composition of Phase 1.2.B.
    """
    d = gene_S.shape[1]
    E = np.zeros((len(markers), d))
    for c, ms in enumerate(markers):
        if ms.gene_idx.size:
            w = np.sign(ms.effect) * np.abs(ms.effect)
            v = (w[:, None] * gene_S[ms.gene_idx]).sum(axis=0)
            n = np.linalg.norm(v)
            E[c] = v / n if n > 0 else v
    if label_kind == "text":
        toks = [" ".join(_TOKEN.findall(str(t))) or str(t) for t in type_names]
        T = _hashed_symbol_features(toks, d, seed=7)
        E = E + T                      # name prior + data-grounded marker prior
        nrm = np.linalg.norm(E, axis=1, keepdims=True)
        E = np.divide(E, nrm, out=np.zeros_like(E), where=nrm > 0)
    return E


# ── Phase 1.2.2 — data backbone, agreement diagnostic, shrinkage blend ───────

def _pca(M: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Deterministic PCA (SVD with a fixed sign convention)."""
    M = np.asarray(M, dtype=np.float64)
    if M.shape[0] < 2:
        return np.zeros((M.shape[0], k))
    Mc = M - M.mean(axis=0, keepdims=True)
    k = int(min(k, Mc.shape[0] - 1, Mc.shape[1]))
    if k < 1:
        return np.zeros((M.shape[0], 1))
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    sign = np.sign(Vt[:k, np.argmax(np.abs(Vt[:k]), axis=1)].diagonal())
    sign[sign == 0] = 1.0
    Z = (U[:, :k] * S[:k]) * sign
    return Z


def pseudobulk_backbone(X: np.ndarray, types: np.ndarray, sections: np.ndarray,
                        n_types: int, cfg: TypeConfig) -> np.ndarray:
    """Phase 1.2.2.1 — the data-driven backbone ``e_c^data``.

    Pseudobulk per (type, section); the section mean profile is subtracted before
    pooling across sections when ``batch_correct_pseudobulk`` is set.  That is the
    caveat the proposal flags: a technical marker would otherwise be inherited by
    *both* embeddings and the agreement check would happily confirm it.
    """
    X = np.asarray(X, dtype=np.float64)
    secs = np.unique(sections)
    acc = np.zeros((n_types, X.shape[1]))
    cnt = np.zeros(n_types)
    for sec in secs:
        sm = sections == sec
        if sm.sum() < 2:
            continue
        offset = X[sm].mean(axis=0) if cfg.batch_correct_pseudobulk else 0.0
        for c in range(n_types):
            m = sm & (types == c)
            if m.sum() == 0:
                continue
            acc[c] += X[m].mean(axis=0) - offset
            cnt[c] += 1
    cnt[cnt == 0] = 1.0
    P = acc / cnt[:, None]
    return _pca(P, cfg.n_components)


def _pairwise(M: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=-1)
    return d


def embedding_agreement(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    """Phase 1.2.2.3 — Mantel correlation + kNN overlap between two embeddings."""
    C = A.shape[0]
    if C < 4:
        return {"mantel_r": float("nan"), "knn_overlap": float("nan")}
    da, db = _pairwise(A), _pairwise(B)
    iu = np.triu_indices(C, k=1)
    a, b = da[iu], db[iu]
    mantel = (float(np.corrcoef(a, b)[0, 1])
              if a.std() > 0 and b.std() > 0 else float("nan"))
    k = int(min(5, C - 1))
    ka = np.argsort(da, axis=1)[:, 1:k + 1]
    kb = np.argsort(db, axis=1)[:, 1:k + 1]
    ov = np.mean([len(set(ka[i]) & set(kb[i])) / k for i in range(C)])
    return {"mantel_r": mantel, "knn_overlap": float(ov)}


def _procrustes_align(src: np.ndarray, dst: np.ndarray,
                      w: np.ndarray) -> np.ndarray:
    """Rotate ``src`` into ``dst``'s frame (weighted orthogonal Procrustes).

    Two independently fitted embeddings live in arbitrary bases; blending them
    without a common frame is meaningless.  The weights are the same shrinkage
    weights used for the blend, so the frame is set by the types we trust.
    """
    if src.shape[1] != dst.shape[1]:
        d = min(src.shape[1], dst.shape[1])
        src, dst = src[:, :d], dst[:, :d]
    W = w[:, None]
    A = (src * W)
    B = (dst * W)
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    R = U @ Vt
    return src @ R


def _unit_scale(M: np.ndarray) -> np.ndarray:
    nrm = np.sqrt((M ** 2).sum(axis=1)).mean()
    return M / nrm if nrm > 0 else M


@dataclass
class TypeRepresentation:
    """The Phase 1.2/1.3 deliverable: a continuous, hierarchical type space."""

    embedding: np.ndarray                  # (C, d) — e_c
    class_of: np.ndarray                   # (C,) int
    subclass_of: np.ndarray                # (C,) int
    class_embedding: np.ndarray            # (n_class, d)
    subclass_embedding: np.ndarray         # (n_subclass, d)
    names: List[str]
    label_kind: str
    weights: np.ndarray                    # (C,) shrinkage weights w_c
    descriptions: List[str] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)

    @property
    def n_types(self) -> int:
        return int(self.embedding.shape[0])

    def hierarchical(self, type_idx: np.ndarray) -> np.ndarray:
        """Concatenated ``[e_cluster, e_subclass, e_class]`` per cell (Phase 3.2)."""
        t = np.clip(np.asarray(type_idx, dtype=int), 0, self.n_types - 1)
        return np.concatenate([
            self.embedding[t],
            self.subclass_embedding[self.subclass_of[t]],
            self.class_embedding[self.class_of[t]],
        ], axis=1)

    @property
    def hier_dim(self) -> int:
        return int(self.embedding.shape[1] * 3)


def _ward_labels(E: np.ndarray, n: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    C = E.shape[0]
    n = int(np.clip(n, 1, C))
    if n <= 1:
        return np.zeros(C, dtype=int)
    if n >= C:
        return np.arange(C, dtype=int)
    Z = linkage(E, method="ward")
    lab = fcluster(Z, t=n, criterion="maxclust") - 1
    return lab.astype(int)


def build_type_representation(X: np.ndarray, types: np.ndarray,
                              sections: np.ndarray, type_names: List[str],
                              gene_names: List[str], cfg: TypeConfig,
                              hcfg: HierarchyConfig) -> TypeRepresentation:
    """Run Phase 1.2 end-to-end and add the Phase 1.3 hierarchy."""
    C = len(type_names)
    label_kind = classify_label_vocabulary(type_names)
    # 1.2.0 "mixed" -> harmonize by treating everything as clusters, then
    # re-annotate from markers.  Never embed a mix of ID conventions directly.
    effective_kind = "numbered" if label_kind == "mixed" else label_kind

    markers = identify_markers(X, types, C, cfg)
    gene_S, encoder_src = load_gene_encoder(gene_names, cfg)
    prior_raw = encode_types_prior(markers, gene_S, type_names, effective_kind, cfg)
    e_prior = _pca(prior_raw, cfg.n_components)
    e_data = pseudobulk_backbone(X, types, sections, C, cfg)

    d = min(e_prior.shape[1], e_data.shape[1])
    e_prior, e_data = e_prior[:, :d], e_data[:, :d]
    e_prior, e_data = _unit_scale(e_prior), _unit_scale(e_data)

    agree = embedding_agreement(e_data, e_prior)

    n_c = np.array([(types == c).sum() for c in range(C)], dtype=np.float64)
    w = n_c / (n_c + cfg.shrinkage_kappa)          # Phase 1.2.2.4 shrinkage
    e_prior_aligned = _procrustes_align(e_prior, e_data, w)
    E = w[:, None] * e_data + (1.0 - w)[:, None] * e_prior_aligned
    E = _unit_scale(E)

    n_class = hcfg.n_classes or int(np.ceil(np.sqrt(max(C, 1))))
    n_sub = hcfg.n_subclasses or int(min(C, 2 * n_class))
    class_of = _ward_labels(E, n_class)
    subclass_of = _ward_labels(E, n_sub)
    cls_emb = np.stack([E[class_of == k].mean(axis=0) if (class_of == k).any()
                        else np.zeros(E.shape[1]) for k in range(class_of.max() + 1)])
    sub_emb = np.stack([E[subclass_of == k].mean(axis=0) if (subclass_of == k).any()
                        else np.zeros(E.shape[1]) for k in range(subclass_of.max() + 1)])

    descs = [markers[c].as_description(
        gene_names, type_names[c] if effective_kind == "text" else None)
        for c in range(C)]

    return TypeRepresentation(
        embedding=E, class_of=class_of, subclass_of=subclass_of,
        class_embedding=cls_emb, subclass_embedding=sub_emb,
        names=list(type_names), label_kind=label_kind, weights=w,
        descriptions=descs,
        diagnostics={
            "label_kind": label_kind,
            "effective_label_kind": effective_kind,
            "encoder": encoder_src,
            "n_types": C,
            "cells_per_type_median": float(np.median(n_c)),
            "shrinkage_weight_median": float(np.median(w)),
            "n_rare_types_prior_dominated": int((w < 0.5).sum()),
            "prior_vs_data_mantel_r": agree["mantel_r"],
            "prior_vs_data_knn_overlap": agree["knn_overlap"],
            "n_classes": int(class_of.max() + 1),
            "n_subclasses": int(subclass_of.max() + 1),
        },
    )


# ── Phase 1.3 — joint typing when the input has no labels ────────────────────

def joint_cell_typing(X: np.ndarray, n_types: int = 12, seed: int = 0
                      ) -> Tuple[np.ndarray, List[str]]:
    """Type *jointly across every section at once* (Phase 1.3).

    Only used when the input carries no cell-type column; the benchmark harness
    normally supplies a training-only label vocabulary.  Clustering is run on the
    pooled matrix so the vocabulary is shared across the stack by construction.
    """
    from sklearn.cluster import KMeans

    Z = _pca(X, 30, seed=seed)
    k = int(min(n_types, max(2, Z.shape[0] // 25)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    return km.labels_.astype(np.int64), [f"joint_{i}" for i in range(k)]
