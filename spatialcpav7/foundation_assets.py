"""
Foundation-model + ligand-receptor assets for SpatialCPA-v7.

Two kinds of external biological prior enter v7 through this module:

1. **Gene-embedding matrices** from single-cell / spatial foundation models
   (scGPT, Geneformer, Gene2vec) or an H&E-paired gene-program matrix, converted
   to the ``.npz`` format the embedder expects (arrays ``genes`` and
   ``embedding``). These are consumed by ``--embedding fm_gene``/``concat``.
   Weights are NOT bundled — point at your own download.

2. **A curated ligand-receptor pair list** (:func:`default_lr_pairs`) used by the
   3D communication term to score whether co-locating cell types can actually
   signal to each other. A compact, panel-agnostic set of canonical pairs
   (matched case-insensitively, so it works for mouse symbols too); the term
   auto-disables when too few pairs overlap the panel.

A no-download, data-derived gene-program embedding
(:func:`build_coexpression_embedding`) is provided so the method always runs.

The morphology hook (:func:`register_he_embedder`) is the extension point for a
paired-H&E foundation model (UNI / CONCH): given per-cell morphology features it
returns a cell-state embedding registered under a name usable by ``--embedding``.
Absent paired images (as in this benchmark) it is simply unused — the external
knowledge then enters through the gene-embedding path instead.
"""

from __future__ import annotations

import argparse
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Curated ligand-receptor pairs (canonical; case-insensitive matching)         #
# --------------------------------------------------------------------------- #
# A compact, broadly-useful subset spanning immune, vascular, neural and
# developmental signalling. Not exhaustive — the LR term is a gentle prior and
# auto-disables when < lr_min_pairs of these appear in the panel.
_LR_PAIRS: List[Tuple[str, str]] = [
    # immune / cytokine
    ("Ccl19", "Ccr7"), ("Ccl21", "Ccr7"), ("Cxcl12", "Cxcr4"),
    ("Cxcl13", "Cxcr5"), ("Cxcl9", "Cxcr3"), ("Cxcl10", "Cxcr3"),
    ("Il7", "Il7r"), ("Il2", "Il2ra"), ("Ifng", "Ifngr1"),
    ("Tnf", "Tnfrsf1a"), ("Tgfb1", "Tgfbr1"), ("Csf1", "Csf1r"),
    ("Cd40lg", "Cd40"), ("Sell", "Selplg"),
    # vascular / angiogenesis
    ("Vegfa", "Kdr"), ("Vegfa", "Flt1"), ("Pdgfb", "Pdgfrb"),
    ("Pdgfa", "Pdgfra"), ("Angpt1", "Tek"), ("Dll4", "Notch1"),
    ("Jag1", "Notch1"), ("Efnb2", "Ephb4"),
    # neural / synaptic / guidance
    ("Bdnf", "Ntrk2"), ("Ngf", "Ntrk1"), ("Nrg1", "Erbb4"),
    ("Sema3a", "Nrp1"), ("Slit2", "Robo1"), ("Nlgn1", "Nrxn1"),
    ("Reln", "Vldlr"), ("Wnt3", "Fzd1"), ("Wnt5a", "Fzd5"),
    ("Shh", "Ptch1"), ("Bmp4", "Bmpr1a"), ("Gdf10", "Bmpr1b"),
    # adhesion / ECM
    ("Nrxn1", "Nlgn1"), ("Efna5", "Epha4"), ("Fn1", "Itgb1"),
    ("Lamb1", "Itgb1"), ("App", "Sorl1"), ("Psap", "Gpr37"),
]


def default_lr_pairs() -> List[Tuple[str, str]]:
    """Return the built-in curated ligand-receptor pair list."""
    return list(_LR_PAIRS)


def load_lr_pairs(path: Optional[str]) -> List[Tuple[str, str]]:
    """Load LR pairs from a 2-column (ligand, receptor) file, else the built-in set."""
    if not path:
        return default_lr_pairs()
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) >= 2 and not parts[0].lower().startswith("ligand"):
                pairs.append((parts[0], parts[1]))
    return pairs or default_lr_pairs()


# --------------------------------------------------------------------------- #
# Morphology (H&E) foundation-model hook                                        #
# --------------------------------------------------------------------------- #
def register_he_embedder(name: str, builder: Callable) -> None:
    """Register a paired-H&E morphology embedder (UNI/CONCH) under ``name``.

    Thin pass-through to :func:`spatialcpav7.embedding.register_embedder`; kept
    here so the morphology extension point lives with the other FM assets. The
    builder receives ``(train_expression, gene_names, cfg)`` and returns a
    callable ``expression -> (N, d)``; a real implementation would additionally
    consume paired per-cell image features from ``cfg``.
    """
    from .embedding import register_embedder
    register_embedder(name, builder)


# --------------------------------------------------------------------------- #
# Gene-embedding writer / converters                                           #
# --------------------------------------------------------------------------- #
def save_npz(genes: Sequence[str], embedding: np.ndarray, out_path: str) -> None:
    genes = np.array([str(g) for g in genes])
    embedding = np.ascontiguousarray(embedding, dtype=np.float32)
    if embedding.shape[0] != len(genes):
        raise ValueError(f"genes ({len(genes)}) and embedding rows "
                         f"({embedding.shape[0]}) must match")
    np.savez(out_path, genes=genes, embedding=embedding)
    print(f"[foundation_assets] wrote {embedding.shape[0]} genes x "
          f"{embedding.shape[1]} dims -> {out_path}")


def _is_number(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


def _rows_at_modal_width(parsed):
    from collections import Counter
    if not parsed:
        raise ValueError("no gene vectors parsed")
    width = Counter(len(v) for _, v in parsed).most_common(1)[0][0]
    genes, rows = [], []
    for g, v in parsed:
        if len(v) == width:
            genes.append(g); rows.append(v)
    return genes, np.array(rows, dtype=np.float32)


def from_gene2vec(path: str) -> Tuple[List[str], np.ndarray]:
    parsed = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or _is_number(parts[0]):
                continue
            try:
                vec = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            parsed.append((parts[0], vec))
    return _rows_at_modal_width(parsed)


def from_delimited(path: str, gene_col: int = 0, sep: Optional[str] = None,
                   skip_header: bool = False) -> Tuple[List[str], np.ndarray]:
    genes, rows = [], []
    with open(path) as f:
        lines = f.readlines()
    if skip_header:
        lines = lines[1:]
    for line in lines:
        parts = line.rstrip("\n").split(sep) if sep else line.split()
        if len(parts) < 2:
            continue
        vals = [p for i, p in enumerate(parts) if i != gene_col]
        try:
            vec = [float(x) for x in vals]
        except ValueError:
            continue
        genes.append(parts[gene_col]); rows.append(vec)
    width = min(len(r) for r in rows)
    emb = np.array([r[:width] for r in rows], dtype=np.float32)
    return genes, emb


def _load_torch_matrix(weights_path: str, key: Optional[str],
                       fallback_keys: Sequence[str]) -> np.ndarray:
    import torch
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    keys = [key] if key else []
    keys += [k for k in fallback_keys if k not in keys]
    for k in keys:
        if k and k in state:
            return np.asarray(state[k].float().cpu().numpy(), dtype=np.float32)
    best = None
    for k, v in state.items():
        if hasattr(v, "ndim") and v.ndim == 2:
            if best is None or v.shape[0] > best[1].shape[0]:
                best = (k, v)
    if best is None:
        raise KeyError(f"no 2-D embedding tensor found in {weights_path}")
    print(f"[foundation_assets] using inferred embedding key '{best[0]}' "
          f"{tuple(best[1].shape)}")
    return np.asarray(best[1].float().cpu().numpy(), dtype=np.float32)


def from_scgpt(vocab_path: str, weights_path: str,
               key: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
    import json
    with open(vocab_path) as f:
        vocab = json.load(f)
    emb = _load_torch_matrix(weights_path, key,
                             ["encoder.embedding.weight",
                              "gene_encoder.embedding.weight",
                              "gene_encoder.weight"])
    id_to_gene = {int(i): g for g, i in vocab.items()}
    genes, rows = [], []
    for i in range(emb.shape[0]):
        g = id_to_gene.get(i)
        if g is not None and not str(g).startswith("<"):
            genes.append(str(g)); rows.append(emb[i])
    return genes, np.array(rows, dtype=np.float32)


def from_geneformer(token_dict_path: str, weights_path: str,
                    symbol_map: Optional[dict] = None,
                    key: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
    import pickle
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    emb = _load_torch_matrix(weights_path, key,
                             ["bert.embeddings.word_embeddings.weight",
                              "embeddings.word_embeddings.weight"])
    genes, rows = [], []
    for g, i in token_dict.items():
        if not isinstance(i, int) or i >= emb.shape[0]:
            continue
        name = symbol_map.get(g, g) if symbol_map else g
        if str(name).startswith("<"):
            continue
        genes.append(str(name)); rows.append(emb[i])
    return genes, np.array(rows, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Data-derived gene embedding (no external asset)                              #
# --------------------------------------------------------------------------- #
def build_coexpression_embedding(expression: np.ndarray, gene_names: Sequence[str],
                                 dim: int = 32) -> Tuple[List[str], np.ndarray]:
    """SVD of the gene-gene correlation matrix — a data-derived gene-program prior."""
    X = np.asarray(expression, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = Xc.std(axis=0, keepdims=True); s[s == 0] = 1.0
    Xz = Xc / s
    C = np.corrcoef(Xz, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    d = int(min(dim, C.shape[0] - 1)) if C.shape[0] > 1 else 1
    U, S, _ = np.linalg.svd(C)
    W = U[:, :d] * np.sqrt(S[:d])[None, :]
    return list(map(str, gene_names)), W.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Convert a pretrained gene embedding into SpatialCPA-v7 .npz")
    ap.add_argument("--source", required=True,
                    choices=["gene2vec", "generic", "scgpt", "geneformer"])
    ap.add_argument("--input")
    ap.add_argument("--vocab")
    ap.add_argument("--token-dict")
    ap.add_argument("--weights")
    ap.add_argument("--key", default=None)
    ap.add_argument("--gene-col", type=int, default=0)
    ap.add_argument("--sep", default=None)
    ap.add_argument("--skip-header", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.source == "gene2vec":
        genes, emb = from_gene2vec(args.input)
    elif args.source == "generic":
        genes, emb = from_delimited(args.input, args.gene_col, args.sep, args.skip_header)
    elif args.source == "scgpt":
        genes, emb = from_scgpt(args.vocab, args.weights, args.key)
    else:
        genes, emb = from_geneformer(args.token_dict, args.weights, key=args.key)
    save_npz(genes, emb, args.output)


if __name__ == "__main__":
    main()
