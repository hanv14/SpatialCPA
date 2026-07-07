"""
Foundation-model gene-embedding assets for SpatialCPA-v6.

v6 injects an external biological prior by projecting expression through a
pretrained *gene embedding* ``W`` (``cell = X_norm @ W``; see
:mod:`spatialcpav6.embedding`, method ``"fm_gene"``/``"concat"``). This module
converts the gene-embedding tables shipped by common spatial-omics / single-cell
foundation models into the ``.npz`` format v6 expects — arrays ``genes`` (str,
G0) and ``embedding`` (G0, d) — and provides a no-download, data-derived
alternative.

Supported pretrained sources (weights are NOT bundled — point at your own
download):

* **scGPT** — ``vocab.json`` (gene -> token id) + a checkpoint holding the gene
  token-embedding matrix (default key ``encoder.embedding.weight``; overridable).
  https://github.com/bowang-lab/scGPT
* **Geneformer** — a token dictionary (pickle, gene -> id) + a checkpoint holding
  ``bert.embeddings.word_embeddings.weight``. Geneformer keys on Ensembl IDs, so
  pass a symbol map if your panel uses symbols.
  https://huggingface.co/ctheodoris/Geneformer
* **gene2vec** — a plain ``GENE v1 v2 ... vd`` text table.
  https://github.com/jingcheng-du/Gene2vec
* **generic** — any delimited table with a gene column + numeric embedding columns.

Data-derived (no external asset):

* :func:`build_coexpression_embedding` — SVD of the training-slice gene-gene
  correlation matrix. A legitimate gene-program prior computed from the data
  itself (leakage-safe when built on the training slices), usable as a stronger
  denoising space than raw PCA for the ``transfer`` expression mode.

CLI::

    python -m spatialcpav6.foundation_assets --source gene2vec \\
        --input gene2vec_dim_200.txt --output gene_emb.npz
    python -m spatialcpav6.foundation_assets --source scgpt \\
        --vocab vocab.json --weights best_model.pt --output gene_emb.npz

Then run the benchmark with ``--embedding fm_gene --fm-gene-embedding gene_emb.npz``.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Writer                                                                       #
# --------------------------------------------------------------------------- #
def save_npz(genes: Sequence[str], embedding: np.ndarray, out_path: str) -> None:
    """Write a gene-embedding table in v6's ``.npz`` format."""
    genes = np.array([str(g) for g in genes])
    embedding = np.ascontiguousarray(embedding, dtype=np.float32)
    if embedding.shape[0] != len(genes):
        raise ValueError(f"genes ({len(genes)}) and embedding rows "
                         f"({embedding.shape[0]}) must match")
    np.savez(out_path, genes=genes, embedding=embedding)
    print(f"[foundation_assets] wrote {embedding.shape[0]} genes x {embedding.shape[1]} "
          f"dims -> {out_path}")


# --------------------------------------------------------------------------- #
# Converters                                                                   #
# --------------------------------------------------------------------------- #
def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _rows_at_modal_width(parsed):
    """Keep rows whose vector length equals the modal length (drops headers)."""
    from collections import Counter
    if not parsed:
        raise ValueError("no gene vectors parsed")
    width = Counter(len(v) for _, v in parsed).most_common(1)[0][0]
    genes, rows = [], []
    for g, v in parsed:
        if len(v) == width:
            genes.append(g)
            rows.append(v)
    return genes, np.array(rows, dtype=np.float32)


def from_gene2vec(path: str) -> Tuple[List[str], np.ndarray]:
    """Parse a gene2vec ``GENE v1 v2 ... vd`` whitespace table.

    Tolerates an optional ``n_genes dim`` header and any stray rows by keeping
    only rows whose vector length matches the modal (embedding) width, and by
    dropping rows whose gene token is purely numeric (header artifacts).
    """
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
    """Parse a generic delimited table (gene column + numeric columns)."""
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
        genes.append(parts[gene_col])
        rows.append(vec)
    width = min(len(r) for r in rows)
    emb = np.array([r[:width] for r in rows], dtype=np.float32)
    return genes, emb


def _load_torch_matrix(weights_path: str, key: Optional[str],
                       fallback_keys: Sequence[str]) -> np.ndarray:
    """Load a 2-D embedding matrix from a torch checkpoint (state dict)."""
    import torch  # local import: only needed for scGPT/Geneformer
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    keys = [key] if key else []
    keys += [k for k in fallback_keys if k not in keys]
    for k in keys:
        if k and k in state:
            return np.asarray(state[k].float().cpu().numpy(), dtype=np.float32)
    # Last resort: the largest 2-D float tensor (typically the token embedding).
    best = None
    for k, v in state.items():
        if hasattr(v, "ndim") and v.ndim == 2:
            if best is None or v.shape[0] > best[1].shape[0]:
                best = (k, v)
    if best is None:
        raise KeyError(f"no 2-D embedding tensor found in {weights_path}; "
                       f"tried keys {keys}")
    print(f"[foundation_assets] using inferred embedding key '{best[0]}' "
          f"{tuple(best[1].shape)}")
    return np.asarray(best[1].float().cpu().numpy(), dtype=np.float32)


def from_scgpt(vocab_path: str, weights_path: str,
               key: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
    """scGPT: ``vocab.json`` (gene -> id) + gene token-embedding matrix."""
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
        if g is not None and not str(g).startswith("<"):   # drop special tokens
            genes.append(str(g))
            rows.append(emb[i])
    return genes, np.array(rows, dtype=np.float32)


def from_geneformer(token_dict_path: str, weights_path: str,
                    symbol_map: Optional[dict] = None,
                    key: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
    """Geneformer: token dict (pickle gene->id) + word-embedding matrix.

    ``symbol_map`` optionally maps Geneformer's Ensembl IDs to the gene symbols
    used by your panel.
    """
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
        genes.append(str(name))
        rows.append(emb[i])
    return genes, np.array(rows, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Data-derived gene embedding (no external asset)                              #
# --------------------------------------------------------------------------- #
def build_coexpression_embedding(expression: np.ndarray, gene_names: Sequence[str],
                                 dim: int = 32) -> Tuple[List[str], np.ndarray]:
    """SVD of the gene-gene correlation matrix — a data-derived gene-program prior.

    Genes that co-vary across cells get similar embeddings, so projecting a cell
    through it (``cell = X_norm @ W``) yields a program-level cell-state vector.
    Build it on the *training* slices only to stay leakage-safe.
    """
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


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Convert a pretrained gene embedding into SpatialCPA-v6 .npz")
    ap.add_argument("--source", required=True,
                    choices=["gene2vec", "generic", "scgpt", "geneformer"])
    ap.add_argument("--input", help="table path (gene2vec / generic)")
    ap.add_argument("--vocab", help="scGPT vocab.json")
    ap.add_argument("--token-dict", help="Geneformer token dictionary pickle")
    ap.add_argument("--weights", help="scGPT / Geneformer checkpoint")
    ap.add_argument("--key", default=None, help="explicit embedding tensor key")
    ap.add_argument("--gene-col", type=int, default=0, help="generic: gene column index")
    ap.add_argument("--sep", default=None, help="generic: delimiter (default whitespace)")
    ap.add_argument("--skip-header", action="store_true", help="generic: drop first line")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.source == "gene2vec":
        genes, emb = from_gene2vec(args.input)
    elif args.source == "generic":
        genes, emb = from_delimited(args.input, args.gene_col, args.sep, args.skip_header)
    elif args.source == "scgpt":
        genes, emb = from_scgpt(args.vocab, args.weights, args.key)
    else:  # geneformer
        genes, emb = from_geneformer(args.token_dict, args.weights, key=args.key)
    save_npz(genes, emb, args.output)


if __name__ == "__main__":
    main()
