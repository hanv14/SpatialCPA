# T02 — Text-grounded gene and context embeddings

**Goal.** Make the model panel-agnostic: genes enter through their *text*, not a fixed vocabulary
index. This is what enables generating genes never measured in the target tissue, and transferring
a trained model across panels.

**Files:** `spatialcpav25_gen/data/text.py`, `spatialcpav25_gen/model/embeddings.py`, `tests/test_text.py`

**Dependencies:** T01.

---

## 1. Descriptor construction — `spatialcpav25_gen/data/text.py`

```python
def gene_descriptor(symbol: str, meta: GeneMeta | None) -> str
def celltype_descriptor(name: str, ontology: dict | None) -> str
def region_descriptor(name: str, hierarchy: list[str] | None) -> str
```

Gene descriptor format (keep it stable — embeddings are cached against a hash of this string):

```
"{symbol}. {full_name}. {summary}. Aliases: {a1}, {a2}."
```

`GeneMeta` comes from a local table shipped in `resources/gene_meta.parquet` (columns:
`symbol, full_name, summary, aliases, ensembl_id`). Provide
`build_gene_meta(symbols) -> pd.DataFrame` that assembles it from **mygene.info** if network is
available, and **degrades gracefully to `symbol` alone** when a gene is unknown. Cache the table;
do not hit the network at training time.

Cell type descriptor: Cell Ontology label + definition when resolvable, else the raw label.
Region descriptor: name plus its ancestor path, e.g.
`"Primary somatosensory area, layer 4. Part of: Isocortex, Cerebral cortex, Brain."` — the ancestor
path matters because it lets unseen regions inherit meaning from their parents.

## 2. Encoder + cache

```python
class TextEncoder:
    def __init__(self, cfg: Config): ...
    def encode(self, texts: list[str]) -> np.ndarray:   # (T, 768) float32, L2-normalised
```

- Model: `cfg.text_model` (MedCPT-Query-Encoder) via `transformers`, **frozen**, eval mode, no grad.
- Mean-pool the last hidden state over non-pad tokens, then L2-normalise. (Do not use the CLS token
  unless you verify it is trained for pooling in this checkpoint — MedCPT's query encoder is
  trained with CLS pooling; **check the model card and use whichever the checkpoint specifies, and
  write the choice in a comment.**)
- Cache to `cfg.text_cache_dir` keyed by `sha256(model_name + descriptor)`. On cache hit, the model
  is never loaded — this keeps CPU-only test runs fast.
- Batch size 32, max length 512, truncation on.

## 3. Learned embedding module — `spatialcpav25_gen/model/embeddings.py`

```python
class TextGroundedEmbedding(nn.Module):
    """Text prior + free residual, with a distillation head for unseen entities.

    Args:
        text_vecs: (V, 768) frozen MedCPT vectors for the V known entities.
        out_dim:   embedding width.
    Returns from forward: (V_query, out_dim)
    """
    def __init__(self, text_vecs: Tensor, out_dim: int, cfg: Config): ...

    def forward(self, idx: Tensor) -> Tensor: ...
        # e = LayerNorm(W @ t[idx] + gamma * r[idx])

    def forward_zero_shot(self, text_vecs_new: Tensor, use_distill: bool = True) -> Tensor: ...
        # r_hat = self.distill(t_new) if use_distill else 0
        # e = LayerNorm(W @ t_new + gamma * r_hat)

    def distillation_loss(self) -> Tensor: ...
        # || distill(t) - stopgrad(r) ||^2  over known entities
```

Components:
- `W`: `Linear(768, out_dim, bias=False)`.
- `r`: `nn.Embedding(V, out_dim)`, initialised to zeros.
- `gamma`: a buffer annealed 0 → 1 over `cfg.residual_gate_warmup_frac` of training. Expose
  `set_progress(frac: float)`; the trainer calls it each epoch. Zeros init + annealing means early
  training is forced to use the text signal, so `W` learns something real before the free residual
  can shortcut it. **This ordering matters — without it `r` absorbs everything and the text channel
  is decorative.**
- `distill`: `MLP(768 -> 256 -> out_dim)` trained by `distillation_loss` with the residual detached.

Instantiate three of these: genes (`gene_emb_dim`), cell types and regions (`ctx_emb_dim`).

## 4. Diagnostics (needed for the paper's ablation A3)

```python
def text_embedding_diagnostics(emb: TextGroundedEmbedding, expr: np.ndarray) -> dict
```

Given the training expression matrix, report:
- Spearman correlation between pairwise cosine similarity in text space and pairwise co-expression
  correlation, over gene pairs. **This number tells you up front whether the text channel carries
  usable signal.** Expect something modest (0.1–0.3). If it is ≈ 0 on real data, say so in
  `PROGRESS.md` — it does not sink the project (the zero-shot capability is the contribution) but
  it changes how the paper frames the claim.
- Norm ratio `||gamma*r|| / ||W t||` at end of training — how much the model relied on the free
  residual.
- kNN purity: for each gene, fraction of its 10 nearest text neighbours that are in the same
  co-expression module (Leiden on the gene-gene correlation graph).

## Acceptance tests

- `test_descriptor_stability` — descriptors are deterministic; changing a gene's alias order does
  not change the string (sort aliases).
- `test_cache_hit_avoids_model_load` — monkeypatch the transformer to raise; second call must
  succeed from cache.
- `test_missing_gene_meta_degrades` — an unknown symbol yields `"{symbol}."` and encodes fine.
- `test_zero_shot_shapes` — `forward_zero_shot` on 10 unseen vectors returns `(10, out_dim)`.
- `test_gamma_anneal` — `set_progress(0)` → gamma 0 and output equals `LayerNorm(W t)` exactly;
  `set_progress(1)` → gamma 1.
- `test_distillation_reduces_error` — train 200 steps on random targets; distillation MSE drops
  ≥ 50%.
- `test_offline` — with no network and no cache, `build_gene_meta` returns symbol-only rows and
  does not raise.

## Definition of done

Tests green on CPU without network. `PROGRESS.md` records the text/co-expression Spearman on the
synthetic fixture (expected ≈ 0 there, since synthetic gene names are arbitrary — that is fine and
expected; the number matters on real data).

## Do NOT

- Do not fine-tune MedCPT. It is frozen; the adaptation lives in `W` and `r`.
- Do not use gene *expression* to build descriptors. That would leak and destroy the zero-shot claim.
- Do not fetch from the network during training or tests.
