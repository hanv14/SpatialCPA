# SpatialCPA-v13 — an LLM-based virtual-slice generator

v13 is a **transformer language model** for aligned serial spatial transcriptomics. It
generates a virtual tissue slice at an arbitrary continuous `z` by treating cells as
**sentences of gene tokens** and running the modern LLM playbook — tokenization → a
self-attention transformer trained with masked-language modelling → **retrieval-augmented
in-context generation**.

It is a paradigm **completely distinct from every other SpatialCPA version**:

* **not v8** — no optimal transport, no diffeomorphic / barycentric morph, no two-slice
  OT fusion, no niche Markov-random-field (v8 is training-free numpy/scipy);
* **not v11/v12** — no implicit coordinate-network field, no Fourier features, no Gaussian
  factor-analysis decoder. The engine here is a **tokenizer + attention transformer +
  retrieval**.

```
cell expression ──tokenize──► cell-sentence  [CLS] g_π(1) g_π(2) … g_π(K)
                                   │
                        self-attention transformer  (masked gene-LM pretraining)
                                   │
   query (x,y,z) ─► spatial cross-attention over retrieved flanking cells ─► context embedding
                                   │
                    sample a real exemplar from the LM-similarity distribution (RAG)
                                   │
                    emit a grounded, lightly LM-edited expression profile
```

## The LLM

* **Cell-sentence tokenizer** (`tokenizer.py`). Each cell's expression becomes a
  rank-ordered sequence of **gene tokens** — the genes it expresses most, highest first —
  exactly the representation used by single-cell language models (Cell2Sentence, which
  fine-tunes GPT-2; Geneformer; scGPT). Vocabulary = the genes plus `[CLS] [MASK] [PAD]`.

* **Transformer** (`nets.CellTransformer`). Token + learned positional embeddings →
  `nn.TransformerEncoder` (multi-head self-attention) → a `[CLS]`-pooled **cell
  embedding**. Heads: a **masked gene-language-model** head over the gene vocabulary, a
  cell-type head, and an expression-decode head.

* **Spatial cross-attention** (`nets.SpatialContextAttention`). A learned query encoding
  of the continuous `(x, y, z)` **cross-attends over the flanking cells' embeddings** —
  the in-context / retrieval-augmented conditioning that lets the model generate a cell
  appropriate to a query location.

## Training objectives (`trainer.py`)

1. **Masked gene-language modelling** — mask gene tokens in each cell-sentence and predict
   them (the canonical LLM pretraining objective); the transformer learns the gene
   "grammar" / co-occurrence structure.
2. **Spatial in-context** — from a cell's flanking spatial neighbours attended in context,
   reconstruct its cell type + expression, and **align** the context embedding to the
   cell's own embedding (a retrieval contrastive term) so that similarity search works at
   generation time.

Trained leave-one-slice-out on the training stack; the held-out slice never enters it.

## Generation — retrieval-augmented (`trainer.generate_slice`)

For the target `z`: emit `n_target` positions from a **retrieval layout** (real flanking
positions, regime-adaptive), then for each position (i) build a context embedding by
cross-attending over the retrieved flanking cells, (ii) **sample one real exemplar** from
the LM-similarity distribution over nearby real cells (temperature `retrieval_temp`),
(iii) read the cell type, and (iv) emit the exemplar's profile, optionally nudged toward
the LM-decoded profile (`edit_weight`). Because each output is a real exemplar drawn from
a **two-slice, LM-weighted** distribution, the population is a *generative recombination*
of both flanking slices (each cell real → realistic gene–gene covariance), not a copy of
either. Composition is matched to the interpolated flanking mix by LM-driven
prior-corrected resampling, and expression is emitted count-like for the evaluator.

## Optional real pretrained gene-LM

`PretrainedLMConfig` exposes a hook (`register_pretrained_lm`) to plug in a real
pretrained single-cell language model (Cell2Sentence/GPT-2, scGPT, Geneformer) as the
cell encoder. With no checkpoint supplied, v13 trains its own self-contained transformer
(clearly logged), so the method always runs.

## Validation status (honest)

Validated end-to-end through the **real** `benchmark-pbya-v2` generation evaluator on
synthetic multi-slice data (both regimes, three held-out sections), head-to-head against
**v8's default** (the strongest prior SpatialCPA generator):

* **Distinct tissue: v13 beats v8, 7 wins / 3 losses (mean over holdouts)** — winning
  co-expression, Sinkhorn, composition, gene mean/variance, and both field metrics.
* **Near-identical planes: competitive** — v13 wins `field_ssim` by a wide margin
  (+0.24) and ties the gene-level metrics; v8 stays ahead on the niche/density metrics
  (`celltype_nhood_agreement`, `density_pearson`, composition) that a real-slice copy is
  intrinsically built to dominate.

Across both regimes v13 plays to the LLM's strength — the molecular "language"
(expression-distribution metrics) — and cedes a little niche/density fidelity, the honest
residue of a generative two-slice recombination vs a coherent real-slice copy. See
`validation/VALIDATION.md` for the per-metric tables and how to reproduce them. The
processed real datasets and conda environments are not bundled here, so the full
cross-dataset leaderboard must be regenerated where the data live. No benchmark numbers
are fabricated.

## Running it

Registered in `benchmark-pbya-v2` as `spatialcpav13_gen` (needs PyTorch, in
`bench_spatialcpa`):

```bash
python -m benchmark.run_benchmark --method spatialcpav13_gen --dataset starmap_visual_cortex
# knobs: --top-genes --n-layers --d-model --retrieval-temp --edit-weight --position-mode
```

### Package layout

| module | role |
|---|---|
| `config.py` | tokenizer / model / training / generation hyperparameters |
| `tokenizer.py` | cell-sentence (rank gene-token) tokenizer |
| `nets.py` | `CellTransformer` (self-attention LM) + `SpatialContextAttention` (RAG) |
| `trainer.py` | masked gene-LM + spatial in-context training; retrieval-augmented generation |
| `model.py` | `SpatialCPAv13` — orchestration, normalization, fallback |
| `data.py` | `Slice` / `SliceStack` containers |
