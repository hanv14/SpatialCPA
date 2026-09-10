"""Where does the spatial structure go? — the generation chain, stage by stage.

T10's pilot found the generated field carrying 15-22% of STARmap's per-gene Moran's I, with
calibration and the layout head both ruled out by direct experiment (``reports/pilot.md`` §6).
That leaves a whole-pipeline number and no location. This script measures autocorrelation at
each link of the chain ``generate_section`` actually runs, so the loss can be attributed to the
prior, the flow, or the decoder rather than to "the expression path".

The chain, mirroring ``infer/generate.py::_expression`` under ``expr_mode="zinb-flow"``::

    h0 = model.prior_latent(xyz, seed)          # the 3-D GRF, at the generated positions
    h  = model.flow.sample(h0, cond, ode_steps) # the latent after the flow
    counts = decode(h) then sample              # the emitted counts

with two references measured on the *real* held-out section:

    real counts                                  # what the tissue has
    h1 = model.encoder(real counts, ...)         # what the encoder makes of it

Reading it:

* ``h0`` low        -> the prior is not delivering structure at these positions at all.
* ``h0`` high, ``h`` low  -> the flow destroys it.
* ``h`` matches ``h1``, counts low -> the decoder destroys it.

Every stage uses the SAME estimator — bench3's row-standardised kNN Moran's I, at
``Config.metric_knn_k`` — so the numbers are comparable down the chain. Latents are compared
per dimension and counts per gene; both are summarised by the median over channels, because a
mean over channels is dominated by whichever few carry the most variance.

**What the comparison is over.** By default every gene and every generated cell, which is what
made the existing artifacts hard to read across datasets: Moran's I on a kNN graph rises with
density, and the runs in ``reports/`` emitted 11 168 / 48 343 / 267 567 cells against a ground
truth of 4 187. Three flags fix the comparison instead of the reader having to:

``--match-density``
    subsample the generated cells to the real section's count, under an explicit seed, *before*
    the kNN graph is built. This deliberately changes the graph — that is the point.
``--top-k-by real --top-k 32``
    restrict the gene-space stages to the ``k`` genes with the highest Moran's I **in the real
    section**. Selecting the panel by the model's own I scores the model on genes it chose;
    ``--top-k-by model`` exists to measure exactly how much that flatters it, and is never the
    number to quote.
``--text-emb-mode`` / ``--expr-pca-dim``
    the shipped arm is ``medcpt`` at the clamp rule's PCA dim; this script was pinned at
    ``lookup`` / 16 with no override, so every artifact under ``reports/chain_*`` is ablation
    A3's ``lookup`` arm at the pilot's PCA dim, whatever it is labelled.

The panel restricts **gene-space stages only** — decoded ``mu``, sampled counts, their calibrated
twins, and the real reference. Stages 1 and 2 are latents with ``Config.latent_dim`` channels that
are not genes and cannot be restricted to one; the report says so where it reports them.

All four default to the previous behaviour, so a run made before they existed produces the
same numbers. The report is not byte-identical to an older one: it gained a block saying which
arm it is, and the JSON sidecar is now an object with the stage list under ``stages`` rather
than a bare list — a stage table that does not say what it measured is what §4.2a-iii is about.

Usage::

    python scripts/t10_chain_diagnostic.py --steps 1200
    python scripts/t10_chain_diagnostic.py --steps 2400 --out reports/chain_2400.md
    python scripts/t10_chain_diagnostic.py --steps 2400 --decoder-mu-link exp \
        --text-emb-mode medcpt --expr-pca-dim 28 --match-density \
        --top-k-by real --top-k 32 --out reports/chain_shipped_tier1.md
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import load_volume
from spatialcpav25_gen.data.schema import TrainingVolume
from spatialcpav25_gen.infer.generate import plane_at_z
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import fit_repulsion
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads

SEED = 1
# Resolved per run by ``_bench3_paths`` (``--bench3`` and friends). They stay as module-level
# names because ``t10_rescore_saved`` imports the two loaders below and sets them once; on a
# machine where bench3 is this repo's own copy the defaults are what the pilot used.
INPUT = "benchmark-pbya-v3/results/_inputs/starmap_visual_cortex/paper_2_4_6/train_registered.h5ad"
GROUND_TRUTH = "benchmark-pbya-v3/data/processed/starmap_visual_cortex/data.h5ad"


def morans_i(xy: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Per-column Moran's I on a row-standardised kNN graph. ``(N, 2)``, ``(N, C)`` -> ``(C,)``.

    Reimplemented here rather than imported from bench3 because this runs on latents, which
    never leave the package — but it is the same estimator: row-standardised kNN weights,
    self excluded. ``tests/conftest.py::morans_i`` is the reference it matches.
    """
    n = xy.shape[0]
    k = min(int(k), n - 1)
    idx = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]
    x = np.asarray(values, dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    denom = (xc**2).sum(axis=0)
    lagged = xc[idx].mean(axis=1)
    numer = (xc * lagged).sum(axis=0)
    out = np.full(x.shape[1], np.nan)
    ok = denom > 0
    out[ok] = numer[ok] / denom[ok]
    return out


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Per-column average ranks, as the scoreboard does before every spatial metric."""
    from scipy.stats import rankdata

    return np.column_stack([rankdata(col, method="average") for col in np.asarray(x).T])


def summarise(
    name: str, xy: np.ndarray, values: np.ndarray, k: int, *, primary: str = "raw"
) -> dict[str, float]:
    """Median / IQR of per-channel Moran's I **under both transforms**, plus the channel count.

    ``values`` is passed **untransformed**; this function ranks it itself. That is the whole point:
    the chain used to rank its count stages at the call site and leave its mean-field and latent
    stages raw, so ``I(4p) = +0.8358`` sat above ``I(3) = +0.7920`` on tier-1 — impossible for a
    field and an independent draw from it, and the symptom of a ratio taken across two transforms
    (``reports/ceiling_review.md`` §2).

    ``primary`` names which of the two ``median_I`` carries, so each stage keeps the figure its
    existing artifacts recorded. ``median_I_raw`` and ``median_I_rank`` are **always both present**,
    and a ratio between two stages must take both sides from the same column.

    Rank-normalising a heavy-tailed field raises its Moran's I by removing the leverage of a few
    extreme cells, which is why the raw column is the smaller one for ``mu`` and why the mixed
    ratios were overstated.
    """
    if primary not in ("raw", "rank"):
        raise ValueError(f"summarise: primary must be 'raw' or 'rank', got {primary!r}")
    out: dict[str, float] = {"stage": name, "transform": primary}
    per: dict[str, np.ndarray] = {}
    for label, array in (("raw", values), ("rank", rank_normalize(values))):
        i = morans_i(xy, array, k)
        finite = i[np.isfinite(i)]
        per[label] = i
        out[f"median_I_{label}"] = float(np.median(finite)) if finite.size else float("nan")
        if label == primary:
            out["median_I"] = out[f"median_I_{label}"]
            out["p25"] = float(np.percentile(finite, 25)) if finite.size else float("nan")
            out["p75"] = float(np.percentile(finite, 75)) if finite.size else float("nan")
            out["n_channels"] = int(finite.size)
    out["per_gene_I_rank"] = per["rank"]
    return out


def morans_i_ranked_blocked(
    xy: np.ndarray, values: np.ndarray, k: int, block: int = 64
) -> np.ndarray:
    """Per-gene Moran's I of ``rank_normalize(values)``, in gene blocks. ``(N, G)`` -> ``(G,)``.

    Identical to ``morans_i(xy, rank_normalize(values), k)`` and asserted so in ``--self-check``;
    the blocking exists only to bound memory. ``morans_i`` forms ``xc[idx]``, which is
    ``(N, k, G)`` — on ``deep_starmap``'s 29 544 cells and 1017 genes that is **24 GB**, so the
    whole-panel call the agreement statistic needs cannot be made directly.

    Ranking is done per block too, so no ``(N, G)`` float64 copy of the ranks exists either.
    """
    x = np.asarray(values)
    out = np.empty(x.shape[1], dtype=np.float64)
    for start in range(0, x.shape[1], int(block)):
        stop = min(start + int(block), x.shape[1])
        out[start:stop] = morans_i(xy, rank_normalize(x[:, start:stop]), k)
    return out


def _json_scalar(obj: object) -> object:
    """Serialise a numpy scalar for the sidecar, and REFUSE anything else.

    ``json.dumps`` has no handler for ``np.float64`` or ``np.bool_``, so one uncast statistic
    anywhere in the sidecar raises ``TypeError`` on the run's very last line -- after the fit, the
    ablation and every measurement have been paid for. A blanket ``default=str`` would avoid that
    by silently writing ``"array([...])"`` into a results file, which is worse. This converts the
    numpy scalars and lets everything else raise, so a genuine bug still surfaces (Convention 6).
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()
    raise TypeError(
        f"the report sidecar cannot serialise {type(obj).__name__}: {obj!r:.80}. Cast it at the "
        "point it is computed rather than widening this handler -- an array in a results file is "
        "a bug that a string conversion would hide."
    )


def knn_index(xy: np.ndarray, k: int) -> np.ndarray:
    """The row-standardised graph's neighbour index. ``(N, 2)`` -> ``(N, k)`` int.

    Split out of :func:`morans_i` so a loop over many draws at the **same** cells builds the tree
    once. The 20-seed permutation null would otherwise rebuild it 320 times on deep's 29 544 cells.
    """
    n = xy.shape[0]
    k = min(int(k), n - 1)
    return np.asarray(cKDTree(xy).query(xy, k=k + 1)[1][:, 1:])


def morans_i_from_index(idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per-column Moran's I given a precomputed neighbour index. ``(N, C)`` -> ``(C,)``.

    The body of :func:`morans_i` after the tree query; ``--self-check`` asserts the two agree
    exactly, so this is an optimisation and never a second estimator.
    """
    x = np.asarray(values, dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    denom = (xc**2).sum(axis=0)
    numer = (xc * xc[idx].mean(axis=1)).sum(axis=0)
    out = np.full(x.shape[1], np.nan)
    ok = denom > 0
    out[ok] = numer[ok] / denom[ok]
    return out


def permutation_null_I(
    xy: np.ndarray, counts: np.ndarray, k: int, seeds: Sequence[int], block: int = 64
) -> np.ndarray:
    """Per-gene ranked Moran's I under ``n_seeds`` independent cell permutations. -> ``(S, G)``.

    ``reports/null_band_preregistration.md`` §2a. Two identities make this cheap enough to run at
    20 seeds on 29 544 x 1017:

    * the tree is built once (:func:`knn_index`);
    * ranks are **permutation-equivariant per column**, so ``rank_normalize(counts[perm])`` equals
      ``rank_normalize(counts)[perm]`` and the expensive ranking is done once rather than per seed.

    Both are asserted in ``--self-check`` against the direct route.
    """
    idx = knn_index(xy, k)
    x = np.asarray(counts)
    out = np.empty((len(seeds), x.shape[1]), dtype=np.float64)
    for start in range(0, x.shape[1], int(block)):
        stop = min(start + int(block), x.shape[1])
        ranks = rank_normalize(x[:, start:stop])
        for si, seed in enumerate(seeds):
            perm = np.random.default_rng(int(seed)).permutation(x.shape[0])
            out[si, start:stop] = morans_i_from_index(idx, ranks[perm])
    return out


def null_centre_test(rs: Sequence[float]) -> dict:
    """Is the permutation arm CENTRED on zero? ``reports/null_band_preregistration.md`` §2a.

    The old check asked whether one draw was small, which conflates a broken construction with a
    wide null. A sound construction has ``mean_r`` at zero; the spread is a resolution limit and
    belongs in :func:`paired_gene_bootstrap`, not here. Fires on
    ``|mean_r| > 2 * sd_r / sqrt(n_seeds)``.
    """
    a = np.asarray([r for r in rs if np.isfinite(r)], dtype=np.float64)
    if a.size < 2:
        return {"n_seeds": int(a.size), "fired": False, "underpowered": True}
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    se = sd / np.sqrt(a.size)
    return {
        "n_seeds": int(a.size),
        "mean_r": mean,
        "sd_r": sd,
        "se_r": float(se),
        "p2.5": float(np.percentile(a, 2.5)),
        "p97.5": float(np.percentile(a, 97.5)),
        "fired": bool(abs(mean) > 2.0 * se),
        "underpowered": False,
    }


def paired_gene_bootstrap(
    vectors: dict[str, np.ndarray],
    ref: np.ndarray,
    pairs: Sequence[tuple[str, str]],
    *,
    n_boot: int = 2000,
    seed: int = 20260910,
) -> list[dict]:
    """Bootstrap interval for each rung DIFFERENCE, resampling genes. §2b of the null band.

    Rungs share the reference vector and the gene set, so ``r(a) - r(b)`` is far better determined
    than either absolute ``r``. The same resampled gene index is applied to every rung, which is
    what makes the difference paired; genes not finite in ``ref`` **and every rung** are dropped
    first, so one resampled index means the same thing everywhere.

    Carries the across-gene sampling error only. Draw-to-draw error is the arms' own across-seed
    spread and the two are reported side by side, never combined (§2b).
    """
    keys = list(vectors)
    stack = np.vstack([np.asarray(vectors[key], dtype=np.float64) for key in keys])
    r = np.asarray(ref, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(stack).all(axis=0)
    if ok.sum() < 8:
        return [{"pair": f"{a} - {b}", "n_genes": int(ok.sum()), "unusable": True} for a, b in pairs]
    stack, r = stack[:, ok], r[ok]
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, r.size, size=(int(n_boot), r.size))

    def _r(vec: np.ndarray, gt: np.ndarray) -> np.ndarray:
        vc = vec - vec.mean(axis=1, keepdims=True)
        gc = gt - gt.mean(axis=1, keepdims=True)
        num = (vc * gc).sum(axis=1)
        den = np.sqrt((vc**2).sum(axis=1) * (gc**2).sum(axis=1))
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    per = {key: _r(stack[i][draws], r[draws]) for i, key in enumerate(keys)}
    point = {key: float(np.corrcoef(stack[i], r)[0, 1]) for i, key in enumerate(keys)}
    out = []
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        d = per[a] - per[b]
        d = d[np.isfinite(d)]
        lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d.size else (
            float("nan"),
            float("nan"),
        )
        out.append(
            {
                "pair": f"{a} - {b}",
                "diff": point[a] - point[b],
                "lo": lo,
                "hi": hi,
                "n_genes": int(r.size),
                "distinguishable": bool(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0),
            }
        )
    return out


def gene_detection(counts: np.ndarray) -> np.ndarray:
    """Per-gene detection rate: the fraction of cells with a non-zero count. ``(N, G)`` -> ``(G,)``.

    R1-R3's control variable. A sparse gene's Moran's I is bounded low whatever its spatial
    structure, so the tissue's own per-gene ``I`` ordering may be substantially a *sparsity*
    ordering — and a model that matched only the sparsity would score on
    ``paper_morans_pearson`` without reproducing any spatial fidelity.
    """
    return (np.asarray(counts) > 0).mean(axis=0)


def partial_correlation(
    x: np.ndarray, y: np.ndarray, control: np.ndarray, degree: int = 2
) -> float:
    """Pearson correlation of ``x`` and ``y``, both residualised on a polynomial in ``control``.

    R3. The question: **given two genes the tissue detects equally often, does the model still
    order them correctly by spatial autocorrelation?** Controlling for detection removes the part
    of the agreement that a model could earn by matching sparsity alone.

    The basis is fixed at ``[1, c, c^2]`` in ``reports/ladder_preregistration.md`` §4 rather than
    chosen here, because "which control specification" is a degree of freedom that would otherwise
    be settled after seeing the answer. The pre-registration also requires the whole thing repeated
    with a second control and the reading dropped if they disagree.
    """
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(control))
    if ok.sum() < degree + 3:
        return float("nan")
    basis = np.vander(np.asarray(control, dtype=np.float64)[ok], degree + 1, increasing=True)
    out = []
    for vec in (np.asarray(x, dtype=np.float64)[ok], np.asarray(y, dtype=np.float64)[ok]):
        coef, *_ = np.linalg.lstsq(basis, vec, rcond=None)
        out.append(vec - basis @ coef)
    # A residual counts as constant when it is negligible *relative to the vector it came from*,
    # not when lstsq happens to return exact zeros. Where the control explains a side completely,
    # the residual is float noise with a non-zero std, and correlating it would report a number
    # made entirely of rounding error -- which reads as a finding. NaN is the honest answer.
    for vec, res in zip((x, y), out, strict=True):
        scale = float(np.asarray(vec, dtype=np.float64)[ok].std())
        if float(res.std()) <= 1e-10 * max(scale, 1.0):
            return float("nan")
    return float(np.corrcoef(out[0], out[1])[0, 1])


def morans_agreement(i_pred: np.ndarray, i_gt: np.ndarray) -> dict[str, float]:
    """bench3's ``_agreement`` on two per-gene Moran's I vectors — **the scored statistic**.

    Reconstructed from ``benchmark-pbya-v3/src/bench3/evaluate_paper.py`` line 102, read from the
    source rather than recalled (``specs/10`` §4.2l): NaN genes dropped **pairwise**, at least
    three survivors and non-zero variance on both sides, then Pearson and Spearman across genes,
    plus the mean absolute deviation and both medians.

    The caller must supply vectors computed the way the evaluator does — rank-normalised counts,
    ``k = SPATIAL_K = 10``, **each side on its own spatial graph** (the evaluator calls this
    "alignment-free"). ``Config.metric_knn_k`` is 10, and the chain measures the generated stages
    at the generated cells and ``REF real counts`` at the real ones, so both hold.

    ⚠️ **This is the statistic, not the score.** bench3 takes all shared genes and medians over
    held-out sections 2/4/6; a chain run has one section and possibly a panel. The number is
    comparable **between stages of one run** and is not comparable to a published `paper_*` value.
    ``reports/q15_preregistration.md`` §2.

    ``morans_mae`` is returned because the evaluator's own docstring names it as the statistic that
    **catches over-smoothing** — *"a blurred reconstruction inflates Moran's I ... while keeping the
    gene ranking intact"* — and the project's scored ``METRICS`` tuple does not include it.
    """
    from scipy.stats import pearsonr, spearmanr

    p = np.asarray(i_pred, dtype=np.float64)
    g = np.asarray(i_gt, dtype=np.float64)
    ok = ~(np.isnan(p) | np.isnan(g))
    out: dict[str, float] = {
        "n_genes": int(ok.sum()),
        "median_pred": float(np.median(p[ok])) if ok.any() else float("nan"),
        "median_gt": float(np.median(g[ok])) if ok.any() else float("nan"),
        "mae": float(np.mean(np.abs(p[ok] - g[ok]))) if ok.any() else float("nan"),
        "pearson": float("nan"),
        "spearman": float("nan"),
    }
    if ok.sum() >= 3 and p[ok].std() > 0 and g[ok].std() > 0:
        out["pearson"] = float(pearsonr(p[ok], g[ok])[0])
        out["spearman"] = float(spearmanr(p[ok], g[ok])[0])
    return out


def load_training_volume(cfg: Config, input_path: str | Path | None = None) -> TrainingVolume:
    """bench3's training-only input as a ``TrainingVolume`` (the wrapper's own path).

    The round-trip through a temporary file goes to a **per-process** directory: the three
    ``layout_mode`` arms are meant to run concurrently, and a shared temp name beside the input
    would have them overwrite and unlink each other's copy.
    """
    import anndata as ad

    src = Path(input_path or INPUT)
    adata = ad.read_h5ad(src)
    with tempfile.TemporaryDirectory(prefix="ctfflow_input_") as tmpdir:
        tmp = Path(tmpdir) / "train_registered.h5ad"
        adata.write_h5ad(tmp)
        vol = load_volume(tmp, cfg, flattened_sections=True)
    return TrainingVolume(
        specimen_id=vol.specimen_id,
        sections=vol.sections,
        gene_names=vol.gene_names,
        celltype_names=vol.celltype_names,
        region_names=vol.region_names,
        flattened_sections=vol.flattened_sections,
    )


def build_embeddings(
    cfg: Config, vol: TrainingVolume, *, live_text: bool = False, for_checkpoint: bool = False
):
    """The entity embeddings this run fits under. **There are three states, not two.**

    ``live_text=False`` (the default, and what every artifact under ``reports/chain_*`` was
    measured with) supplies **zero text vectors**, whatever ``cfg.text_emb_mode`` says. Written
    when the MedCPT encoder was unreachable off the campaign machine, it makes the text channel
    dead: ``W t = 0`` for every entity. That is a third state, and calling it "the lookup arm"
    is the mislabelling this flag exists to end — A3's ``lookup`` arm has the *vectors* and
    zeroes the *projection*, so ``distillation_loss``, which reads ``text_vecs`` directly, sees
    something in that arm and nothing here.

    ``live_text=True`` builds the real channel through ``model.build_entity_embeddings`` — the
    panel's own gene-metadata table, encoded by MedCPT. Both arms of the ``text_emb_mode`` gate
    then get the same vectors (the gate is applied inside ``TextGroundedEmbedding._text_channel``),
    which is what makes A3 a comparison in one variable. It **raises** if the table or the encoder
    is not reachable and does not fall back to zeros: a silent fall back to zeros is how every
    prior STARmap number became a ``lookup`` measurement under a ``medcpt`` label (Convention 6).

    ``for_checkpoint`` says the caller is about to ``load_state_dict`` over this module. The zero
    vectors then exist for microseconds and never reach the model — ``text_vecs`` is a registered
    buffer, so the checkpoint's own vectors replace them — and **saying "ZERO VECTORS" in that
    case publishes the opposite of what the run measured**. It did: the A1 runs' console said
    ``ZERO VECTORS`` while the report they wrote said ``live, medcpt, 28/28 non-zero``, one run
    asserting two arms (``specs/10`` §4.2k). The report line reads the tensor
    (:func:`describe_text_state`); this one now declines to name an arm it is about to lose.

    The default is kept so ``reports/chain_2400*.md`` still reproduces. It prints what it is,
    rather than leaving the reader to find this docstring.
    """
    if live_text:
        from _starmap_run import describe_text_channel
        from spatialcpav25_gen.model.embeddings import build_entity_embeddings

        described = describe_text_channel(cfg, vol)
        print(
            f"  text channel: LIVE (text_emb_mode={cfg.text_emb_mode!r}, "
            f"gene_meta_path={described.get('gene_meta_path')!r}, "
            f"bare symbols {described.get('n_bare')}/{described.get('n_genes')})",
            flush=True,
        )
        return build_entity_embeddings(cfg, vol.gene_names, vol.celltype_names, None)

    from spatialcpav25_gen.model.embeddings import EntityEmbeddings

    if for_checkpoint:
        print(
            "  text channel: constructed with zero vectors, to be REPLACED by the checkpoint's "
            "own text_vecs buffer — the arm this run measures is reported after the load",
            flush=True,
        )
    else:
        print(
            f"  text channel: ZERO VECTORS (cfg.text_emb_mode={cfg.text_emb_mode!r} is NOT "
            "exercised here; this is neither A3 arm — pass --text-emb-mode for a live channel)",
            flush=True,
        )
    zeros = torch.zeros((vol.n_genes, cfg.text_dim_in), dtype=torch.float32)
    types = torch.zeros((len(vol.celltype_names), cfg.text_dim_in), dtype=torch.float32)
    return EntityEmbeddings(cfg, zeros, types, None)


@dataclass(frozen=True)
class RealSection:
    """The held-out section as the tissue has it. Loaded once, before the fit.

    Attributes
    ----------
    section_id
        The value of ``obs[cfg.section_key]`` this section was selected by.
    xy
        ``(N, 2)`` float64, physical um.
    counts
        ``(N, G)`` float32 raw counts, columns in the ground truth's ``var_names`` order —
        which :func:`load_real_section` has checked equals the training volume's gene order,
        because every consumer here indexes ``model.embeddings.gene`` with a column number.
    z
        The section's own z (median over its cells), or NaN if the build carries no third
        spatial column.
    z_gap
        The smallest gap between distinct section z's in the build, or NaN. Used only to say
        whether ``--target-z`` names this section's plane or a different one.
    """

    section_id: str
    xy: np.ndarray
    counts: np.ndarray
    z: float
    z_gap: float


def load_real_section(
    section_id: str,
    ground_truth: str | Path | None = None,
    *,
    section_key: str = "section",
    coord_key: str = "spatial",
    expected_genes: list[str] | tuple[str, ...] | None = None,
) -> RealSection:
    """Read one held-out section out of the built dataset. Raises rather than returning empty.

    An unknown ``section_id`` used to produce an all-False mask and a zero-row reference that
    the chain then summarised as NaN, several hours after the fit that paid for it. It now
    fails here, before anything is trained, and the message lists the ids the file has
    (Convention 6). Same for a gene order that disagrees with the training volume's: every
    caller downstream indexes ``model.embeddings.gene`` by column number, so a disagreement is
    a silently wrong gene, not a warning.
    """
    import anndata as ad
    import scipy.sparse as sp

    path = Path(ground_truth or GROUND_TRUTH)
    gt = ad.read_h5ad(path)
    if section_key not in gt.obs:
        raise SystemExit(
            f"{path} has no obs[{section_key!r}]; it carries {sorted(gt.obs.columns)}. "
            "That key names the sections and is not guessable."
        )
    sections = gt.obs[section_key].values.astype(str)
    mask = sections == str(section_id)
    if not mask.any():
        raise SystemExit(
            f"{path} has no section {section_id!r} under obs[{section_key!r}]. "
            f"It carries {sorted(set(sections))}. Pass --section with one of those; the "
            "default is tier-1 STARmap's and does not transfer to another dataset."
        )
    if expected_genes is not None:
        got, want = list(map(str, gt.var_names)), list(map(str, expected_genes))
        if got != want:
            first = next(
                (i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b),
                min(len(got), len(want)),
            )
            raise SystemExit(
                f"gene order disagrees between the ground truth ({len(got)} genes) and the "
                f"training volume ({len(want)} genes); first difference at column {first}: "
                f"{got[first : first + 1]} vs {want[first : first + 1]}. Every stage here indexes "
                "the gene embedding by column number, so this would silently score the wrong "
                "genes."
            )

    spatial = np.asarray(gt.obsm[coord_key], dtype=np.float64)
    xy = spatial[mask, :2]
    counts = gt.X[mask]
    counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    counts = np.asarray(counts, dtype=np.float32)

    z, z_gap = float("nan"), float("nan")
    if spatial.shape[1] >= 3:
        z = float(np.median(spatial[mask, 2]))
        per_section = sorted(
            {float(np.median(spatial[sections == s, 2])) for s in set(sections.tolist())}
        )
        if len(per_section) >= 2:
            z_gap = float(min(np.diff(per_section)))
    return RealSection(section_id=str(section_id), xy=xy, counts=counts, z=z, z_gap=z_gap)


def top_k_panel(xy: np.ndarray, values: np.ndarray, k: int, top_k: int) -> np.ndarray:
    """Columns with the ``top_k`` highest Moran's I. ``(N, 2)``, ``(N, C)`` -> ``(top_k,)`` int64.

    Deterministic with no generator: ties break by column index, and the returned indices are
    sorted ascending so the panel is in the volume's own gene order rather than in rank order.
    Non-finite I (a constant column) sorts last and is only selected if fewer than ``top_k``
    columns have a finite one.
    """
    i = morans_i(xy, values, k)
    i = np.where(np.isfinite(i), i, -np.inf)
    order = np.argsort(-i, kind="stable")[: max(int(top_k), 0)]
    return np.sort(order).astype(np.int64)


def density_subsample(n_available: int, n_target: int, seed: int) -> np.ndarray:
    """Row indices of ``n_target`` cells drawn without replacement. ``(min(n_target, n),)``.

    Moran's I on a kNN graph is a function of the graph, and the graph is a function of density:
    at 48 343 generated cells the k nearest neighbours of a cell sit inside a radius the real
    section at 4 187 cells never reaches. Comparing the two without this is comparing two
    estimators. ``seed`` is explicit (Convention 3); there is no global RNG here.

    Upsampling is not attempted — the caller is told and proceeds with what it has.
    """
    if n_target >= n_available:
        return np.arange(int(n_available), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n_available), size=int(n_target), replace=False)).astype(np.int64)


def knn_mean_field(xy: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Row-stochastic kNN mean of ``values``, self included. ``(N, 2)``, ``(N, G)`` -> ``(N, G)``.

    A shrinkage estimate of the tissue's own mean field: ``E[y_i]`` approximated by the average
    over cell ``i`` and its ``k`` nearest neighbours, on the same graph the metric uses. Column
    means are preserved up to boundary effects, because the weight matrix is row-stochastic —
    which is what makes an oracle-``mu`` arm comparable to the real counts in *level* as well as
    in structure.

    Built as a sparse product rather than ``values[idx].mean(axis=1)``: that intermediate is
    ``(N, k+1, G)``, which on ``deep_starmap`` is 30 097 x 11 x 1017 float64 = 2.4 GB.

    Smoothing **creates** autocorrelation, so ``I(knn_mean_field(y))`` is an upper bound on the
    tissue's own ``I(mu)``. Every arm built on it inherits that, and reads as an upper bound.
    """
    from scipy.sparse import csr_matrix

    n = int(xy.shape[0])
    kk = min(int(k), n - 1)
    idx = cKDTree(xy).query(xy, k=kk + 1)[1]
    rows = np.repeat(np.arange(n), kk + 1)
    weights = np.full(rows.size, 1.0 / (kk + 1))
    w = csr_matrix((weights, (rows, idx.ravel())), shape=(n, n))
    return np.asarray(w @ np.asarray(values, dtype=np.float64))


def log_mu_sd_from_field(field: np.ndarray) -> np.ndarray:
    """``sd(log mu)`` implied by a mean field's own dispersion. ``(N, G)`` -> ``(G,)``.

    Uses the lognormal identity ``sd(log mu) = sqrt(log(1 + CV^2))`` — an approximation, labelled
    as one wherever it is reported — so no pseudocount is needed and genes that are zero in most
    cells contribute without a choice of ``eps`` deciding the answer.

    Applied to ``mu_oracle`` this is a **lower** bound on the tissue's spread, because the kNN mean
    that produced the field shrinks variance.
    """
    x = np.asarray(field, dtype=np.float64)
    mean = x.mean(axis=0)
    cv2 = np.where(mean > 0, x.var(axis=0) / np.maximum(mean, 1e-30) ** 2, np.nan)
    return np.sqrt(np.log1p(np.maximum(cv2, 0.0)))


def log_mu_sd_from_counts(counts: np.ndarray) -> np.ndarray:
    """``sd(log mu)`` implied by Poisson-deconvolved counts. ``(N, G)`` -> ``(G,)``.

    For any conditional draw with ``Var(y | mu) >= mu``, the law of total variance gives
    ``Var(mu) = Var(y) - E[Var(y | mu)] <= Var(y) - mean(y)``, so

        CV^2(mu) = (Var(y) - mean(y)) / mean(y)^2

    clipped at zero, is an **upper** bound on the structured part — any over-dispersion in the
    tissue inflates it. With :func:`log_mu_sd_from_field` on ``mu_oracle`` it brackets the tissue's
    own ``sd(log mu)``, and neither route passes through the model's decoder, which is what
    ``reports/emission_repair_options.md`` §8.3's gate could not manage.
    """
    y = np.asarray(counts, dtype=np.float64)
    mean = y.mean(axis=0)
    cv2 = np.where(
        mean > 0, np.maximum(y.var(axis=0) - mean, 0.0) / np.maximum(mean, 1e-30) ** 2, np.nan
    )
    return np.sqrt(np.log1p(cv2))


def theta_report(
    model, cfg: Config, real: RealSection, panel: np.ndarray | None, floor: float
) -> dict:
    """Percentiles of the decoder's learned ``theta`` at the real cells, and what a floor binds on.

    M3's floor has to be chosen by a rule fixed in advance rather than by eye, and its failure
    mode — *"the floor binds on so few genes it does nothing"*
    (``reports/emission_repair_options.md`` §5) — is invisible unless the binding fraction is
    measured. This produces both: the distribution the pre-registration picks the floor from, and,
    given a floor, the fraction of ``(cell, gene)`` pairs at or below it.

    Read at the **real** section's cells through ``encoder(real counts)``, so the same quantity is
    comparable between a baseline and a floored refit without either depending on what the layout
    or the flow produced.
    """
    from spatialcpav25_gen.infer.generate import _decode

    gene_idx = torch.arange(real.counts.shape[1], dtype=torch.long)
    with torch.no_grad():
        gene_emb = model.embeddings.gene(gene_idx)
        totals = torch.from_numpy(real.counts.sum(axis=1))
        size_factor = totals / max(float(model.stats.median_total), 1.0)
        h1 = model.encoder(torch.from_numpy(real.counts), gene_emb, size_factor)
        _mu, theta, _pi = _decode(model, h1, cfg, None)
    t = theta.numpy()
    t = t if panel is None else t[:, panel]
    out: dict = {
        "n_pairs": int(t.size),
        "percentiles": {str(q): float(np.percentile(t, q)) for q in (1, 5, 10, 25, 50, 75, 90, 99)},
        "median_per_gene": float(np.median(np.median(t, axis=0))),
        "floor": float(floor),
    }
    if floor > 0:
        out["fraction_at_or_below_floor"] = float((t <= floor).mean())
        out["genes_with_any_binding"] = int((t.min(axis=0) <= floor).sum())
        out["n_genes"] = int(t.shape[1])
    return out


def decomposition_of_r(
    i_pred: np.ndarray, i_gt: np.ndarray, real_counts: np.ndarray, *, r4: float
) -> dict:
    """R1-R3 — how much of the scored correlation is spatial fidelity and how much is sparsity.

    A sparse gene's Moran's I is bounded low whatever its spatial structure, so the tissue's own
    per-gene ``I`` ordering may be substantially a **sparsity** ordering — and a model matching
    only the sparsity would score on ``paper_morans_pearson`` while reproducing no spatial
    fidelity. R3 asks the question that separates them: *given two genes the tissue detects
    equally often, does the model still order them correctly by ``I``?*

    **Three** control specifications are computed — detection rate, log mean count and log count
    variance — because "which control" is a degree of freedom the pre-registration fixes rather
    than leaves to be chosen after the answer. Two were not enough: they disagreed on both datasets
    (0.247 and 0.256 against a 0.150 tolerance), and both are location statistics of the same
    distribution, so ``flanking_copy_preregistration.md`` §3 adds the variance before any of them
    is read. The verdict is **dropped** when the three span more than 0.15 in the retained fraction,
    and the bands must be met by **every** specification, not by their mean -- averaging would let
    one favourable control carry an unfavourable one.
    """
    x = np.asarray(real_counts, dtype=np.float64)
    ok = np.isfinite(i_gt) & np.isfinite(i_pred)
    controls = {
        "detection": gene_detection(real_counts),
        "logmean": np.log(x.mean(axis=0) + 1e-9),
        # C3 is a different MOMENT. C1 and C2 are both location statistics of the same count
        # distribution, so their disagreement may say only that the abundance-I relation is not
        # quadratic in either -- which is not the same as saying the control does not matter.
        # `flanking_copy_preregistration.md` §3 adds it before any of the three is computed.
        "logvar": np.log(x.var(axis=0) + 1e-9),
    }
    out: dict = {"r4": float(r4), "n_genes": int(ok.sum()), "controls": list(controls)}
    for name, c in controls.items():
        out[f"R1_corr_Ireal_{name}"] = float(np.corrcoef(i_gt[ok], c[ok])[0, 1])
        out[f"R2_corr_Ipred_{name}"] = float(np.corrcoef(i_pred[ok], c[ok])[0, 1])
        out[f"R3_partial_{name}"] = partial_correlation(i_pred, i_gt, c)
        out[f"retained_{name}"] = (
            float(out[f"R3_partial_{name}"] / r4)
            if r4 and np.isfinite(out[f"R3_partial_{name}"]) and abs(r4) > 1e-12
            else float("nan")
        )
    fracs = [out[f"retained_{name}"] for name in controls]
    spread = float(np.nanmax(fracs) - np.nanmin(fracs)) if np.isfinite(fracs).any() else float("nan")
    out["spec_disagreement"] = spread
    if not np.isfinite(spread) or not np.isfinite(fracs).all() or spread > 0.15:
        out["verdict"] = "UNINFORMATIVE — the control specifications disagree"
    elif abs(r4) < 0.20:
        out["verdict"] = "UNINFORMATIVE — r below 0.20, the retained fraction is unstable"
    elif min(fracs) >= 0.60:
        out["verdict"] = "SPATIAL — the correlation survives controlling for sparsity"
    elif max(fracs) <= 0.30:
        out["verdict"] = "SPARSITY — most of r is matching which genes are sparse"
    else:
        out["verdict"] = "PARTIAL"
    return out


def stratified_relabel_r(
    i_pred: np.ndarray,
    i_gt: np.ndarray,
    control: np.ndarray,
    width: int,
    seeds: Sequence[int],
) -> dict:
    """F3 — the abundance-matched relabelling. ``flanking_copy_preregistration.md`` §5a.

    Rank genes by the target section's ``control`` (its detection rate), cut into strata of
    ``width`` genes, and permute **which predicted gene is compared to which real gene** inside
    each stratum. The abundance-``I`` relationship survives exactly; gene-specific spatial identity
    does not. So:

    * ``r`` staying near the unpermuted value  => the score is carried by abundance;
    * ``r`` collapsing toward zero             => the score is gene-specific spatial fidelity.

    This is the **decisive** instrument and the partial correlation is the corroborating one, not
    the other way round: it has no control-specification degree of freedom, only a stratum width,
    and the pre-registration requires all three widths reported and the reading stable across them.
    """
    ok = np.isfinite(i_pred) & np.isfinite(i_gt) & np.isfinite(control)
    pred, gt, c = i_pred[ok], i_gt[ok], control[ok]
    order = np.argsort(c, kind="stable")
    rs = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        mapping = np.empty(order.size, dtype=np.int64)
        for start in range(0, order.size, int(width)):
            block_idx = order[start : start + int(width)]
            mapping[block_idx] = block_idx[rng.permutation(block_idx.size)]
        den = pred[mapping].std() * gt.std()
        rs.append(float(np.corrcoef(pred[mapping], gt)[0, 1]) if den > 0 else float("nan"))
    a = np.asarray([r for r in rs if np.isfinite(r)])
    return {
        "width": int(width),
        "n_genes": int(ok.sum()),
        "n_seeds": int(a.size),
        "median_r": float(np.median(a)) if a.size else float("nan"),
        "p2.5": float(np.percentile(a, 2.5)) if a.size else float("nan"),
        "p97.5": float(np.percentile(a, 97.5)) if a.size else float("nan"),
    }


def flanking_copy_arm(
    vol: TrainingVolume,
    real: RealSection,
    k: int,
    *,
    null_seeds: Sequence[int],
    f3_widths: Sequence[int] = (10, 25, 50),
    stage4_r: float | None = None,
) -> dict:
    """`flanking_copy` on the ladder, plus F3, R3 and the `spatial_scramble` positive control.

    ``reports/flanking_copy_preregistration.md``. The probe is reproduced from
    ``bench3/selftest.py::make_probe``: the **nearest training section by z, verbatim** -- its
    counts at its own cells, in its own cell count. ``vol`` is the ``TrainingVolume``, so the
    held-out sections are already removed by type and the source cannot be one of them.

    Two asymmetries the pre-registration fixes and this function preserves:

    * §2a -- the copy is **not** at the target's cells, so it is comparable to stage 4 and to no
      other rung. It is returned in its own block and never merged into the A1 ladder.
    * §2b -- 0.9836 is tier-1's 28-gene figure. What this measures on deep's 1017 genes is a
      different number, and if it does not exceed stage 4 the premise of the test is absent.
    """
    import scipy.sparse as sp

    if not vol.sections:
        raise SystemExit("flanking_copy: the training volume carries no sections")
    src = min(vol.sections, key=lambda sec: (abs(float(sec.z) - real.z), str(sec.section_id)))
    counts = src.counts
    counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    counts = np.asarray(counts, dtype=np.float64)
    xy = np.asarray(src.coords, dtype=np.float64)[:, :2]
    if counts.shape[1] != real.counts.shape[1]:
        raise SystemExit(
            f"flanking_copy: section {src.section_id!r} has {counts.shape[1]} genes and the target "
            f"has {real.counts.shape[1]}; the two must index the same gene order"
        )

    i_real = morans_i_ranked_blocked(real.xy, np.asarray(real.counts, dtype=np.float64), k)
    i_flank = morans_i_ranked_blocked(xy, counts, k)
    agree = morans_agreement(i_flank, i_real)
    r_flank = float(agree["pearson"])

    # §5c -- the positive control. spatial_scramble keeps every per-gene marginal and destroys
    # only position. The prediction that it scores ~0 is RECORDED IN §6 BEFORE this ran; if it
    # holds, the maximal defensible critique is already the narrow one §5c writes out.
    scramble = permutation_null_I(
        real.xy, np.asarray(real.counts, dtype=np.float64), k, list(null_seeds)[:5]
    )
    scramble_rs = [morans_agreement(row, i_real)["pearson"] for row in scramble]
    detection = gene_detection(real.counts)
    out: dict = {
        "source_section": str(src.section_id),
        "source_z": float(src.z),
        "target_section": str(real.section_id),
        "target_z": float(real.z),
        "n_cells_source": int(xy.shape[0]),
        "n_cells_target": int(real.xy.shape[0]),
        "n_genes": int(agree["n_genes"]),
        "r_flank": r_flank,
        "spearman": float(agree["spearman"]),
        "mae": float(agree["mae"]),
        "stage4_r": None if stage4_r is None else float(stage4_r),
        "scramble": null_centre_test(scramble_rs),
        "f3": [
            stratified_relabel_r(i_flank, i_real, detection, w, list(null_seeds))
            for w in f3_widths
        ],
        "R3": decomposition_of_r(i_flank, i_real, np.asarray(real.counts), r4=r_flank),
    }
    out["verdict"] = _flanking_verdict(out)
    return out


def _flanking_verdict(f: dict) -> str:
    """The outcome table of ``flanking_copy_preregistration.md`` §4, in its stated order.

    §1's rule is what makes the order matter: **the default absent a clear result is the outcome
    that does not suit us.** PARTIAL and UNINFORMATIVE both resolve to outcome 1's reading, so a
    test this project has an interest in cannot return "unclear" and have that count as support.
    """
    r, s4 = f["r_flank"], f.get("stage4_r")
    if not np.isfinite(r):
        return "5. PREMISE ABSENT — the copy's r is not finite on this scope"
    if s4 is not None and r <= s4:
        return (
            f"5. PREMISE ABSENT — the copy scores {r:+.4f} against stage 4's {s4:+.4f}, so there "
            "is no high model-free floor on this scope to explain (§2b)"
        )
    fracs = [f["R3"][f"retained_{name}"] for name in f["R3"]["controls"]]
    f3 = [w["median_r"] for w in f["f3"]]
    if not np.isfinite(fracs).all() or "UNINFORMATIVE" in f["R3"]["verdict"]:
        return (
            "4. UNINFORMATIVE — the control specifications disagree; §4 resolves this to "
            "outcome 1's reading: THE COPY FLOOR STANDS AND THE DEFICIT IS OURS"
        )
    if min(fracs) >= 0.60:
        return (
            "1. FLOOR IS GENUINE — the copy's score survives every control. The deficit is ours "
            "and the paper reports a negative result with NO benchmark claim"
        )
    if max(fracs) <= 0.30:
        if np.isfinite(f3).all() and max(f3) >= 0.60 * r:
            return (
                "2. FLOOR IS ABUNDANCE — R3 and F3 agree. THIS IS THE OUTCOME THAT SUITS US and "
                "it is NOT reportable on this evidence: §5's preconditions govern, and F3 is the "
                "instrument that carries it, not R3"
            )
        return (
            "3. PARTIAL — R3 says abundance but F3 does not corroborate it (§5a makes F3 "
            "decisive). §4 resolves this to outcome 1's reading: THE COPY FLOOR STANDS"
        )
    return (
        "3. PARTIAL — a control lands between 0.30 and 0.60. §4 resolves this to outcome 1's "
        "reading: THE COPY FLOOR STANDS AND THE DEFICIT IS OURS"
    )


def md_cell(text: object) -> str:
    """Escape a value for a markdown table cell.

    Several stage and arm labels contain a literal ``|`` -- ``A1a. counts ~ emission(mu | h1)``
    is the one that bit. Unescaped, it opens a new column and every cell on that row after it
    renders one place to the left, silently: the reader sees a number under the wrong heading
    rather than a broken table. This was live in `reports/a1_*.md` for four rounds.
    """
    return str(text).replace("|", "\\|")


def agreement_block(agreement: dict[str, dict], invariant_note: str = "") -> list[str]:
    """Q1.5 — the scored statistic beside the median the campaign has been measuring.

    ``paper_morans_pearson`` is a **correlation across genes**; every number in the chain work is a
    **median**. A model can match the tissue's median exactly and get every gene wrong, so this
    table is the first thing in the campaign that reports both quantities side by side.

    Criteria and the three ways this differs from the published score are fixed in
    ``reports/q15_preregistration.md``; the reading is not applied here, because §5 there puts the
    bands on a comparison **between** stages and this function does not know which run it is in.
    """
    out: list[str] = []
    if invariant_note:
        out += [
            "",
            f"## 🚨 INVARIANT VIOLATED — {invariant_note}.",
            "",
            "No number in this report that divides one stage by another may be read until that is",
            "explained. `reports/ceiling_review.md` §2 is the last time this fired.",
        ]
    if not agreement:
        return out
    out += [
        "",
        "## Q1.5 — the scored statistic, beside the median",
        "",
        "`paper_morans_pearson` is the **correlation across genes** between the model's per-gene",
        "Moran's I vector and the tissue's. Every other number in this report is a **median**, and",
        "a model can match the median exactly while getting every gene wrong. Reconstructed here",
        "by bench3's own construction (`evaluate_paper.py::_agreement`, read from source): ranked",
        "counts both sides, `k=10`, each side on its own graph, NaN genes dropped pairwise.",
        "",
        "⚠️ **This is the statistic, not the score** — bench3 takes all shared genes and medians",
        "over sections 2/4/6, and this is one section. Comparable **between stages**, not to a",
        "published `paper_*` number (`reports/q15_preregistration.md` §2).",
        "",
        "🚩 `mae` is emitted by the evaluator, documented there as **the metric that catches",
        "over-smoothing**, and is **not** in the project's scored `METRICS` tuple.",
        "",
        "| gene set | stage | **pearson** | spearman | mae | median pred | median gt | genes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, m in agreement.items():
        gene_set, _, stage = key.partition("::")
        out.append(
            f"| {gene_set} | {md_cell(stage)} | **{m['pearson']:+.4f}** "
            f"| {m['spearman']:+.4f} "
            f"| {m['mae']:.4f} | {m['median_pred']:+.4f} | {m['median_gt']:+.4f} "
            f"| {m['n_genes']} |"
        )
    return out


def ladder_block(
    ladder: dict,
    sparsity: dict,
    agreement: dict | None = None,
    null: dict | None = None,
    boot: dict | None = None,
) -> list[str]:
    """The ladder and R1-R3 — is the gap to the copy floor closeable inside this architecture?

    Criteria in ``reports/ladder_preregistration.md``. The bands are **not** applied here: §3c
    reads them across two datasets and this function knows only one run. What it does enforce is
    the null check, now **per scope** and on the null's centre rather than on one draw against an
    absolute 0.15 — see ``reports/null_band_preregistration.md`` for what that threshold got wrong
    and why neither defect is repaired retroactively.
    """
    out: list[str] = []
    if not ladder:
        return out
    out += [
        "",
        "## The ladder — what each rung holds at the truth",
        "",
        "Every arm is drawn at the **real section's own cells**, so `pred_xy == gt_xy`. That is an",
        "advantage stage 4 and bench3 do not have — they compare a generated cell set against the",
        "real one, each on its own graph — so **a LOW rung here is conclusive and a high one is",
        "permissive** (`ladder_preregistration.md` §2a).",
        "",
        "Reference points on tier-1: the model-free copy floor `flanking_copy` = **0.9836**,",
        "SpatialZ **0.932**, v25 shipped **0.5574**.",
        "",
        "| gene set | rung | **pearson** | across seeds | spearman | mae | genes |",
        "|---|---|---|---|---|---|---|",
    ]
    # Stage 4 -- where we actually are -- is folded into the ladder rather than left in the
    # agreement block above. The reading in ladder_preregistration.md is entirely about the
    # DISTANCE between A1a and stage 4, and a rung the reader has to go and find is not beside it.
    merged = dict(ladder)
    for scope in ("panel", "all genes"):
        m4 = (agreement or {}).get(f"{scope}::4. sampled counts")
        if m4 is not None and any(key.startswith(f"{scope}::") for key in ladder):
            merged[f"{scope}::4"] = {
                "arm": "4.  counts ~ emission(mu | model latent)   [where we are]",
                "median_r": m4["pearson"],
                "min_r": m4["pearson"],
                "max_r": m4["pearson"],
                "n_seeds": 1,
                "spearman": m4["spearman"],
                "mae": m4["mae"],
                "n_genes": m4["n_genes"],
            }
    order = {"A1c": 0, "A1b": 1, "A1b-t": 2, "A1b-p": 3, "A1a": 4, "4": 5, "A1n": 6}
    for key in sorted(merged, key=lambda x: (x.partition("::")[0] != "panel", order.get(x.partition("::")[2], 9))):
        m = merged[key]
        scope, _, arm = key.partition("::")
        spread = (
            f"{m['min_r']:+.4f} .. {m['max_r']:+.4f} ({m['n_seeds']})" if m["n_seeds"] > 1 else "—"
        )
        # Several arm labels contain a literal "|" (``emission(mu | h1)``); unescaped it splits
        # the markdown column and the table renders one cell short from that row on.
        out.append(
            f"| {scope} | {md_cell(m['arm'])} "
            f"| **{m['median_r']:+.4f}** | {spread} "
            f"| {m['spearman']:+.4f} | {m['mae']:.4f} | {m['n_genes']} |"
        )
    out += _null_block(null or {})
    out += _bootstrap_block(boot or {})
    if sparsity:
        out += [
            "",
            "### R1-R3 - is the correlation spatial fidelity, or sparsity matching?",
            "",
            f"Computed on **{sparsity.get('gene_set', 'the scored panel')}** — the gene set the",
            "agreement table above says governs.",
            "",
            "A sparse gene's Moran's I is bounded low whatever its spatial structure, so a model",
            "that matched only *which genes are sparse* would score on `paper_morans_pearson`",
            "without reproducing any spatial fidelity. R3 controls for the tissue's own detection",
            "rate and asks whether the model still orders genes correctly.",
            "",
            "| quantity | detection rate | log mean count |",
            "|---|---|---|",
            "| **R1** `corr(I_real, control)` - is the tissue's ordering a sparsity ordering? | "
            + f"{sparsity['R1_corr_Ireal_detection']:+.4f} | "
            + f"{sparsity['R1_corr_Ireal_logmean']:+.4f} |",
            f"| **R2** `corr(I_model, control)` | {sparsity['R2_corr_Ipred_detection']:+.4f} "
            f"| {sparsity['R2_corr_Ipred_logmean']:+.4f} |",
            "| **R3** partial `corr(I_4, I_real given control)` | "
            + f"{sparsity['R3_partial_detection']:+.4f} | {sparsity['R3_partial_logmean']:+.4f} |",
            f"| retained fraction of `r(4)` = {sparsity['r4']:+.4f} | "
            f"{sparsity['retained_detection']:.1%} | {sparsity['retained_logmean']:.1%} |",
            "",
            f"**{sparsity['verdict']}** — the two specifications differ by "
            f"{sparsity['spec_disagreement']:.3f} against a 0.150 tolerance "
            "(`ladder_preregistration.md` §4).",
        ]
    return out


def _null_block(null: dict) -> list[str]:
    """Per-scope: is the permutation arm CENTRED on zero? ``null_band_preregistration.md`` §2a.

    Per scope is the point. The check this replaces assigned one variable inside the render loop,
    so with two scopes the second overwrote the first and deep's panel null of +0.2578 was hidden
    behind all-genes' +0.0594 -- ``specs/10`` §4.2f-i's family, in a check built for it.
    """
    if not null:
        return []
    out = [
        "",
        "### Is the null null? — per scope, on the null's CENTRE",
        "",
        "The permutation arm shares one shuffle across every gene, so a single draw has a wide",
        "spread and says nothing about the centre. A sound construction has `mean_r` at zero; the",
        "spread is a resolution limit and lives in the bootstrap below, not here",
        "(`null_band_preregistration.md` §2a). Fires on `|mean_r| > 2 * sd_r / sqrt(seeds)`.",
        "",
        "| scope | seeds | mean r | sd | 2 x se | 2.5% .. 97.5% | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    fired = []
    for scope, m in null.items():
        if m.get("underpowered"):
            out.append(f"| {md_cell(scope)} | {m['n_seeds']} | — | — | — | — | too few seeds |")
            continue
        verdict = "🚨 **NOT CENTRED**" if m["fired"] else "centred"
        out.append(
            f"| {md_cell(scope)} | {m['n_seeds']} | {m['mean_r']:+.4f} | {m['sd_r']:.4f} "
            f"| {2 * m['se_r']:.4f} | {m['p2.5']:+.4f} .. {m['p97.5']:+.4f} | {verdict} |"
        )
        if m["fired"]:
            fired.append(scope)
    if fired:
        out += [
            "",
            f"🚨 **NULL IS NOT CENTRED on {', '.join(fired)}** — the construction is wrong on "
            "that scope and **none of its rungs may be read.** Other scopes are unaffected: the "
            "check is per scope (`null_band_preregistration.md` §2a).",
        ]
    return out


def _bootstrap_block(boot: dict) -> list[str]:
    """Rung differences with a paired gene bootstrap. ``null_band_preregistration.md`` §2b.

    An ordering printed without this is an ordering of point estimates. The verdict leans on
    ``r(4) - r(A1a)``, so that difference in particular gets an interval rather than a sign.
    """
    if not boot:
        return []
    out = [
        "",
        "### Which rung differences are real? — paired gene bootstrap",
        "",
        "2000 replicates resampling **genes**, the same index applied to every rung so the",
        "difference is paired. This carries the across-gene sampling error only; draw-to-draw",
        "error is the `across seeds` column above and the two are never combined (§2b).",
        "",
        "| scope | difference | point | 95% interval | genes | |",
        "|---|---|---|---|---|---|",
    ]
    for scope, rows in boot.items():
        for row in rows:
            if row.get("unusable"):
                out.append(
                    f"| {md_cell(scope)} | {md_cell(row['pair'])} | — | — | {row['n_genes']} "
                    "| too few genes |"
                )
                continue
            mark = "distinguishable" if row["distinguishable"] else "**contains zero**"
            out.append(
                f"| {md_cell(scope)} | {md_cell(row['pair'])} | {row['diff']:+.4f} "
                f"| {row['lo']:+.4f} .. {row['hi']:+.4f} | {row['n_genes']} | {mark} |"
            )
    return out


def stage4_seed_block(rows: list[dict]) -> list[str]:
    """Stage 4 across whole generations. ``null_band_preregistration.md`` §2c.

    Every A1 arm redraws counts at fixed cells, so its across-seed spread is emission noise alone.
    Stage 4's is not: the layout and the flow sample vary too, and stage 4 is the rung the ladder's
    verdict leans on. Reported as its own block so the two spreads are never read as the same
    quantity.
    """
    if not rows:
        return []
    out = [
        "",
        "### Stage 4 across whole generations",
        "",
        "Each row is a **complete** regeneration — layout, prior, flow, decode, draw — not a",
        "redraw at fixed cells. The A1 arms' `across seeds` column is emission noise alone; this",
        "is the whole pipeline's, and it is the one the verdict leans on",
        "(`null_band_preregistration.md` §2c).",
        "",
        "| seed | cells | median I (rank) | r, panel | r, all genes |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        panel_r = r.get("r_panel")
        all_r = r.get("r_all")
        out.append(
            f"| {r['seed']} | {r['n_used']} | {r['median_I']:+.4f} "
            f"| {'—' if panel_r is None else f'{panel_r:+.4f}'} "
            f"| {'—' if all_r is None else f'{all_r:+.4f}'} |"
        )
    for key, label in (("r_panel", "panel"), ("r_all", "all genes")):
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if len(vals) > 1:
            out += [
                "",
                f"**{label}: r spans {min(vals):+.4f} .. {max(vals):+.4f} across "
                f"{len(vals)} generations** (sd {float(np.std(vals, ddof=1)):.4f}). Any rung "
                "difference smaller than this is not resolved by a single generation.",
            ]
    return out


def flanking_block(f: dict) -> list[str]:
    """`flanking_copy` on the ladder, F3, R3 and the positive control.

    ``reports/flanking_copy_preregistration.md``, whose §1 is reproduced at the top of the block
    rather than left in the file: this is the one test with an outcome that suits this project,
    and a reader meeting the number should meet the conflict of interest in the same place.
    """
    if not f:
        return []
    s4 = f.get("stage4_r")
    out = [
        "",
        "## `flanking_copy` — is the floor that beats us spatial fidelity?",
        "",
        "⚠️ **This is the one test whose favourable outcome this project has an interest in.**",
        "`flanking_copy_preregistration.md` §1 fixes two rules before any number here existed:",
        "the default absent a clear result is **the outcome that does not suit us**, and **the",
        "negative result is reported either way** — v25 loses to a model-free copy, and that",
        "sentence goes in the paper whatever this block says.",
        "",
        f"Source: **{f['source_section']}** at z={f['source_z']:.1f} "
        f"({f['n_cells_source']} cells) — the nearest *training* section to "
        f"{f['target_section']} at z={f['target_z']:.1f} ({f['n_cells_target']} cells), emitted",
        "verbatim, exactly as `bench3/selftest.py::make_probe` does. It is **not** at the",
        "target's cells, so it is comparable to stage 4 and to no other rung (§2a).",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| `r_flank` on {f['n_genes']} genes | **{f['r_flank']:+.4f}** |",
        f"| stage 4, same scope | {'—' if s4 is None else f'{s4:+.4f}'} |",
        f"| spearman / mae | {f['spearman']:+.4f} / {f['mae']:.4f} |",
        "",
        "🚩 **0.9836 is tier-1's 28-gene figure and does not transfer here** (§2b). The test is",
        "against whatever the copy scores on *this* scope.",
    ]
    sc = f["scramble"]
    if not sc.get("underpowered"):
        out += [
            "",
            "### §5c — the positive control",
            "",
            f"`spatial_scramble` keeps every per-gene marginal and destroys only position: "
            f"**{sc['mean_r']:+.4f}** (sd {sc['sd_r']:.4f}, {sc['n_seeds']} seeds).",
            "",
            "§6 predicted this scores ~0 **before it ran**. If it does, the metric plainly does",
            "respond to position and the maximal defensible critique is already the narrow one:",
            "*among predictions carrying realistic within-gene autocorrelation, the across-gene",
            "correlation is dominated by per-gene abundance*. A broader sentence than that cannot",
            "be written later. If it scores high instead, that is the headline result.",
        ]
    out += [
        "",
        "### §5a — F3, the abundance-matched relabelling (the DECISIVE instrument)",
        "",
        "Genes are ranked by the target's detection rate, cut into strata, and the *pairing*",
        "between predicted and real genes is permuted **within** each stratum. The abundance-`I`",
        "relationship survives exactly; gene-specific spatial identity does not. R3 below is the",
        "corroborating instrument, not the other way round — it has a control specification we",
        "chose and this does not.",
        "",
        "| stratum width | median r | 2.5% .. 97.5% | as a fraction of `r_flank` | seeds |",
        "|---|---|---|---|---|",
    ]
    for w in f["f3"]:
        frac = w["median_r"] / f["r_flank"] if f["r_flank"] else float("nan")
        out.append(
            f"| {w['width']} genes | **{w['median_r']:+.4f}** | {w['p2.5']:+.4f} .. "
            f"{w['p97.5']:+.4f} | {frac:.1%} | {w['n_seeds']} |"
        )
    out += [
        "",
        "The reading must be **stable across all three widths** (§5a); a result that appears at",
        "one width and not the others is a stratum-width artefact.",
    ]
    r3 = f["R3"]
    out += [
        "",
        "### §3 — R3 for the copy, under all three controls",
        "",
        "| quantity | detection | log mean | log variance |",
        "|---|---|---|---|",
        "| **R1** `corr(I_real, control)` | "
        + " | ".join(f"{r3[f'R1_corr_Ireal_{n}']:+.4f}" for n in r3["controls"])
        + " |",
        "| **R2** `corr(I_copy, control)` | "
        + " | ".join(f"{r3[f'R2_corr_Ipred_{n}']:+.4f}" for n in r3["controls"])
        + " |",
        "| **R3** partial `corr(I_copy, I_real given control)` | "
        + " | ".join(f"{r3[f'R3_partial_{n}']:+.4f}" for n in r3["controls"])
        + " |",
        f"| retained fraction of `r_flank` = {r3['r4']:+.4f} | "
        + " | ".join(f"{r3[f'retained_{n}']:.1%}" for n in r3["controls"])
        + " |",
        "",
        f"The three specifications span **{r3['spec_disagreement']:.3f}** against a 0.150",
        "tolerance, and the bands must be met by **every** control, not by their mean (§3-§4).",
        "",
        f"## **{f['verdict']}**",
        "",
        "§5's preconditions — independence from v25's own numbers, the positive control,",
        "stability across F3's widths, replication across sections 2/4/6 and both datasets — are",
        "**preconditions, not follow-ups**. One section of one dataset does not make a benchmark",
        "claim, and this run is one section of one dataset.",
    ]
    return out


def cancelling_defects_block(
    rows: list[dict], decomposition: dict[str, float], mu_spread: list[dict]
) -> list[str]:
    """Why a generative model can sit **above** real tissue on Moran's I — beside the number.

    Tier-1 read healthy for three rounds on exactly this: the latent is too smooth, which pushes
    ``I`` up; the emission adds spatially independent noise, which pushes it down; and on that
    dataset the two cancel to within 11 %. A number at or above the tissue's therefore says the
    two errors cancel, **not** that either is right — and repairing one of them alone moves ``I``
    *away* from the tissue, in whichever direction that one was wrong.

    Built entirely from the run's own rows, so it cannot disagree with the table above it, and
    printed here rather than left to a report the reader may not have — ``specs/10`` §4.2f-i's
    lesson applied to an explanation rather than to an alarm.

    Returns the markdown lines, or ``[]`` when the run lacks the stages to say anything.
    """

    def find(prefix: str) -> float | None:
        for row in rows:
            if str(row["stage"]).startswith(prefix):
                return float(row["median_I"])
        return None

    i_model, i_real = find("4. sampled counts"), find("REF real counts")
    i_h, i_h1 = find("2. latent h"), find("REF real latent")
    i_ceiling = find("4p.")
    if i_real in (None, 0.0):
        return []

    bracket = {
        row["quantity"].split(" (")[0]: row["sd_log_mu"]
        for row in mu_spread
        if "bound" in str(row.get("quantity", ""))
    }
    lower, upper = bracket.get("mu_oracle"), bracket.get("real counts")

    out: list[str] = []
    if None not in (i_model, i_h, i_h1) and i_model > i_real and i_h1:
        out += [
            "",
            "## Why the model sits ABOVE the tissue here, and why that is not fidelity",
            "",
            f"`I(model counts)` = **{i_model:+.4f}** against the real section's **{i_real:+.4f}** "
            f"— {i_model / i_real:.2f}x. That is **not** a reconstruction result. Two defects "
            "point in opposite directions on this dataset and partly cancel:",
            "",
            "| defect | this run | direction on `I` |",
            "|---|---|---|",
            f"| the latent is **{i_h / i_h1:.2f}x smoother** than the tissue's ({i_h:+.4f} "
            f"against `h1`'s {i_h1:+.4f}) | too smooth | pushes `I` **up** |",
        ]
        if lower is not None and upper is not None:
            out.append(
                f"| `mu`'s spread is **{decomposition['sd_log_mu']:.4f}** against the tissue's "
                f"model-free bracket [{lower:.4f}, {upper:.4f}] | too narrow | — |"
            )
        out += [
            "| the emission adds spatially independent noise (`theta`, and `pi` where it is "
            "non-zero) | too much | pushes `I` **down** |",
            "",
            "So a number at or above the tissue's says the two happen to cancel, not that either "
            "is right. Repairing one alone moves `I` **away** from the tissue: a faithful latent "
            "lowers it (A1's `A1a` arm), and removing the emission's noise raises it (stage 4p).",
        ]
    if i_ceiling is not None:
        out += [
            "",
            "### Stage 4p — the emission-free ceiling",
            "",
            f"With the emission's noise removed from the model's **own** mean field, `I` = "
            f"**{i_ceiling:+.4f}** against the real section's **{i_real:+.4f}** "
            f"({i_ceiling / i_real:.2f}x). That is the most any repair to `theta` or `pi` can "
            "reach on this fit — it bounds the emission-side work from above. **If it exceeds the "
            "tissue, the emission is not the only defect** (`reports/n5_and_m3_review.md` §5), and "
            "a repair to it alone cannot land the model on the tissue.",
        ]
    return out


def n5_verdict(i_c: float, i_b: float, i_t: float, i_p: float) -> dict[str, float | str]:
    """Attribute the ``A1c -> A1b`` loss to ``theta``, to ``pi``, to both, or to neither.

    The decision rule of ``reports/a1_escalation_preregistration.md`` §2, in code rather than in
    the reader's head, because a pre-registered rule executed by hand is a rule that drifts.

    ``L_total = I(A1c) - I(A1b)``, ``L_theta = I(A1c) - I(A1b-t)``, ``L_pi = I(A1c) - I(A1b-p)``;
    the verdict is **over-dispersion** at ``L_theta/L_total >= 0.70`` with ``L_pi/L_total <= 0.30``,
    **dropout** at the reverse, **both, additively** when each share is in ``(0.30, 0.70)`` *and*
    the additivity gap is ``<= 0.02`` in ``I``, and **not decomposable** otherwise — which is the
    honest outcome when zero-inflation and over-dispersion interact rather than compose.
    """
    l_total, l_theta, l_pi = i_c - i_b, i_c - i_t, i_c - i_p
    share_t = l_theta / l_total if l_total else float("nan")
    share_p = l_pi / l_total if l_total else float("nan")
    gap = abs(l_theta + l_pi - l_total)
    if share_t >= 0.70 and share_p <= 0.30:
        verdict = "**over-dispersion**"
    elif share_p >= 0.70 and share_t <= 0.30:
        verdict = "**dropout**"
    elif 0.30 < share_t < 0.70 and 0.30 < share_p < 0.70 and gap <= 0.02:
        verdict = "**both, additively**"
    else:
        verdict = "**not decomposable** — the two interact"
    return {
        "l_total": l_total,
        "l_theta": l_theta,
        "l_pi": l_pi,
        "share_theta": share_t,
        "share_pi": share_p,
        "additivity_gap": gap,
        "predicted_i_b": (i_t * i_p / i_c) if i_c else float("nan"),
        "verdict": verdict,
    }


def describe_text_state(model, cfg: Config) -> str:
    """What the text channel **holds**, read off the buffer rather than inferred from a flag.

    The line this replaces reported how the module was *constructed*, so a run that loaded a
    ``medcpt`` checkpoint printed "zero vectors": ``text_vecs`` is a registered buffer and
    ``load_state_dict`` restores it, which the constructor's argument knows nothing about. Two
    reports were published saying the opposite of what they measured (``specs/10`` §4.2k, and
    §4.2j's rule that a report may only state what it can establish from the run's own record).

    So this counts all-zero rows in the tensor the model will actually use.
    """
    with torch.no_grad():
        vecs = model.embeddings.gene.text_vecs
        n_rows = int(vecs.shape[0])
        n_zero = int((vecs.abs().sum(dim=1) == 0).sum())
    if n_zero == n_rows:
        return f"**zero vectors** — all {n_rows} gene rows are zero, so this is neither A3 arm"
    live = (
        f"live, `text_emb_mode={cfg.text_emb_mode}`, {n_rows - n_zero}/{n_rows} gene rows non-zero"
    )
    return live if n_zero == 0 else f"{live} (**{n_zero} bare**)"


def _over_seeds(rows: list[dict]) -> dict:
    """Collapse one arm's per-seed rows into a single row. ``median_I`` is the median over seeds.

    ``reports/a1_escalation_preregistration.md`` §1 reads the band from the median and applies a
    **stability override** when the seeds do not all agree, so ``min_I`` / ``max_I`` and the full
    per-seed list are carried rather than summarised away.
    """
    values = [r["median_I"] for r in rows]
    head = dict(rows[0])
    head.pop("seed", None)
    head["median_I"] = float(np.median(values))
    head["min_I"] = float(np.min(values))
    head["max_I"] = float(np.max(values))
    head["n_seeds"] = len(rows)
    head["per_seed"] = [{"seed": r["seed"], "median_I": r["median_I"]} for r in rows]
    # The ladder correlates each arm's per-gene I vector against the tissue's, at every seed the arm
    # was drawn at, as the median rows already are. Keeping only the first seed's would make the
    # ladder the one single-seed statistic in the ablation.
    head["per_seed_I_rank"] = [r["per_gene_I_rank"] for r in rows]
    return head


def emission_ablation(
    model,
    cfg: Config,
    real: RealSection,
    h1,
    k: int,
    seeds: list[int],
    panel: np.ndarray | None = None,
    *,
    mu_gen: np.ndarray | None = None,
    counts_gen: np.ndarray | None = None,
    null_seeds: Sequence[int] = tuple(range(101, 121)),
) -> tuple[list[dict], dict, list[dict], dict, dict, dict, dict]:
    """A1 — which stage of the emission loses the structure. Measured at the REAL cells.

    Every arm is drawn at ``real.xy``, so the kNN graph, the cell count and the density are the
    ground truth's own and identical to the ``REF real counts`` row. Nothing here touches the
    layout, the prior or the flow: positions are held at the tissue's, and the only thing that
    varies between the arms is what the counts were drawn from.

    The arms, in the order the pre-registration reads them
    (``reports/a1_preregistration.md``):

    ``A1c`` ``Poisson(mu_oracle)``
        **Model-free.** The tissue's own mean field through the leanest possible emission. If
        this cannot reach the real section's ``I``, no independent per-cell draw can, whatever
        the mean model — and the deficit is counting statistics, not the decoder.
    ``A1b`` ``ZINB(mu_oracle, theta_model, pi_model)``
        The same mean field through **this model's dispersion and dropout**. A1c minus A1b is
        what the fitted ``theta``/``pi`` cost.
    ``A1a`` ``ZINB(decode(h1))``
        The model's whole emission on the best latent it can form from the truth. A1b minus A1a
        is what the decode path costs.

    ``A1b-t`` ``NB(mu_oracle, theta_model)``, ``pi`` forced to 0
        over-dispersion alone. ``theta`` cannot move the mean, so this arm's level ratio must come
        back at ~1.00x — a self-check that can fail.
    ``A1b-p`` the **same Poisson realisation as A1c**, then zeroed with ``pi_model``
        dropout alone. Sharing A1c's draw rather than taking a fresh one is deliberate: it makes
        A1c -> A1b-p an exact within-realisation contrast, so the difference is the dropout and
        nothing else.

    N5, pre-registered in ``reports/a1_escalation_preregistration.md`` §2, which fixes the shares
    that attribute the ``A1c -> A1b`` loss and the additivity check reported either way.

    ``A1n`` is the permutation null: the real counts shuffled across cells, which is what "no
    spatial structure" measures as on this panel and this graph.

    ``seeds`` redraws every sampled arm once per seed — N4, the escalation
    ``reports/a1_escalation_preregistration.md`` §1 fixes the combination rule for. The two
    **mean-field** rows do not depend on the draw and are computed once; each drawn row carries its
    per-seed values, and its ``median_I`` is the median over seeds, which is the statistic the band
    is read from. ``min_I`` / ``max_I`` are carried so the margin is visible rather than inferred.

    Returns the summary rows, a dict of per-arm level ratios (median over seeds of the median over
    the panel of the arm's per-gene mean over the real section's — an arm at a different count level
    is not comparable in ``I`` however it was drawn), the ``sd(log mu)`` table of N2, and **the
    ladder**: each arm's per-gene ``I`` correlated against the real section's, which is the
    **scored** statistic rather than the median every other row reports
    (``reports/ladder_preregistration.md``).
    """
    from spatialcpav25_gen.infer.generate import _decode
    from spatialcpav25_gen.model.expression import sample_counts

    def sel(a: np.ndarray) -> np.ndarray:
        return a if panel is None else a[:, panel]

    real_counts = np.asarray(real.counts, dtype=np.float64)
    with torch.no_grad():
        mu1, theta1, pi1 = _decode(model, h1, cfg, None)
    mu1_np, theta1_np, pi1_np = mu1.numpy(), theta1.numpy(), pi1.numpy()
    mu_oracle = knn_mean_field(real.xy, real.counts, k)

    zero_pi = np.zeros_like(pi1_np)

    def draw(seed: int) -> dict[str, np.ndarray]:
        # Not through sample_counts for A1c: that arm is a Poisson draw, not a ZINB one with theta
        # pushed to an extreme the sampler's own docstring flags. Saying "Poisson" and drawing
        # Poisson is the whole point of the arm.
        poisson = np.random.default_rng(seed).poisson(np.maximum(mu_oracle, 0.0)).astype(np.float64)
        dropout = np.random.default_rng(seed + 2).random(pi1_np.shape) < pi1_np
        return {
            "A1a": sample_counts(mu1_np, theta1_np, pi1_np, np.random.default_rng(seed)).numpy(),
            "A1b": sample_counts(mu_oracle, theta1_np, pi1_np, np.random.default_rng(seed)).numpy(),
            "A1bt": sample_counts(
                mu_oracle, theta1_np, zero_pi, np.random.default_rng(seed)
            ).numpy(),
            "A1bp": np.where(dropout, 0.0, poisson),
            "A1c": poisson,
            "A1n": real_counts[np.random.default_rng(seed + 1).permutation(real_counts.shape[0])],
        }

    labels = {
        "A1a": "A1a. counts ~ emission(mu | h1)",
        "A1b": "A1b. counts ~ ZINB(mu_oracle, model theta/pi)",
        "A1bt": "A1b-t. counts ~ NB(mu_oracle, model theta), pi=0",
        "A1bp": "A1b-p. counts ~ A1c's Poisson draw, then model pi",
        "A1c": "A1c. counts ~ Poisson(mu_oracle)   [model-free]",
        "A1n": "A1n. permutation null (real counts shuffled)",
    }
    ref_mean = sel(real_counts).mean(axis=0)
    ok = ref_mean > 0

    per_seed: dict[str, list[dict]] = {key: [] for key in labels}
    per_seed_level: dict[str, list[float]] = {key: [] for key in labels}
    drawn: dict[str, np.ndarray] = {}
    for seed in seeds:
        arms = draw(int(seed))
        for key, counts in arms.items():
            row = summarise(labels[key], real.xy, sel(counts), k, primary="rank")
            row["seed"] = int(seed)
            per_seed[key].append(row)
            ratio = sel(np.asarray(counts, dtype=np.float64)).mean(axis=0)[ok] / ref_mean[ok]
            per_seed_level[key].append(float(np.median(ratio)) if ratio.size else float("nan"))
            drawn.setdefault(key, counts)

    rows = [summarise("A1a'. mu decoded from h1", real.xy, sel(mu1_np), k, primary="raw")]
    rows.append(_over_seeds(per_seed["A1a"]))
    rows.append(
        summarise(
            "A1b'. mu_oracle = kNN mean of real counts", real.xy, sel(mu_oracle), k, primary="raw"
        )
    )
    rows.append(_over_seeds(per_seed["A1b"]))
    rows.append(_over_seeds(per_seed["A1bt"]))
    rows.append(_over_seeds(per_seed["A1bp"]))
    rows.append(_over_seeds(per_seed["A1c"]))
    rows.append(_over_seeds(per_seed["A1n"]))
    levels = {
        labels[key].split(".")[0]: float(np.median(vals))
        for key, vals in per_seed_level.items()
        if key != "A1n"
    }

    # N2: two model-free routes to the tissue's own sd(log mu), bracketing it from below and above,
    # neither passing through the decoder. Reported beside the decoder's own figure so the
    # comparison §6 of chain_shipped_review.md asked for can be made on a matched estimator.
    spread = [
        (
            "real counts (Poisson-deconvolved) — UPPER bound",
            log_mu_sd_from_counts(sel(real_counts)),
        ),
        ("mu_oracle (kNN mean field) — LOWER bound", log_mu_sd_from_field(sel(mu_oracle))),
        ("mu decoded from h1", log_mu_sd_from_field(sel(mu1_np))),
    ]
    if mu_gen is not None:
        spread.append(("mu decoded from the generated h", log_mu_sd_from_field(np.asarray(mu_gen))))
    if counts_gen is not None:
        spread.append(
            (
                "model counts (Poisson-deconvolved)",
                log_mu_sd_from_counts(np.asarray(counts_gen)),
            )
        )
    spread.append(
        (
            "A1c counts (Poisson-deconvolved) — estimator check",
            log_mu_sd_from_counts(sel(drawn["A1c"])),
        )
    )
    mu_spread = [
        {
            "quantity": name,
            "sd_log_mu": float(np.nanmedian(values)) if values.size else float("nan"),
            "n_genes": int(np.isfinite(values).sum()),
        }
        for name, values in spread
    ]

    # --- the ladder (reports/ladder_preregistration.md) -------------------------------------
    # Every arm is drawn at the REAL cells, so pred_xy == gt_xy: an advantage stage 4 and bench3
    # do not have, which is why §2a(1) reads a LOW rung as conclusive and a high one as permissive.
    ref_panel = morans_i(real.xy, rank_normalize(sel(real_counts)), k)
    ladder: dict = {}
    ladder_extra: dict = {}
    null: dict = {}
    vecs_panel: dict[str, np.ndarray] = {}
    for key, label in labels.items():
        per = [morans_agreement(row["per_gene_I_rank"], ref_panel) for row in per_seed[key]]
        rs = [m["pearson"] for m in per]
        ladder[f"panel::{label.split('.')[0]}"] = {
            "arm": label,
            "median_r": float(np.median(rs)),
            "min_r": float(np.min(rs)),
            "max_r": float(np.max(rs)),
            "n_seeds": len(rs),
            "spearman": float(np.median([m["spearman"] for m in per])),
            "mae": float(np.median([m["mae"] for m in per])),
            "n_genes": per[0]["n_genes"],
        }
        vecs_panel[label.split(".")[0]] = per_seed[key][0]["per_gene_I_rank"]

    # --- the corrected null check (reports/null_band_preregistration.md §2a) -------------------
    # 20 permutations, per scope, on the null's CENTRE. The arm is a shuffle and a Moran's I: no
    # model, no decoder. `permutation_null_I` builds the tree once and ranks once, which is what
    # makes 20 seeds affordable at all-genes.
    t = time.time()
    nulls_panel = permutation_null_I(real.xy, sel(real_counts), k, list(null_seeds))
    null["panel"] = null_centre_test(
        [morans_agreement(row, ref_panel)["pearson"] for row in nulls_panel]
    )
    print(f"  null x{len(null_seeds)}, panel, in {time.time() - t:.1f}s", flush=True)
    ladder_extra["ref_panel"] = ref_panel
    ladder_extra["vecs_panel"] = vecs_panel

    if panel is not None and len(panel) < real_counts.shape[1]:
        # The panel is the top few per cent by the real section's own I, so its I vector has a
        # compressed range and every correlation on it is attenuated. bench3 scores all shared
        # genes, so the all-genes pass governs -- as it did in Q1.5.
        t = time.time()
        ref_all = morans_i_ranked_blocked(real.xy, real_counts, k)
        vecs_all: dict[str, np.ndarray] = {}
        for key, label in labels.items():
            vec = morans_i_ranked_blocked(real.xy, drawn[key], k)
            vecs_all[label.split(".")[0]] = vec
            m = morans_agreement(vec, ref_all)
            ladder[f"all genes::{label.split('.')[0]}"] = {
                "arm": label,
                "median_r": m["pearson"],
                "min_r": m["pearson"],
                "max_r": m["pearson"],
                "n_seeds": 1,
                "spearman": m["spearman"],
                "mae": m["mae"],
                "n_genes": m["n_genes"],
            }
        print(f"  ladder, all {real_counts.shape[1]} genes in {time.time() - t:.1f}s", flush=True)
        t = time.time()
        nulls_all = permutation_null_I(real.xy, real_counts, k, list(null_seeds))
        null["all genes"] = null_centre_test(
            [morans_agreement(row, ref_all)["pearson"] for row in nulls_all]
        )
        print(f"  null x{len(null_seeds)}, all genes, in {time.time() - t:.1f}s", flush=True)
        ladder_extra["ref_all"] = ref_all
        ladder_extra["vecs_all"] = vecs_all
    return rows, levels, mu_spread, ladder, null, ladder_extra


def real_section_reference(
    cfg: Config,
    model: CTFFlow,
    section_id: str,
    k: int,
    ground_truth: str | Path | None = None,
    *,
    real: RealSection | None = None,
    panel: np.ndarray | None = None,
) -> tuple[list[dict], torch.Tensor]:
    """Moran's I of the real held-out section's counts, and of the latent the encoder makes.

    Returns the two summary rows **and** ``h1``, the encoder's latent on the real counts —
    ``(N, Config.latent_dim)``. ``h1`` is the tissue side of step 1's ``sd(log mu)`` comparison,
    and returning it here is what lets that comparison cost nothing beyond this run.

    ``panel`` restricts the *counts* row to those gene columns, and only that row: ``h1`` has
    ``Config.latent_dim`` channels which are not genes. The **encoder always sees the whole
    panel**, because its size factor and its trunk are functions of the full expression vector;
    restricting its input would measure a different model, not the same model on fewer genes.
    """
    real = (
        real
        if real is not None
        else load_real_section(
            section_id, ground_truth, section_key=str(cfg.section_key), coord_key=str(cfg.coord_key)
        )
    )
    counts, xy = real.counts, real.xy
    selected = counts if panel is None else counts[:, panel]

    rows = [summarise("REF real counts (rank-normalised)", xy, selected, k, primary="rank")]

    gene_idx = torch.arange(counts.shape[1], dtype=torch.long)
    with torch.no_grad():
        gene_emb = model.embeddings.gene(gene_idx)
        totals = torch.from_numpy(counts.sum(axis=1))  # (N,), not (N, 1)
        size_factor = totals / max(float(model.stats.median_total), 1.0)
        h1 = model.encoder(torch.from_numpy(counts), gene_emb, size_factor)
    rows.append(
        summarise("REF real latent h1 = encoder(real counts)", xy, h1.numpy(), k, primary="raw")
    )
    return rows, h1


def mu_log_variance_terms(
    model, h, cfg: Config, panel: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-gene ``Var(shape)``, ``Var(log s)`` and ``Cov``, given a latent. Each ``(G,)``.

    The decoder builds ``mu = link(MLP_mu(u)) * size_factor`` (``model/expression.py``), so in
    logs the split is exact and additive::

        log mu = shape + log s,   shape = log link(MLP_mu(u)),   s = size_factor

    and ``Var(log mu) = Var(shape) + Var(log s) + 2 Cov(shape, log s)``.

    Kept separate from the summary because step 1 needs the **per-gene** arrays: the generated
    and the real-latent runs are compared gene by gene on one panel, and a ratio of two medians
    is not the median of the ratios.

    ``h`` is ``(N, Config.latent_dim)`` — it may be the flow's sample or the encoder's latent on
    the real section; the decoder applied to it is the same decoder either way, which is what
    makes the two sides comparable in one variable.
    """
    idx = (
        torch.arange(len(model.data.vol.gene_names), dtype=torch.long)
        if panel is None
        else torch.from_numpy(np.asarray(panel, dtype=np.int64))
    )
    with torch.no_grad():
        gene_emb = model.embeddings.gene(idx)
        size_factor = model.size_head(h)
        features = model.decoder.trunk(h, gene_emb)
        raw = model.decoder.head_mu(features).squeeze(-1)
        if cfg.decoder_mu_link == "exp":
            shape = torch.clamp(raw, min=model.decoder._log_mu_min, max=model.decoder._log_mu_max)
        else:
            shape = torch.log(torch.nn.functional.softplus(raw) + float(cfg.zinb_eps))
        log_s = torch.log(torch.clamp(size_factor, min=float(cfg.zinb_eps)))[:, None]

    sh = shape.numpy()
    ls = np.broadcast_to(log_s.numpy(), sh.shape)
    v_shape = sh.var(axis=0)
    v_size = ls.var(axis=0)
    cov = np.array([np.cov(sh[:, g], ls[:, g])[0, 1] for g in range(sh.shape[1])])
    return {
        "var_shape": v_shape,
        "var_logsize": v_size,
        "cov": cov,
        "total": v_shape + v_size + 2.0 * cov,
    }


def summarise_mu_terms(terms: dict[str, np.ndarray]) -> dict[str, float]:
    """Medians over genes of :func:`mu_log_variance_terms`. T10 candidate 2.

    If ``Var(log s)`` dominates, ``mu``'s between-cell dynamic range is the size factor and the
    latent is only modulating it weakly — which is the shape T10's structured-share finding
    would have if the decoder were mostly reproducing library size.

    ``share_shape`` is the statistic R12's 15.3% / 61.4% / 62.2% are on and is kept for that
    reason, but it is **not bounded by 1**: a negative covariance makes the total smaller than
    ``Var(shape)``. ``share_shape_bounded`` — ``Var(shape) / (Var(shape) + Var(log s))`` — drops
    the covariance, answers the same question, and is the one a threshold may be placed on
    (``scripts/t09_structured_share.py``, which measured 1.21 on the unbounded form).
    """
    v_shape, v_size = terms["var_shape"], terms["var_logsize"]
    total = terms["total"]
    ok = total > 0
    return {
        "n_genes_decomposed": int(v_shape.size),
        "var_shape": float(np.median(v_shape)),
        "var_logsize": float(np.median(v_size)),
        "cov": float(np.median(terms["cov"])),
        "share_shape": float(np.median(v_shape[ok] / total[ok])) if ok.any() else float("nan"),
        "share_logsize": float(np.median(v_size[ok] / total[ok])) if ok.any() else float("nan"),
        "share_shape_bounded": float(np.median(v_shape / np.maximum(v_shape + v_size, 1e-30))),
        "sd_log_mu": float(np.median(np.sqrt(np.maximum(total, 0.0)))),
    }


def mu_variance_decomposition(
    model, h, cfg: Config, panel: np.ndarray | None = None
) -> dict[str, float]:
    """Split ``Var(log mu)`` into its size-factor and latent-driven parts, over ``panel``."""
    return summarise_mu_terms(mu_log_variance_terms(model, h, cfg, panel))


def mean_variance_slope(counts: np.ndarray) -> float:
    """Log-log slope of per-gene variance against per-gene mean. The tissue's is ~1.74."""
    m = np.asarray(counts, dtype=np.float64).mean(axis=0)
    v = np.asarray(counts, dtype=np.float64).var(axis=0)
    return float(np.polyfit(np.log(m + 1e-9), np.log(v + 1e-9), 1)[0])


def _verdict(
    rows: list[dict],
    emitted: dict,
    cfg: Config,
    args,
    real_section: RealSection,
    panel: np.ndarray | None = None,
) -> list[str]:
    """The three numbers the decision turns on: retention, slope, and the counts' own Moran's I.

    ``panel`` restricts the mean-variance slopes to the same genes the Moran's I rows are on;
    a slope over 1017 genes against an ``I`` over 32 of them would be two different comparisons
    in one table.
    """
    by = {r["stage"]: r["median_I"] for r in rows}
    real = np.asarray(real_section.counts, dtype=np.float64)
    if panel is not None:
        real = real[:, panel]

    real_latent = by.get("REF real latent h1 = encoder(real counts)", float("nan"))
    real_counts = by.get("REF real counts (rank-normalised)", float("nan"))
    real_retention = real_counts / real_latent if real_latent else float("nan")
    latent = by.get("2. latent h after the flow", float("nan"))

    out = [
        "",
        "## The three numbers",
        "",
        "**Retention across the latent -> counts step** — what the emission costs, against what",
        "the tissue's own sampling noise costs.",
        "",
        "⚠️ **The retention column is OVERSTATED and is not a like-for-like ratio.** Its numerator",
        "is a **rank-normalised** count stage and its denominator a **raw** latent stage; rank-",
        "normalising a heavy-tailed field raises its Moran's I, so the denominator is too small.",
        "Both arms carry the same bias, so the *comparison* between them stands and the",
        "*percentages* do not (`reports/ceiling_review.md` §2).",
        "",
        "| arm | counts I | latent I | retention | slope | tissue slope |",
        "|---|---|---|---|---|---|",
    ]
    real_slope = mean_variance_slope(real)
    out.append(
        f"| **real tissue** | {real_counts:+.4f} | {real_latent:+.4f} | "
        f"**{real_retention:.1%}** | {real_slope:.3f} | — |"
    )
    for label, stage in (
        ("uncalibrated", "4. sampled counts (rank-normalised)"),
        ("calibrated", "4c. sampled counts, CALIBRATED (rank-norm)"),
    ):
        if label not in emitted:
            continue
        ci = by.get(stage, float("nan"))
        counts = emitted[label]
        counts = counts if panel is None else counts[:, panel]
        out.append(
            f"| {label} | {ci:+.4f} | {latent:+.4f} | **{ci / latent:.1%}** | "
            f"{mean_variance_slope(counts):.3f} | {real_slope:.3f} |"
        )
    return out


def _raises(fn) -> bool:
    """True when ``fn()`` raises. For asserting that a guard actually guards."""
    try:
        fn()
    except Exception:
        return True
    return False


def _self_check() -> int:
    """Assert the panel and density logic on synthetic fields. Seconds, no fit, no data.

    These two flags decide *what is compared* rather than what is measured, so a defect in
    either silently changes every number in the report rather than failing. Run this on the
    campaign machine before spending an hour of fit time on the flags.
    """
    checks: list[tuple[str, bool]] = []

    rng = np.random.default_rng(0)
    xy = rng.random((300, 2)) * 100.0
    smooth = np.sin(xy[:, 0] / 20.0)
    vals = np.column_stack(
        [smooth * (g / 9.0) + rng.normal(size=300) * (1.0 - g / 9.0) for g in range(10)]
    )
    i = morans_i(xy, vals, 8)
    panel = top_k_panel(xy, vals, 8, 4)
    checks += [
        ("top_k_panel returns exactly top_k indices", panel.shape == (4,)),
        ("top_k_panel indices are sorted ascending", list(panel) == sorted(panel)),
        ("top_k_panel picks the highest-I columns", set(panel) == set(np.argsort(-i)[:4])),
        ("top_k_panel is deterministic", np.array_equal(panel, top_k_panel(xy, vals, 8, 4))),
        (
            "top_k >= n_columns keeps every column",
            list(top_k_panel(xy, vals, 8, 99)) == list(range(10)),
        ),
    ]

    with_const = np.column_stack([vals, np.zeros(300)])
    checks += [
        (
            "a constant column loses to every finite one",
            10 not in set(top_k_panel(xy, with_const, 8, 10)),
        ),
        (
            "and is taken only when nothing else is left",
            10 in set(top_k_panel(xy, with_const, 8, 11)),
        ),
        (
            "ties break by column index",
            list(top_k_panel(xy, np.column_stack([vals[:, 9], vals[:, 9], vals[:, 0]]), 8, 2))
            == [0, 1],
        ),
    ]

    sel = np.array([1, 4, 7])
    checks += [
        (
            "rank_normalize commutes with the panel, so the panel does not move a stage's own I",
            bool(
                np.allclose(rank_normalize(vals)[:, sel], rank_normalize(vals[:, sel]))
                and np.allclose(
                    morans_i(xy, rank_normalize(vals), 8)[sel],
                    morans_i(xy, rank_normalize(vals[:, sel]), 8),
                )
            ),
        ),
    ]

    keep = density_subsample(1000, 250, 1)
    checks += [
        ("density_subsample returns n_target rows", keep.shape == (250,)),
        ("its indices are sorted and unique", list(keep) == sorted(set(keep.tolist()))),
        ("it is deterministic in the seed", np.array_equal(keep, density_subsample(1000, 250, 1))),
        (
            "a different seed draws differently",
            not np.array_equal(keep, density_subsample(1000, 250, 2)),
        ),
        ("it never upsamples", list(density_subsample(50, 90, 1)) == list(range(50))),
    ]

    dense = rng.random((4000, 2)) * 100.0
    field = np.sin(dense[:, 0] / 20.0)[:, None] + rng.normal(size=(4000, 1)) * 2.0
    thin = density_subsample(4000, 400, 3)
    i_dense, i_thin = (
        float(morans_i(dense, field, 8)[0]),
        float(morans_i(dense[thin], field[thin], 8)[0]),
    )
    checks.append(
        (
            f"density moves I on one unchanged field ({i_dense:.4f} -> {i_thin:.4f}), "
            "which is why --match-density exists",
            abs(i_dense - i_thin) > 0.02,
        )
    )

    counts = np.abs(rng.normal(size=(600, 5))) * np.array([0.05, 0.5, 1.0, 5.0, 50.0])
    cxy = rng.random((600, 2)) * 100.0
    smoothed = knn_mean_field(cxy, counts, 10)
    noisy = counts[:, 2:3] + np.sin(cxy[:, 0:1] / 20.0)
    i_raw = float(morans_i(cxy, noisy, 8)[0])
    i_sm = float(morans_i(cxy, knn_mean_field(cxy, noisy, 10), 8)[0])
    checks += [
        ("knn_mean_field preserves shape", smoothed.shape == counts.shape),
        (
            "it is row-stochastic, so per-column means are preserved",
            bool(np.allclose(smoothed.mean(axis=0), counts.mean(axis=0), rtol=0.05, atol=1e-9)),
        ),
        ("it keeps non-negative input non-negative", bool(smoothed.min() >= 0.0)),
        ("it is deterministic", bool(np.array_equal(smoothed, knn_mean_field(cxy, counts, 10)))),
        (
            f"it RAISES Moran's I ({i_raw:.4f} -> {i_sm:.4f}), which is why every oracle-mu arm "
            "reads as an upper bound",
            i_sm > i_raw,
        ),
    ]

    # N2's two estimators, against a lognormal mu whose sd(log mu) is known exactly.
    sigma = 0.8
    mu_true = np.exp(rng.normal(0.0, sigma, size=(20000, 4))) * np.array([1.0, 5.0, 20.0, 100.0])
    y_pois = rng.poisson(mu_true).astype(np.float64)
    y_over = rng.negative_binomial(6.0, 6.0 / (6.0 + mu_true)).astype(np.float64)
    sd_field = float(np.median(log_mu_sd_from_field(mu_true)))
    sd_counts = float(np.median(log_mu_sd_from_counts(y_pois)))
    sd_over = float(np.median(log_mu_sd_from_counts(y_over)))
    smooth_field = knn_mean_field(cxy, np.exp(rng.normal(0.0, sigma, size=(600, 5))), 10)
    checks += [
        (
            f"log_mu_sd_from_field recovers a known sd(log mu) ({sd_field:.3f} vs {sigma})",
            abs(sd_field - sigma) < 0.05,
        ),
        (
            f"log_mu_sd_from_counts deconvolves the Poisson term ({sd_counts:.3f} vs {sigma})",
            abs(sd_counts - sigma) < 0.10,
        ),
        (
            f"it counts over-dispersion as signal, so it over-states ({sd_over:.3f} > {sigma})",
            sd_over > sigma,
        ),
        (
            "smoothing shrinks the field estimate, so mu_oracle is a LOWER bound "
            f"({float(np.median(log_mu_sd_from_field(smooth_field))):.3f} < {sigma})",
            float(np.median(log_mu_sd_from_field(smooth_field))) < sigma,
        ),
        (
            "both estimators are NaN, not zero, for a gene with no counts",
            bool(np.isnan(log_mu_sd_from_counts(np.zeros((10, 1)))[0])),
        ),
    ]

    # N5's decision rule, against the four outcomes a1_escalation_preregistration.md §2 names.
    n5_cases = [
        ("over-dispersion", 0.50, 0.17, 0.20, 0.48, "**over-dispersion**"),
        ("dropout", 0.50, 0.17, 0.48, 0.20, "**dropout**"),
        ("both, additively", 0.50, 0.20, 0.35, 0.35, "**both, additively**"),
        ("interaction", 0.50, 0.20, 0.45, 0.45, "**not decomposable** — the two interact"),
    ]
    for name, i_c, i_b, i_t, i_p in [(c[0], *c[1:5]) for c in n5_cases]:
        want = next(c[5] for c in n5_cases if c[0] == name)
        got = n5_verdict(i_c, i_b, i_t, i_p)
        checks.append((f"n5_verdict reads the {name} case as written", got["verdict"] == want))
    lopsided = n5_verdict(0.50, 0.17, 0.20, 0.48)
    checks += [
        (
            "its shares are of the total loss, not of I",
            abs(lopsided["share_theta"] - (0.50 - 0.20) / (0.50 - 0.17)) < 1e-12,
        ),
        (
            "and the additivity gap is reported even when the verdict does not use it",
            abs(lopsided["additivity_gap"] - abs(0.30 + 0.02 - 0.33)) < 1e-12,
        ),
    ]

    # Q1: summarise must produce BOTH transforms and must rank its own input, or the ratios go
    # back to crossing transforms. Q1.5: the blocked estimator must equal the direct one exactly,
    # and morans_agreement must reproduce bench3's _agreement construction.
    sm_raw = summarise("x", cxy, counts, 10, primary="raw")
    sm_rank = summarise("x", cxy, counts, 10, primary="rank")
    direct_rank = float(np.median(morans_i(cxy, rank_normalize(counts), 10)))
    checks += [
        ("summarise reports both transforms", {"median_I_raw", "median_I_rank"} <= set(sm_raw)),
        (
            "primary='raw' puts the raw figure in median_I",
            sm_raw["median_I"] == sm_raw["median_I_raw"],
        ),
        (
            "primary='rank' puts the ranked one there",
            sm_rank["median_I"] == sm_rank["median_I_rank"],
        ),
        (
            "the two differ, which is why the mixed ratios were wrong",
            abs(sm_raw["median_I_raw"] - sm_raw["median_I_rank"]) > 1e-9,
        ),
        (
            "summarise ranks its own input rather than trusting the caller",
            abs(sm_raw["median_I_rank"] - direct_rank) < 1e-12,
        ),
        (
            "it carries the per-gene ranked vector the agreement statistic needs",
            sm_raw["per_gene_I_rank"].shape == (counts.shape[1],),
        ),
        (
            "it refuses an unknown primary",
            _raises(lambda: summarise("x", cxy, counts, 10, primary="?")),
        ),
    ]
    blocked = morans_i_ranked_blocked(cxy, counts, 10, block=2)
    checks.append(
        (
            "morans_i_ranked_blocked equals the direct call exactly, it only bounds memory",
            bool(np.allclose(blocked, morans_i(cxy, rank_normalize(counts), 10), equal_nan=True)),
        )
    )

    perfect = morans_agreement(np.array([0.1, 0.4, 0.7, 0.9]), np.array([0.1, 0.4, 0.7, 0.9]))
    shifted = morans_agreement(np.array([0.5, 0.8, 1.1, 1.3]), np.array([0.1, 0.4, 0.7, 0.9]))
    withnan = morans_agreement(np.array([0.1, np.nan, 0.7, 0.9]), np.array([0.1, 0.4, 0.7, np.nan]))
    flat = morans_agreement(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.1, 0.4, 0.7, 0.9]))
    checks += [
        ("morans_agreement is 1.0 on an identical vector", abs(perfect["pearson"] - 1.0) < 1e-12),
        ("and 0.0 mae there", perfect["mae"] == 0.0),
        (
            "a constant offset keeps pearson at 1.0 while mae rises — the ranking/level split "
            "that is the whole point of Q1.5",
            abs(shifted["pearson"] - 1.0) < 1e-12 and abs(shifted["mae"] - 0.4) < 1e-12,
        ),
        ("NaN genes are dropped PAIRWISE, as the evaluator does", withnan["n_genes"] == 2),
        ("fewer than 3 survivors leaves pearson NaN", np.isnan(withnan["pearson"])),
        ("a zero-variance vector leaves pearson NaN", np.isnan(flat["pearson"])),
    ]

    # The ladder's null rung and R1-R3's controls. A partial correlation whose control is chosen
    # after the answer is not a control, so the specification is asserted here as well as fixed in
    # the pre-registration.
    rng2 = np.random.default_rng(7)
    d_ctrl = rng2.uniform(0.02, 0.9, size=200)
    common = 3.0 * d_ctrl  # both vectors driven ONLY by the control
    x_sparse, y_sparse = common + rng2.normal(0, 0.05, 200), common + rng2.normal(0, 0.05, 200)
    shared = rng2.normal(0, 1, 200)  # and a version with genuine shared signal
    x_spat, y_spat = common + shared, common + shared + rng2.normal(0, 0.05, 200)
    r_sparse = float(np.corrcoef(x_sparse, y_sparse)[0, 1])
    r_spat = float(np.corrcoef(x_spat, y_spat)[0, 1])
    p_sparse = partial_correlation(x_sparse, y_sparse, d_ctrl)
    p_spat = partial_correlation(x_spat, y_spat, d_ctrl)
    checks += [
        (
            f"two vectors driven only by the control correlate highly ({r_sparse:.3f}) and the "
            f"partial correlation removes it ({p_sparse:.3f}) — R3's whole point",
            r_sparse > 0.9 and abs(p_sparse) < 0.4,
        ),
        (
            f"genuine shared signal survives the same control ({r_spat:.3f} -> {p_spat:.3f})",
            p_spat > 0.9,
        ),
        (
            "partial_correlation returns NaN rather than a number when a residual is constant",
            bool(np.isnan(partial_correlation(d_ctrl, y_spat, d_ctrl))),
        ),
        (
            "gene_detection is the fraction of NON-ZERO cells",
            float(gene_detection(np.array([[0.0, 1.0], [2.0, 0.0], [0.0, 0.0], [1.0, 3.0]]))[0])
            == 0.5,
        ),
    ]
    agree = decomposition_of_r(x_spat, y_spat, np.ones((4, 200)), r4=r_spat)
    checks.append(
        (
            "decomposition_of_r drops the reading when its two control specifications disagree",
            "UNINFORMATIVE" in agree["verdict"] or agree["spec_disagreement"] <= 0.15,
        )
    )
    low = decomposition_of_r(x_spat, y_spat, np.ones((4, 200)), r4=0.05)
    checks.append(
        (
            "and when r(4) is below 0.20, where the retained fraction is unstable",
            "UNINFORMATIVE" in low["verdict"],
        )
    )

    # The report block that explains a model sitting above the tissue. Report-generating code
    # that silently produces nothing is what §4.2k is about, so both branches are asserted.
    def stage(name, value):
        return {"stage": name, "median_I": value, "p25": 0.0, "p75": 0.0, "n_channels": 1}

    tier1_like = [
        stage("2. latent h after the flow", 0.8011),
        stage("4. sampled counts (rank-normalised)", 0.5134),
        stage("4p. counts ~ Poisson(mu) — emission noise removed", 0.7775),
        stage("REF real counts (rank-normalised)", 0.4635),
        stage("REF real latent h1 = encoder(real counts)", 0.6253),
    ]
    spread_rows = [
        {"quantity": "real counts (Poisson-deconvolved) — UPPER bound", "sd_log_mu": 0.9438},
        {"quantity": "mu_oracle (kNN mean field) — LOWER bound", "sd_log_mu": 0.7165},
    ]
    above = "\n".join(cancelling_defects_block(tier1_like, {"sd_log_mu": 0.6728}, spread_rows))
    below = "\n".join(
        cancelling_defects_block(
            [r for r in tier1_like if not r["stage"].startswith("4.")]
            + [stage("4. sampled counts (rank-normalised)", 0.1154)],
            {"sd_log_mu": 0.6728},
            spread_rows,
        )
    )
    def _rung(arm: str, r: float) -> dict:
        return {
            "arm": arm,
            "median_r": r,
            "min_r": r - 0.01,
            "max_r": r + 0.01,
            "n_seeds": 3,
            "spearman": r,
            "mae": 0.10,
            "n_genes": 28,
        }

    clean_ladder = {
        "panel::A1c": _rung("A1c. Poisson(mu_oracle)", 0.94),
        "panel::A1a": _rung("A1a. ZINB(decode(h1))", 0.61),
        "panel::A1n": _rung("A1n. permutation null", 0.02),
    }
    dirty_ladder = dict(clean_ladder)
    dirty_ladder["panel::A1n"] = _rung("A1n. permutation null", 0.42)
    ag4 = {"panel::4. sampled counts": {"pearson": 0.51, "spearman": 0.5, "mae": 0.1, "n_genes": 28}}
    clean_txt = "\n".join(ladder_block(clean_ladder, {}))
    with4 = "\n".join(ladder_block(clean_ladder, {}, ag4))
    dirty_txt = "\n".join(ladder_block(dirty_ladder, {}))
    spars = {
        "R1_corr_Ireal_detection": 0.5,
        "R1_corr_Ireal_logmean": 0.5,
        "R2_corr_Ipred_detection": 0.4,
        "R2_corr_Ipred_logmean": 0.4,
        "R3_partial_detection": 0.3,
        "R3_partial_logmean": 0.3,
        "r4": 0.5,
        "retained_detection": 0.6,
        "retained_logmean": 0.6,
        "spec_disagreement": 0.0,
        "verdict": "SPATIAL",
    }
    checks += [
        ("ladder_block returns nothing when the ladder is empty", ladder_block({}, {}) == []),
        ("it renders every rung it was given", clean_txt.count("| panel |") == 3),
        (
            "a rung label containing '|' is escaped, so the markdown table does not split",
            all(
                "\\|" in row and row.replace("\\|", "").count("|") == 8
                for row in "\n".join(
                    ladder_block({"panel::A1a": _rung("A1a. emission(mu | h1)", 0.6)}, {})
                ).splitlines()
                if row.startswith("| panel |")
            ),
        ),
        (
            "it folds stage 4 INTO the ladder, so the A1a-to-4 distance is on one table",
            "where we are" in with4 and with4.count("| panel |") == 4,
        ),
        (
            "the rungs are ordered ceiling -> emission -> perfect latent -> where we are -> null",
            [seg for seg in ("A1c", "A1a", "where we are", "A1n") if seg in with4]
            == sorted(("A1c", "A1a", "where we are", "A1n"), key=with4.index),
        ),
        (
            "stage 4 is NOT invented for a scope the ladder has no rungs in",
            "where we are" not in "\n".join(ladder_block({}, {}, ag4)),
        ),
        (
            "it quotes the copy floor, not SpatialZ, as the target",
            "0.9836" in clean_txt and "0.5574" in clean_txt,
        ),
        ("it says a LOW rung is conclusive and a high one permissive", "permissive" in clean_txt),
        ("the ladder table itself no longer carries the alarm", "NULL RUNG" not in dirty_txt),
        ("R1-R3 are omitted when the decomposition was not computed", "R1" not in clean_txt),
        (
            "and rendered under BOTH control specifications when it was",
            "\n".join(ladder_block(clean_ladder, spars)).count("| detection rate |") == 1,
        ),
        (
            "the R1-R3 block carries its own verdict and the disagreement it rests on",
            "SPATIAL" in "\n".join(ladder_block(clean_ladder, spars)),
        ),
    ]

    bare = cancelling_defects_block([r for r in tier1_like if "4p." not in r["stage"]][:2], {}, [])
    checks += [
        ("the cancelling-defects block fires when the model is above the tissue", "ABOVE" in above),
        ("it quotes the latent's smoothness ratio", "1.28x smoother" in above),
        ("it quotes mu against the tissue's bracket", "[0.7165, 0.9438]" in above),
        ("it always reports stage 4p's ceiling", "emission-free ceiling" in above),
        ("it does NOT claim cancellation when the model is below the tissue", "ABOVE" not in below),
        ("but still reports the ceiling there", "emission-free ceiling" in below),
        ("and returns nothing when the run has no real reference", bare == []),
    ]

    # Every markdown row this script emits must have the same cell count as its header, whatever
    # is in the labels. This walks the rendered blocks rather than the format strings, so a new
    # renderer that forgets md_cell is caught by the output, not by a reviewer.
    def _table_is_square(text: str) -> bool:
        header_cells = None
        for row in text.splitlines():
            if not row.startswith("|"):
                header_cells = None
                continue
            n = row.replace("\\|", "").count("|")
            if header_cells is None:
                header_cells = n
            elif n != header_cells:
                return False
        return True

    piped = {
        "panel::A1a": _rung("A1a. counts ~ emission(mu | h1)", 0.6),
        "panel::A1c": _rung("A1c. counts ~ Poisson(mu_oracle)", 0.9),
    }
    checks += [
        ("md_cell escapes a pipe", md_cell("a | b") == "a \\| b"),
        ("and leaves a clean label alone", md_cell("A1c. Poisson") == "A1c. Poisson"),
        (
            "every rendered ladder row has the header's cell count, pipes in labels or not",
            _table_is_square("\n".join(ladder_block(piped, {}, ag4))),
        ),
        (
            "the squareness check would FAIL on an unescaped pipe, so it is not vacuous",
            not _table_is_square("| a | b |\n|---|---|\n| x | y | z |"),
        ),
    ]

    # --- the corrected null check and the bootstrap (null_band_preregistration.md) -----------
    rng_c = np.random.default_rng(7)
    centred = list(rng_c.normal(0.0, 0.20, size=20))
    offset = [r + 0.30 for r in centred]
    nt_c, nt_o = null_centre_test(centred), null_centre_test(offset)
    null_txt = "\n".join(_null_block({"panel": nt_o, "all genes": nt_c}))
    checks += [
        ("null_centre_test does NOT fire on a wide but centred null", not nt_c["fired"]),
        ("it DOES fire on a null shifted off zero by less than its own sd", nt_o["fired"]),
        (
            "so the test is on the CENTRE, not on whether one draw is small",
            nt_c["sd_r"] > 0.15 and abs(nt_c["mean_r"]) < 0.15,
        ),
        ("it reports the empirical interval, not a formula", nt_c["p2.5"] < nt_c["p97.5"]),
        ("two seeds are too few to test a centre", null_centre_test([0.9, 0.9])["n_seeds"] == 2),
        ("one seed is refused outright", null_centre_test([0.9])["underpowered"]),
        (
            "the alarm names the scope that fired and only that scope",
            "NOT CENTRED on panel" in null_txt and "all genes" not in null_txt.split("🚨")[-1],
        ),
        (
            "both scopes are rendered, so a passing one is not hidden by a firing one",
            null_txt.count("| panel |") == 1 and null_txt.count("| all genes |") == 1,
        ),
    ]

    g = 200
    base_v = rng_c.normal(size=g)
    ref_v = base_v + rng_c.normal(0, 0.5, size=g)
    vecs = {
        "hi": base_v + rng_c.normal(0, 0.2, size=g),
        "lo": base_v + rng_c.normal(0, 2.0, size=g),
        "same": base_v + rng_c.normal(0, 0.5, size=g),
    }
    bt = paired_gene_bootstrap(vecs, ref_v, [("hi", "lo"), ("same", "same")], n_boot=400)
    by = {row["pair"]: row for row in bt}
    nan_vecs = {"a": np.r_[vecs["hi"][:5], np.full(g - 5, np.nan)], "b": vecs["lo"]}
    checks += [
        (
            "the bootstrap separates a clearly better rung from a worse one",
            by["hi - lo"]["distinguishable"] and by["hi - lo"]["diff"] > 0,
        ),
        (
            "and a rung against ITSELF is never distinguishable — the check is not vacuous",
            not by["same - same"]["distinguishable"] and by["same - same"]["diff"] == 0.0,
        ),
        (
            "it drops genes not finite in every rung, so one index means the same everywhere",
            paired_gene_bootstrap(nan_vecs, ref_v, [("a", "b")], n_boot=50)[0].get("unusable")
            is True,
        ),
        (
            "it is deterministic — two runs at the same seed agree exactly",
            paired_gene_bootstrap(vecs, ref_v, [("hi", "lo")], n_boot=200)[0]["lo"]
            == paired_gene_bootstrap(vecs, ref_v, [("hi", "lo")], n_boot=200)[0]["lo"],
        ),
    ]

    # --- the permutation null's two identities, against the direct route ----------------------
    xy_p = rng_c.random((60, 2)) * 100.0
    cts_p = rng_c.poisson(3.0, size=(60, 5)).astype(np.float64)
    idx_p = knn_index(xy_p, 6)
    direct = morans_i(xy_p, rank_normalize(cts_p), 6)
    perm = np.random.default_rng(3).permutation(60)
    checks += [
        (
            "morans_i_from_index equals morans_i exactly — an optimisation, not a 2nd estimator",
            np.allclose(morans_i_from_index(idx_p, rank_normalize(cts_p)), direct, atol=0, rtol=0),
        ),
        (
            "ranking is permutation-equivariant, which is what makes 20 seeds affordable",
            np.array_equal(rank_normalize(cts_p[perm]), rank_normalize(cts_p)[perm]),
        ),
        (
            "permutation_null_I matches the direct permute-then-rank-then-measure route",
            np.allclose(
                permutation_null_I(xy_p, cts_p, 6, [3])[0],
                morans_i(xy_p, rank_normalize(cts_p[perm]), 6),
            ),
        ),
        (
            "and it is centred on zero, which is the property the check tests",
            abs(float(np.mean(permutation_null_I(xy_p, cts_p, 6, range(101, 121))))) < 0.10,
        ),
    ]

    # --- F3 and the flanking verdict ordering -------------------------------------------------
    ctrl_f = np.arange(g, dtype=np.float64)
    abundance_only = ctrl_f + rng_c.normal(0, 0.5, size=g)
    gene_specific = rng_c.normal(size=g)
    f3_ab = stratified_relabel_r(abundance_only, ctrl_f, ctrl_f, 10, range(101, 111))
    f3_gs = stratified_relabel_r(gene_specific, gene_specific, ctrl_f, 10, range(101, 111))
    checks += [
        (
            f"F3 SURVIVES relabelling when the agreement is abundance-driven "
            f"({f3_ab['median_r']:+.3f})",
            f3_ab["median_r"] > 0.9,
        ),
        (
            f"F3 COLLAPSES when the agreement is gene-specific ({f3_gs['median_r']:+.3f}) — "
            "which is the whole point of the instrument",
            abs(f3_gs["median_r"]) < 0.3,
        ),
        (
            "a wider stratum destroys more, so the width is a real degree of freedom",
            stratified_relabel_r(gene_specific, gene_specific, ctrl_f, 50, range(101, 111))[
                "median_r"
            ]
            <= f3_gs["median_r"] + 0.2,
        ),
    ]

    def _fk(r, retained, f3r, s4=0.40):
        return {
            "r_flank": r,
            "stage4_r": s4,
            "f3": [{"width": 10, "median_r": f3r}],
            "R3": {
                "controls": ["detection", "logmean", "logvar"],
                "verdict": "PARTIAL",
                **{f"retained_{n}": retained for n in ("detection", "logmean", "logvar")},
            },
        }

    unin = _fk(0.9, 0.1, 0.8)
    unin["R3"]["verdict"] = "UNINFORMATIVE — the control specifications disagree"
    checks += [
        (
            "flanking verdict: a copy that does not beat stage 4 returns PREMISE ABSENT",
            "PREMISE ABSENT" in _flanking_verdict(_fk(0.30, 0.9, 0.1)),
        ),
        (
            "a floor that survives every control is outcome 1 and blocks any benchmark claim",
            "1. FLOOR IS GENUINE" in _flanking_verdict(_fk(0.9, 0.8, 0.1)),
        ),
        (
            "R3 saying abundance WITHOUT F3 corroborating is outcome 3, not outcome 2",
            "3. PARTIAL" in _flanking_verdict(_fk(0.9, 0.1, 0.05)),
        ),
        (
            "outcome 2 needs BOTH, and says in the verdict that it is the favourable one",
            "2. FLOOR IS ABUNDANCE" in _flanking_verdict(_fk(0.9, 0.1, 0.8))
            and "SUITS US" in _flanking_verdict(_fk(0.9, 0.1, 0.8)),
        ),
        (
            "UNINFORMATIVE resolves to outcome 1's reading, never to 'unclear'",
            "COPY FLOOR STANDS" in _flanking_verdict(unin),
        ),
        (
            "a middling retained fraction also resolves to outcome 1's reading",
            "COPY FLOOR STANDS" in _flanking_verdict(_fk(0.9, 0.45, 0.8)),
        ),
    ]

    checks += [
        ("the sidecar handler converts a numpy scalar", _json_scalar(np.float64(1.5)) == 1.5),
        ("and a numpy bool, which json also refuses", _json_scalar(np.bool_(True)) is True),
        (
            "but REFUSES an array rather than writing a string into a results file",
            _raises(lambda: _json_scalar(np.arange(3))),
        ),
        (
            "and the whole sidecar shape round-trips through it",
            json.loads(
                json.dumps(
                    {"n": nt_c, "b": bt, "f": f3_ab, "s": [{"seed": np.int64(1)}]},
                    default=_json_scalar,
                )
            )["s"][0]["seed"]
            == 1,
        ),
    ]

    s4_txt = "\n".join(
        stage4_seed_block(
            [
                {"seed": 1, "n_used": 10, "median_I": 0.1, "r_panel": 0.50, "r_all": 0.70},
                {"seed": 2, "n_used": 10, "median_I": 0.1, "r_panel": 0.55, "r_all": 0.74},
            ]
        )
    )
    checks += [
        ("stage4_seed_block returns nothing with no seeds", stage4_seed_block([]) == []),
        ("it reports the span across whole generations", "+0.5000 .. +0.5500" in s4_txt),
        (
            "and says a smaller rung difference is not resolved by one generation",
            "not resolved by a single generation" in s4_txt,
        ),
        (
            "it distinguishes pipeline spread from the arms' emission-only spread",
            "emission noise alone" in s4_txt,
        ),
    ]

    seed_rows = [
        {
            "stage": "A1b. x",
            "median_I": v_,
            "p25": 0.0,
            "p75": 0.0,
            "n_channels": 3,
            "seed": s_,
            "per_gene_I_rank": np.full(3, v_),
        }
        for s_, v_ in ((1, 0.10), (2, 0.30), (3, 0.20))
    ]
    collapsed = _over_seeds(seed_rows)
    checks += [
        ("_over_seeds reports the MEDIAN over seeds", collapsed["median_I"] == 0.20),
        ("it carries the range", (collapsed["min_I"], collapsed["max_I"]) == (0.10, 0.30)),
        ("it keeps every seed, so a straddle is visible", len(collapsed["per_seed"]) == 3),
        ("and drops the single-seed key", "seed" not in collapsed),
        (
            "it carries EVERY seed's per-gene I vector, so the ladder is not single-seed",
            len(collapsed["per_seed_I_rank"]) == 3
            and [float(v[0]) for v in collapsed["per_seed_I_rank"]] == [0.10, 0.30, 0.20],
        ),
    ]

    terms = {"var_shape": np.array([1.0, 4.0, 9.0]), "var_logsize": np.ones(3), "cov": np.zeros(3)}
    terms["total"] = terms["var_shape"] + terms["var_logsize"] + 2.0 * terms["cov"]
    d = summarise_mu_terms(terms)
    neg = {"var_shape": np.array([1.0]), "var_logsize": np.array([1.0]), "cov": np.array([-0.6])}
    neg["total"] = neg["var_shape"] + neg["var_logsize"] + 2.0 * neg["cov"]
    dn = summarise_mu_terms(neg)
    checks += [
        ("summarise_mu_terms counts the panel it was given", d["n_genes_decomposed"] == 3),
        (
            "it reports medians over genes",
            d["var_shape"] == 4.0 and abs(d["sd_log_mu"] - 5.0**0.5) < 1e-12,
        ),
        (
            "share_shape is the median ratio, not the ratio of medians",
            abs(d["share_shape"] - 0.8) < 1e-12,
        ),
        (
            f"a negative covariance pushes the unbounded share past 1 ({dn['share_shape']:.3f})",
            dn["share_shape"] > 1.0,
        ),
        (
            f"the bounded share stays in [0, 1] ({dn['share_shape_bounded']:.3f})",
            0.0 <= dn["share_shape_bounded"] <= 1.0,
        ),
    ]

    failed = 0
    for label, passed in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {label}")
        failed += 0 if passed else 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="assert the panel and density logic on synthetic fields and exit. No fit, no data, "
        "seconds. --match-density and --top-k-by decide what is compared rather than what is "
        "measured, so a defect in either changes every number here without failing.",
    )
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--section", default="section_2")
    ap.add_argument(
        "--target-z",
        type=float,
        default=None,
        help="depth of the generated plane, in um. Default: **the named section's own z**, read "
        "from the ground truth. The old default was tier-1's 30.0 for every dataset, which put "
        "the deep_starmap run's plane 4.9 um from the section it was compared against.",
    )
    ap.add_argument("--out", default="reports/chain_diagnostic.md")
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="also fit T09 §2's pi / log-theta calibration and re-measure the emission stages",
    )
    ap.add_argument(
        "--decoder-mu-link",
        default=None,
        choices=["softplus", "exp"],
        help="override Config.decoder_mu_link (T10 candidate 1: softplus compresses dynamic "
        "range at means in the thousands, which is STARmap's regime and not the fixture's)",
    )
    ap.add_argument(
        "--save-model",
        default=None,
        help="torch.save the fitted model here, so a later analysis needs no refit. Written "
        "immediately after the fit, before the chain is measured: an hour of training must "
        "not be lost to a failure in an analysis stage downstream of it.",
    )
    ap.add_argument(
        "--fit-checkpoint",
        default=None,
        help="train_ctfflow(checkpoint=...): resume an interrupted fit from here. Safe to pass "
        "on a first run — it is written every Config.checkpoint_every_n_steps and read only if "
        "it exists and matches this config, seed and budget.",
    )
    ap.add_argument(
        "--load-model",
        default=None,
        help="read a model saved by --save-model and skip the fit entirely. The CONFIG comes "
        "from the checkpoint, not from this command line, so a measurement on a saved fit is a "
        "measurement of the arm that was fitted. --text-emb-mode / --expr-pca-dim / "
        "--decoder-mu-link are refused with it, and --steps is ignored.",
    )
    ap.add_argument(
        "--theta-floor",
        type=float,
        default=None,
        help="fit under Config.decoder_theta_floor: a lower bound on the ZINB theta, i.e. an "
        "upper bound on over-dispersion. M3's instrument. Refused with --load-model, whose "
        "config comes from the checkpoint — except with --report-theta, where it names the "
        "candidate floor to measure the binding fraction of rather than to fit under.",
    )
    ap.add_argument(
        "--report-theta",
        action="store_true",
        help="report the decoder's learned theta at the real cells — percentiles, and with "
        "--theta-floor the fraction of (cell, gene) pairs that floor would bind on. Combine "
        "with --load-model to measure a baseline before M3's floor is chosen.",
    )
    ap.add_argument(
        "--emission-ablation",
        action="store_true",
        help="A1: draw counts at the REAL cells from three mean fields — the model's decode of "
        "encoder(real counts), the tissue's own kNN mean field through the model's theta/pi, and "
        "the same field through a bare Poisson draw — and report each against the real section. "
        "Pre-registered in reports/a1_preregistration.md; read that before reading the numbers.",
    )
    ap.add_argument(
        "--ablation-seed",
        type=int,
        nargs="+",
        default=[SEED],
        help="generator seed(s) for --emission-ablation's draws (Convention 3). Several redraw "
        "every sampled arm once per seed; the band is read from the median and an arm whose "
        "seeds straddle a boundary reads UNRESOLVED (a1_escalation_preregistration.md §1).",
    )
    ap.add_argument(
        "--null-seed",
        type=int,
        nargs="+",
        default=list(range(101, 121)),
        help="permutation seeds for the ladder's null arm. Twenty by default: the check is on "
        "whether the null is CENTRED, and one draw cannot test a centre "
        "(reports/null_band_preregistration.md §2a).",
    )
    ap.add_argument(
        "--boot-reps",
        type=int,
        default=2000,
        help="replicates for the paired gene bootstrap on rung differences (§2b). The verdict "
        "leans on r(4) - r(A1a), so that difference gets an interval rather than a sign.",
    )
    ap.add_argument(
        "--stage4-seeds",
        type=int,
        nargs="+",
        default=None,
        help="regenerate the whole section under each seed and report stage 4's scored r per "
        "seed. Stage 4 is otherwise a SINGLE generation, and its variance includes the layout "
        "and the flow sample, not only the emission draw -- it is the rung the ladder's verdict "
        "leans on. Costs one full generation per seed.",
    )
    ap.add_argument(
        "--flanking-copy",
        action="store_true",
        help="score the bench3 `flanking_copy` probe -- the nearest TRAINING section by z, "
        "emitted verbatim -- on the same statistic, plus F3, R3 under three controls and the "
        "spatial_scramble positive control. Read reports/flanking_copy_preregistration.md "
        "first: §1 states the interest this project has in one of the two outcomes.",
    )
    ap.add_argument(
        "--fit-only",
        action="store_true",
        help="fit, save, and stop — skip the chain measurement. For producing the checkpoint "
        "the layout-mode arms re-score.",
    )
    ap.add_argument(
        "--layout-sampler",
        default=None,
        choices=["grid", "rejection"],
        help="override Config.layout_sampler for the generated chain. Generation-time only: "
        "the fit does not draw positions, so this does not change the weights.",
    )
    ap.add_argument(
        "--text-emb-mode",
        default=None,
        choices=["medcpt", "lookup"],
        help="fit with a LIVE text channel at this mode, through the panel's gene-metadata "
        "table. Omitted (the default) means zero text vectors, which is what every artifact "
        "under reports/chain_* was measured with and is NEITHER A3 arm — see build_embeddings. "
        "The shipped configuration is medcpt.",
    )
    ap.add_argument(
        "--expr-pca-dim",
        type=int,
        default=None,
        help="override Config.expr_pca_dim (default: the pilot's 16, which this script was "
        "pinned at). The clamp rule gives 28 on tier-1 STARmap and 32 on deep_starmap.",
    )
    ap.add_argument(
        "--match-density",
        action="store_true",
        help="subsample the generated cells to the real section's cell count before building "
        "the kNN graph, so the two sides' Moran's I is the same estimator. Off by default.",
    )
    ap.add_argument(
        "--density-seed",
        type=int,
        default=SEED,
        help="generator seed for --match-density (Convention 3). Default: the run seed.",
    )
    ap.add_argument(
        "--top-k-by",
        default="all",
        choices=["all", "real", "model"],
        help="restrict the gene-space stages to the --top-k genes with the highest Moran's I, "
        "ranked on the REAL section ('real') or on the model's own emitted counts ('model'). "
        "'model' is the flattering selection and exists to measure how much it flatters; do "
        "not quote it. Default 'all' — every gene, as before.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="panel size for --top-k-by (ignored when it is 'all')",
    )
    add_path_args(ap)
    args = ap.parse_args(argv)

    if args.self_check:
        return _self_check()

    # Before path resolution: this is a pure argument check, and it should fail on the command
    # line rather than after a missing-file error has sent the operator looking somewhere else.
    if args.load_model:
        refused = [
            name
            for name, value in (
                ("--text-emb-mode", args.text_emb_mode),
                ("--expr-pca-dim", args.expr_pca_dim),
                ("--decoder-mu-link", args.decoder_mu_link),
                # --theta-floor shapes the fit, so it is refused with a checkpoint — unless
                # --report-theta, where it is only the candidate value being measured against.
                ("--theta-floor", None if args.report_theta else args.theta_floor),
            )
            if value is not None
        ]
        if refused:
            raise SystemExit(
                f"--load-model was given with {', '.join(refused)}. Those three shape the FIT, "
                "and the weights being loaded were fitted under the checkpoint's values; "
                "honouring the flags would report one arm's config over another arm's weights "
                "(specs/10 §4.2a-ii). Drop them — the checkpoint supplies its own config — or "
                "refit. --layout-sampler is allowed because generation does not change the fit."
            )
        if args.save_model:
            raise SystemExit("--save-model with --load-model would rewrite the checkpoint it read.")

    global INPUT, GROUND_TRUTH
    paths = resolve(args)
    INPUT, GROUND_TRUTH = str(paths.input), str(paths.ground_truth)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")

    cfg = Config(
        seed=SEED,
        text_emb_mode=args.text_emb_mode or "lookup",
        train_steps=args.steps,
        expr_pca_dim=16 if args.expr_pca_dim is None else int(args.expr_pca_dim),
        ell_xy=116.3,
        ell_z=132.0,
    ).replace(section_key="section", coord_key="spatial", celltype_key="cell_type", region_key=None)
    if args.decoder_mu_link is not None:
        cfg = cfg.replace(decoder_mu_link=args.decoder_mu_link)
    if args.layout_sampler is not None:
        cfg = cfg.replace(layout_sampler=args.layout_sampler)
    # specs/10 section 0's clamp, from the input's header: expr_pca_dim cannot exceed the panel
    # width, and on tier-1 STARmap (28 genes against a default of 32) that is what makes the
    # dataset fittable at all. Reading it here rather than after the volume is built is
    # deliberate -- building the volume runs the very validation the clamp exists to satisfy.
    # A no-op at the default 16, so an artifact measured before this existed is unaffected.
    # Imported here, not at module scope: _starmap_run pulls in the embeddings package, and
    # four other scripts import build_embeddings from this file for its zero-vector branch.
    from _starmap_run import clamp_config_to_input

    checkpoint = None
    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location="cpu")
        cfg = Config(**checkpoint["config"])
        if args.layout_sampler is not None:
            cfg = cfg.replace(layout_sampler=args.layout_sampler)
        print(
            f"  loaded {args.load_model}: train_steps={cfg.train_steps}, "
            f"text_emb_mode={cfg.text_emb_mode}, expr_pca_dim={cfg.expr_pca_dim}, "
            f"decoder_mu_link={cfg.decoder_mu_link} (--steps ignored)"
        )
    else:
        cfg = clamp_config_to_input(cfg, paths.input)
        if args.theta_floor is not None:
            cfg = cfg.replace(decoder_theta_floor=float(args.theta_floor))
    print(f"  decoder_theta_floor = {cfg.decoder_theta_floor}")
    live_text = args.text_emb_mode is not None
    print(f"  decoder_mu_link = {cfg.decoder_mu_link}")
    print(f"  layout_sampler  = {cfg.layout_sampler}, layout_mode = {cfg.layout_mode}")
    print(f"  expr_pca_dim    = {cfg.expr_pca_dim}")
    k = int(cfg.metric_knn_k)

    print(f"chain diagnostic: {int(cfg.train_steps)} steps, section {args.section}")
    vol = load_training_volume(cfg, paths.input)

    # Before the fit, not after: an unknown --section or a gene order that disagrees with the
    # volume's is a wrong answer, and finding it out downstream of an hour of training has
    # already cost this project a run. Loading it here also means the reference stage and the
    # verdict read the ground truth once between them instead of three times.
    real = load_real_section(
        args.section,
        paths.ground_truth,
        section_key=str(cfg.section_key),
        coord_key=str(cfg.coord_key),
        expected_genes=list(vol.gene_names),
    )
    print(
        f"  real {real.section_id}: {real.counts.shape[0]} cells x {real.counts.shape[1]} genes, "
        f"z = {real.z:.1f} (section gap {real.z_gap:.1f})"
    )
    if args.target_z is None:
        if not np.isfinite(real.z):
            raise SystemExit(
                f"--target-z was not given and {real.section_id!r} has no z: the build's "
                f"obsm[{cfg.coord_key!r}] has fewer than three columns, so the section's own "
                "plane cannot be read. Pass --target-z explicitly."
            )
        target_z = float(real.z)
        print(f"  --target-z defaulted to {real.section_id}'s own plane, z = {target_z:.1f}")
    else:
        target_z = float(args.target_z)
    z_note = ""
    if (
        np.isfinite(real.z)
        and np.isfinite(real.z_gap)
        and abs(real.z - target_z) > (0.5 * real.z_gap)
    ):
        z_note = (
            f"--target-z {target_z} is {abs(real.z - target_z):.1f} um from "
            f"{real.section_id}'s own plane at z = {real.z:.1f}, more than half the "
            f"{real.z_gap:.1f} um section gap: the generated plane and the reference section "
            "are not the same plane"
        )
        print(f"  !! {z_note}", file=sys.stderr)

    data = TrainingData.build(vol, cfg)
    if checkpoint is not None:
        # Zero text vectors here even for a medcpt checkpoint: ``text_vecs`` is a registered
        # buffer, so ``load_state_dict`` restores the fitted MedCPT vectors and this path needs
        # no encoder and no network. A shape mismatch raises — the load is strict.
        model = CTFFlow(cfg, data, build_embeddings(cfg, vol, for_checkpoint=True), grf_seed=SEED)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"  text channel: {describe_text_state(model, cfg)}", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BBoxClampWarning)
            if cfg.repulsion:
                model.repulsion = fit_repulsion(vol, cfg, seed=SEED + 1)
        print("  weights loaded; no fit run", flush=True)
    else:
        model = CTFFlow(cfg, data, build_embeddings(cfg, vol, live_text=live_text), grf_seed=SEED)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BBoxClampWarning)
            train_ctfflow(
                model, cfg, steps=int(cfg.train_steps), seed=SEED, checkpoint=args.fit_checkpoint
            )
            if cfg.repulsion:
                model.repulsion = fit_repulsion(vol, cfg, seed=SEED + 1)
        print(f"  fit: {cfg.train_steps} steps in {time.time() - t0:.1f}s", flush=True)

    # Save before anything else runs. The chain stages below generate a section, and under a
    # broken layout that is exactly where a run dies; losing the fit to it would cost the hour
    # again for nothing.
    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, args.save_model)
        print(f"  model saved to {args.save_model}", flush=True)
    if args.fit_only:
        print("  --fit-only: stopping before the chain measurement")
        return 0

    # --- the generated chain, reproducing infer/generate.py::_expression step by step ---
    from spatialcpav25_gen.infer.generate import (
        _decode,
        _default_exclusions,
        _layout_on,
        is_boundary_plane,
    )
    from spatialcpav25_gen.model.expression import sample_counts

    plane = plane_at_z(vol, target_z, cfg)
    # plane_at_z takes the depth verbatim; it is the *field* that clamps queries to the bbox,
    # under a BBoxClampWarning this run suppresses. So the two ways --target-z can be wrong are
    # checked here rather than left to a warning nobody sees.
    z_lo, z_hi = float(np.asarray(vol.bbox)[0, 2]), float(np.asarray(vol.bbox)[1, 2])
    if not (z_lo <= target_z <= z_hi):
        plane_note = (
            f"--target-z {target_z} is outside the training volume's z range "
            f"[{z_lo:.1f}, {z_hi:.1f}]; every GRF query on this plane is clamped to the "
            "bounding box, so the prior is being read off a face rather than a slice"
        )
    elif is_boundary_plane(vol, plane, cfg):
        plane_note = (
            f"--target-z {target_z} is a boundary plane (within "
            f"{cfg.boundary_margin_spacings} median spacings of the stack's end): evidence "
            "there is one-sided, which T04 measured as a 20-35% reconstruction deficit (R3)"
        )
    else:
        plane_note = ""
    if plane_note:
        print(f"  !! {plane_note}", file=sys.stderr)
    rows: list[dict] = []
    density = {
        "matched": bool(args.match_density),
        "n_target": int(real.counts.shape[0]),
        "seed": int(args.density_seed),
    }
    def _generate(gen_seed: int) -> dict:
        """One whole generation at ``gen_seed``: layout, prior, flow, decode, draw.

        Factored out so ``--stage4-seeds`` can repeat *all* of it. Stage 4's variance includes the
        layout and the flow sample, not only the emission draw, so re-drawing counts under a new
        seed while holding the layout would understate it -- and stage 4 is the rung the ladder's
        verdict leans on.
        """
        out: dict = {"seed": int(gen_seed)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layout = _layout_on(model, plane, vol, cfg, int(gen_seed))
            xyz = layout.coords_xyz.astype(np.float64)
            cell_type_idx = layout.cell_type.astype(np.int64)
            out["n_generated"] = int(xyz.shape[0])
            if args.match_density:
                # Thin *before* the neighbour query, the conditioning and the metric: the kNN
                # graph is what density changes, so subsampling afterwards would leave the
                # estimator that this flag exists to equalise untouched.
                keep = density_subsample(
                    xyz.shape[0], real.counts.shape[0], args.density_seed
                )
                xyz, cell_type_idx = xyz[keep], cell_type_idx[keep]
            out["n_used"] = int(xyz.shape[0])
            cell_type = torch.from_numpy(cell_type_idx)
            neighbours, _w = model.data.index.query(
                xyz, _default_exclusions(vol, float(plane.origin[2])), seed=int(gen_seed)
            )
            points = torch.from_numpy(xyz.astype(np.float32))
            with torch.no_grad():
                tokens, mask = model.data.index.neighbour_tokens(xyz, neighbours)
                cond, _ = model.conditioning(points, points, cell_type, None, tokens, mask)
                h0 = model.prior_latent(xyz, seed=int(gen_seed))
                h = model.flow.sample(h0, cond, int(cfg.ode_steps))
                mu_g, theta_g, pi_g = _decode(model, h, cfg, None)
                counts_g = sample_counts(
                    mu_g, theta_g, pi_g, np.random.default_rng(int(gen_seed))
                )
        out.update(
            xyz=xyz, xy=xyz[:, :2], h0=h0, h=h, mu=mu_g, theta=theta_g, pi=pi_g, counts=counts_g
        )
        return out

    gen = _generate(SEED)
    xyz, xy = gen["xyz"], gen["xy"]
    h0, h, mu, theta, pi, counts = (
        gen["h0"], gen["h"], gen["mu"], gen["theta"], gen["pi"], gen["counts"]
    )
    density["n_generated"], density["n_used"] = gen["n_generated"], gen["n_used"]
    if args.match_density and density["n_generated"] < density["n_target"]:
        print(
            f"  !! --match-density asked for {density['n_target']} cells and the layout produced "
            f"{density['n_generated']}; running with all of them, NOT density-matched",
            file=sys.stderr,
        )

    counts_np = counts.numpy()
    if args.top_k_by == "real":
        panel = top_k_panel(real.xy, rank_normalize(real.counts), k, args.top_k)
    elif args.top_k_by == "model":
        panel = top_k_panel(xy, rank_normalize(counts_np), k, args.top_k)
    else:
        panel = None
    genes = list(map(str, vol.gene_names))
    panel_genes = genes if panel is None else [genes[i] for i in panel]
    if panel is not None:
        print(f"  panel: {len(panel_genes)} genes by --top-k-by {args.top_k_by}")

    def _sel(a: np.ndarray) -> np.ndarray:
        """Restrict a ``(N, G)`` gene-space array to the panel. Latents are never passed here."""
        return a if panel is None else a[:, panel]

    rows.append(summarise("1. prior h0 = GRF at generated xyz", xy, h0.numpy(), k, primary="raw"))
    rows.append(summarise("2. latent h after the flow", xy, h.numpy(), k, primary="raw"))
    rows.append(
        summarise("3. decoded mu (before sampling)", xy, _sel(mu.numpy()), k, primary="raw")
    )
    rows.append(
        summarise("4. sampled counts (rank-normalised)", xy, _sel(counts_np), k, primary="rank")
    )
    # P1: the same mean field with the emission's noise removed — the ceiling any repair to
    # theta/pi could reach on the model as it actually is. It belongs here and not in the
    # ablation because ``mu`` lives at the *generated* cells; every A1 arm is at the real ones.
    # Drawn on the panel only (Moran's I is per column, so selecting before or after the draw is
    # the same) and at the run seed, like stage 4, so the two differ in the emission and nothing
    # else.
    rows.append(
        summarise(
            "4p. counts ~ Poisson(mu) — emission noise removed",
            xy,
            np.random.default_rng(SEED)
            .poisson(np.maximum(_sel(mu.numpy()), 0.0))
            .astype(np.float64),
            k,
            primary="rank",
        )
    )
    emitted = {"uncalibrated": counts_np}
    gen_terms = mu_log_variance_terms(model, h, cfg, panel)
    decomposition = summarise_mu_terms(gen_terms)

    if args.calibrate:
        # T09 §2's calibrator solves log theta per gene against the mean-variance relation at the
        # model's own mean — the quantity the pilot measured wrong on real data (slope 2.120
        # against the tissue's 1.738). It ships unapplied because the fixture gave it no headroom.
        from spatialcpav25_gen.infer.calibrate import calibrate_detection

        t1 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibration = calibrate_detection(model, vol, cfg, seed=SEED)
            with torch.no_grad():
                mu_c, theta_c, pi_c = _decode(model, h, cfg, calibration)
                counts_c = sample_counts(mu_c, theta_c, pi_c, np.random.default_rng(SEED))
        print(f"  calibration fitted in {time.time() - t1:.1f}s on {list(calibration.section_ids)}")
        rows.append(
            summarise("3c. decoded mu, CALIBRATED", xy, _sel(mu_c.numpy()), k, primary="raw")
        )
        rows.append(
            summarise(
                "4c. sampled counts, CALIBRATED (rank-norm)",
                xy,
                _sel(counts_c.numpy()),
                k,
                primary="rank",
            )
        )
        emitted["calibrated"] = counts_c.numpy()

    for r in rows:  # print the generated chain before anything else can fail
        print(f"  {r['stage']:<48s} median I = {r['median_I']:+.4f}  (n={r['n_channels']})")
    ladder: dict = {}
    ladder_extra: dict = {}
    null_check: dict = {}
    boot: dict = {}
    flanking: dict = {}
    stage4_seeds: list[dict] = []
    theta_stats: dict | None = None
    if args.report_theta:
        theta_stats = theta_report(
            model, cfg, real, panel, float(args.theta_floor or cfg.decoder_theta_floor)
        )
        print(f"  theta percentiles at the real cells: {theta_stats['percentiles']}")
        if "fraction_at_or_below_floor" in theta_stats:
            print(
                f"  a floor at {theta_stats['floor']} would bind on "
                f"{theta_stats['fraction_at_or_below_floor']:.1%} of (cell, gene) pairs and "
                f"{theta_stats['genes_with_any_binding']}/{theta_stats['n_genes']} genes"
            )
    real_decomposition: dict[str, float] | None = None
    mu_var_ratio = float("nan")
    ablation_rows: list[dict] = []
    ablation_levels: dict[str, float] = {}
    mu_spread: list[dict] = []
    try:
        ref_rows, h1 = real_section_reference(
            cfg, model, args.section, k, paths.ground_truth, real=real, panel=panel
        )
        rows.extend(ref_rows)
        # Step 1: the same decoder, the same size head, the same panel — the only thing that
        # differs between these two decompositions is the latent it is applied to.
        real_terms = mu_log_variance_terms(model, h1, cfg, panel)
        real_decomposition = summarise_mu_terms(real_terms)
        denom = real_terms["total"]
        ok = denom > 0
        if ok.any():
            mu_var_ratio = float(np.median(gen_terms["total"][ok] / denom[ok]))
        if args.emission_ablation:
            t2 = time.time()
            (
                ablation_rows,
                ablation_levels,
                mu_spread,
                ladder,
                null_check,
                ladder_extra,
            ) = emission_ablation(
                model,
                cfg,
                real,
                h1,
                k,
                [int(x) for x in args.ablation_seed],
                panel,
                mu_gen=_sel(mu.numpy()),
                counts_gen=_sel(counts_np),
                null_seeds=[int(x) for x in args.null_seed],
            )
            print(f"  emission ablation in {time.time() - t2:.1f}s", flush=True)

            for r in ablation_rows:
                print(f"  {r['stage']:<48s} median I = {r['median_I']:+.4f}  (n={r['n_channels']})")
    except Exception as exc:  # a reference failure must not discard the chain above
        print(
            f"  !! reference stage failed ({type(exc).__name__}: {exc}); "
            f"the generated chain above still stands",
            file=sys.stderr,
        )

    text_channel = describe_text_state(model, cfg)
    if checkpoint is not None:
        text_channel += f" — from `{args.load_model}`"
    if panel is None:
        panel_rule = f"all {len(panel_genes)} genes (`--top-k-by all`)"
    elif len(panel_genes) >= len(genes):
        panel_rule = (
            f"🚩 **vacuous**: `--top-k {args.top_k}` >= the panel's {len(genes)} genes, so all "
            f"{len(genes)} were kept and **no selection took place**"
        )
    else:
        panel_rule = (
            f"top {len(panel_genes)} of {len(genes)} "
            f"({len(panel_genes) / len(genes):.1%}) by Moran's I on the **{args.top_k_by}** side"
        )
    if not args.match_density:
        density_rule = (
            f"NOT matched (flag not given): {density['n_used']} generated against "
            f"{density['n_target']} real"
        )
    elif density["n_generated"] <= density["n_target"]:
        density_rule = (
            f"🚩 **vacuous**: requested {density['n_target']}, the layout produced only "
            f"{density['n_generated']}, so nothing was subsampled and the arms are **NOT** "
            "density-matched"
        )
    else:
        density_rule = (
            f"matched: {density['n_generated']} generated -> {density['n_used']} kept "
            f"(seed {density['seed']})"
        )
    width = max(len(r["stage"]) for r in rows)
    lines = [
        f"# Chain diagnostic — where the spatial structure is lost ({int(cfg.train_steps)} steps)",
        "",
        f"`{paths.dataset}` / `{paths.holdout}`, `{args.section}` at z={target_z:.1f}, "
        f"{xy.shape[0]} generated cells.",
        "Median per-channel Moran's I on a row-standardised kNN graph "
        f"(k={k}), the same estimator at every stage.",
        "",
        "## What this run is",
        "",
        "| | |",
        "|---|---|",
        f"| text channel | {text_channel} |",
        f"| `expr_pca_dim` | {cfg.expr_pca_dim} |",
        f"| `decoder_mu_link` | `{cfg.decoder_mu_link}` |",
        f"| `layout_mode` / `layout_sampler` | `{cfg.layout_mode}` / `{cfg.layout_sampler}` |",
        f"| cell density | {density_rule} |",
        f"| gene panel | {panel_rule} |",
        f"| real section | {real.counts.shape[0]} cells, z = {real.z:.1f} |",
    ]
    for note in (z_note, plane_note):
        if note:
            lines += ["", f"🚩 {note}."]
    if panel is not None:
        vacuous = len(panel_genes) >= len(genes)
        lines += [
            "",
            (
                f"**Panel** (**no selection** — all {len(genes)} genes): "
                if vacuous
                else f"**Panel** ({args.top_k_by}-selected, {len(panel_genes)} of {len(genes)}): "
            )
            + ", ".join(f"`{g}`" for g in panel_genes),
            "",
            "The panel restricts the **gene-space stages only** — 3, 4, their calibrated twins, "
            "and `REF real counts`. Stages 1 and 2 and `REF real latent h1` are latents with "
            f"{int(cfg.latent_dim)} channels that are not genes; their `channels` column "
            "shows that, and their `median I` is over all of them.",
        ]
        if args.top_k_by == "model":
            lines += [
                "",
                "🚩 This panel was selected by **the model's own** Moran's I. It scores the "
                "model on the genes it did best at and is not a number to quote; it exists to "
                "bound how much the `real`-selected panel differs from a flattering one.",
            ]
    lines += [
        "",
        "## The chain",
        "",
        "Every stage now carries Moran's I under **both** transforms. A ratio between two stages",
        "must take both sides from the **same** column: the chain used to rank its count stages",
        "and leave its mean-field and latent stages raw, which made every `counts / mu` retention",
        "a cross-transform ratio (`reports/ceiling_review.md` §2). The **primary** column is the",
        "one that stage's earlier artifacts recorded, and is marked `*`.",
        "",
        f"| {'stage':<{width}} | median I (raw) | median I (rank) | p25 | p75 | channels |",
        f"|{'-' * (width + 2)}|---|---|---|---|---|",
    ]
    for r in rows:
        raw = f"{r.get('median_I_raw', float('nan')):+.4f}"
        rank = f"{r.get('median_I_rank', float('nan')):+.4f}"
        if r.get("transform") == "raw":
            raw = f"**{raw}** *"
        else:
            rank = f"**{rank}** *"
        lines.append(
            f"| {md_cell(r['stage']):<{width}} | {raw} | {rank} | {r['p25']:+.4f} "
            f"| {r['p75']:+.4f} | {r['n_channels']} |"
        )
    # Q1: the invariant that would have caught the transform mismatch. 4p is an independent draw
    # from stage 3's field, and independent noise dilutes autocorrelation — it cannot create it —
    # so on ONE transform I(4p) <= I(3) must hold. It did not, and that is what sent the mismatch
    # to the code (reports/ceiling_review.md §2). Checked on the ranked column, which both stages
    # now carry; the tolerance is for one draw's sampling noise, not for a transform.
    by_rank = {r["stage"].split(".")[0]: r.get("median_I_rank") for r in rows}
    i3, i4p = by_rank.get("3"), by_rank.get("4p")
    invariant_note = ""
    readable = i3 is not None and i4p is not None and np.isfinite(i3) and np.isfinite(i4p)
    if readable and i4p > i3 + 0.02:
        invariant_note = (
            f"stage 4p ({i4p:+.4f}) exceeds stage 3 ({i3:+.4f}) on the SAME transform by "
            f"{i4p - i3:+.4f}. 4p is an independent draw from stage 3's field and independent "
            "noise cannot raise Moran's I, so one of the two stages is not measuring what it "
            "says it is"
        )
        print(f"  !! INVARIANT VIOLATED: {invariant_note}", file=sys.stderr)

    # Q1.5: the scored statistic, from vectors this run already has.
    agreement: dict[str, dict] = {}
    sparsity: dict = {}
    ref_gene_I_all: np.ndarray | None = ladder_extra.get("ref_all")
    model_all: np.ndarray | None = None
    ref_gene_I = next(
        (r.get("per_gene_I_rank") for r in rows if r["stage"].startswith("REF real counts")), None
    )
    if ref_gene_I is not None:
        for prefix, label in (
            ("3.", "3. decoded mu"),
            ("4.", "4. sampled counts"),
            ("4p.", "4p. Poisson(mu) — emission-free"),
        ):
            vec = next(
                (r.get("per_gene_I_rank") for r in rows if r["stage"].startswith(prefix)), None
            )
            if vec is not None:
                agreement[f"panel::{label}"] = morans_agreement(vec, ref_gene_I)
        if panel is not None and len(panel) < len(genes):
            # The panel is the top few per cent by the real section's own I, so its I vector has a
            # compressed range and its correlation is attenuated. bench3 scores ALL shared genes,
            # so the all-genes pass is the one closer to the benchmark and it governs.
            t3 = time.time()
            # The ablation already paid for this vector when it ran the all-genes ladder pass;
            # recomputing it would be a second 1017-gene blocked pass for an identical result.
            ref_all = (
                ref_gene_I_all
                if ref_gene_I_all is not None
                else morans_i_ranked_blocked(real.xy, real.counts, k)
            )
            ref_gene_I_all = ref_all
            for label, arr in (
                ("3. decoded mu", mu.numpy()),
                ("4. sampled counts", counts_np),
                (
                    "4p. Poisson(mu) — emission-free",
                    np.random.default_rng(SEED).poisson(np.maximum(mu.numpy(), 0.0)),
                ),
            ):
                vec_all = morans_i_ranked_blocked(xy, np.asarray(arr, dtype=np.float64), k)
                agreement[f"all genes::{label}"] = morans_agreement(vec_all, ref_all)
                if label.startswith("4."):
                    model_all = vec_all
            print(f"  all-gene agreement over {len(genes)} genes in {time.time() - t3:.1f}s")
            # R1-R3 runs on whichever gene set GOVERNS, which is the all-genes pass wherever it
            # ran. The panel is the top few per cent by the tissue's OWN I, so on it `I_real` has
            # almost no spread and the partial correlation is attenuated for a reason that has
            # nothing to do with sparsity; and a panel chosen by I is not independent of detection
            # rate, so the control would be conditioned on. Neither is true of all genes.
            sparsity = decomposition_of_r(
                model_all,
                ref_all,
                real.counts,
                r4=agreement["all genes::4. sampled counts"]["pearson"],
            )
            sparsity["gene_set"] = f"all {len(genes)} genes"
        else:
            sparsity = decomposition_of_r(
                next(r["per_gene_I_rank"] for r in rows if r["stage"].startswith("4. ")),
                ref_gene_I,
                _sel(real.counts),
                r4=agreement["panel::4. sampled counts"]["pearson"],
            )
            sparsity["gene_set"] = "the scored panel"

    # --- the paired gene bootstrap (null_band_preregistration.md §2b) ------------------------
    # Run here rather than inside the ablation because the difference the verdict leans on is
    # `r(4) - r(A1a)`, and stage 4's per-gene vector is generated in this scope. Stage 4 sits at
    # the GENERATED cells while the A1 arms sit at the real ones; the bootstrap resamples genes,
    # which both sides share, so the pairing is over genes and not over cells.
    _pairs = [
        ("A1c", "A1b"),
        ("A1c", "A1b-t"),  # theta alone
        ("A1c", "A1b-p"),  # pi alone
        ("A1b", "A1a"),
        ("4", "A1a"),  # the one the verdict leans on
        ("A1a", "A1n"),
    ]
    for _scope, _vkey, _rkey, _s4 in (
        ("panel", "vecs_panel", "ref_panel", "panel::4. sampled counts"),
        ("all genes", "vecs_all", "ref_all", "all genes::4. sampled counts"),
    ):
        _vecs = dict(ladder_extra.get(_vkey) or {})
        _ref = ladder_extra.get(_rkey)
        if not _vecs or _ref is None:
            continue
        _v4 = (
            model_all
            if _scope == "all genes"
            else next(
                (r.get("per_gene_I_rank") for r in rows if r["stage"].startswith("4. ")), None
            )
        )
        if _v4 is not None and _s4 in agreement:
            _vecs["4"] = _v4
        boot[_scope] = paired_gene_bootstrap(
            _vecs, _ref, _pairs, n_boot=int(args.boot_reps)
        )

    # --- stage 4 across whole generations (null_band_preregistration.md §2c) -----------------
    if args.stage4_seeds:
        ref_panel_g = morans_i(real.xy, rank_normalize(_sel(real.counts)), k)
        ref_all_g = ref_gene_I_all if ref_gene_I_all is not None else None
        for gs in args.stage4_seeds:
            t4 = time.time()
            g = _generate(int(gs)) if int(gs) != SEED else gen
            cg = g["counts"].numpy()
            row = {
                "seed": int(gs),
                "n_used": int(g["n_used"]),
                "median_I": summarise("4", g["xy"], _sel(cg), k, primary="rank")["median_I"],
                "r_panel": morans_agreement(
                    morans_i(g["xy"], rank_normalize(_sel(cg)), k), ref_panel_g
                )["pearson"],
            }
            if ref_all_g is not None:
                row["r_all"] = morans_agreement(
                    morans_i_ranked_blocked(g["xy"], np.asarray(cg, dtype=np.float64), k),
                    ref_all_g,
                )["pearson"]
            stage4_seeds.append(row)
            print(f"  generation seed {gs} in {time.time() - t4:.1f}s", flush=True)

    # --- flanking_copy (reports/flanking_copy_preregistration.md) ----------------------------
    if args.flanking_copy:
        t5 = time.time()
        flanking = flanking_copy_arm(
            vol,
            real,
            k,
            null_seeds=[int(x) for x in args.null_seed],
            stage4_r=(agreement.get("all genes::4. sampled counts") or {}).get("pearson"),
        )
        print(
            f"  flanking_copy from {flanking['source_section']} in {time.time() - t5:.1f}s",
            flush=True,
        )
        print(f"  {flanking['verdict']}")

    lines.extend(_verdict(rows, emitted, cfg, args, real, panel))
    lines.extend(agreement_block(agreement, invariant_note))
    lines.extend(ladder_block(ladder, sparsity, agreement, null_check, boot))
    lines.extend(stage4_seed_block(stage4_seeds))
    lines.extend(flanking_block(flanking))
    lines.extend(cancelling_defects_block(rows, decomposition, mu_spread))

    def _column(d: dict[str, float] | None, key: str, fmt: str) -> str:
        if d is None:
            return "—"
        return format(d[key], fmt)

    lines.extend(
        [
            "",
            "## Candidate 2 — is `mu`'s dynamic range the size factor?",
            "",
            "`mu = link(MLP_mu(u)) * size_factor`, so `log mu = shape + log s` and the",
            "variance splits exactly. Per gene, medians over the panel above.",
            "",
            "The **real latent** column is the same decoder and the same size head applied to",
            "`h1 = encoder(real counts)` instead of to the flow's sample: the two columns differ",
            "in the latent and in nothing else. It is the matched tissue-side quantity the",
            'record has been quoting as "tissue\'s 1.213" without a source.',
            "",
            "| quantity | generated `h` | real latent `h1` |",
            "|---|---|---|",
            f"| genes decomposed | {decomposition['n_genes_decomposed']} | "
            f"{_column(real_decomposition, 'n_genes_decomposed', 'd')} |",
            f"| `Var(shape)` — the latent-driven part | {decomposition['var_shape']:.5f} | "
            f"{_column(real_decomposition, 'var_shape', '.5f')} |",
            f"| `Var(log s)` — the size-factor part | {decomposition['var_logsize']:.5f} | "
            f"{_column(real_decomposition, 'var_logsize', '.5f')} |",
            f"| `2 Cov` | {2 * decomposition['cov']:+.5f} | "
            + ("—" if real_decomposition is None else f"{2 * real_decomposition['cov']:+.5f}")
            + " |",
            f"| share of `Var(log mu)` from the latent (unbounded) | "
            f"{decomposition['share_shape']:.1%} | "
            f"{_column(real_decomposition, 'share_shape', '.1%')} |",
            "| **bounded share** `Var(shape)/(Var(shape)+Var(log s))` | "
            f"**{decomposition['share_shape_bounded']:.1%}** | "
            f"**{_column(real_decomposition, 'share_shape_bounded', '.1%')}** |",
            f"| **`sd(log mu)` across cells** | **{decomposition['sd_log_mu']:.4f}** | "
            f"**{_column(real_decomposition, 'sd_log_mu', '.4f')}** |",
            "",
            f"**`Var(log mu)` generated / real, median per gene: {mu_var_ratio:.3f}.** "
            "Pre-registered in `reports/emission_repair_options.md` §8.3: >= 0.8 means the "
            "structured component is intact and §2's binding constraint does not exist; "
            "<= 0.4 confirms it; between the two is uninformative and needs the three-seed "
            "version.",
        ]
    )
    if ablation_rows:
        i_model = next(
            (r["median_I"] for r in rows if r["stage"].startswith("4. sampled counts")),
            float("nan"),
        )
        i_real = next(
            (r["median_I"] for r in rows if r["stage"].startswith("REF real counts")),
            float("nan"),
        )
        deficit = i_real - i_model
        width_a = max(len(r["stage"]) for r in ablation_rows)
        lines += [
            "",
            "## A1 — emission ablation, at the real cells",
            "",
            "Every arm is drawn at the ground truth's own positions, so the kNN graph, the cell",
            "count and the density are identical to `REF real counts` and to each other. The",
            "layout, the prior and the flow are held out of it entirely: only what the counts",
            "were drawn from varies.",
            "",
            "**Read `reports/a1_preregistration.md` before reading these numbers.** The outcome",
            "table, the thresholds, the level guard and the two stated asymmetries were committed",
            "before the run.",
            "",
            f"| {'arm':<{width_a}} | median I | across seeds | ch | level | R | band |",
            f"|{'-' * (width_a + 2)}|---|---|---|---|---|---|",
        ]
        readable = bool(np.isfinite(deficit) and deficit > 0)

        def band(recovery: float) -> str:
            if not np.isfinite(recovery):
                return "—"
            if recovery >= 0.70:
                return "RECOVERS"
            return "DOES NOT RECOVER" if recovery <= 0.30 else "UNINFORMATIVE"

        for r in ablation_rows:
            key = r["stage"].split(".")[0]
            level = ablation_levels.get(key)
            level_s = "—" if level is None else f"{level:.2f}x"
            drawn = "per_seed" in r
            recovery = (r["median_I"] - i_model) / deficit if readable and drawn else float("nan")
            rec_s = "—" if not np.isfinite(recovery) else f"**{recovery:+.2f}**"
            if drawn and r.get("n_seeds", 1) > 1:
                seeds_s = f"{r['min_I']:+.4f} .. {r['max_I']:+.4f} ({r['n_seeds']})"
            else:
                seeds_s = "—"
            lines.append(
                f"| {md_cell(r['stage']):<{width_a}} | **{r['median_I']:+.4f}** | {seeds_s} "
                f"| {r['n_channels']} | {level_s} | {rec_s} | {band(recovery)} |"
            )
        lines += [
            "",
            f"Anchors from this run: `I(model counts)` = **{i_model:+.4f}**, "
            f"`I(real counts)` = **{i_real:+.4f}**, deficit = **{deficit:+.4f}**.",
        ]

        # a1_escalation_preregistration.md §1: an arm whose seeds do not all land in one band reads
        # UNRESOLVED whatever its median says. Applied here rather than left to the reader,
        # because a median quoted without its spread is the verdict a re-draw could reverse.
        if readable and any(r.get("n_seeds", 1) > 1 for r in ablation_rows):
            lines += [
                "",
                "**Three-seed stability** (`a1_escalation_preregistration.md` §1): an arm whose "
                "per-seed bands disagree reads **UNRESOLVED** regardless of its median.",
                "",
                "| arm | per-seed R | bands | verdict |",
                "|---|---|---|---|",
            ]
            for r in ablation_rows:
                if r.get("n_seeds", 1) <= 1 or "per_seed" not in r:
                    continue
                recoveries = [(e["median_I"] - i_model) / deficit for e in r["per_seed"]]
                bands = [band(x) for x in recoveries]
                stable = len(set(bands)) == 1
                median_band = band(float(np.median(recoveries)))
                verdict = median_band if stable else "**UNRESOLVED** (seeds straddle)"
                lines.append(
                    f"| {md_cell(r['stage'].split('.')[0])} | "
                    + ", ".join(f"{x:+.2f}" for x in recoveries)
                    + f" | {', '.join(dict.fromkeys(bands))} | {verdict} |"
                )
        # N5 (a1_escalation_preregistration.md §2): attribute the A1c -> A1b loss to
        # over-dispersion, to dropout, to both additively, or to neither. Criteria fixed there;
        # applied here so the verdict is not left to whoever reads the table.
        by_arm = {r["stage"].split(".")[0]: r["median_I"] for r in ablation_rows}
        i_c, i_b = by_arm.get("A1c"), by_arm.get("A1b")
        i_t, i_p = by_arm.get("A1b-t"), by_arm.get("A1b-p")
        if None not in (i_c, i_b, i_t, i_p):
            n5 = n5_verdict(i_c, i_b, i_t, i_p)
            l_total, l_theta, l_pi = n5["l_total"], n5["l_theta"], n5["l_pi"]
            share_t, share_p = n5["share_theta"], n5["share_pi"]
            additive_gap, predicted = n5["additivity_gap"], n5["predicted_i_b"]
            level_t = ablation_levels.get("A1b-t", float("nan"))
            lines += [
                "",
                "### N5 — which of `theta` and `pi` costs the `A1c -> A1b` loss",
                "",
                "Criteria fixed in `a1_escalation_preregistration.md` §2, before these arms were",
                "built. `A1b-p` shares `A1c`'s Poisson realisation, so `A1c -> A1b-p` is an exact",
                "within-realisation contrast.",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| `L_total = I(A1c) - I(A1b)` | {l_total:+.4f} |",
                f"| `L_theta = I(A1c) - I(A1b-t)` | {l_theta:+.4f} (**{share_t:.1%}** of total) |",
                f"| `L_pi = I(A1c) - I(A1b-p)` | {l_pi:+.4f} (**{share_p:.1%}** of total) |",
                f"| additivity gap `abs(L_theta + L_pi - L_total)` | {additive_gap:.4f} "
                f"(criterion <= 0.0200) |",
                f"| multiplicative prediction of `I(A1b)` | {predicted:+.4f} against the measured "
                f"{i_b:+.4f}, gap {abs(predicted - i_b):.4f} |",
                f"| **verdict** | {n5['verdict']} |",
                "",
                f"🔎 **Instrument self-check**: `A1b-t`'s level ratio is **{level_t:.2f}x**. "
                "`theta` cannot move the mean, so anything away from 1.00x means that arm is not "
                "what it claims and the verdict above does not stand.",
            ]
        if theta_stats:
            lines += [
                "",
                "### `theta` at the real cells"
                + (
                    f", and what a floor at {theta_stats['floor']} binds on"
                    if "fraction_at_or_below_floor" in theta_stats
                    else ""
                ),
                "",
                "| percentile | " + " | ".join(theta_stats["percentiles"]) + " |",
                "|---" * (len(theta_stats["percentiles"]) + 1) + "|",
                "| `theta` | "
                + " | ".join(f"{v:.4g}" for v in theta_stats["percentiles"].values())
                + " |",
            ]
            if "fraction_at_or_below_floor" in theta_stats:
                lines += [
                    "",
                    f"A floor at **{theta_stats['floor']}** binds on "
                    f"**{theta_stats['fraction_at_or_below_floor']:.1%}** of (cell, gene) pairs "
                    f"and touches **{theta_stats['genes_with_any_binding']} of "
                    f"{theta_stats['n_genes']}** genes. 🚩 **A floor that binds on nothing is a "
                    "null experiment, not a null result** — this figure must travel with any "
                    "conclusion drawn from the floored arm.",
                ]
        if mu_spread:
            lines += [
                "",
                "### N2 — the tissue's own `sd(log mu)`, two model-free routes",
                "",
                "Both use `sd(log mu) = sqrt(log(1 + CV^2))` — an approximation — so neither",
                "needs a pseudocount, and **neither passes through the decoder**, which is what",
                "the gate in `emission_repair_options.md` §8.3 could not manage. The kNN mean",
                "field shrinks variance, so its figure is a **lower** bound; the",
                "Poisson-deconvolved one counts any tissue over-dispersion as signal, so it is an",
                "**upper** bound. Together they bracket the tissue. Estimator fixed in",
                "`a1_escalation_preregistration.md` §3.",
                "",
                "| quantity | `sd(log mu)` | genes |",
                "|---|---|---|",
            ]
            for entry in mu_spread:
                lines.append(
                    f"| {entry['quantity']} | **{entry['sd_log_mu']:.4f}** | {entry['n_genes']} |"
                )
            lines += [
                "",
                "Read against the decoder's own figure in the Candidate 2 block above. "
                "`chain_shipped_review.md` §6's narrow-`mu` mechanism is **supported** if the "
                "tissue's lower bound exceeds it by >= 1.5x, **refuted** if the tissue's upper "
                "bound falls below it, and **untested still** in between.",
                "",
                "⚠️ How loose the lower bound is depends on how autocorrelated the field already "
                "is — a kNN mean destroys the variance of a field whose neighbours are unrelated "
                "and preserves it where they are not (the self-check measures 0.800 -> 0.279 on "
                "an unstructured field). Read it beside `I(mu_oracle)` in the table above. **A "
                'lower bound below the decoder\'s figure is "untested still", never '
                '"refuted"** — only the upper bound can refute.',
            ]
        if not (np.isfinite(deficit) and deficit > 0):
            lines += [
                "",
                "🚩 `I(model counts)` is at or above `I(real counts)` on this dataset, so there is",
                "no deficit to recover and **`R` is undefined**. This run is the",
                "pre-registration's",
                "**instrument control**: every drawn arm must reach 0.6x `I(real counts)` = "
                f"**{0.6 * i_real:+.4f}**, or the ablation is measuring something other than what",
                "it claims and no result on the other dataset may be read.",
            ]
    if real_decomposition is None:
        lines += [
            "",
            "🚩 The real-latent column is missing because the reference stage failed; see the "
            "run's stderr. The generated column stands.",
        ]
    text = "\n".join(lines)
    print()
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    sidecar = {
        "run": {
            "dataset": paths.dataset,
            "holdout": paths.holdout,
            "section": args.section,
            "target_z": target_z,
            "section_z": real.z,
            "steps": int(args.steps),
            "seed": SEED,
            "text_channel": "live" if live_text else "zero_vectors",
            "text_emb_mode": cfg.text_emb_mode,
            "expr_pca_dim": int(cfg.expr_pca_dim),
            "decoder_mu_link": cfg.decoder_mu_link,
            "layout_mode": cfg.layout_mode,
            "layout_sampler": cfg.layout_sampler,
            "decoder_theta_floor": float(cfg.decoder_theta_floor),
            "calibrated": bool(args.calibrate),
            "z_note": z_note or None,
            "plane_note": plane_note or None,
        },
        "density": density,
        "theta": theta_stats,
        "panel": {
            "rule": args.top_k_by,
            "top_k": int(args.top_k) if panel is not None else None,
            "n_genes": len(panel_genes),
            "genes": panel_genes if panel is not None else None,
            "indices": [int(i) for i in panel] if panel is not None else None,
        },
        "stages": [{k: v for k, v in r.items() if k != "per_gene_I_rank"} for r in rows],
        "agreement": agreement,
        "ladder": ladder,
        "sparsity": sparsity,
        "null_check": null_check,
        "bootstrap": boot,
        "stage4_seeds": stage4_seeds,
        "flanking_copy": {k: v for k, v in flanking.items() if k != "R3"} or None,
        "flanking_R3": flanking.get("R3"),
        "invariant_violated": invariant_note or None,
        "emission_ablation": {
            "ran": bool(ablation_rows),
            "seeds": [int(x) for x in args.ablation_seed],
            "arms": [{k: v for k, v in r.items() if k != "per_gene_I_rank"} for r in ablation_rows],
            "level_vs_real": ablation_levels,
            "mu_spread": mu_spread,
        },
        "mu_variance": {
            "generated": decomposition,
            "real_latent": real_decomposition,
            "var_log_mu_ratio_gen_over_real": mu_var_ratio,
        },
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=_json_scalar))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
