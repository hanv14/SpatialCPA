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


def summarise(name: str, xy: np.ndarray, values: np.ndarray, k: int) -> dict[str, float]:
    """Median / IQR of per-channel Moran's I, plus the channel count."""
    i = morans_i(xy, values, k)
    finite = i[np.isfinite(i)]
    return {
        "stage": name,
        "median_I": float(np.median(finite)) if finite.size else float("nan"),
        "p25": float(np.percentile(finite, 25)) if finite.size else float("nan"),
        "p75": float(np.percentile(finite, 75)) if finite.size else float("nan"),
        "n_channels": int(finite.size),
    }


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
) -> tuple[list[dict], dict, list[dict]]:
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
    is not comparable in ``I`` however it was drawn), and the ``sd(log mu)`` table of N2.
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
            row = summarise(labels[key], real.xy, rank_normalize(sel(counts)), k)
            row["seed"] = int(seed)
            per_seed[key].append(row)
            ratio = sel(np.asarray(counts, dtype=np.float64)).mean(axis=0)[ok] / ref_mean[ok]
            per_seed_level[key].append(float(np.median(ratio)) if ratio.size else float("nan"))
            drawn.setdefault(key, counts)

    rows = [summarise("A1a'. mu decoded from h1", real.xy, sel(mu1_np), k)]
    rows.append(_over_seeds(per_seed["A1a"]))
    rows.append(summarise("A1b'. mu_oracle = kNN mean of real counts", real.xy, sel(mu_oracle), k))
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
    return rows, levels, mu_spread


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

    rows = [summarise("REF real counts (rank-normalised)", xy, rank_normalize(selected), k)]

    gene_idx = torch.arange(counts.shape[1], dtype=torch.long)
    with torch.no_grad():
        gene_emb = model.embeddings.gene(gene_idx)
        totals = torch.from_numpy(counts.sum(axis=1))  # (N,), not (N, 1)
        size_factor = totals / max(float(model.stats.median_total), 1.0)
        h1 = model.encoder(torch.from_numpy(counts), gene_emb, size_factor)
    rows.append(summarise("REF real latent h1 = encoder(real counts)", xy, h1.numpy(), k))
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

    seed_rows = [
        {"stage": "A1b. x", "median_I": 0.10, "p25": 0.0, "p75": 0.0, "n_channels": 3, "seed": 1},
        {"stage": "A1b. x", "median_I": 0.30, "p25": 0.0, "p75": 0.0, "n_channels": 3, "seed": 2},
        {"stage": "A1b. x", "median_I": 0.20, "p25": 0.0, "p75": 0.0, "n_channels": 3, "seed": 3},
    ]
    collapsed = _over_seeds(seed_rows)
    checks += [
        ("_over_seeds reports the MEDIAN over seeds", collapsed["median_I"] == 0.20),
        ("it carries the range", (collapsed["min_I"], collapsed["max_I"]) == (0.10, 0.30)),
        ("it keeps every seed, so a straddle is visible", len(collapsed["per_seed"]) == 3),
        ("and drops the single-seed key", "seed" not in collapsed),
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layout = _layout_on(model, plane, vol, cfg, SEED)
        xyz = layout.coords_xyz.astype(np.float64)
        cell_type_idx = layout.cell_type.astype(np.int64)
        density["n_generated"] = int(xyz.shape[0])
        if args.match_density:
            # Thin *before* the neighbour query, the conditioning and the metric: the kNN graph
            # is what density changes, so subsampling afterwards would leave the estimator that
            # this flag exists to equalise untouched.
            keep = density_subsample(xyz.shape[0], real.counts.shape[0], args.density_seed)
            xyz, cell_type_idx = xyz[keep], cell_type_idx[keep]
        density["n_used"] = int(xyz.shape[0])
        xy = xyz[:, :2]
        cell_type = torch.from_numpy(cell_type_idx)
        neighbours, _w = model.data.index.query(
            xyz, _default_exclusions(vol, float(plane.origin[2])), seed=SEED
        )
        points = torch.from_numpy(xyz.astype(np.float32))
        with torch.no_grad():
            tokens, mask = model.data.index.neighbour_tokens(xyz, neighbours)
            cond, _ = model.conditioning(points, points, cell_type, None, tokens, mask)
            h0 = model.prior_latent(xyz, seed=SEED)
            h = model.flow.sample(h0, cond, int(cfg.ode_steps))
            mu, theta, pi = _decode(model, h, cfg, None)
            counts = sample_counts(mu, theta, pi, np.random.default_rng(SEED))
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

    rows.append(summarise("1. prior h0 = GRF at generated xyz", xy, h0.numpy(), k))
    rows.append(summarise("2. latent h after the flow", xy, h.numpy(), k))
    rows.append(summarise("3. decoded mu (before sampling)", xy, _sel(mu.numpy()), k))
    rows.append(
        summarise("4. sampled counts (rank-normalised)", xy, rank_normalize(_sel(counts_np)), k)
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
        rows.append(summarise("3c. decoded mu, CALIBRATED", xy, _sel(mu_c.numpy()), k))
        rows.append(
            summarise(
                "4c. sampled counts, CALIBRATED (rank-norm)",
                xy,
                rank_normalize(_sel(counts_c.numpy())),
                k,
            )
        )
        emitted["calibrated"] = counts_c.numpy()

    for r in rows:  # print the generated chain before anything else can fail
        print(f"  {r['stage']:<48s} median I = {r['median_I']:+.4f}  (n={r['n_channels']})")
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
            ablation_rows, ablation_levels, mu_spread = emission_ablation(
                model,
                cfg,
                real,
                h1,
                k,
                [int(x) for x in args.ablation_seed],
                panel,
                mu_gen=_sel(mu.numpy()),
                counts_gen=_sel(counts_np),
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
        f"| {'stage':<{width}} | median I | p25 | p75 | channels |",
        f"|{'-' * (width + 2)}|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['stage']:<{width}} | **{r['median_I']:+.4f}** | {r['p25']:+.4f} "
            f"| {r['p75']:+.4f} | {r['n_channels']} |"
        )
    lines.extend(_verdict(rows, emitted, cfg, args, real, panel))

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
                f"| {r['stage']:<{width_a}} | **{r['median_I']:+.4f}** | {seeds_s} "
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
                    f"| {r['stage'].split('.')[0]} | "
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
            "calibrated": bool(args.calibrate),
            "z_note": z_note or None,
            "plane_note": plane_note or None,
        },
        "density": density,
        "panel": {
            "rule": args.top_k_by,
            "top_k": int(args.top_k) if panel is not None else None,
            "n_genes": len(panel_genes),
            "genes": panel_genes if panel is not None else None,
            "indices": [int(i) for i in panel] if panel is not None else None,
        },
        "stages": rows,
        "emission_ablation": {
            "ran": bool(ablation_rows),
            "seeds": [int(x) for x in args.ablation_seed],
            "arms": ablation_rows,
            "level_vs_real": ablation_levels,
            "mu_spread": mu_spread,
        },
        "mu_variance": {
            "generated": decomposition,
            "real_latent": real_decomposition,
            "var_log_mu_ratio_gen_over_real": mu_var_ratio,
        },
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
