"""Why does `marker_depth_r` split both gates the same way? — the per-gene decomposition.

On ``deep_starmap`` the same metric flipped both gates in the machinery's favour —
``zinb-flow`` over ``cross-mix`` by 7.8x the envelope, ``medcpt`` over ``lookup`` by 5.5x —
while every count-realism metric moved the other way and further.

**Correction (2026-08-27).** The two gates are **not** independent. ``zinb-flow`` under
``prior_mode=correlated`` and ``medcpt`` under ``expr_mode=zinb-flow`` are the *same fitted
config* — hash ``336cbc6a491faa51`` — so the winning arm is one model measured against two
different losers, and its per-gene profile correlations are numerically identical in both gates
(+0.2447 on ``section_3``, +0.3044 on ``section_5``). A single favourable draw of that one fit
inflates both margins at once. "One metric splitting two gates the same way" was the motivation
for building this, and it was weaker evidence than it looked; the decomposition below is
worth reading anyway, but not as two gates corroborating each other.

**The hypothesis under test.** Semantic gene embeddings group functionally related genes, so a
gene can borrow laminar structure from its neighbours in text space; a lookup table memorises
each gene independently and wins per-gene reproduction instead. If that is right, the gain
should **concentrate on genes whose text neighbours carry strong depth gradients**, over and
above the gene's own gradient.

**Why this is a stronger test than the margin it explains.** ``marker_depth_r`` is a *mean over*
``Config.metric_marker_genes`` *genes* of a per-gene depth-profile correlation, so it decomposes
exactly. That is 32 internal degrees of freedom against the 2 LOSO folds the margin rests on —
and a *structured* pattern across genes is far harder to obtain by chance than a difference of
two means. A concentration result would therefore carry weight the n = 2 margin cannot, which
matters most for ``text_emb_mode``, whose margin (0.1850) is **inside its own worst within-arm
fold spread** (0.2033) and is not established on the margin alone.

**The confound, and the controls for it.** A gene with a stronger depth gradient has more
variance in its real profile, so its Pearson r is better conditioned and *any* difference
between two arms has more room to show. Per-gene gain will therefore correlate with gradient
strength for a purely statistical reason, with no semantics involved. Three controls separate
them:

1. a **partial** correlation of the gain against the *neighbours'* gradient strength, holding
   the gene's **own** trend *and* its own bin-to-bin contrast fixed — borrowing predicts the
   neighbours matter beyond the gene, the conditioning artefact does not;
2. a **permutation null** that shuffles the text vectors across the marker genes and rebuilds
   the neighbourhood predictor. If the shuffled null reproduces the partial correlation, the
   structure is in the metric's conditioning and not in MedCPT's geometry;
3. the same decomposition run on ``expr_mode`` as well as ``text_emb_mode``. Only the second
   gate touches the text channel, so a partial correlation that appears on *both* is not
   about text space at all.

Everything is computed with the metric's own code — ``marker_genes``, ``soft_depth_profile``,
``profile_axis``, ``_normalised`` — so the per-gene terms average to the reported
``marker_depth_r`` exactly. That identity is **asserted** against ``section_scores``, not
assumed: a decomposition that does not add back up explains a different number than the one the
audit reported.

No refit: the audit's fit checkpoints are resume points, and re-entering a finished fit is a
no-op that restores it.

Usage::

    python scripts/t09_depth_mechanism.py --dataset deep_starmap \\
        --workdir runs/audit_deep --train-steps 2400
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from spatialcpav25_gen.infer.generate import emitted_counts, generate_section
from spatialcpav25_gen.infer.planes import section_plane
from spatialcpav25_gen.losses.metric_aware import (
    knn_weight_graph,
    marker_genes,
    profile_axis,
    soft_depth_profile,
)
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

# `_normalised` is private, and is imported deliberately: the per-gene terms have to be built
# on exactly the normalisation `section_scores` builds them on, or the decomposition explains a
# number the audit never reported. The assertion below is what makes that a checked claim.
from spatialcpav25_gen.train.select import _normalised, section_scores, selection_folds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import (
    base_config,
    clamp_config_to_input,
    embeddings_factory,
    load_training_volume,
)

# The two comparisons the finding is about, each as (gate, arm_that_won, arm_that_lost, under).
# `expr_mode` is included as much as a control as a subject: it does not touch the text channel,
# so a neighbourhood effect that shows up on both gates is not about text space.
COMPARISONS = (
    ("expr_mode", "zinb-flow", "cross-mix", {"prior_mode": "correlated"}),
    ("text_emb_mode", "medcpt", "lookup", {"expr_mode": "zinb-flow"}),
)
N_PERMUTATIONS = 2000
IDENTITY_TOL = 1e-6


def restore(cfg, volume, embeddings, *, seed: int, checkpoint: Path) -> CTFFlow:
    """Rebuild the model and restore the finished fit from the audit's checkpoint. No refit."""
    if not checkpoint.exists():
        raise SystemExit(
            f"no fit checkpoint at {checkpoint}. Run the audit for this arm first "
            "(scripts/t09_audit_starmap.py --fit-only), which writes it."
        )
    model = CTFFlow(cfg, TrainingData.build(volume, cfg), embeddings(cfg), grf_seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        train_ctfflow(model, cfg, steps=int(cfg.train_steps), seed=seed, checkpoint=str(checkpoint))
    model.eval()
    return model


def depth_profiles(counts, coords, real, markers, axis, cfg):
    """``(n_bins, n_markers)`` soft depth profile, built exactly as ``section_scores`` builds it.

    The bounds and the kernel width are taken from the **real** section on both sides, which is
    what makes a generated profile and a real one comparable bin by bin.
    """
    x = _normalised(np.asarray(counts, dtype=np.float64), cfg)
    p = np.asarray(coords, dtype=np.float64)[:, :2]
    real_p = np.asarray(real.coords, dtype=np.float64)
    real_xyz = torch.from_numpy(
        np.concatenate([real_p, np.full((real_p.shape[0], 1), float(real.z))], axis=1).astype(
            np.float32
        )
    )
    projected = (real_xyz @ axis).numpy()
    bounds = (float(projected.min()), float(projected.max()))
    sigma = float(cfg.profile_sigma_frac) * (bounds[1] - bounds[0]) / int(cfg.profile_n_bins)
    xyz = torch.from_numpy(
        np.concatenate([p, np.full((p.shape[0], 1), float(real.z))], axis=1).astype(np.float32)
    )
    return soft_depth_profile(
        x.index_select(1, markers), xyz, axis, int(cfg.profile_n_bins), sigma, bounds=bounds
    ).numpy()


def safe_r(a, b) -> float:
    """Pearson r, returning 0.0 on a degenerate side. Mirrors ``select._safe_r`` exactly."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def gradient_strength(profile) -> tuple[float, float]:
    """``(trend, contrast)`` of one real depth profile.

    ``trend`` is |Spearman(value, bin index)| — how monotonically the gene rises or falls with
    depth, which is what "laminar gradient" means. ``contrast`` is the coefficient of variation
    across bins — how much the gene varies with depth at all, monotone or not. Both are kept
    because the confound (a flat profile conditions its own Pearson r badly) attaches to
    *contrast*, while the hypothesis is about *trend*, and the partial below holds both.
    """
    bins = np.arange(len(profile), dtype=np.float64)
    trend = 0.0 if np.std(profile) == 0 else abs(float(spearmanr(bins, profile).statistic))
    mean = float(np.mean(profile))
    contrast = 0.0 if mean == 0 else float(np.std(profile) / abs(mean))
    return (0.0 if trend != trend else trend), (0.0 if contrast != contrast else contrast)


def partial_spearman(y, x, controls) -> float:
    """Spearman(y, x) with every column of ``controls`` partialled out, on ranks.

    Returns NaN when the controls explain one of the two sides fully, which is the honest
    answer: there is no residual variation left to correlate.
    """
    ry, rx = (rankdata(np.asarray(v, dtype=np.float64)) for v in (y, x))
    rc = np.column_stack(
        [rankdata(np.asarray(c, dtype=np.float64)) for c in np.atleast_2d(controls)]
    )
    design = np.column_stack([np.ones(len(ry)), rc])
    resid = []
    for v in (ry, rx):
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        resid.append(v - design @ beta)
    if resid[0].std() == 0 or resid[1].std() == 0:
        return float("nan")
    return float(np.corrcoef(resid[0], resid[1])[0, 1])


def neighbour_gradient(text_vecs, trend, k: int):
    """Mean ``trend`` of each gene's ``k`` nearest neighbours in text space, self excluded.

    ``(G, 768)`` text vectors and ``(G,)`` trends in, ``(G,)`` out. Cosine similarity, matching
    the geometry ``embeddings._knn_purity`` reads.
    """
    v = np.asarray(text_vecs, dtype=np.float64)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    cos = v @ v.T
    np.fill_diagonal(cos, -np.inf)
    idx = np.argsort(-cos, axis=1)[:, : min(k, v.shape[0] - 1)]
    return np.asarray(trend)[idx].mean(axis=1)


def _line(row) -> str:
    """One progress line, identical in shape for a fold row and the pooled row."""
    return (
        f"    {row['fold']}: mean gain {row['mean_gain']:+.4f} over {row['n_genes']} genes, "
        f"{100 * row['frac_genes_improved']:.0f}% improved | "
        f"rho(own) {row['rho_gain_vs_own_trend']:+.3f} "
        f"rho(nbr) {row['rho_gain_vs_neighbour_trend']:+.3f} "
        f"partial {row['partial_rho_neighbour_given_own']:+.3f} "
        f"p1={row['permutation_p_one_sided']:.3f} p2={row['permutation_p_two_sided']:.3f}"
    )


def statistics(gain, trend, contrast, text_vecs, k: int, *, n_perm: int, seed: int) -> dict:
    """The three correlations, the two permutation p-values, and the null spread. One block.

    Shared by the per-fold rows and the pooled row so they cannot drift apart. ``text_vecs`` is
    ``(G, 768)`` for the same G genes ``gain``/``trend``/``contrast`` describe, in the same order.
    """
    nbr = neighbour_gradient(text_vecs, trend, k)
    controls = np.column_stack([trend, contrast]).T
    rho = partial_spearman(gain, nbr, controls)
    rng = np.random.default_rng(seed)
    order = np.arange(len(gain))
    null = np.array(
        [
            partial_spearman(
                gain, neighbour_gradient(text_vecs[rng.permutation(order)], trend, k), controls
            )
            for _ in range(n_perm)
        ]
    )
    null = null[np.isfinite(null)]
    finite = null.size > 0 and np.isfinite(rho)
    return {
        "n_genes": int(gain.size),
        "mean_gain": float(gain.mean()),
        "frac_genes_improved": float((gain > 0).mean()),
        "rho_gain_vs_own_trend": float(spearmanr(gain, trend).statistic),
        "rho_gain_vs_own_contrast": float(spearmanr(gain, contrast).statistic),
        "rho_gain_vs_neighbour_trend": float(spearmanr(gain, nbr).statistic),
        "partial_rho_neighbour_given_own": float(rho),
        # One-sided against the *predicted* sign: borrowing predicts a positive partial, so a
        # negative one is not evidence for the hypothesis and should not be spent as if it were.
        # The two-sided p is kept beside it because the sign was predicted before measuring, not
        # after, and a reader is entitled to check that claim against the symmetric test.
        "permutation_p_one_sided": float((null >= rho).mean()) if finite else float("nan"),
        "permutation_p_two_sided": (
            float((np.abs(null) >= abs(rho)).mean()) if finite else float("nan")
        ),
        "n_permutations": int(null.size),
        "null_abs_rho_p95": float(np.percentile(np.abs(null), 95)) if null.size else None,
        "neighbour_trend": [float(v) for v in nbr],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workdir", default="runs/audit_deep", help="where the audit's fits live")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument(
        "--knn", type=int, default=None, help="text neighbours (default text_diag_knn_k)"
    )
    ap.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--out", default="reports/t09_depth_mechanism.md")
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--gene-meta", default=None)
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")

    overrides: dict = {"train_steps": int(args.train_steps)}
    if args.text_cache:
        overrides["text_cache_dir"] = args.text_cache
    if args.gene_meta:
        overrides["gene_meta_path"] = args.gene_meta
    if args.expr_pca_dim is not None:
        overrides["expr_pca_dim"] = int(args.expr_pca_dim)
    base = clamp_config_to_input(base_config(args.seed, **overrides), paths.input)

    volume = load_training_volume(base, paths.input, flattened=args.flattened)
    embeddings = embeddings_factory(volume)
    folds = selection_folds(volume, base)
    axis = profile_axis(volume, base)
    k = int(args.knn or base.text_diag_knn_k)
    print(
        f"  {volume.n_cells} cells x {volume.n_genes} genes, folds "
        f"{[s.section_id for s in folds]}, {base.metric_marker_genes} marker genes, "
        f"{base.profile_n_bins} depth bins, text kNN {k}"
    )

    # The MedCPT vectors the panel is embedded with, taken from the model's own frozen buffer,
    # so this is the geometry the fit actually saw rather than a re-encode. `text_emb_mode` is
    # applied inside `TextGroundedEmbedding._text_channel` and does not change the buffer, so
    # both arms of the text gate share these vectors — which is the point: the question is
    # whether *using* them helps, not whether they differ.
    text_vecs = (
        embeddings(base.replace(text_emb_mode="medcpt")).gene.text_vecs.detach().cpu().numpy()
    )

    results: list[dict] = []
    for gate, won, lost, under in COMPARISONS:
        cfg = base.replace(**under)
        models = {}
        for option in (won, lost):
            arm = cfg.replace(**{gate: option})
            ckpt = Path(args.workdir) / f"fit_{gate}_{option}_seed{args.seed}.pt"
            print(
                f"\n  restoring {gate}={option}  ({arm.content_hash()}) from {ckpt.name}",
                flush=True,
            )
            models[option] = (
                arm,
                restore(arm, volume, embeddings, seed=args.seed, checkpoint=ckpt),
            )

        per_gate: list[tuple] = []
        for fold_i, hidden in enumerate(folds):
            real_counts = np.asarray(hidden.counts.todense(), dtype=np.float64)
            real_x = _normalised(real_counts, cfg)
            w_real = knn_weight_graph(np.asarray(hidden.coords, dtype=np.float64), cfg)
            markers = marker_genes(real_x, w_real, cfg)
            real_profile = None
            per_arm: dict[str, np.ndarray] = {}
            reported: dict[str, float] = {}
            for option, (arm, model) in models.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adata = generate_section(
                        model,
                        section_plane(hidden),
                        volume,
                        arm,
                        args.seed + fold_i,
                        exclude_z={float(hidden.z)},
                    )
                gen_counts = emitted_counts(adata)
                # obsm["xyz"], not obsm[coord_key]: the latter is the plane-local (u, v),
                # which puts every coordinate-referenced metric on the floor.
                gen_coords = np.asarray(adata.obsm["xyz"], dtype=np.float64)[:, :2]
                gen = depth_profiles(gen_counts, gen_coords, hidden, markers, axis, arm)
                if real_profile is None:
                    real_profile = depth_profiles(
                        real_counts, np.asarray(hidden.coords), hidden, markers, axis, arm
                    )
                per_arm[option] = np.array(
                    [safe_r(gen[:, g], real_profile[:, g]) for g in range(gen.shape[1])]
                )

                # The identity the decomposition rests on: these per-gene terms must average to
                # the `marker_depth_r` the audit reported for this arm and this fold.
                types = np.asarray(
                    [volume.celltype_names.index(v) for v in adata.obs[arm.celltype_key]],
                    dtype=np.int64,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    six = section_scores(
                        gen_counts,
                        gen_coords,
                        types,
                        hidden,
                        axis,
                        arm,
                        n_types=len(volume.celltype_names),
                    )
                reported[option] = float(six["marker_depth_r"])
                drift = abs(reported[option] - float(per_arm[option].mean()))
                if drift > IDENTITY_TOL:
                    raise SystemExit(
                        f"decomposition does not add back up for {gate}={option} on "
                        f"{hidden.section_id}: per-gene mean {per_arm[option].mean():.8f} vs "
                        f"section_scores marker_depth_r {reported[option]:.8f} (drift {drift:.2e} "
                        f"> {IDENTITY_TOL:g}). The per-gene terms are explaining a different "
                        "number than the audit reported; fix `depth_profiles` before reading "
                        "anything below."
                    )

            gain = per_arm[won] - per_arm[lost]
            grads = [gradient_strength(real_profile[:, g]) for g in range(real_profile.shape[1])]
            trend = np.array([t for t, _ in grads])
            contrast = np.array([c for _, c in grads])
            m = markers.numpy()

            stats = statistics(
                gain,
                trend,
                contrast,
                text_vecs[m],
                k,
                n_perm=int(args.permutations),
                seed=args.seed,
            )
            nbr = np.asarray(stats.pop("neighbour_trend"))
            per_gate.append((m, gain, trend, contrast))

            row = {
                "gate": gate,
                "won": won,
                "lost": lost,
                "fold": hidden.section_id,
                "dataset": paths.dataset,
                "holdout": paths.holdout,
                "under": dict(under),
                "marker_depth_r_won": reported[won],
                "marker_depth_r_lost": reported[lost],
                **stats,
                "per_gene": [
                    {
                        "gene": volume.gene_names[int(m[g])],
                        "gain": float(gain[g]),
                        "r_won": float(per_arm[won][g]),
                        "r_lost": float(per_arm[lost][g]),
                        "own_trend": float(trend[g]),
                        "own_contrast": float(contrast[g]),
                        "neighbour_trend": float(nbr[g]),
                    }
                    for g in range(gain.size)
                ],
            }
            results.append(row)
            print(_line(row), flush=True)

        # The pooled row: the genes both folds selected as markers, with their gains averaged.
        # Two folds are not independent replicates, so this is not a second measurement — it is
        # the *same* measurement at lower noise, which is what the modest power at 32 genes
        # calls for. Reported beside the per-fold rows, never instead of them.
        shared = sorted(set.intersection(*[set(mm.tolist()) for mm, _, _, _ in per_gate]))
        if len(shared) >= 4 and len(per_gate) > 1:
            picks = [[list(mm).index(g) for g in shared] for mm, _, _, _ in per_gate]
            gain_p = np.mean(
                [gg[ix] for (_, gg, _, _), ix in zip(per_gate, picks, strict=True)], axis=0
            )
            trend_p = np.mean(
                [tt[ix] for (_, _, tt, _), ix in zip(per_gate, picks, strict=True)], axis=0
            )
            contrast_p = np.mean(
                [cc[ix] for (_, _, _, cc), ix in zip(per_gate, picks, strict=True)], axis=0
            )
            idx = np.asarray(shared, dtype=np.int64)
            stats = statistics(
                gain_p,
                trend_p,
                contrast_p,
                text_vecs[idx],
                k,
                n_perm=int(args.permutations),
                seed=args.seed,
            )
            stats.pop("neighbour_trend")
            row = {
                "gate": gate,
                "won": won,
                "lost": lost,
                "fold": f"pooled ({len(folds)} folds)",
                "dataset": paths.dataset,
                "holdout": paths.holdout,
                "under": dict(under),
                "marker_depth_r_won": None,
                "marker_depth_r_lost": None,
                **stats,
                "pooled_genes": [volume.gene_names[int(g)] for g in shared],
            }
            results.append(row)
            print(_line(row), flush=True)
        elif len(per_gate) > 1:
            print(
                f"    pooled row skipped: only {len(shared)} genes are markers in every fold",
                flush=True,
            )

    text = _report(results, base, volume, folds, k, paths, int(args.permutations))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2, default=str))
    print("\n" + text)
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


def _report(results, cfg, volume, folds, k, paths, n_perm) -> str:
    lines = [
        "# Why `marker_depth_r` splits both gates — the per-gene decomposition",
        "",
        f"Dataset **`{paths.dataset}`**, holdout **`{paths.holdout}`** — {volume.n_cells} cells x "
        f"{volume.n_genes} genes, {len(folds)} LOSO folds "
        f"(`{'`, `'.join(s.section_id for s in folds)}`). Up to {cfg.metric_marker_genes} marker "
        f"genes per fold (the `genes` column is what each fold actually had), "
        f"{cfg.profile_n_bins} depth bins, text kNN {k}, {cfg.train_steps} steps, seed "
        f"{cfg.seed}. No refit: the audit's finished fits were restored from their checkpoints.",
        "",
        "`marker_depth_r` is a **mean over marker genes** of a per-gene depth-profile "
        "correlation, so it decomposes exactly — asserted here against `section_scores` to "
        f"{IDENTITY_TOL:g}. That is why this test can carry weight the n = {len(folds)} fold "
        "margin cannot.",
        "",
        "| gate | fold | genes | mean gain | improved | rho(gain, own trend) | rho(gain, nbr "
        "trend) | **partial** | p (1-sided) | p (2-sided) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        pooled = str(r["fold"]).startswith("pooled")
        cell = f"**{r['partial_rho_neighbour_given_own']:+.3f}**"
        lines.append(
            f"| `{r['gate']}` {r['won']} vs {r['lost']} | "
            f"{'**' + str(r['fold']) + '**' if pooled else '`' + str(r['fold']) + '`'} | "
            f"{r['n_genes']} | {r['mean_gain']:+.4f} | "
            f"{100 * r['frac_genes_improved']:.0f}% | {r['rho_gain_vs_own_trend']:+.3f} | "
            f"{r['rho_gain_vs_neighbour_trend']:+.3f} | {cell} | "
            f"{r['permutation_p_one_sided']:.3f} | {r['permutation_p_two_sided']:.3f} |"
        )
    lines += [
        "",
        "**How to read it.** `rho(gain, own trend)` is expected to be positive under *either* "
        "explanation — a gene with a flat real profile conditions its own Pearson r badly, so "
        "any arm difference has more room where the gradient is strong. The hypothesis is "
        "tested by the **partial** column: the gain against the gradient strength of a gene's "
        "*text neighbours*, holding the gene's own trend **and** its own bin-to-bin contrast "
        "fixed. Borrowing predicts that positive with a small permutation p; the conditioning "
        "artefact predicts it near zero whatever the text geometry says.",
        "",
        f"`p` shuffles the text vectors across the marker genes {n_perm} times and rebuilds the "
        "neighbourhood predictor, so it asks whether MedCPT's *actual* geometry matters or only "
        "the shape of the gradient distribution. The **one-sided** p tests the sign the "
        "hypothesis predicts (positive partial); the two-sided p is printed beside it so a "
        "reader can check that claim against the symmetric test.",
        "",
        "**`expr_mode` is the control gate.** It does not touch the text channel. A partial "
        "correlation that appears on both gates is a property of the metric, not of text space; "
        "only a `text_emb_mode` effect *without* an `expr_mode` effect supports the hypothesis.",
        "",
        "**The pooled row is not a third measurement.** The folds are two sections of one volume "
        "scored against one fit per arm, so pooling them averages noise out of the *same* "
        "measurement; it does not add a replicate. It is reported because of the power limit "
        "below, and never instead of the per-fold rows.",
        "",
        "**Power — read a null result carefully.** Calibrated on planted data at "
        f"{cfg.metric_marker_genes} genes by "
        "`scripts/t09_depth_mechanism_calibration.py` "
        "(`reports/t09_depth_mechanism_calibration.md`): the false-positive rate is nominal "
        "(5-8% at p < 0.05, s.e. +-2.8) and the **confound world** — text space encodes the "
        "gradient, but the gain depends only on each gene's own gradient — rejects at **5-8%** "
        "too, i.e. the partial does strip it. But power against a real borrowing effect is only "
        "**18-28% per fold and 35-42% pooled**. A positive here is informative; a null is "
        "**not** evidence of absence, and does not settle what the three-seed run settles.",
        "",
        f"**One seed, and the {len(folds)} folds are not independent replicates** — they are "
        "sections of one volume scored against one fit per arm. Consistency across folds is "
        "evidence; it is not a second seed.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
