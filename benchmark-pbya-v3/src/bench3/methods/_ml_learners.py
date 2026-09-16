"""The learner registry shared by the v14, v18 and v21 expression ablations.

The three wrappers (``run_spatialcpav14_ml.py``, ``run_spatialcpav18_ml.py``,
``run_spatialcpav21_ml.py``) replace their host method's expression step with the
same set of regressors. Those regressors live here, in one file, so that
"identical learners across the three families" is true **by construction** rather
than by three copies staying in step — they had already drifted in comments by the
time the fourth and fifth learners were added, which is how this module came to
exist.

What stays in each wrapper: its parity guard, its feature builder (each host
method has its own ``morphology_features`` and its own config field names), its
target-scale decision and its output tail. What lives here: the learners, the
determinism fix, the parameter record and the dense-output pre-flight check.
"""

import pathlib

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# Order is the order rows appear in tables. `lasso` sits beside `ridge` (both
# linear) and `xgb` beside `gbm` (both boosted trees).
LEARNERS = ("ridge", "lasso", "knn", "rf", "gbm", "xgb", "lgbm", "mlp", "tabm")

# `mlp` and `tabm` are neural, not classical. They are in the set because the
# question is "can a learned regressor beat copying", and excluding the model
# class most likely to win would stack the answer.


class FracAlphaLasso(RegressorMixin, BaseEstimator):
    """Lasso whose penalty is a FRACTION of the value that would zero everything.

    A fixed ``alpha`` is not portable across this benchmark, and the failure is
    silent. Measured on a standardized-feature fixture with log-scale targets,
    scikit-learn's conventional ``alpha=1.0`` leaves **0 of 640 coefficients
    non-zero** (R² = 0.000): every prediction collapses to the per-gene mean, and
    the row still scores, looking like a method. Meanwhile v18 trains on the RAW
    scale, where EASI-FISH intensities are ~10³ times larger, so the *same* alpha
    is effectively no penalty at all. One number cannot mean the same thing in
    both places.

    So the penalty is expressed relative to ``alpha_max = max |Xᵀy| / n`` — the
    smallest penalty that drives every coefficient to zero, and the standard
    starting point of a lasso path. ``alpha = alpha_frac * alpha_max`` is then
    scale-free: it means the same thing on log counts and on raw intensities,
    and the same thing on a 28-gene panel and a 3000-gene one.

    ``coef_``/``intercept_`` are forwarded so the degeneracy check can read them.
    Inherits ``BaseEstimator``/``RegressorMixin`` because scikit-learn's
    ``Pipeline`` needs the tag machinery they provide — a plain class fits but
    raises on ``predict``.
    """

    def __init__(self, alpha_frac=0.01, max_iter=5000, random_state=42):
        self.alpha_frac = alpha_frac
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.linear_model import Lasso
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y2 = y if y.ndim == 2 else y[:, None]
        self.alpha_max_ = float(np.abs(X.T @ y2).max() / max(X.shape[0], 1))
        self.alpha_ = float(self.alpha_frac * self.alpha_max_)
        self.estimator_ = Lasso(alpha=self.alpha_, max_iter=self.max_iter,
                                random_state=self.random_state)
        self.estimator_.fit(X, y)
        self.coef_ = self.estimator_.coef_
        self.intercept_ = self.estimator_.intercept_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class TabMRegressor(RegressorMixin, BaseEstimator):
    """TabM (Gorishniy et al., ICLR 2025) with a scikit-learn surface.

    The published package ships a raw ``nn.Module`` and no estimator API, so the
    training loop is here. Nothing about the model is reimplemented — ``tabm.TabM``
    is constructed and trained as published.

    Multi-gene output is native: ``d_out`` is the head width, so one model covers
    every gene, the same as ``rf`` and unlike ``gbm``/``xgb``/``lgbm`` which need
    one booster each. The forward pass returns ``(batch, k, d_out)`` — TabM's
    parameter-efficient ensemble of ``k`` members, all trained against the same
    target — and prediction is their mean.

    Determinism: the seed is set before construction and the shuffler carries its
    own generator, so two runs at one seed agree bitwise. On CUDA that additionally
    requires deterministic kernels, which ``--tabm-device`` leaves to the caller —
    the ``cpu`` default is reproducible everywhere.
    """

    def __init__(self, n_blocks=3, d_block=256, dropout=0.1, k=32,
                 arch_type="tabm", lr=2e-3, weight_decay=0.0, batch_size=256,
                 max_epochs=100, patience=10, val_frac=0.2, device="cpu",
                 predict_batch=1024, random_state=42):
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.k = k
        self.arch_type = arch_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.device = device
        self.predict_batch = predict_batch
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        from tabm import TabM

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        y2 = y if y.ndim == 2 else y[:, None]
        # Targets are standardized per gene: expression ranges over orders of
        # magnitude between genes (and v18 trains on the raw intensity scale),
        # and an unstandardized MSE would be a handful of loud genes.
        self._y_mean = y2.mean(0)
        self._y_std = y2.std(0) + 1e-8
        yz = (y2 - self._y_mean) / self._y_std

        dev = torch.device(self.device)
        torch.manual_seed(int(self.random_state))
        model = TabM(n_num_features=X.shape[1], cat_cardinalities=None,
                     d_out=y2.shape[1], n_blocks=self.n_blocks,
                     d_block=self.d_block, dropout=self.dropout, k=self.k,
                     arch_type=self.arch_type,
                     start_scaling_init=(None if self.arch_type == "tabm-packed"
                                         else "random-signs")).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        rng = np.random.default_rng(int(self.random_state))
        perm = rng.permutation(X.shape[0])
        n_val = max(1, int(self.val_frac * X.shape[0])) if X.shape[0] > 10 else 0
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xtr = torch.as_tensor(X[tr_idx], device=dev)
        ytr = torch.as_tensor(yz[tr_idx], device=dev)
        Xva = torch.as_tensor(X[val_idx], device=dev) if n_val else None
        yva = torch.as_tensor(yz[val_idx], device=dev) if n_val else None

        gen = torch.Generator().manual_seed(int(self.random_state))
        best, best_state, bad = float("inf"), None, 0
        n = Xtr.shape[0]
        for _ in range(int(self.max_epochs)):
            model.train()
            order = torch.randperm(n, generator=gen).to(dev)
            for i in range(0, n, self.batch_size):
                b = order[i:i + self.batch_size]
                opt.zero_grad(set_to_none=True)
                # (batch, k, d_out) against (batch, 1, d_out): every ensemble
                # member is trained on the same target, which is the method.
                loss = torch.nn.functional.mse_loss(model(Xtr[b]), ytr[b][:, None, :])
                loss.backward()
                opt.step()
            if Xva is None:
                continue
            model.eval()
            with torch.no_grad():
                v = float(torch.nn.functional.mse_loss(model(Xva), yva[:, None, :]))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= int(self.patience):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.val_mse_ = best if Xva is not None else None
        return self

    def predict(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        dev = next(self.model_.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], int(self.predict_batch)):
                xb = torch.as_tensor(X[i:i + int(self.predict_batch)], device=dev)
                out.append(self.model_(xb).mean(1).cpu().numpy())   # mean over k
        p = np.vstack(out) * self._y_std + self._y_mean
        return p


def make_learner(name, args):
    """A scikit-learn-compatible estimator, seeded, with features standardized.

    Standardization matters most for ``knn`` (raw micrometres would swamp the
    one-hot and morphology channels, making it pure geometry), for ``mlp``, and
    for the two linear learners; it is harmless for the tree learners. It is
    applied to every learner so the pipeline is one shape.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    seed = int(args.seed)

    if name == "ridge":
        from sklearn.linear_model import Ridge
        est = Ridge(alpha=args.ridge_alpha, random_state=seed)
    elif name == "lasso":
        est = FracAlphaLasso(alpha_frac=args.lasso_alpha_frac,
                             max_iter=args.lasso_max_iter, random_state=seed)
    elif name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        est = KNeighborsRegressor(n_neighbors=args.knn_k, weights="distance",
                                  n_jobs=args.n_jobs)
    elif name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(
            n_estimators=args.rf_trees, max_features="sqrt",
            min_samples_leaf=args.rf_min_leaf, random_state=seed, n_jobs=args.n_jobs)
    elif name == "gbm":
        # HistGradientBoosting is single-output: one booster per gene. With `xgb`
        # this is the campaign's dominant cost on the wide-panel datasets, and it
        # is inherent to the learner rather than to the wrapper.
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        est = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=args.gbm_iters, learning_rate=args.gbm_lr,
                early_stopping=True, random_state=seed),
            n_jobs=args.n_jobs)
    elif name == "xgb":
        # XGBoost takes 2-D y natively. `multi_strategy` chooses how:
        # "one_output_per_tree" trains an independent booster per gene (the
        # default, and what `gbm` also does); "multi_output_tree" grows trees
        # with vector leaves, one ensemble for all genes. They are different
        # models, not two implementations of one, so the choice is a flag with
        # the measured default rather than a silent pick.
        from xgboost import XGBRegressor
        est = XGBRegressor(
            n_estimators=args.xgb_rounds, max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_lr, subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample, tree_method="hist",
            multi_strategy=args.xgb_multi_strategy,
            random_state=seed, n_jobs=args.n_jobs, verbosity=0)
    elif name == "lgbm":
        # LightGBM is single-output, like `gbm` and `xgb`: one booster per gene.
        # `deterministic` + `force_row_wise` are required for a reproducible fit —
        # without them the histogram construction varies with thread scheduling.
        #
        # `n_jobs=1` on the INNER estimator is not a typo. `MultiOutputRegressor`
        # already parallelises across genes, and leaving the booster at n_jobs=-1
        # inside it oversubscribes every core G times over: measured here as a fit
        # that had not finished G=24 after several minutes, against seconds once
        # the inner threads were pinned to one. Parallelise on one axis only.
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor
        est = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=args.lgbm_rounds, learning_rate=args.lgbm_lr,
                num_leaves=args.lgbm_leaves, min_child_samples=args.lgbm_min_child,
                subsample=1.0, colsample_bytree=1.0,
                deterministic=True, force_row_wise=True,
                random_state=seed, n_jobs=1, verbose=-1),
            n_jobs=args.n_jobs)
    elif name == "tabm":
        est = TabMRegressor(
            n_blocks=args.tabm_blocks, d_block=args.tabm_width,
            dropout=args.tabm_dropout, k=args.tabm_k, arch_type=args.tabm_arch,
            lr=args.tabm_lr, batch_size=args.tabm_batch,
            max_epochs=args.tabm_epochs, patience=args.tabm_patience,
            device=args.tabm_device, random_state=seed)
    elif name == "mlp":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.neural_network import MLPRegressor
        est = TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(args.mlp_hidden, args.mlp_hidden),
                max_iter=args.mlp_iters, early_stopping=True, random_state=seed),
            transformer=StandardScaler())
    else:
        raise ValueError(f"unknown learner {name!r}; choose from {LEARNERS}")
    return Pipeline([("scale", StandardScaler()), ("est", est)])


def freeze_for_determinism(model):
    """Make prediction bitwise reproducible.

    ``RandomForestRegressor`` *fits* deterministically given ``random_state`` —
    measured: identical ``tree_.value`` across two fits at ``n_jobs=-1``. Its
    ``predict`` does not: joblib threads accumulate each tree's contribution into
    a shared array, so the summation order varies and the float result differs run
    to run. Two runs at the same seed must be bitwise identical, so the forest is
    switched to single-threaded prediction after the (parallel) fit.

    Nothing else needs it: ridge, lasso, knn, gbm, xgb and mlp all predict
    identically at ``n_jobs=-1`` (measured, including XGBoost under both
    ``multi_strategy`` settings).
    """
    est = model.named_steps["est"] if hasattr(model, "named_steps") else model
    if hasattr(est, "estimators_") and hasattr(est, "n_jobs") and hasattr(est, "n_estimators"):
        est.n_jobs = 1
    return model


def report_degeneracy(model, name):
    """Say so, loudly, when a linear learner has collapsed to the intercept.

    An all-zero coefficient matrix is not an error — it is a legitimate (if
    useless) fit, and it scores: the prediction becomes the per-gene training
    mean, which a benchmark will happily rank. It must not be mistaken for a
    method. Returns the non-zero fraction, or None for non-linear learners.
    """
    est = model.named_steps["est"] if hasattr(model, "named_steps") else model
    coef = getattr(est, "coef_", None)
    if coef is None:
        return None
    nz = float((np.asarray(coef) != 0).mean())
    extra = ""
    if hasattr(est, "alpha_"):
        extra = (f", alpha={est.alpha_:.4g} "
                 f"({est.alpha_frac:g} x alpha_max={est.alpha_max_:.4g})")
    print(f"    {name}: {nz:.1%} of coefficients non-zero{extra}")
    if nz == 0.0:
        print(f"    ⚠️  WARNING: every coefficient is zero. This row is the "
              f"PER-GENE TRAINING MEAN, not a fitted model. It will still score. "
              f"Lower --lasso-alpha-frac (or raise it, for ridge) before reading "
              f"this row as a learner.")
    return nz


# ── field-guided donor selection ─────────────────────────────────────────────
# The `*_gbm` variants EMIT the learner's prediction. These helpers use the same
# prediction as a SELECTION TARGET instead, and emit a real donor profile. The
# motivation is that the two halves of a regressor's behaviour are separable:
# its conditional mean is a good estimate of the local expression field, but
# emitting that mean is what destroys sparsity and spatial autocorrelation. Using
# it to choose among real cells keeps the estimate and discards the emission.
#
# The bounded-swap design (noise floor, relative margin, worst-first budget) is
# taken from v21's own `_field_align`, which solves the same problem with a
# kNN-interpolated field target instead of a learned one.


def training_index(stack):
    """Row-aligned (xy, z, type, section) for the rows of ``training_matrix``.

    Built by the same iteration order over ``stack.slices``, so row *i* here
    describes the training cell whose profile is row *i* of ``Ytr``.
    """
    xy, z, t, sec = [], [], [], []
    for j, s in enumerate(stack.slices):
        n = int(np.asarray(s.coords_xy).shape[0])
        xy.append(np.asarray(s.coords_xy, dtype=np.float64))
        z.append(np.asarray(s.z_values, dtype=np.float64))
        t.append(np.asarray(s.cell_type_indices, dtype=np.int64)
                 if s.cell_type_indices is not None else np.zeros(n, np.int64))
        sec.append(np.full(n, j, dtype=np.int64))
    return (np.vstack(xy), np.concatenate(z), np.concatenate(t),
            np.concatenate(sec))


def flanking_rows(stack, sec_of, z):
    """Boolean mask over training rows selecting the two sections bracketing z.

    Mirrors ``pick_flanking_slices``: the nearest section at or below z and the
    nearest above it, falling back to the two nearest overall when z sits outside
    the stack. Restricting candidates to these two keeps the donor pool the same
    one the host method grounds from, so the comparison stays like for like.
    """
    centres = np.array([float(np.median(np.asarray(s.z_values)))
                        for s in stack.slices], dtype=np.float64)
    below = np.where(centres <= z)[0]
    above = np.where(centres > z)[0]
    if below.size and above.size:
        pair = [int(below[np.argmax(centres[below])]), int(above[np.argmin(centres[above])])]
    else:
        pair = [int(i) for i in np.argsort(np.abs(centres - z))[:2]]
    return np.isin(sec_of, pair)


def to_target_space(host_expr, scale):
    """Map a host method's EMITTED profile back to the learner's target space.

    Exact in both directions this benchmark uses. On the ``log`` path the host
    emits ``expm1(clip(y, 0, 20))``, so ``log1p`` recovers ``y`` (up to the clip,
    which real log-normalized values do not reach). On the ``raw`` path the host
    emits the measurement itself, which already is the target space.
    """
    x = np.asarray(host_expr, dtype=np.float64)
    return np.log1p(np.clip(x, 0.0, None)) if scale == "log" else x


def select_donor_by_field(pred, incumbent, Ytr, tr_xy, tr_type, cand_mask,
                          q_xy, q_type, args):
    """Swap a bounded set of cells to the real donor that best matches ``pred``.

    All arrays are in the learner's target space. ``pred`` is the learner's
    predicted field at each generated cell; ``incumbent`` is what the host method
    emitted there (its own donor's profile); ``Ytr`` holds the candidate real
    profiles, row-aligned with ``tr_xy`` / ``tr_type``.

    Three guards, all of them from v21's ``_field_align``:

    * a **noise floor** — only cells whose deviation from the predicted field
      exceeds ``noise_mult`` times the real cells' own median deviation are
      eligible. Real cells are noisy too, and aligning a prediction closer to the
      field than real cells sit would over-smooth and inflate Moran's I past the
      ground truth, which is the failure this whole design exists to avoid;
    * a **relative margin** — a swap must reduce the deviation by more than
      ``margin`` times its current value, so marginal swaps are not taken;
    * a **worst-first budget** — at most ``frac`` of cells are touched.

    ⚠️ The floor is estimated from the model's residual on the training cells it
    was fit on. For a flexible learner that residual is optimistically small, so
    the floor admits more cells than a held-out estimate would. The budget and the
    margin, not the floor, are therefore the binding safeguards; ``--gbmfield-frac``
    is the knob to reach for first. Computing an out-of-fold floor would cost a
    second fit of the most expensive learner in the set and is deliberately not
    done here.

    Returns ``(new_target_space_matrix, stats)``; nothing is emitted directly.
    """
    from scipy.spatial import cKDTree

    pred = np.asarray(pred, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    Q, G = pred.shape
    stats = {"n_cells": int(Q), "n_eligible": 0, "n_swapped": 0,
             "dev_before": None, "dev_after": None, "sigma0": None}
    cand_idx = np.where(np.asarray(cand_mask))[0]
    if Q == 0 or cand_idx.size == 0:
        return incumbent.copy(), stats

    rms = lambda A: np.sqrt((A ** 2).mean(axis=1))          # noqa: E731

    # Real cells' own deviation from the predicted field -> the noise floor.
    sigma0 = float(np.median(rms(Ytr[cand_idx] - args._pool_pred[cand_idx])))
    dev = rms(incumbent - pred)
    stats["sigma0"] = sigma0
    stats["dev_before"] = float(np.median(dev))

    eligible = np.where(dev > args.gbmfield_noise_mult * sigma0)[0]
    stats["n_eligible"] = int(eligible.size)
    if eligible.size == 0:
        return incumbent.copy(), stats
    budget = int(round(args.gbmfield_frac * Q))
    eligible = eligible[np.argsort(-dev[eligible])][:max(budget, 0)]
    if eligible.size == 0:
        return incumbent.copy(), stats

    K = int(min(args.gbmfield_k, cand_idx.size))
    tree = cKDTree(tr_xy[cand_idx])
    _, nn = tree.query(q_xy[eligible], k=K)
    if nn.ndim == 1:
        nn = nn[:, None]

    out = incumbent.copy()
    n_swap = 0
    for r, i in enumerate(eligible):
        ci = cand_idx[nn[r]]
        same = ci[tr_type[ci] == q_type[i]]
        if same.size == 0:
            same = ci
        d = rms(Ytr[same] - pred[i][None, :])
        b = int(np.argmin(d))
        if d[b] < (1.0 - args.gbmfield_margin) * dev[i]:
            out[i] = Ytr[same[b]]
            n_swap += 1
    stats["n_swapped"] = int(n_swap)
    stats["dev_after"] = float(np.median(rms(out - pred)))
    return out, stats


def learner_params(name, args):
    """What actually ran, for ``method_params`` — so a prediction is self-describing."""
    common = {"learner": name, "n_jobs": args.n_jobs,
              "emit": getattr(args, "emit", "learner")}
    if getattr(args, "emit", "learner") == "donor":
        common.update({"gbmfield_k": args.gbmfield_k,
                       "gbmfield_frac": args.gbmfield_frac,
                       "gbmfield_margin": args.gbmfield_margin,
                       "gbmfield_noise_mult": args.gbmfield_noise_mult})
    per = {
        "ridge": {"ridge_alpha": args.ridge_alpha},
        "lasso": {"lasso_alpha_frac": args.lasso_alpha_frac,
                  "lasso_max_iter": args.lasso_max_iter,
                  "alpha_rule": "alpha_frac * max|X'y|/n (scale-free)"},
        "knn": {"knn_k": args.knn_k, "weights": "distance"},
        "rf": {"rf_trees": args.rf_trees, "rf_min_leaf": args.rf_min_leaf,
               "max_features": "sqrt"},
        "gbm": {"gbm_iters": args.gbm_iters, "gbm_lr": args.gbm_lr,
                "early_stopping": True, "per_gene_models": True},
        "xgb": {"xgb_rounds": args.xgb_rounds, "xgb_max_depth": args.xgb_max_depth,
                "xgb_lr": args.xgb_lr, "xgb_subsample": args.xgb_subsample,
                "xgb_colsample": args.xgb_colsample,
                "xgb_multi_strategy": args.xgb_multi_strategy,
                "tree_method": "hist"},
        "lgbm": {"lgbm_rounds": args.lgbm_rounds, "lgbm_lr": args.lgbm_lr,
                 "lgbm_leaves": args.lgbm_leaves,
                 "lgbm_min_child": args.lgbm_min_child,
                 "deterministic": True, "per_gene_models": True},
        "tabm": {"tabm_blocks": args.tabm_blocks, "tabm_width": args.tabm_width,
                 "tabm_k": args.tabm_k, "tabm_arch": args.tabm_arch,
                 "tabm_lr": args.tabm_lr, "tabm_epochs": args.tabm_epochs,
                 "tabm_patience": args.tabm_patience,
                 "tabm_device": args.tabm_device,
                 "early_stopping": True, "target_standardized": True},
        "mlp": {"mlp_hidden": args.mlp_hidden, "mlp_iters": args.mlp_iters,
                "early_stopping": True, "target_standardized": True},
    }[name]
    return {**common, **per}


def add_learner_args(p):
    """The learner CLI, identical for all three families."""
    p.add_argument("--learner", required=True, choices=list(LEARNERS),
                   help="expression regressor replacing the host method's "
                        "donor-copy step")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--max-dense-gb", type=float, default=8.0,
                   help="refuse before training if the dense Q x G prediction "
                        "would exceed this (a regressor emits no zeros)")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--lasso-alpha-frac", type=float, default=0.01,
                   help="lasso penalty as a fraction of alpha_max = max|X'y|/n, "
                        "the value that zeroes every coefficient. Scale-free, so "
                        "it means the same thing on log counts and on raw "
                        "intensities. A fixed alpha does not: at the conventional "
                        "alpha=1.0 the fit collapses to the per-gene mean")
    p.add_argument("--lasso-max-iter", type=int, default=5000)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--rf-trees", type=int, default=100)
    p.add_argument("--rf-min-leaf", type=int, default=5)
    p.add_argument("--gbm-iters", type=int, default=100)
    p.add_argument("--gbm-lr", type=float, default=0.1)
    p.add_argument("--xgb-rounds", type=int, default=100)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-lr", type=float, default=0.1)
    p.add_argument("--xgb-subsample", type=float, default=1.0)
    p.add_argument("--xgb-colsample", type=float, default=1.0)
    p.add_argument("--xgb-multi-strategy", default="one_output_per_tree",
                   choices=["one_output_per_tree", "multi_output_tree"],
                   help="how XGBoost handles the multi-gene target: an "
                        "independent booster per gene (default), or one ensemble "
                        "of vector-leaf trees. Different models, not two "
                        "implementations of one")
    p.add_argument("--lgbm-rounds", type=int, default=100)
    p.add_argument("--lgbm-lr", type=float, default=0.1)
    p.add_argument("--lgbm-leaves", type=int, default=31)
    p.add_argument("--lgbm-min-child", type=int, default=20)
    p.add_argument("--tabm-blocks", type=int, default=3)
    p.add_argument("--tabm-width", type=int, default=256)
    p.add_argument("--tabm-dropout", type=float, default=0.1)
    p.add_argument("--tabm-k", type=int, default=32,
                   help="TabM ensemble members (its parameter-efficient ensemble)")
    p.add_argument("--tabm-arch", default="tabm",
                   choices=["tabm", "tabm-mini", "tabm-packed"])
    p.add_argument("--tabm-lr", type=float, default=2e-3)
    p.add_argument("--tabm-batch", type=int, default=256)
    p.add_argument("--tabm-epochs", type=int, default=100)
    p.add_argument("--tabm-patience", type=int, default=10)
    p.add_argument("--tabm-device", default="cpu", choices=["cpu", "cuda"],
                   help="TabM is a torch model. 'cpu' is bitwise reproducible "
                        "everywhere; 'cuda' is much faster on wide panels but "
                        "its determinism depends on the kernels available")
    p.add_argument("--emit", default="learner", choices=["learner", "donor"],
                   help="what the generated cell emits. 'learner' (default, and "
                        "the behaviour of every v*_<learner> method) emits the "
                        "prediction itself. 'donor' uses the prediction only as a "
                        "field target and emits the best-matching REAL local "
                        "profile, so every emitted value stays a measurement")
    p.add_argument("--gbmfield-k", type=int, default=12,
                   help="local same-type real candidates per cell (--emit donor)")
    p.add_argument("--gbmfield-frac", type=float, default=0.35,
                   help="max fraction of cells re-grounded, worst mismatch first")
    p.add_argument("--gbmfield-margin", type=float, default=0.10,
                   help="relative deviation improvement a swap must beat")
    p.add_argument("--gbmfield-noise-mult", type=float, default=1.25,
                   help="eligibility floor, as a multiple of the real cells' own "
                        "median deviation from the predicted field")
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-iters", type=int, default=300)
    return p


def require_learner_deps(learner):
    """Import what this learner needs, or fail naming the package. No fallbacks."""
    try:
        import sklearn
    except Exception as e:
        print(f"ERROR: scikit-learn is required and is not importable: {e}")
        print("  conda install -n bench_spatialcpa scikit-learn")
        return None
    versions = f"scikit-learn {sklearn.__version__}"
    extra = {"xgb": ("xgboost", "conda install -n bench_spatialcpa -c conda-forge xgboost"),
             "lgbm": ("lightgbm", "conda install -n bench_spatialcpa -c conda-forge lightgbm"),
             "tabm": ("tabm", "pip install tabm   (needs torch; see the TabM paper's package)")}
    if learner in extra:
        mod, how = extra[learner]
        try:
            m = __import__(mod)
        except Exception as e:
            print(f"ERROR: --learner {learner} requires {mod} and it is not "
                  f"importable: {e}")
            print(f"  {how}")
            return None
        versions += f", {mod} {getattr(m, '__version__', '?')}"
        if learner == "tabm":
            import torch
            versions += f", torch {torch.__version__}"
    return versions


def assert_args_declared(args, wrapper_file):
    """Every ``args.<name>`` the wrapper reads must exist on the parsed namespace.

    This exists because it was needed. Refactoring the learner flags into this
    module sliced v21's ``--device`` out of its parser along with them — the flag
    sat between two learner flags — and nothing noticed until the run reached
    ``cfg.device = args.device`` and died *after* loading the input and building
    the cell-type vocabulary. v14 and v18 were untouched because their flag-parity
    guards compare against their host wrapper's whole flag table; v21_ml declares
    no host flags at all (it builds ``V14Config()`` directly), so it had no
    equivalent check.

    Walks the wrapper's own AST for attribute reads on the name ``args`` — exact,
    so a mention in a docstring or comment cannot produce a false failure — and
    fails before any data is read rather than partway through a run.
    """
    import ast
    tree = ast.parse(pathlib.Path(wrapper_file).read_text())
    used = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "args" and isinstance(n.ctx, ast.Load)}
    missing = sorted(a for a in used if not hasattr(args, a))
    if missing:
        raise SystemExit(
            f"ERROR: {pathlib.Path(wrapper_file).name} reads "
            f"{', '.join('args.' + a for a in missing)}, which its parser does not "
            f"declare. This is a wrapper bug, not a configuration problem — the run "
            f"would have failed partway through. Add the flag(s) to build_parser().")
    return len(used)


def check_output_size(adata, n_targets, max_gb):
    """Refuse before training if the dense prediction cannot be written.

    A donor-copy method emits sparse real measurements; a squared-loss regressor
    emits the conditional mean, which has NO zeros. The prediction is stored as
    CSR, so every one of the Q x G entries costs its value plus its column index,
    on top of a dense Q x G array that must be materialized first. Reported up
    front rather than discovered after the host method's flow has trained.
    """
    n_sections = int(adata.obs["section"].nunique())
    est_q = (adata.n_obs / max(n_sections, 1)) * max(n_targets, 1)
    dense_gb = est_q * adata.n_vars * 4 / 1e9
    print(f"  dense-output estimate: ~{est_q:,.0f} cells x {adata.n_vars} genes "
          f"= {dense_gb:.2f} GB dense (~{dense_gb * 2:.2f} GB as CSR)")
    if dense_gb > max_gb:
        raise SystemExit(
            f"ERROR: this prediction would be ~{dense_gb:.1f} GB dense, over the "
            f"--max-dense-gb limit of {max_gb}. A regressor emits no zeros, so an "
            f"uncapped whole-transcriptome panel ({adata.n_vars} genes here) "
            f"cannot be written as a prediction.\n"
            f"  Options, in the order I would take them:\n"
            f"    1. Leave this dataset out of the ablation and say so — the "
            f"targeted panels answer the question.\n"
            f"    2. Build a gene-capped copy of it (prepare_dataset --n-hvg "
            f"3000) and register it as a SEPARATE dataset; capping in place "
            f"would change the panel that previously-reported rows were "
            f"measured on.\n"
            f"    3. Raise --max-dense-gb deliberately, if you really have the "
            f"disk.\n"
            f"  Do NOT threshold small predictions to zero: that contaminates "
            f"paper_gene_detection_spearman, which is the column this ablation "
            f"exists to read.")
