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
LEARNERS = ("ridge", "lasso", "knn", "rf", "gbm", "xgb", "mlp")


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


def learner_params(name, args):
    """What actually ran, for ``method_params`` — so a prediction is self-describing."""
    common = {"learner": name, "n_jobs": args.n_jobs}
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
    if learner == "xgb":
        try:
            import xgboost
        except Exception as e:
            print(f"ERROR: --learner xgb requires xgboost and it is not "
                  f"importable: {e}")
            print("  conda install -n bench_spatialcpa -c conda-forge xgboost")
            return None
        versions += f", xgboost {xgboost.__version__}"
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
