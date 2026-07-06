"""Central leakage-prevention utilities for benchmark-pbya-v2.

benchmark-pbya-v2 is a leakage-hardened variant of the benchmark. Every
potential path by which the held-out section could influence what a method sees
is closed *here*, in one place, so the individual method wrappers cannot
accidentally reintroduce leakage.

Leakage vectors addressed
-------------------------
1. **Membership**: the held-out section's cells must never appear in a method's
   input. :func:`split_holdout` separates train vs. held-out and
   :func:`assert_no_leakage` hard-fails if any held-out cell survives.
2. **Upstream registration**: several datasets ship coordinates that were
   globally registered by the data provider using *all* sections (including the
   one we hold out). :func:`reregister_training` throws that away and re-registers
   the **training slices only** into a common frame, so the coordinate frame the
   method consumes never depended on the held-out slice.
3. **Held-out geometry**: methods are given only a scalar **target z** (a
   position), never the held-out (x, y). Enforced by the v2 wrapper interface.
4. **Global statistics**: label vocabularies / clustering / any pooled statistic
   must be computed on training cells only. :func:`build_labels_train_only`
   provides a leakage-safe label builder; expression normalization used by the
   wrappers is per-cell (see note in the wrappers).

The registration helpers (Kabsch/Umeyama + ICP) are dependency-light (numpy +
scipy) so they run in every method's conda env. An optional PASTE hook is used
when the ``paste`` package is available for expression-aware alignment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# Holdout split + membership guard                                             #
# --------------------------------------------------------------------------- #
def split_holdout(adata, holdout_sections):
    """Split an AnnData into (train, held-out) by section label.

    Returns
    -------
    train_adata : AnnData
        All cells NOT in ``holdout_sections`` (a copy).
    holdout_adata : AnnData
        The held-out cells (a copy) — used ONLY for evaluation, never fed to a
        method.
    """
    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, [str(s) for s in holdout_sections])
    if holdout_mask.sum() == 0:
        raise ValueError(f"No cells found for holdout sections {holdout_sections}")
    train_adata = adata[~holdout_mask].copy()
    holdout_adata = adata[holdout_mask].copy()
    assert_no_leakage(train_adata, holdout_sections)
    return train_adata, holdout_adata


def assert_no_leakage(train_adata, holdout_sections):
    """Hard-fail if any held-out section leaked into the training set."""
    train_secs = set(train_adata.obs["section"].values.astype(str))
    holds = {str(s) for s in holdout_sections}
    overlap = train_secs & holds
    if overlap:
        raise AssertionError(
            f"LEAKAGE: held-out sections {sorted(overlap)} present in training data"
        )


# --------------------------------------------------------------------------- #
# Rigid / similarity registration primitives                                   #
# --------------------------------------------------------------------------- #
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False
            ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Solves ``dst ≈ s · R · src + t`` (Umeyama 1991) given *corresponding* point
    sets. Works in 2-D or 3-D.

    Parameters
    ----------
    src, dst : (N, D) arrays of corresponding points.
    with_scale : bool
        If False, ``s = 1`` (pure rigid: rotation + translation).

    Returns
    -------
    R : (D, D) rotation, t : (D,) translation, s : float scale.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, d = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 0 else 1.0
    else:
        s = 1.0
    t = mu_dst - s * (R @ mu_src)
    return R, t, s


def apply_transform(xy: np.ndarray, R: np.ndarray, t: np.ndarray, s: float
                    ) -> np.ndarray:
    """Apply ``s · R · xy + t``."""
    return (s * (xy @ R.T)) + t


def icp_align(src_xy: np.ndarray, dst_xy: np.ndarray,
              with_scale: bool = False, max_iter: int = 50, tol: float = 1e-6,
              init_pca: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """Align ``src_xy`` onto ``dst_xy`` via Iterative Closest Point.

    Correspondence-free rigid/similarity registration for two point clouds that
    represent *different* cells of neighboring tissue sections. Initialised by
    centroid (and optionally PCA principal-axis) matching, then refined by
    alternating nearest-neighbor correspondences and :func:`umeyama`.

    Returns the cumulative transform (R, t, s) mapping src -> dst frame.

    Notes
    -----
    This is a coordinate-only heuristic. For expression-aware alignment install
    ``paste`` and use ``method="paste"`` in :func:`reregister_training`.
    """
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    d = src.shape[1]

    # --- initialisation: centroid + (optional) PCA axis alignment ---------- #
    R = np.eye(d)
    t = dst.mean(axis=0) - src.mean(axis=0)
    s = 1.0
    if init_pca and src.shape[0] >= 3 and dst.shape[0] >= 3:
        def principal_axes(p):
            pc = p - p.mean(axis=0)
            _, _, vt = np.linalg.svd(pc, full_matrices=False)
            return vt
        Vs = principal_axes(src)
        Vd = principal_axes(dst)
        R0 = Vd.T @ Vs
        if np.linalg.det(R0) < 0:  # reflect -> proper rotation
            Vd = Vd.copy(); Vd[-1] *= -1
            R0 = Vd.T @ Vs
        R = R0
        t = dst.mean(axis=0) - (R @ src.mean(axis=0))

    cur = apply_transform(src, R, t, s)
    tree = cKDTree(dst)
    prev_err = np.inf
    for _ in range(max_iter):
        _, idx = tree.query(cur, k=1)
        Rn, tn, sn = umeyama(src, dst[idx], with_scale=with_scale)
        cur = apply_transform(src, Rn, tn, sn)
        err = float(np.mean(np.sum((cur - dst[idx]) ** 2, axis=1)))
        R, t, s = Rn, tn, sn
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return R, t, s


# --------------------------------------------------------------------------- #
# Training-only re-registration                                                #
# --------------------------------------------------------------------------- #
def _sorted_train_sections(adata) -> List[str]:
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    labels = np.unique(sections)
    return sorted(labels, key=lambda s: np.median(coords[sections == s, 2]))


def reregister_training(adata, method: str = "rigid", with_scale: bool = False,
                        verbose: bool = False):
    """Re-register the **training** slices into a common frame (leakage-free).

    The held-out section must already be removed (call after
    :func:`split_holdout`). Consecutive training slices (sorted by z) are chained:
    the first is the anchor (identity); each subsequent slice's (x, y) is aligned
    to the cumulative frame of the previous slice. Because only training slices
    participate, the resulting frame never depended on the held-out slice.

    Parameters
    ----------
    adata : AnnData
        Training-only data with 3-D ``obsm['spatial']`` (z in column 2).
    method : str
        ``"rigid"``  — coordinate-only ICP (numpy/scipy, always available).
        ``"paste"``  — expression-aware PASTE if installed, else falls back.
        ``"none"``   — assume already aligned; identity (records identity xforms).
    with_scale : bool
        Allow an isotropic scale factor in the alignment.

    Returns
    -------
    adata : AnnData
        Same object with ``obsm['spatial']`` xy replaced by re-registered coords
        (z unchanged) and per-section transforms stored in
        ``uns['v2_registration']``.
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64).copy()
    sections = adata.obs["section"].values.astype(str)
    order = _sorted_train_sections(adata)

    transforms: Dict[str, Dict] = {}
    d = 2  # register in-plane (x, y)

    if method == "paste":
        aligned = _try_paste(adata, order, verbose=verbose)
        if aligned is not None:
            coords[:, :2] = aligned
            adata.obsm["spatial"] = coords
            adata.uns["v2_registration"] = {"method": "paste", "sections": order}
            return adata
        if verbose:
            print("  PASTE unavailable; falling back to rigid ICP")
        method = "rigid"

    # Anchor: first section identity.
    anchor = order[0]
    transforms[anchor] = _identity_transform(d)
    prev_mask = sections == anchor
    prev_xy = coords[prev_mask, :2]

    for sec in order[1:]:
        m = sections == sec
        cur_xy = coords[m, :2]
        if method == "none":
            R, t, s = np.eye(d), np.zeros(d), 1.0
        else:  # rigid ICP against the previous (already-aligned) slice
            R, t, s = icp_align(cur_xy, prev_xy, with_scale=with_scale)
        new_xy = apply_transform(cur_xy, R, t, s)
        coords[m, :2] = new_xy
        transforms[sec] = {"R": R.tolist(), "t": t.tolist(), "s": float(s)}
        prev_xy = new_xy
        if verbose:
            print(f"  registered {sec} -> anchor frame (scale={s:.3f})")

    adata.obsm["spatial"] = coords
    adata.uns["v2_registration"] = {"method": method, "sections": order,
                                    "transforms": transforms}
    return adata


def _identity_transform(d: int) -> Dict:
    return {"R": np.eye(d).tolist(), "t": np.zeros(d).tolist(), "s": 1.0}


def _try_paste(adata, order, verbose=False):
    """Attempt PASTE pairwise alignment; return (N,2) xy or None if unavailable."""
    try:
        import paste as pst  # noqa: F401
        import anndata as ad
    except Exception:
        return None
    try:
        sections = adata.obs["section"].values.astype(str)
        slices = []
        idx_map = []
        for sec in order:
            m = np.where(sections == sec)[0]
            sub = adata[m].copy()
            sub.obsm["spatial"] = np.asarray(sub.obsm["spatial"])[:, :2]
            slices.append(sub)
            idx_map.append(m)
        # Sequential pairwise PASTE alignment onto the anchor frame.
        new_coords = [np.asarray(slices[0].obsm["spatial"], dtype=np.float64)]
        for i in range(1, len(slices)):
            pi = pst.pairwise_align(slices[i - 1], slices[i])
            # Map slice i onto slice i-1's (already-updated) frame via the OT plan.
            prev = new_coords[i - 1]
            w = np.asarray(pi)
            w = w / (w.sum(axis=0, keepdims=True) + 1e-12)
            mapped = w.T @ prev
            new_coords.append(mapped)
        out = np.zeros((adata.n_obs, 2), dtype=np.float64)
        for m, xy in zip(idx_map, new_coords):
            out[m] = xy
        return out
    except Exception as e:
        if verbose:
            print(f"  PASTE alignment failed ({e})")
        return None


# --------------------------------------------------------------------------- #
# Train-only label building                                                     #
# --------------------------------------------------------------------------- #
def build_labels_train_only(adata, key, train_mask, seed=42, leiden_fallback=True):
    """Leakage-safe categorical label indices (vocabulary from training cells).

    Mirrors the fix used in the spatialcpav4 wrapper but reusable by any v2
    wrapper. The vocabulary — and any clustering — is derived from ``train_mask``
    cells only; held-out cells map to -1 when their label is unseen and are never
    consumed in training.

    Returns
    -------
    (indices int64[n_all], names list[str]) or (None, None) if absent and no
    fallback applies.
    """
    import pandas as pd

    train_idx = np.where(train_mask)[0]

    if key in adata.obs.columns:
        labels = adata.obs[key].values.astype(str)
        names = sorted(pd.unique(labels[train_mask]).tolist())
        idx_map = {n: i for i, n in enumerate(names)}
        idx = np.array([idx_map.get(l, -1) for l in labels], dtype=np.int64)
        return idx, names

    if key != "cell_type" or not leiden_fallback:
        return None, None

    try:
        import scanpy as sc
        tmp = adata[train_mask].copy()
        sc.pp.pca(tmp, n_comps=min(30, tmp.n_vars - 1, tmp.n_obs - 1))
        sc.pp.neighbors(tmp, n_neighbors=15)
        sc.tl.leiden(tmp, resolution=1.0, random_state=seed)
        train_labels = tmp.obs["leiden"].values.astype(str)
        names = sorted(pd.unique(train_labels).tolist(), key=lambda s: int(s))
        idx_map = {n: i for i, n in enumerate(names)}
        idx = np.full(adata.n_obs, -1, dtype=np.int64)
        idx[train_idx] = np.array([idx_map[l] for l in train_labels], dtype=np.int64)
        names = [f"leiden_{n}" for n in names]
        return idx, names
    except Exception:
        return np.zeros(adata.n_obs, dtype=np.int64), ["type_0"]


# --------------------------------------------------------------------------- #
# Evaluation-side rigid alignment (predictions -> ground truth)                #
# --------------------------------------------------------------------------- #
def align_prediction_to_gt(pred_xy: np.ndarray, gt_xy: np.ndarray,
                           with_scale: bool = True, n_init_angles: int = 12
                           ) -> np.ndarray:
    """Rigidly align predicted coordinates onto GT before matching.

    In a generation benchmark the absolute coordinate frame is arbitrary (the
    method synthesizes in the training frame; the training frame was re-registered
    training-only, so it differs from the held-out GT frame by a rigid transform).
    This aligns the predicted point cloud onto the GT cloud so that downstream
    nearest-neighbor matching is frame-invariant.

    A single ICP run initialised from the PCA principal axis is unreliable when
    the tissue outline is near-symmetric (e.g. a roundish occupancy blob): ICP
    can settle in a rotated local minimum, mismatching interior cells and
    depressing per-cell correlation even for a good prediction. To avoid that we
    try multiple initial rotations (and a reflection), ICP-refine each, and keep
    the transform with the lowest nearest-neighbor residual — a global,
    orientation-robust alignment.

    This is an EVALUATION-side operation: it uses the GT (which evaluation is
    allowed to see) and does not feed anything back to the method — so it is not
    leakage. Returns the aligned predicted coordinates.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    if len(pred_xy) < 3 or len(gt_xy) < 3:
        return pred_xy

    gt_c = gt_xy.mean(axis=0)
    p_c = pred_xy.mean(axis=0)
    pred_centered = pred_xy - p_c
    tree = cKDTree(gt_xy)

    best_out = None
    best_err = np.inf
    for k in range(max(n_init_angles, 1)):
        ang = 2.0 * np.pi * k / max(n_init_angles, 1)
        c, sn = np.cos(ang), np.sin(ang)
        Rot = np.array([[c, -sn], [sn, c]])
        for refl in (1.0, -1.0):
            R0 = Rot @ np.array([[refl, 0.0], [0.0, 1.0]])
            init = pred_centered @ R0.T + gt_c
            R, t, s = icp_align(init, gt_xy, with_scale=with_scale, init_pca=False)
            out = apply_transform(init, R, t, s)
            err = float(np.mean(tree.query(out, k=1)[0]))
            if err < best_err:
                best_err = err
                best_out = out
    return best_out if best_out is not None else pred_xy
