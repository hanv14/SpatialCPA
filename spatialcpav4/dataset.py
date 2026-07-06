"""
PyTorch dataset that turns triplet samples into transformer token batches.

Each sample corresponds to one target location.  ``__getitem__`` gathers the
``2 * k`` neighbor spots (k from the lower slice, k from the upper slice) from
the shared :class:`~spatialcpav4.data.SliceStack` global table and builds:

* per-token expression vectors,
* per-token cell-type / region indices,
* per-token relative coordinates ``[Δx, Δy, Δz, ‖Δ‖]`` (raw physical units; the
  model normalises them with a learned/estimated ``coord_scale`` buffer),
* per-token "side" ids (0 = lower slice, 1 = upper slice),
* a padding mask (True where a token is padding),

together with the supervision targets (expression, cell type, region,
occupancy) and a ``has_target`` flag that disables expression/label loss for
background (occupancy-only) samples.

Keeping only *indices* per sample — and gathering the heavy expression rows
lazily here — keeps memory flat and lets ``num_workers`` scale.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import SliceStack, TripletSamples


class TripletTokenDataset(Dataset):
    """Neighbor-token dataset for the ``{i-1, i+1} -> i`` transformer.

    Parameters
    ----------
    stack
        The :class:`SliceStack` whose global table neighbor features are gathered
        from.
    samples
        The :class:`TripletSamples` produced by
        :func:`~spatialcpav4.data.build_triplet_samples`.
    n_neighbors
        Neighbors per side (``k``); ``2k`` tokens total.
    indices
        Optional subset of sample rows (used for train/val splitting).
    """

    def __init__(
        self,
        stack: SliceStack,
        samples: TripletSamples,
        n_neighbors: int,
        indices: np.ndarray | None = None,
    ) -> None:
        self.stack = stack
        self.samples = samples
        self.k = n_neighbors
        self.n_tokens = 2 * n_neighbors
        self.indices = (
            np.arange(len(samples)) if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

        # Side ids are identical for every sample: [0]*k + [1]*k.
        self._side = np.concatenate(
            [np.zeros(self.k, dtype=np.int64), np.ones(self.k, dtype=np.int64)]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s = self.samples
        row = self.indices[i]

        # Global neighbor indices: (2k,)
        tok_idx = np.concatenate([s.lower_idx[row], s.upper_idx[row]])

        # Gather neighbor features from the shared global table.
        tok_expr = self.stack.expr[tok_idx]                    # (2k, G)
        tok_ct = self.stack.cell_type[tok_idx]                 # (2k,)
        tok_reg = self.stack.region[tok_idx]                   # (2k,)
        tok_coords = self.stack.coords[tok_idx]                # (2k, 3)

        # Relative coordinates w.r.t. the target location.
        target_coord = s.target_coords[row]                    # (3,)
        delta = tok_coords - target_coord[None, :]             # (2k, 3)
        dist = np.linalg.norm(delta, axis=1, keepdims=True)    # (2k, 1)
        rel = np.concatenate([delta, dist], axis=1).astype(np.float32)  # (2k, 4)

        # Padding mask: knn padding repeats indices, so a token is "pad" when
        # it duplicates an earlier token on the same side.  We reconstruct this
        # cheaply by flagging exact-duplicate (idx, side) pairs after the first.
        pad_mask = _duplicate_pad_mask(tok_idx, self.k)

        # Supervision targets.
        has_target = float(s.has_target[row])
        if s.target_spot[row] >= 0:
            target_expr = self.stack.expr[s.target_spot[row]]
            target_ct = int(self.stack.cell_type[s.target_spot[row]])
            target_reg = int(self.stack.region[s.target_spot[row]])
        else:  # background sample — no expression/label supervision
            target_expr = np.zeros(self.stack.n_genes, dtype=np.float32)
            target_ct = -1
            target_reg = -1

        return {
            "token_expr": torch.from_numpy(np.ascontiguousarray(tok_expr)),
            "token_ct": torch.from_numpy(np.ascontiguousarray(tok_ct)),
            "token_reg": torch.from_numpy(np.ascontiguousarray(tok_reg)),
            "token_relcoord": torch.from_numpy(rel),
            "token_side": torch.from_numpy(self._side.copy()),
            "token_pad_mask": torch.from_numpy(pad_mask),
            "target_expr": torch.from_numpy(np.ascontiguousarray(target_expr)),
            "target_ct": torch.tensor(target_ct, dtype=torch.long),
            "target_reg": torch.tensor(target_reg, dtype=torch.long),
            "target_occ": torch.tensor(s.occupancy[row], dtype=torch.float32),
            "has_target": torch.tensor(has_target, dtype=torch.float32),
            # log1p of local intensity λ (stabilises the wide density range).
            "target_density": torch.tensor(
                float(np.log1p(s.target_density[row])), dtype=torch.float32),
        }


def _duplicate_pad_mask(tok_idx: np.ndarray, k: int) -> np.ndarray:
    """Flag padded tokens (duplicate neighbors introduced by knn padding).

    Padding is only produced when a source slice has fewer than ``k`` spots, in
    which case :func:`~spatialcpav4.data.knn_indices` repeats the last neighbor.
    We mark repeats (per side) so the transformer can ignore them via
    ``src_key_padding_mask``.  ``True`` means "pad / ignore".
    """
    mask = np.zeros(tok_idx.shape[0], dtype=bool)
    for start in (0, k):  # lower side, then upper side
        seen = set()
        for j in range(start, start + k):
            v = int(tok_idx[j])
            if v in seen:
                mask[j] = True
            else:
                seen.add(v)
    return mask
