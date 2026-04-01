"""
Data preprocessing and dataset utilities for SpatialCPA.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import issparse


class SpatialSection:
    """
    Represents a single physical tissue section at a specific z-layer.

    Each cell retains its actual z-coordinate from the data.

    Parameters
    ----------
    expression : (N, G) array
        Gene expression matrix.
    coords_xy : (N, 2) array
        2D spatial coordinates within the slice.
    z_values : (N,) array
        Per-cell z-coordinates.
    cell_type_indices : (N,) array of int
        Cell type label indices.
    region_indices : (N,) array of int or None
        Region label indices.
    section_id : str
        Identifier for this section.
    """

    def __init__(self, expression, coords_xy, z_values,
                 cell_type_indices, region_indices=None, section_id=''):
        self.expression = np.asarray(expression, dtype=np.float32)
        self.coords_xy = np.asarray(coords_xy, dtype=np.float32)
        self.z_values = np.asarray(z_values, dtype=np.float32)
        self.cell_type_indices = np.asarray(cell_type_indices, dtype=np.int64)
        self.region_indices = region_indices
        if region_indices is not None:
            self.region_indices = np.asarray(region_indices, dtype=np.int64)
        self.section_id = section_id
        self.n_cells = len(expression)
        self.z_center = float(np.median(self.z_values))

    def get_3d_coords(self):
        """Return (N, 3) coords with each cell's actual z."""
        return np.hstack([self.coords_xy, self.z_values.reshape(-1, 1)])


class SectionDataset(Dataset):
    """
    PyTorch dataset that samples cells from a list of sections.

    Each cell retains its actual (x, y, z) coordinate.
    """

    def __init__(self, sections):
        self.sections = sections
        self.cell_indices = []
        for s_idx, sec in enumerate(sections):
            for c_idx in range(sec.n_cells):
                self.cell_indices.append((s_idx, c_idx))

    def __len__(self):
        return len(self.cell_indices)

    def __getitem__(self, idx):
        s_idx, c_idx = self.cell_indices[idx]
        sec = self.sections[s_idx]

        return {
            'xy': torch.tensor(sec.coords_xy[c_idx], dtype=torch.float32),
            'z': torch.tensor(sec.z_values[c_idx], dtype=torch.float32),
            'expression': torch.tensor(sec.expression[c_idx], dtype=torch.float32),
            'cell_type': torch.tensor(sec.cell_type_indices[c_idx], dtype=torch.long),
            'section_idx': torch.tensor(s_idx, dtype=torch.long),
            'region': torch.tensor(
                sec.region_indices[c_idx] if sec.region_indices is not None else -1,
                dtype=torch.long,
            ),
        }


def compute_gap_weights(sections):
    """
    Compute gap sizes between consecutive sections and sampling weights
    proportional to gap size for LOO training.
    """
    z_centers = [s.z_center for s in sections]
    n = len(z_centers)

    if n < 3:
        return [0.0] * max(n - 1, 1), np.ones(n) / n

    gap_sizes = [z_centers[i + 1] - z_centers[i] for i in range(n - 1)]

    weights = np.zeros(n)
    for i in range(n):
        left_gap = gap_sizes[i - 1] if i > 0 else 0.0
        right_gap = gap_sizes[i] if i < n - 1 else 0.0
        weights[i] = (left_gap + right_gap) / 2.0

    weights[0] = 0.0
    weights[-1] = 0.0

    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        interior = np.zeros(n)
        interior[1:-1] = 1.0
        weights = interior / interior.sum()

    return gap_sizes, weights
