"""
Data preprocessing and dataset utilities for SpatialCPA.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import issparse


class SpatialSection:
    """
    Represents a single physical tissue section.

    Parameters
    ----------
    expression : (N, G) array
        Gene expression matrix.
    coords_xy : (N, 2) array
        2D spatial coordinates within the slice.
    z_position : float
        Physical z-position of section center (µm).
    thickness : float
        Physical thickness of the section (µm).
    cell_type_indices : (N,) array of int
        Cell type label indices.
    region_indices : (N,) array of int or None
        Region label indices.
    section_id : str
        Identifier for this section.
    """

    def __init__(self, expression, coords_xy, z_position, thickness,
                 cell_type_indices, region_indices=None, section_id=''):
        self.expression = np.asarray(expression, dtype=np.float32)
        self.coords_xy = np.asarray(coords_xy, dtype=np.float32)
        self.z_position = float(z_position)
        self.thickness = float(thickness)
        self.cell_type_indices = np.asarray(cell_type_indices, dtype=np.int64)
        self.region_indices = region_indices
        if region_indices is not None:
            self.region_indices = np.asarray(region_indices, dtype=np.int64)
        self.section_id = section_id
        self.n_cells = len(expression)

    def get_3d_coords(self, z_offset=0.0):
        """Return (N, 3) coords with z = z_position + z_offset."""
        z = np.full((self.n_cells, 1), self.z_position + z_offset,
                    dtype=np.float32)
        return np.hstack([self.coords_xy, z])


def adata_to_sections(adata, z_key='z', cell_type_key='cell_class',
                      region_key=None, section_thickness=10.0,
                      spatial_key=None):
    """
    Convert an AnnData object into a list of SpatialSection objects.

    Parameters
    ----------
    adata : AnnData
        Input data with obs columns for z and cell type.
    z_key : str
        Column in adata.obs containing z-slice identifiers.
    cell_type_key : str
        Column in adata.obs containing cell type labels.
    region_key : str or None
        Column for region labels.
    section_thickness : float
        Physical section thickness in µm.
    spatial_key : str or None
        Key in adata.obsm for 2D coordinates. If None, tries 'spatial',
        then falls back to obs columns 'x'/'y'.

    Returns
    -------
    sections : list of SpatialSection
    cell_type_names : list of str
    region_names : list of str or None
    gene_names : list of str
    """
    # Get expression matrix
    X = adata.X
    if issparse(X):
        X = np.asarray(X.todense())
    X = np.asarray(X, dtype=np.float32)

    # Get gene names
    gene_names = list(adata.var_names)

    # Get 2D coordinates
    if spatial_key is not None and spatial_key in adata.obsm:
        coords_xy = np.asarray(adata.obsm[spatial_key])[:, :2]
    elif 'spatial' in adata.obsm:
        coords_xy = np.asarray(adata.obsm['spatial'])[:, :2]
    elif 'x' in adata.obs.columns and 'y' in adata.obs.columns:
        coords_xy = np.column_stack([
            adata.obs['x'].values.astype(np.float32),
            adata.obs['y'].values.astype(np.float32),
        ])
    else:
        raise ValueError("Cannot find spatial coordinates. Provide spatial_key "
                         "or ensure 'spatial' in obsm or 'x','y' in obs.")

    # Build cell type mapping
    ct_labels = adata.obs[cell_type_key].values
    cell_type_names = sorted(set(ct_labels))
    ct_to_idx = {name: i for i, name in enumerate(cell_type_names)}
    ct_indices = np.array([ct_to_idx[c] for c in ct_labels], dtype=np.int64)

    # Build region mapping
    region_names = None
    region_indices = None
    if region_key is not None and region_key in adata.obs.columns:
        reg_labels = adata.obs[region_key].values
        region_names = sorted(set(reg_labels))
        reg_to_idx = {name: i for i, name in enumerate(region_names)}
        region_indices = np.array([reg_to_idx[r] for r in reg_labels],
                                  dtype=np.int64)

    # Split by z-value into sections
    z_values = adata.obs[z_key].values
    unique_z = np.sort(np.unique(z_values))

    sections = []
    for z_val in unique_z:
        mask = z_values == z_val
        idx = np.where(mask)[0]

        reg_idx = None
        if region_indices is not None:
            reg_idx = region_indices[idx]

        sec = SpatialSection(
            expression=X[idx],
            coords_xy=coords_xy[idx],
            z_position=float(z_val),
            thickness=section_thickness,
            cell_type_indices=ct_indices[idx],
            region_indices=reg_idx,
            section_id=f'z={z_val}',
        )
        sections.append(sec)

    return sections, cell_type_names, region_names, gene_names


class SectionDataset(Dataset):
    """
    PyTorch dataset that samples cells from a list of sections.

    For each item, returns a cell's (x, y, z) coordinates, expression,
    cell type index, and section metadata for z-marginalization.
    """

    def __init__(self, sections):
        self.sections = sections
        # Flatten all cells with section references
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
            'z_center': torch.tensor(sec.z_position, dtype=torch.float32),
            'z_thickness': torch.tensor(sec.thickness, dtype=torch.float32),
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
    proportional to gap size.

    Returns
    -------
    gap_sizes : list of float
        Gap between section i and section i+1.
    section_weights : (n_sections,) array
        Probability of selecting each section for LOO, proportional to
        the average gap on both sides.
    """
    z_positions = [s.z_position for s in sections]
    n = len(z_positions)

    if n < 3:
        return [0.0] * max(n - 1, 1), np.ones(n) / n

    gap_sizes = [z_positions[i + 1] - z_positions[i] for i in range(n - 1)]

    # Weight per section = average of gaps on both sides
    weights = np.zeros(n)
    for i in range(n):
        left_gap = gap_sizes[i - 1] if i > 0 else 0.0
        right_gap = gap_sizes[i] if i < n - 1 else 0.0
        weights[i] = (left_gap + right_gap) / 2.0

    # Don't hold out first or last section (no interpolation possible)
    weights[0] = 0.0
    weights[-1] = 0.0

    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        # Uniform over interior sections
        interior = np.zeros(n)
        interior[1:-1] = 1.0
        weights = interior / interior.sum()

    return gap_sizes, weights
