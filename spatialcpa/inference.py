"""
Virtual Slice Generation and Inference for SpatialCPA.
"""

import numpy as np
import torch
import anndata as ad
from scipy.spatial import cKDTree


class VirtualSliceGenerator:
    """
    Generate virtual tissue slices at arbitrary z-positions.

    Parameters
    ----------
    model : SpatialCPA
        Trained SpatialCPA model.
    cell_type_names : list of str
        Names of cell types (ordered by index).
    gene_names : list of str
        Names of genes (ordered by index).
    region_names : list of str or None
        Names of regions.
    device : str
        Device for inference.
    """

    def __init__(self, model, cell_type_names, gene_names,
                 region_names=None, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.cell_type_names = cell_type_names
        self.gene_names = gene_names
        self.region_names = region_names
        self.device = device

    @torch.no_grad()
    def _query_grid(self, z, xy_range, grid_size=100):
        """
        Query cell type probabilities on a coarse 2D grid at given z.

        Returns grid positions and max cell type probability at each position.
        """
        x_min, x_max, y_min, y_max = xy_range
        xs = np.linspace(x_min, x_max, grid_size)
        ys = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
        grid_z = np.full((len(grid_xy), 1), z)
        grid_3d = np.hstack([grid_xy, grid_z]).astype(np.float32)

        coords = torch.tensor(grid_3d, device=self.device)
        probs = self.model.predict_cell_type(coords)
        max_prob = probs.max(dim=1).values.cpu().numpy()

        return grid_xy, max_prob

    def _poisson_disk_sample(self, boundary_xy, n_cells, min_dist=None):
        """
        Sample cell positions within the tissue boundary using
        rejection sampling with minimum distance constraint.
        """
        if min_dist is None:
            # Estimate from desired cell count and area
            area = (boundary_xy[:, 0].max() - boundary_xy[:, 0].min()) * \
                   (boundary_xy[:, 1].max() - boundary_xy[:, 1].min())
            min_dist = np.sqrt(area / n_cells) * 0.5

        x_min, x_max = boundary_xy[:, 0].min(), boundary_xy[:, 0].max()
        y_min, y_max = boundary_xy[:, 1].min(), boundary_xy[:, 1].max()

        # Build KD-tree of boundary points for "inside tissue" check
        boundary_tree = cKDTree(boundary_xy)
        boundary_radius = min_dist * 5  # max distance to be considered "in tissue"

        points = []
        max_attempts = n_cells * 20

        for _ in range(max_attempts):
            if len(points) >= n_cells:
                break
            # Random candidate
            candidate = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
            ])
            # Check if near boundary (inside tissue)
            dist, _ = boundary_tree.query(candidate)
            if dist > boundary_radius:
                continue
            # Check minimum distance to existing points
            if len(points) > 0:
                pts = np.array(points)
                dists = np.sqrt(((pts - candidate) ** 2).sum(axis=1))
                if dists.min() < min_dist:
                    continue
            points.append(candidate)

        if len(points) < n_cells:
            # Fall back to random sampling if Poisson disk is too slow
            extra_needed = n_cells - len(points)
            for _ in range(extra_needed * 50):
                if len(points) >= n_cells:
                    break
                candidate = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max),
                ])
                dist, _ = boundary_tree.query(candidate)
                if dist <= boundary_radius:
                    points.append(candidate)

        return np.array(points[:n_cells])

    @torch.no_grad()
    def generate(self, z, xy_range, n_cells=None, reference_adata=None,
                 confidence_threshold=0.3, grid_size=80, sample_expression=True,
                 batch_size=2048):
        """
        Generate a virtual tissue slice at position z.

        Parameters
        ----------
        z : float
            Z-position for the virtual slice (µm).
        xy_range : tuple (x_min, x_max, y_min, y_max)
            Spatial extent of the slice.
        n_cells : int or None
            Target number of cells. If None, uses reference_adata.n_obs.
        reference_adata : AnnData or None
            Reference slice for cell count and coordinate range.
        confidence_threshold : float
            Minimum cell type probability to consider a position as tissue.
        grid_size : int
            Resolution for tissue boundary detection.
        sample_expression : bool
            If True, sample from ZINB. If False, return mean expression.
        batch_size : int
            Batch size for model inference.

        Returns
        -------
        virtual_adata : AnnData
            Virtual slice with expression, cell types, and coordinates.
        """
        if reference_adata is not None:
            if n_cells is None:
                n_cells = reference_adata.n_obs
            if xy_range is None:
                if 'spatial' in reference_adata.obsm:
                    sp = np.asarray(reference_adata.obsm['spatial'])
                elif 'x' in reference_adata.obs.columns:
                    sp = np.column_stack([
                        reference_adata.obs['x'].values,
                        reference_adata.obs['y'].values,
                    ])
                else:
                    raise ValueError("Cannot determine xy_range from reference")
                margin = 10.0
                xy_range = (sp[:, 0].min() - margin, sp[:, 0].max() + margin,
                            sp[:, 1].min() - margin, sp[:, 1].max() + margin)

        if n_cells is None:
            n_cells = 5000

        # Step A: Determine tissue boundary
        grid_xy, max_prob = self._query_grid(z, xy_range, grid_size)
        tissue_mask = max_prob > confidence_threshold
        tissue_xy = grid_xy[tissue_mask]

        if len(tissue_xy) < 10:
            # Fallback: use full grid
            tissue_xy = grid_xy

        # Step B: Sample cell positions
        cell_positions = self._poisson_disk_sample(tissue_xy, n_cells)
        if len(cell_positions) == 0:
            # Ultimate fallback
            cell_positions = tissue_xy[
                np.random.choice(len(tissue_xy), min(n_cells, len(tissue_xy)),
                                 replace=True)
            ]

        actual_n = len(cell_positions)
        coords_3d = np.hstack([
            cell_positions,
            np.full((actual_n, 1), z),
        ]).astype(np.float32)

        # Process in batches
        all_ct_probs = []
        all_expr = []
        all_region_probs = []

        for start in range(0, actual_n, batch_size):
            end = min(start + batch_size, actual_n)
            batch_coords = torch.tensor(coords_3d[start:end],
                                        device=self.device)

            # Step C: Cell type probabilities
            ct_probs = self.model.predict_cell_type(batch_coords)
            all_ct_probs.append(ct_probs.cpu().numpy())

            # Step D: Region predictions
            if self.model.region_head is not None:
                h = self.model.encode_coords(batch_coords)
                reg_probs = self.model.region_head.predict_proba(h)
                all_region_probs.append(reg_probs.cpu().numpy())

        ct_probs_all = np.concatenate(all_ct_probs, axis=0)

        # Sample cell types from distribution
        cell_types = np.array([
            np.random.choice(len(self.cell_type_names), p=p)
            for p in ct_probs_all
        ])
        cell_type_labels = [self.cell_type_names[i] for i in cell_types]

        # Step E: Generate expression
        for start in range(0, actual_n, batch_size):
            end = min(start + batch_size, actual_n)
            batch_coords = torch.tensor(coords_3d[start:end],
                                        device=self.device)
            batch_ct = torch.tensor(cell_types[start:end],
                                    dtype=torch.long, device=self.device)

            if sample_expression:
                expr = self.model.predict_expression(
                    batch_coords, batch_ct, sample=True
                ).cpu().numpy()
            else:
                expr = self.model.predict_expression(
                    batch_coords, batch_ct, sample=False
                ).cpu().numpy()
            all_expr.append(expr)

        expr_all = np.concatenate(all_expr, axis=0)

        # Build AnnData
        import pandas as pd
        obs = pd.DataFrame({
            'cell_class': cell_type_labels,
            'cell_type_idx': cell_types,
            'x': cell_positions[:, 0],
            'y': cell_positions[:, 1],
            'z': z,
        })

        if len(all_region_probs) > 0:
            region_probs_all = np.concatenate(all_region_probs, axis=0)
            regions = region_probs_all.argmax(axis=1)
            obs['region'] = [self.region_names[i] for i in regions]

        virtual_adata = ad.AnnData(
            X=expr_all.astype(np.float32),
            obs=obs,
            var=pd.DataFrame(index=self.gene_names),
        )
        virtual_adata.obsm['spatial'] = cell_positions.astype(np.float32)

        return virtual_adata

    @torch.no_grad()
    def generate_matching(self, z, reference_adata, cell_type_key='cell_class',
                          sample_expression=True, batch_size=2048):
        """
        Generate a virtual slice that matches a reference slice's cell positions.

        This is useful for evaluation: use the real slice's positions and
        predict expression/cell types at those exact locations.

        Parameters
        ----------
        z : float
            Z-position for predictions.
        reference_adata : AnnData
            Reference slice whose cell positions to use.
        cell_type_key : str
            Key for cell type labels.
        sample_expression : bool
            Sample from ZINB or use mean.
        batch_size : int
            Inference batch size.

        Returns
        -------
        virtual_adata : AnnData matching reference positions.
        """
        # Get reference positions
        if 'spatial' in reference_adata.obsm:
            positions = np.asarray(reference_adata.obsm['spatial'])[:, :2]
        elif 'x' in reference_adata.obs.columns:
            positions = np.column_stack([
                reference_adata.obs['x'].values.astype(np.float32),
                reference_adata.obs['y'].values.astype(np.float32),
            ])
        else:
            raise ValueError("Cannot find spatial coordinates in reference")

        n = len(positions)
        coords_3d = np.hstack([
            positions,
            np.full((n, 1), z, dtype=np.float32),
        ]).astype(np.float32)

        all_ct_probs = []
        all_expr = []

        # First pass: get cell types
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_coords = torch.tensor(coords_3d[start:end],
                                        device=self.device)
            ct_probs = self.model.predict_cell_type(batch_coords)
            all_ct_probs.append(ct_probs.cpu().numpy())

        ct_probs_all = np.concatenate(all_ct_probs, axis=0)
        cell_types = ct_probs_all.argmax(axis=1)
        cell_type_labels = [self.cell_type_names[i] for i in cell_types]

        # Second pass: get expression
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_coords = torch.tensor(coords_3d[start:end],
                                        device=self.device)
            batch_ct = torch.tensor(cell_types[start:end],
                                    dtype=torch.long, device=self.device)

            if sample_expression:
                expr = self.model.predict_expression(
                    batch_coords, batch_ct, sample=True
                ).cpu().numpy()
            else:
                expr = self.model.predict_expression(
                    batch_coords, batch_ct, sample=False
                ).cpu().numpy()
            all_expr.append(expr)

        expr_all = np.concatenate(all_expr, axis=0)

        import pandas as pd
        obs = pd.DataFrame({
            'cell_class': cell_type_labels,
            'cell_type_idx': cell_types,
        })
        obs.index = reference_adata.obs.index.copy()

        virtual_adata = ad.AnnData(
            X=expr_all.astype(np.float32),
            obs=obs,
            var=reference_adata.var.copy(),
        )
        virtual_adata.obsm['spatial'] = positions.astype(np.float32)

        return virtual_adata
