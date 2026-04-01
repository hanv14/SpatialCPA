# SpatialCPA: Continuous Neural Fields for 3D Spatial Transcriptomics Reconstruction
## Comprehensive Method Development Plan (v2 — with Gap-Aware Design)

---

## 1. Problem Statement

Given sparsely sampled 2D spatial transcriptomics sections of a tissue, learn a continuous
function that maps any 3D coordinate (x, y, z) to:
- A predicted gene expression profile
- A cell-type annotation
- Optional higher-level annotations (region, subclass, supertype, etc.)

The method must:
- Accept **arbitrary z-values** (not just positions between adjacent slice pairs)
- Work with **heterogeneous datasets** that have different annotation schemas
- Operate at **single-cell resolution**
- Handle **non-uniform inter-slice gaps** and **variable section thicknesses**
- Outperform SpatialZ on standard evaluation metrics

---

## 2. Data Inventory & Harmonization Strategy

### 2.1 Available Datasets

| Dataset | Coordinates | Cell Typing | Region/Domain | Hierarchy | QC Metrics | z-info |
|---------|------------|-------------|---------------|-----------|------------|--------|
| DS1 (MERFISH brain) | obsm['spatial'] (2D) | cell_type | brain_section_label | class→subclass→supertype→cluster_alias | average_correlation_score | section |
| DS2 (MERFISH brain) | obsm['spatial'] (2D) | cell_type | brain_section_label | class→subclass→supertype→cluster_alias | subclass/cluster_confidence_score | section |
| DS3 (sectioned tissue) | X, Y (+ scaled versions) | cell_type | Subregion | clusters | standard scanpy QC | z_plane, section |
| DS4 (integrated atlas) | obsm['spatial'] (2D) | cell_type | — | Harmony_labels, FUSEmap_main_level | — | section |
| DS5 (imaging-based) | x_pos, y_pos | cell_type | — | — | pct_counts_mt, entropy, compression | section |

### 2.2 Unified Data Schema

Before training, all datasets must be converted to a common schema. Below is the
target AnnData structure:

```
adata_unified.obs columns:
  - cell_type          : str    (REQUIRED - present in all datasets)
  - section            : str    (REQUIRED - present in all datasets)
  - x_coord            : float  (REQUIRED - extracted from obs or obsm)
  - y_coord            : float  (REQUIRED - extracted from obs or obsm)
  - z_coord            : float  (REQUIRED - derived from section ordering)
  - z_physical         : float  (REQUIRED - physical z in μm or mm; see Section 2.4)
  - section_thickness  : float  (REQUIRED - physical thickness of this section in μm)
  - gap_to_prev        : float  (OPTIONAL - gap to previous section in μm)
  - gap_to_next        : float  (OPTIONAL - gap to next section in μm)
  - region             : str    (OPTIONAL - from brain_section_label / Subregion / None)
  - subclass           : str    (OPTIONAL - from subclass / Harmony_labels / None)
  - supertype          : str    (OPTIONAL - from supertype / FUSEmap_main_level / None)
  - confidence_score   : float  (OPTIONAL - from avg_corr_score / confidence scores / None)
  - dataset_id         : str    (REQUIRED - identifier for batch correction)
  - donor_id           : str    (OPTIONAL - from donor_label / None)

adata_unified.X:
  - Gene expression matrix (cells × genes)
  - Log-normalized, with raw counts in adata.layers['counts'] if available

adata_unified.obsm:
  - 'spatial_3d': np.ndarray of shape (n_cells, 3) containing [x, y, z_physical]

adata_unified.uns:
  - 'section_metadata': DataFrame with per-section z_physical, thickness, gaps
  - 'coordinate_scales': dict with xy_min_spacing, xy_max_range, z_min_spacing,
                          z_max_range for Fourier feature calibration
```

### 2.3 Data Harmonization Pipeline (preprocessing code structure)

```
harmonize_dataset(adata, dataset_config) → adata_unified

Step 1: Extract spatial coordinates
  - DS1, DS2, DS4: coords = adata.obsm['spatial']  # verify with labmate
  - DS3: coords = adata.obs[['X_Scaled', 'Y_Scaled']].values
  - DS5: coords = adata.obs[['x_pos', 'y_pos']].values

Step 2: Derive z-coordinates from section labels (see Section 2.4 for details)

Step 3: Map annotations to unified schema
  - cell_type: use directly (already present in all)
  - region: DS1/DS2 → brain_section_label; DS3 → Subregion; DS4/DS5 → None
  - subclass: DS1/DS2 → subclass; DS4 → Harmony_labels; DS3/DS5 → None
  - supertype: DS1/DS2 → supertype; DS4 → FUSEmap_main_level; DS3/DS5 → None

Step 4: Quality filtering
  - DS2: filter cells where high_quality_transfer == False
  - DS5: filter cells where pct_counts_mt > threshold (e.g., 20%)
  - DS2: optionally weight training loss by cluster_confidence_score
  - All: filter genes detected in < 5% of cells

Step 5: Normalize coordinates (see Section 2.5 for gap-aware normalization)
```

### 2.4 Physical z-Coordinate Registration (NEW — Gap-Aware)

A critical preprocessing step. Each cell must receive a z-coordinate in real physical
units (μm or mm), NOT sequential indices. This ensures the model learns that large
physical gaps correspond to large changes in tissue architecture.

```python
def assign_physical_z(adata, dataset_type, section_metadata=None):
    """
    Assign physically meaningful z-coordinates to each cell.
    
    The z-coordinate should be in micrometers (or millimeters) from a 
    reference point, NOT arbitrary indices.
    
    Args:
        adata: AnnData object
        dataset_type: one of 'DS1', 'DS2', 'DS3', 'DS4', 'DS5'
        section_metadata: optional DataFrame with columns:
            - section_id: str
            - z_center_um: float (center z-position in micrometers)
            - thickness_um: float (section thickness in micrometers)
            If not provided, values are inferred from dataset conventions.
    
    Returns:
        adata with new obs columns: z_physical, section_thickness, 
        gap_to_prev, gap_to_next
    """
    
    if dataset_type == 'DS3':
        # Dataset 3 already has z_plane — use directly
        # But verify: is z_plane in physical units or indices?
        # If indices, multiply by known section spacing
        z_values = adata.obs['z_plane'].values.astype(float)
        section_spacing_um = 10  # GET FROM EXPERIMENTAL PROTOCOL
        section_thickness_um = 10  # GET FROM EXPERIMENTAL PROTOCOL
        adata.obs['z_physical'] = z_values * section_spacing_um
        adata.obs['section_thickness'] = section_thickness_um
        
    elif dataset_type in ['DS1', 'DS2']:
        # Parse brain_section_label to get registered coordinates
        # e.g., "C57BL/6J-1.050" → 1.050 (in mm, from CCFv3 registration)
        def parse_z_from_label(label):
            parts = label.split('-')
            return float(parts[-1])  # e.g., 1.050 in mm
        
        adata.obs['z_physical'] = (
            adata.obs['brain_section_label']
            .apply(parse_z_from_label)
            * 1000  # convert mm to μm for consistency
        )
        # MERFISH sections are typically 10 μm thick
        adata.obs['section_thickness'] = 10.0
        
    elif dataset_type in ['DS4', 'DS5']:
        # Need to reconstruct z from section ordering + known spacing
        # CRITICAL: ask labmate for the actual section spacing
        section_order = sorted(adata.obs['section'].unique())
        
        if section_metadata is not None:
            # Use provided metadata for precise positioning
            z_map = dict(zip(
                section_metadata['section_id'], 
                section_metadata['z_center_um']
            ))
            thickness_map = dict(zip(
                section_metadata['section_id'], 
                section_metadata['thickness_um']
            ))
            adata.obs['z_physical'] = adata.obs['section'].map(z_map)
            adata.obs['section_thickness'] = adata.obs['section'].map(thickness_map)
        else:
            # Fallback: assume uniform spacing (GET REAL VALUE)
            section_spacing_um = 100  # PLACEHOLDER — ASK LABMATE
            section_thickness_um = 10  # PLACEHOLDER — ASK LABMATE
            section_to_z = {
                s: i * section_spacing_um 
                for i, s in enumerate(section_order)
            }
            adata.obs['z_physical'] = adata.obs['section'].map(section_to_z)
            adata.obs['section_thickness'] = section_thickness_um
    
    # Compute inter-section gaps
    sections = adata.obs.groupby('section')['z_physical'].first().sort_values()
    section_z = sections.values
    section_names = sections.index.values
    
    gap_prev = {}
    gap_next = {}
    for i, name in enumerate(section_names):
        gap_prev[name] = (section_z[i] - section_z[i-1]) if i > 0 else np.nan
        gap_next[name] = (section_z[i+1] - section_z[i]) if i < len(section_z)-1 else np.nan
    
    adata.obs['gap_to_prev'] = adata.obs['section'].map(gap_prev)
    adata.obs['gap_to_next'] = adata.obs['section'].map(gap_next)
    
    # Store section metadata in uns for downstream use
    adata.uns['section_metadata'] = pd.DataFrame({
        'section_id': section_names,
        'z_center_um': section_z,
        'thickness_um': [adata.obs.loc[adata.obs['section']==s, 'section_thickness'].iloc[0] 
                         for s in section_names],
        'gap_to_prev': [gap_prev[s] for s in section_names],
        'gap_to_next': [gap_next[s] for s in section_names],
    })
    
    return adata
```

### 2.5 Gap-Aware Coordinate Normalization (NEW)

Standard min-max normalization to [0,1] distorts the physical relationship between
x/y spacing (μm between cells) and z spacing (tens to hundreds of μm between sections).
This matters because the Fourier features must reflect real physical distances.

```python
def normalize_coordinates_gap_aware(adata):
    """
    Normalize spatial coordinates while preserving physical aspect ratios.
    
    Strategy: normalize each axis by the SAME global scale factor, so that
    distances in the normalized space remain proportional to physical distances.
    
    This is different from independently normalizing each axis to [0,1],
    which would distort the aspect ratio.
    """
    coords_3d = adata.obsm['spatial_3d']  # (n_cells, 3) in μm
    
    # Find the largest range across all three axes
    ranges = coords_3d.max(axis=0) - coords_3d.min(axis=0)  # [x_range, y_range, z_range]
    global_scale = ranges.max()  # normalize by the largest range
    
    # Center and scale
    center = (coords_3d.max(axis=0) + coords_3d.min(axis=0)) / 2
    coords_normalized = (coords_3d - center) / global_scale  # now in roughly [-0.5, 0.5]
    
    adata.obsm['spatial_3d_normalized'] = coords_normalized
    
    # Store scale parameters for inverse transform at inference
    adata.uns['coordinate_normalization'] = {
        'center': center,
        'global_scale': global_scale,
        'axis_ranges': ranges,
    }
    
    # Compute spacing statistics for Fourier feature calibration
    # xy spacing: typical nearest-neighbor distance within a slice
    from sklearn.neighbors import NearestNeighbors
    sample_section = adata.obs['section'].value_counts().idxmax()
    sample_mask = adata.obs['section'] == sample_section
    sample_xy = adata.obsm['spatial_3d'][sample_mask, :2]
    nn = NearestNeighbors(n_neighbors=2).fit(sample_xy)
    dists, _ = nn.kneighbors(sample_xy)
    xy_min_spacing = np.median(dists[:, 1])  # median NN distance
    xy_max_range = max(ranges[0], ranges[1])
    
    # z spacing: distances between section centers
    section_z = adata.uns['section_metadata']['z_center_um'].values
    z_gaps = np.diff(np.sort(section_z))
    z_min_spacing = z_gaps.min()
    z_max_range = ranges[2]
    
    adata.uns['coordinate_scales'] = {
        'xy_min_spacing': xy_min_spacing / global_scale,  # in normalized units
        'xy_max_range': xy_max_range / global_scale,
        'z_min_spacing': z_min_spacing / global_scale,
        'z_max_range': z_max_range / global_scale,
    }
    
    return adata
```

---

## 3. Model Architecture

### 3.1 Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │     SpatialCPA Architecture (Gap-Aware v2)      │
                    └─────────────────────────────────────────────────┘

Input: (x, y, z) continuous 3D coordinate in physical units
         │
         ▼
┌──────────────────────────┐
│  Gap-Aware Coordinate    │   Normalize using physical scales
│  Preprocessing           │   Preserve aspect ratio (Section 2.5)
│  (Section 2.5)           │
└─────────┬────────────────┘
          │  normalized coords ∈ ℝ^3
          ▼
┌──────────────────────────┐
│  Adaptive Fourier Feature│   Separate frequency bands for xy vs z
│  Positional Encoding     │   Calibrated to actual data spacing
│  (Section 3.2)           │   γ(x,y,z) ∈ ℝ^{2 * n_frequencies}
└─────────┬────────────────┘
          │  encoded coords ∈ ℝ^{2L}
          ▼
┌──────────────────────────┐
│  Shared Spatial          │   MLP with skip connections
│  Backbone Network        │   + optional dataset-specific bias
│  (Section 3.3)           │   h(x,y,z) ∈ ℝ^{256}
└────┬──────┬─────┬────────┘
     │      │     │
     ▼      ▼     ▼
┌────────┐ ┌────────────┐ ┌──────────────────┐
│ Cell   │ │ Region     │ │ Expression       │
│ Type   │ │ Annotation │ │ Decoder          │
│ Head   │ │ Head       │ │ (Section 3.5)    │
│(3.4)   │ │ (3.4)      │ │                  │
└────┬───┘ └─────┬──────┘ └────────┬─────────┘
     │           │                  │
     ▼           ▼                  ▼
 P(cell_type)  P(region)     Expression profile
 P(subclass)   P(supertype)  ∈ ℝ^G

During TRAINING: z-marginalization over section thickness (Section 3.6)
During INFERENCE: query at exact z-values (Section 5.1)
```

### 3.2 Adaptive Fourier Feature Positional Encoding (Gap-Aware)

Standard Fourier features use the same frequency range for all three spatial axes.
This fails for spatial transcriptomics because the spatial scales are fundamentally
different: within a slice, cells are ~1-10 μm apart; between slices, gaps range from
~50 to ~300 μm. A single frequency range that captures within-slice variation would
miss between-slice patterns, and vice versa.

Our adaptive encoding calibrates frequency bands separately for xy and z using the
actual spacing statistics computed during preprocessing (Section 2.5).

```python
class AdaptiveFourierFeatures(nn.Module):
    """
    Fourier features with frequency scales adapted to the actual data 
    spacing, separately for x/y vs z.
    
    Key design: 
    - Half the frequencies are tuned to within-slice (xy) variation
    - Half are tuned to between-slice (z) variation
    - Each frequency vector has components in all 3 axes, but with
      dominant contribution from its target axis
    
    This ensures the model can simultaneously capture:
    - Fine-grained cellular neighborhoods within slices (high xy freq)
    - Smooth tissue gradients across slices (low z freq)
    - Sharp region boundaries along z (high z freq)
    - Global organ-level structure (low xy freq)
    """
    
    def __init__(
        self, 
        n_frequencies=128, 
        coordinate_scales=None,
        learnable=False
    ):
        """
        Args:
            n_frequencies: total number of frequency bands (split between xy and z)
            coordinate_scales: dict from adata.uns['coordinate_scales'] with keys:
                xy_min_spacing, xy_max_range, z_min_spacing, z_max_range
                (all in normalized coordinate units)
            learnable: if True, frequency matrix B is a trainable parameter.
                       if False, B is fixed (more stable, recommended to start).
        """
        super().__init__()
        
        n_xy = n_frequencies // 2
        n_z = n_frequencies - n_xy  # handle odd n_frequencies
        
        # Compute appropriate frequency ranges from data spacing
        if coordinate_scales is not None:
            # Nyquist-inspired frequency bounds
            # Highest freq should resolve smallest spacing
            # Lowest freq should capture full spatial range
            xy_freq_min = 1.0 / coordinate_scales['xy_max_range']
            xy_freq_max = 1.0 / (2 * coordinate_scales['xy_min_spacing'])
            z_freq_min = 1.0 / coordinate_scales['z_max_range']
            z_freq_max = 1.0 / (2 * coordinate_scales['z_min_spacing'])
        else:
            # Conservative defaults if scales not provided
            xy_freq_min, xy_freq_max = 0.1, 50.0
            z_freq_min, z_freq_max = 0.01, 10.0
        
        # Log-linear frequency bands (covers multiple octaves)
        xy_freqs = torch.logspace(
            np.log10(xy_freq_min), np.log10(xy_freq_max), n_xy
        )
        z_freqs = torch.logspace(
            np.log10(z_freq_min), np.log10(z_freq_max), n_z
        )
        
        # Build 3D frequency matrix B ∈ ℝ^{n_frequencies × 3}
        # Each row is a 3D frequency vector [fx, fy, fz]
        B = torch.zeros(n_frequencies, 3)
        
        # XY-dominant frequencies: strong in x,y; weak in z
        # Random signs for diverse orientation coverage
        B[:n_xy, 0] = xy_freqs * torch.sign(torch.randn(n_xy))
        B[:n_xy, 1] = xy_freqs * torch.sign(torch.randn(n_xy))
        B[:n_xy, 2] = z_freqs[:n_xy] * 0.1  # weak z coupling
        
        # Z-dominant frequencies: strong in z; weak in x,y
        B[n_xy:, 0] = xy_freqs[:n_z] * 0.1  # weak xy coupling
        B[n_xy:, 1] = xy_freqs[:n_z] * 0.1
        B[n_xy:, 2] = z_freqs  # strong z component
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
        
        self.output_dim = 2 * n_frequencies
        
        # Store scale info for logging/debugging
        self.register_buffer('_xy_freq_range', 
                             torch.tensor([xy_freq_min, xy_freq_max]))
        self.register_buffer('_z_freq_range', 
                             torch.tensor([z_freq_min, z_freq_max]))
    
    def forward(self, coords):
        """
        Args:
            coords: (batch, 3) — [x, y, z] in normalized units
        Returns:
            (batch, 2 * n_frequencies) — sin/cos encoded features
        """
        proj = coords @ self.B.T  # (batch, n_frequencies)
        return torch.cat([
            torch.sin(2 * torch.pi * proj),
            torch.cos(2 * torch.pi * proj)
        ], dim=-1)
```

### 3.3 Shared Spatial Backbone

```python
class SpatialBackbone(nn.Module):
    """
    Transforms Fourier-encoded coordinates into a rich spatial context
    embedding. Uses skip connections to preserve coordinate information
    at multiple scales.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for i in range(1, n_layers):
            if i == n_layers // 2:
                # Skip connection: re-inject input at midpoint
                self.layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.skip_layer = n_layers // 2

    def forward(self, encoded_coords):
        h = encoded_coords
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, encoded_coords], dim=-1)  # skip connection
            h = F.relu(layer(h))
        return self.output_layer(h)  # (batch, 256)


class BatchAwareSpatialBackbone(SpatialBackbone):
    """
    Extension that handles dataset-specific batch effects via learnable
    per-dataset bias terms.
    
    When training on multiple datasets from different technologies/labs,
    each dataset may have systematic shifts. This module adds a small,
    regularized offset per dataset.
    """
    def __init__(self, n_datasets, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_dim = kwargs.get('output_dim', 256)
        self.dataset_bias = nn.Embedding(n_datasets, output_dim)
        nn.init.zeros_(self.dataset_bias.weight)  # start with no correction

    def forward(self, encoded_coords, dataset_idx=None):
        spatial_ctx = super().forward(encoded_coords)
        if dataset_idx is not None:
            bias = self.dataset_bias(dataset_idx)
            return spatial_ctx + bias  # additive correction
        return spatial_ctx
```

### 3.4 Annotation Prediction Heads

```python
class AnnotationHead(nn.Module):
    """
    Predicts categorical annotations from spatial context.
    Handles the variable annotation schema across datasets by
    maintaining separate output layers for each annotation level.
    """
    def __init__(self, input_dim=256, hidden_dim=128, annotation_config=None):
        """
        annotation_config: dict mapping annotation names to number of classes
        Example: {
            'cell_type': 30,    # present in all datasets
            'class': 8,         # only DS1, DS2
            'subclass': 45,     # DS1, DS2, partially DS4
            'supertype': 120,   # DS1, DS2, partially DS4
            'region': 14,       # DS1, DS2, DS3
        }
        """
        super().__init__()
        self.shared_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        # Separate output layer per annotation level
        self.classifiers = nn.ModuleDict({
            name: nn.Linear(hidden_dim, n_classes)
            for name, n_classes in annotation_config.items()
        })

    def forward(self, spatial_context, requested_annotations=None):
        """
        Returns predictions only for requested annotation levels.
        This handles missing annotations gracefully — if a dataset
        doesn't have 'supertype', we simply don't compute or backprop
        through that head for those cells.
        """
        h = self.shared_hidden(spatial_context)
        predictions = {}
        targets = requested_annotations or self.classifiers.keys()
        for name in targets:
            if name in self.classifiers:
                predictions[name] = self.classifiers[name](h)
        return predictions
```

### 3.5 Expression Decoder

```python
class ExpressionDecoder(nn.Module):
    """
    Generates gene expression profiles conditioned on spatial context
    and cell-type identity.

    Key design principle: the decoder outputs DISTRIBUTION PARAMETERS,
    not point estimates. This captures cell-to-cell variability.
    
    Uses Zero-Inflated Negative Binomial (ZINB) because single-cell
    expression data has two key properties:
    - Overdispersion (variance exceeds mean)
    - Excess zeros (many genes detected in only a fraction of cells)
    """
    def __init__(
        self,
        spatial_dim=256,
        n_cell_types=30,
        cell_type_embed_dim=64,
        hidden_dim=512,
        n_genes=1100,
        n_layers=4
    ):
        super().__init__()
        self.cell_type_embedding = nn.Embedding(n_cell_types, cell_type_embed_dim)

        combined_dim = spatial_dim + cell_type_embed_dim
        layers = [nn.Linear(combined_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.decoder_body = nn.Sequential(*layers)

        # Output heads: parameters of a Zero-Inflated Negative Binomial
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, n_genes), nn.Softplus()
        )
        self.dispersion_head = nn.Sequential(
            nn.Linear(hidden_dim, n_genes), nn.Softplus()
        )
        self.dropout_head = nn.Sequential(
            nn.Linear(hidden_dim, n_genes), nn.Sigmoid()
        )

    def forward(self, spatial_context, cell_type_idx):
        ct_embed = self.cell_type_embedding(cell_type_idx)
        combined = torch.cat([spatial_context, ct_embed], dim=-1)
        h = self.decoder_body(combined)
        return {
            'mean': self.mean_head(h),
            'dispersion': self.dispersion_head(h),
            'dropout': self.dropout_head(h)
        }

    def sample(self, spatial_context, cell_type_idx, n_samples=1):
        """
        Generate concrete expression profiles by sampling from
        the predicted distribution. Used at inference time.
        """
        params = self.forward(spatial_context, cell_type_idx)
        mu = params['mean']
        theta = params['dispersion']
        pi = params['dropout']

        # Sample from ZINB
        is_zero = torch.bernoulli(pi)
        concentration = theta
        rate = theta / mu
        nb_samples = torch.distributions.NegativeBinomial(
            total_count=concentration, probs=1 - 1/(1+rate)
        ).sample()
        expression = (1 - is_zero) * nb_samples
        return expression
```

### 3.6 Slice-as-Slab: z-Marginalization Module (NEW — Gap-Aware)

Each physical tissue section has a real thickness (typically ~10 μm). Cells within a
single "slice" don't all sit at exactly the same z-coordinate — they are distributed
across the thickness of that section. Ignoring this creates artificial z-discontinuities
at section boundaries.

During training, we marginalize over the z-uncertainty within each section. During
inference at a specific z-value, we query at that exact point.

```python
class SliceAsSlabModule(nn.Module):
    """
    Models the within-slice z-uncertainty by sampling multiple z-values
    within the section thickness and averaging predictions.
    
    This teaches the model that cells in the same section could be at
    slightly different z-positions, preventing artificial z-discontinuities 
    at section boundaries and forcing the model to learn genuinely smooth 
    spatial variation along z.
    
    Analogy: if you photograph a scene through a thick glass window,
    the image is an average over the window's depth. Similarly, a 
    tissue section captures cells across its thickness.
    """
    
    def __init__(self, default_thickness_um=10.0, n_z_samples=5):
        """
        Args:
            default_thickness_um: default section thickness in μm
                (used only if per-section thickness is not provided)
            n_z_samples: number of z-positions to sample within 
                each section during training
        """
        super().__init__()
        self.n_z_samples = n_z_samples
        # Learnable default thickness (log-space for positivity)
        self.log_default_thickness = nn.Parameter(
            torch.log(torch.tensor(default_thickness_um))
        )
    
    def sample_z_within_slab(self, z_nominal, section_thickness=None):
        """
        Sample z-positions within the section thickness around the
        nominal (center) z-coordinate.
        
        Uses a truncated Gaussian: most cells are near the center,
        but some are near the edges of the section.
        
        Args:
            z_nominal: (batch,) center z-coordinate of each cell's section
            section_thickness: (batch,) thickness of each cell's section in 
                              the same units as z_nominal. If None, uses
                              learned default.
        
        Returns:
            z_samples: (batch, n_z_samples) sampled z-positions
        """
        if section_thickness is None:
            thickness = torch.exp(self.log_default_thickness)
            section_thickness = thickness.expand_as(z_nominal)
        
        # Sample from truncated Gaussian
        # σ = thickness/4 → ~95% of samples within the slab
        sigma = section_thickness / 4.0
        z_offsets = torch.randn(
            z_nominal.shape[0], self.n_z_samples, 
            device=z_nominal.device
        ) * sigma.unsqueeze(-1)
        
        # Hard clamp to section boundaries
        half_thickness = section_thickness.unsqueeze(-1) / 2.0
        z_offsets = z_offsets.clamp(-half_thickness, half_thickness)
        
        z_samples = z_nominal.unsqueeze(-1) + z_offsets
        return z_samples
    
    def forward_marginalized(self, model, xy_coords, z_nominal, 
                              section_thickness, cell_type_idx,
                              requested_annotations=None):
        """
        Forward pass with z-marginalization: evaluate the model at
        multiple z-positions within the section thickness and average.
        
        Args:
            model: the SpatialCPA model (backbone + heads)
            xy_coords: (batch, 2)
            z_nominal: (batch,) section center z
            section_thickness: (batch,) section thickness
            cell_type_idx: (batch,) integer cell type indices
            requested_annotations: which annotation levels to predict
        
        Returns:
            averaged predictions across z-samples
        """
        z_samples = self.sample_z_within_slab(z_nominal, section_thickness)
        # z_samples: (batch, n_z_samples)
        
        all_outputs = []
        for i in range(self.n_z_samples):
            coords_3d = torch.stack([
                xy_coords[:, 0],
                xy_coords[:, 1],
                z_samples[:, i]
            ], dim=-1)  # (batch, 3)
            
            output = model(coords_3d, cell_type_idx, requested_annotations)
            all_outputs.append(output)
        
        # Average predictions across z-samples
        averaged = {
            'expression_params': {},
            'annotation_logits': {},
            'spatial_context': torch.stack(
                [o['spatial_context'] for o in all_outputs]
            ).mean(dim=0)
        }
        
        # Average expression distribution parameters
        if all_outputs[0]['expression_params'] is not None:
            for key in ['mean', 'dispersion', 'dropout']:
                averaged['expression_params'][key] = torch.stack(
                    [o['expression_params'][key] for o in all_outputs]
                ).mean(dim=0)
        
        # Average annotation logits (before softmax)
        for name in all_outputs[0]['annotation_logits']:
            averaged['annotation_logits'][name] = torch.stack(
                [o['annotation_logits'][name] for o in all_outputs]
            ).mean(dim=0)
        
        return averaged
```

### 3.7 Full Model Assembly (Gap-Aware)

```python
class SpatialCPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fourier_encoder = AdaptiveFourierFeatures(
            n_frequencies=config.n_frequencies,
            coordinate_scales=config.coordinate_scales,
            learnable=config.learnable_frequencies
        )
        self.backbone = BatchAwareSpatialBackbone(
            n_datasets=config.n_datasets,
            input_dim=self.fourier_encoder.output_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            n_layers=config.n_backbone_layers
        )
        self.annotation_head = AnnotationHead(
            input_dim=config.latent_dim,
            annotation_config=config.annotation_config
        )
        self.expression_decoder = ExpressionDecoder(
            spatial_dim=config.latent_dim,
            n_cell_types=config.n_cell_types,
            n_genes=config.n_genes
        )
        self.slab_module = SliceAsSlabModule(
            default_thickness_um=config.default_section_thickness,
            n_z_samples=config.n_z_samples
        )

    def forward(self, coords_3d, cell_type_idx=None, 
                requested_annotations=None, dataset_idx=None):
        """
        Standard forward pass at EXACT coordinates.
        Used during inference and for the non-marginalized path.
        """
        encoded = self.fourier_encoder(coords_3d)
        spatial_ctx = self.backbone(encoded, dataset_idx)
        ann_logits = self.annotation_head(spatial_ctx, requested_annotations)

        expr_params = None
        if cell_type_idx is not None:
            expr_params = self.expression_decoder(spatial_ctx, cell_type_idx)

        return {
            'annotation_logits': ann_logits,
            'expression_params': expr_params,
            'spatial_context': spatial_ctx
        }

    def forward_training(self, xy_coords, z_nominal, section_thickness,
                         cell_type_idx, requested_annotations=None,
                         dataset_idx=None, use_z_marginalization=True):
        """
        Training forward pass with optional z-marginalization.
        
        During training, use z-marginalization to account for within-slice
        position uncertainty. During inference, call forward() directly
        with exact coordinates.
        """
        if use_z_marginalization:
            return self.slab_module.forward_marginalized(
                model=self,
                xy_coords=xy_coords,
                z_nominal=z_nominal,
                section_thickness=section_thickness,
                cell_type_idx=cell_type_idx,
                requested_annotations=requested_annotations
            )
        else:
            coords_3d = torch.stack([
                xy_coords[:, 0], xy_coords[:, 1], z_nominal
            ], dim=-1)
            return self.forward(
                coords_3d, cell_type_idx, 
                requested_annotations, dataset_idx
            )
```

---

## 4. Training Strategy

### 4.1 Loss Function (Gap-Aware)

```python
class SpatialCPALoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_expr = config.lambda_expr        # 1.0
        self.lambda_type = config.lambda_type        # 1.0
        self.lambda_region = config.lambda_region    # 0.5
        self.lambda_smooth = config.lambda_smooth    # 0.01
        self.lambda_loo = config.lambda_loo          # 0.5

    def zinb_loss(self, x_true, mean, dispersion, dropout):
        """
        Negative log-likelihood of Zero-Inflated Negative Binomial.
        """
        theta = dispersion
        nb_logprob = (
            torch.lgamma(x_true + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x_true + 1)
            + theta * torch.log(theta / (theta + mean))
            + x_true * torch.log(mean / (theta + mean))
        )

        zero_case = torch.log(dropout + (1 - dropout) * torch.exp(nb_logprob))
        nonzero_case = torch.log(1 - dropout) + nb_logprob

        is_zero = (x_true < 1e-8).float()
        log_likelihood = is_zero * zero_case + (1 - is_zero) * nonzero_case
        return -log_likelihood.mean()

    def annotation_loss(self, logits_dict, labels_dict, available_masks):
        """
        Cross-entropy loss computed ONLY for annotation levels that
        exist in the current batch. Handles heterogeneous datasets.
        """
        total_loss = 0
        for name, logits in logits_dict.items():
            if name in labels_dict and name in available_masks:
                mask = available_masks[name]
                if mask.sum() > 0:
                    loss = F.cross_entropy(
                        logits[mask], labels_dict[name][mask],
                        reduction='mean'
                    )
                    weight = {
                        'cell_type': 1.0, 'class': 0.5,
                        'subclass': 0.7, 'supertype': 0.5,
                        'region': 0.8
                    }.get(name, 0.5)
                    total_loss += weight * loss
        return total_loss

    def gap_aware_spatial_smoothness_loss(self, model, coords, 
                                          coordinate_scales, delta=0.01):
        """
        Gap-aware spatial smoothness regularization.
        
        Key difference from standard smoothness: the perturbation scale
        is adapted to the local spacing. In the xy plane, we perturb at 
        the scale of cell-cell distances. In z, we perturb at the scale 
        of inter-slice gaps.
        
        This means the model is regularized appropriately for each axis:
        - Within-slice: smooth over cellular neighborhoods
        - Between-slice: smooth over inter-section distances
        
        We also implement PIECEWISE smoothness — the penalty is modulated 
        so that the model CAN have sharp transitions at region boundaries 
        while being smooth within regions.
        """
        batch_size = coords.shape[0]
        
        # Axis-specific perturbation scales
        xy_scale = coordinate_scales['xy_min_spacing'] * 2  # ~2 cell widths
        z_scale = coordinate_scales['z_min_spacing'] * 0.1  # fraction of min gap
        
        noise = torch.zeros_like(coords)
        noise[:, 0] = torch.randn(batch_size, device=coords.device) * xy_scale
        noise[:, 1] = torch.randn(batch_size, device=coords.device) * xy_scale
        noise[:, 2] = torch.randn(batch_size, device=coords.device) * z_scale
        
        perturbed_coords = coords + noise
        
        ctx_original = model.backbone(model.fourier_encoder(coords))
        ctx_perturbed = model.backbone(model.fourier_encoder(perturbed_coords))
        
        # Embedding change relative to perturbation magnitude
        embedding_change = (ctx_original - ctx_perturbed).norm(dim=-1)
        perturbation_magnitude = noise.norm(dim=-1)
        
        smoothness = (embedding_change / (perturbation_magnitude + 1e-8)).mean()
        return smoothness

    def confidence_weighted_loss(self, per_cell_loss, confidence_scores):
        """
        Weight the loss by cell confidence scores (from DS1/DS2).
        High-confidence cells contribute more to the loss.
        Falls back to uniform weighting if no scores available.
        """
        if confidence_scores is not None:
            weights = confidence_scores.clamp(0.1, 1.0)  # floor at 0.1
            return (per_cell_loss * weights).sum() / weights.sum()
        return per_cell_loss.mean()

    def forward(self, model_output, batch, model=None, coords=None,
                coordinate_scales=None):
        """
        Compute total loss.
        """
        losses = {}

        # 1. Expression reconstruction loss (ZINB NLL)
        if model_output['expression_params'] is not None:
            per_cell_expr_loss = self._per_cell_zinb(
                batch['expression'],
                model_output['expression_params']
            )
            losses['expr'] = self.lambda_expr * self.confidence_weighted_loss(
                per_cell_expr_loss,
                batch.get('confidence_scores', None)
            )

        # 2. Annotation classification losses
        losses['annotation'] = self.lambda_type * self.annotation_loss(
            model_output['annotation_logits'],
            batch['labels'],
            batch['available_masks']
        )

        # 3. Gap-aware spatial smoothness regularization
        if model is not None and coords is not None:
            losses['smooth'] = self.lambda_smooth * \
                self.gap_aware_spatial_smoothness_loss(
                    model, coords, coordinate_scales
                )

        losses['total'] = sum(losses.values())
        return losses
```

### 4.2 Gap-Aware Leave-One-Out Self-Supervision (NEW)

The difficulty of predicting a virtual slice depends heavily on the gap size to the
nearest real slices. The LOO training must account for this: larger gaps are harder
to predict but more important to learn from.

```python
class GapAwareLOOTrainer:
    """
    Implements leave-one-out self-supervision with gap-aware weighting.
    
    Two key gap-aware behaviors:
    1. LOO sampling probability is proportional to gap size, so the model
       practices predicting across large gaps more often
    2. LOO loss weight is proportional to gap size, so getting large gaps
       right matters more than getting small gaps right
    """
    
    def __init__(self, section_metadata):
        """
        Args:
            section_metadata: DataFrame with columns 
                section_id, z_center_um, thickness_um, gap_to_prev, gap_to_next
        """
        self.section_metadata = section_metadata.sort_values('z_center_um')
        self.section_ids = self.section_metadata['section_id'].values
        self.z_positions = self.section_metadata['z_center_um'].values
        
        # Compute the "isolation score" for each section:
        # how far is it from its nearest neighbors?
        self.isolation_scores = {}
        for i, sid in enumerate(self.section_ids):
            gaps = []
            if i > 0:
                gaps.append(self.z_positions[i] - self.z_positions[i-1])
            if i < len(self.z_positions) - 1:
                gaps.append(self.z_positions[i+1] - self.z_positions[i])
            # Max gap to either neighbor
            self.isolation_scores[sid] = max(gaps) if gaps else 0
        
        # Sampling probability proportional to isolation
        # (but not too extreme — clamp the ratio)
        median_isolation = np.median(list(self.isolation_scores.values()))
        self.sampling_weights = {
            sid: np.clip(score / median_isolation, 0.3, 3.0)
            for sid, score in self.isolation_scores.items()
        }
        # Normalize to probability distribution
        total = sum(self.sampling_weights.values())
        self.sampling_probs = {
            sid: w / total for sid, w in self.sampling_weights.items()
        }
        
    def sample_holdout_section(self):
        """
        Sample a section to hold out, with probability proportional
        to its isolation score (larger gaps → more likely to be held out).
        
        Excludes the first and last sections to avoid extrapolation during LOO.
        """
        # Exclude boundary sections
        eligible = self.section_ids[1:-1]
        probs = np.array([self.sampling_probs[s] for s in eligible])
        probs = probs / probs.sum()
        
        chosen = np.random.choice(eligible, p=probs)
        return chosen
    
    def get_loo_loss_weight(self, held_out_section_id):
        """
        Weight for the LOO loss: larger gaps get higher weight.
        
        Rationale: if the model can accurately predict across a 200 μm gap,
        it can certainly handle a 50 μm gap. The reverse is not true.
        So we want to prioritize learning from large-gap cases.
        """
        isolation = self.isolation_scores[held_out_section_id]
        median_isolation = np.median(list(self.isolation_scores.values()))
        return np.clip(isolation / median_isolation, 0.5, 3.0)
    
    def compute_loo_loss(self, model, held_out_adata, loss_fn, config):
        """
        Compute the leave-one-out reconstruction loss for a held-out section.
        
        Note: during LOO, we do NOT use z-marginalization for the held-out 
        section. We query at the exact z-position, because the goal is to
        test interpolation accuracy at a specific location.
        """
        held_out_section = held_out_adata.obs['section'].iloc[0]
        loss_weight = self.get_loo_loss_weight(held_out_section)
        
        # Query the model at the held-out section's exact coordinates
        coords_3d = torch.tensor(
            held_out_adata.obsm['spatial_3d_normalized'],
            dtype=torch.float32, device=config.device
        )
        cell_type_idx = torch.tensor(
            held_out_adata.obs['cell_type_idx'].values,
            dtype=torch.long, device=config.device
        )
        expression = torch.tensor(
            held_out_adata.X if issparse(held_out_adata.X) 
            else held_out_adata.X,
            dtype=torch.float32, device=config.device
        )
        
        # Forward pass WITHOUT z-marginalization (exact position query)
        output = model(coords_3d, cell_type_idx)
        
        # Compute reconstruction loss
        batch = {
            'expression': expression,
            'labels': {
                name: torch.tensor(
                    held_out_adata.obs[f'{name}_idx'].values,
                    dtype=torch.long, device=config.device
                )
                for name in config.annotation_levels
                if f'{name}_idx' in held_out_adata.obs.columns
            },
            'available_masks': {
                name: torch.ones(len(held_out_adata), dtype=torch.bool,
                                 device=config.device)
                for name in config.annotation_levels
                if f'{name}_idx' in held_out_adata.obs.columns
            }
        }
        
        raw_loss = loss_fn(output, batch)
        return loss_weight * raw_loss['total']
```

### 4.3 Full Training Loop (Gap-Aware)

```
Algorithm: SpatialCPA Training (Gap-Aware v2)

Input: List of harmonized datasets D = {D1, D2, ..., D5}
       Each with physical z-coordinates and section thickness (Section 2.4)
       Coordinate scales for Fourier feature calibration (Section 2.5)

Initialization:
  1. Build unified cell-type vocabulary across all datasets
  2. Initialize AdaptiveFourierFeatures with coordinate_scales from data
  3. Initialize GapAwareLOOTrainer with section_metadata from each dataset
  4. Initialize SliceAsSlabModule with default thickness from data

For each epoch:
  ┌────────────────────────────────────────────────────────────────────┐
  │ PHASE 1: Supervised training WITH z-marginalization (80% of iters)│
  │                                                                    │
  │  For each mini-batch:                                              │
  │    1. Sample dataset D_i proportional to its size                  │
  │    2. Sample cells from D_i (stratified by cell type)              │
  │    3. Extract:                                                     │
  │       - xy_coords: (batch, 2) spatial coordinates                  │
  │       - z_nominal: (batch,) section center z (physical units)      │
  │       - section_thickness: (batch,) per-cell section thickness     │
  │       - expression: (batch, n_genes) gene expression               │
  │       - cell_type_idx: (batch,) cell type indices                  │
  │       - labels + available_masks for this dataset's annotation     │
  │       - confidence_scores (if available, e.g. DS1/DS2)             │
  │       - dataset_idx: (batch,) dataset identifier                   │
  │                                                                    │
  │    4. Forward pass with z-marginalization:                         │
  │       output = model.forward_training(                             │
  │           xy_coords, z_nominal, section_thickness,                 │
  │           cell_type_idx, use_z_marginalization=True                │
  │       )                                                            │
  │                                                                    │
  │    5. Compute gap-aware loss:                                      │
  │       - ZINB expression loss (confidence-weighted if DS1/DS2)      │
  │       - Masked annotation loss (only available levels)             │
  │       - Gap-aware spatial smoothness (axis-specific perturbation)  │
  │                                                                    │
  │    6. Backpropagate and update weights                             │
  └────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │ PHASE 2: Gap-aware LOO self-supervision (20% of iterations)       │
  │                                                                    │
  │  For each LOO iteration:                                           │
  │    1. Sample holdout section (gap-proportional probability)        │
  │    2. Temporarily exclude all cells from held-out section          │
  │    3. Train on neighboring sections (standard Phase 1 step)        │
  │    4. Query model at held-out section coordinates                  │
  │       WITHOUT z-marginalization (testing exact prediction)         │
  │    5. Compute gap-weighted LOO loss:                               │
  │       loo_loss = gap_weight * reconstruction_loss                  │
  │       (larger gaps → higher weight → prioritize hard cases)        │
  │    6. Total loss = standard_loss + λ_loo * loo_loss                │
  │    7. Backpropagate and update                                     │
  └────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │ PHASE 3: Validation (end of epoch)                                 │
  │                                                                    │
  │  1. Hold out 2-3 sections NEVER used in Phase 1/2                  │
  │     (select sections spanning different gap sizes)                 │
  │  2. Predict at held-out z-positions (no z-marginalization)         │
  │  3. Compute evaluation metrics (Section 6.1)                       │
  │  4. Additionally report metrics STRATIFIED BY GAP SIZE:            │
  │     - Small gap (<50 μm): expect very high accuracy                │
  │     - Medium gap (50-150 μm): main operating regime                │
  │     - Large gap (>150 μm): hardest case, most informative          │
  │  5. Early stopping based on validation loss                        │
  └────────────────────────────────────────────────────────────────────┘

Hyperparameters:
  - Learning rate: 1e-4 with cosine annealing
  - Batch size: 4096 cells
  - Optimizer: AdamW (weight_decay=1e-5)
  - n_z_samples: 5 (for slab marginalization)
  - LOO fraction: 20% of batches per epoch
  - Training duration: ~200 epochs (early stopping patience=20)
```

### 4.4 Handling Confidence Scores During Training

DS2 provides `subclass_confidence_score` and `cluster_confidence_score`.
DS1 provides `average_correlation_score`. These are used as per-cell sample weights
in the loss function (implemented in SpatialCPALoss.confidence_weighted_loss).

### 4.5 Handling Dataset-Specific Batch Effects

Different datasets from different technologies/labs may have systematic shifts.
The BatchAwareSpatialBackbone (Section 3.3) adds a small, learnable, L2-regularized
per-dataset offset. Initialized at zero so it only activates when needed.

---

## 5. Inference Pipeline

### 5.1 Generating a Virtual Slice at Arbitrary z

Key difference from training: during inference, we query at EXACT z-coordinates
with NO z-marginalization. The model has learned smooth z-variation during training
(thanks to the slab module), so point queries produce clean predictions.

```python
def generate_virtual_slice(model, z_query, xy_bounds, config):
    """
    Generate a complete virtual slice at any z-value.

    Args:
        model: trained SpatialCPA model
        z_query: float, the z-coordinate in PHYSICAL UNITS (μm or mm)
                 Can be any value — between observed slices, at observed 
                 positions, or even slight extrapolation beyond range.
        xy_bounds: dict with x_min, x_max, y_min, y_max (physical units)
        config: generation parameters

    Returns:
        adata: AnnData with expression, cell types, regions, coordinates
    """
    model.eval()
    
    # Convert z_query to normalized coordinates
    norm_params = config.coordinate_normalization
    z_norm = (z_query - norm_params['center'][2]) / norm_params['global_scale']

    # Step 1: Determine tissue boundary at this z
    grid_x = torch.linspace(
        (xy_bounds['x_min'] - norm_params['center'][0]) / norm_params['global_scale'],
        (xy_bounds['x_max'] - norm_params['center'][0]) / norm_params['global_scale'],
        100
    )
    grid_y = torch.linspace(
        (xy_bounds['y_min'] - norm_params['center'][1]) / norm_params['global_scale'],
        (xy_bounds['y_max'] - norm_params['center'][1]) / norm_params['global_scale'],
        100
    )
    xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
    grid_coords = torch.stack([
        xx.flatten(), yy.flatten(),
        torch.full_like(xx.flatten(), z_norm)
    ], dim=-1).to(config.device)

    with torch.no_grad():
        coarse_output = model(grid_coords)
        ct_probs = F.softmax(
            coarse_output['annotation_logits']['cell_type'], dim=-1
        )
        max_prob = ct_probs.max(dim=-1).values
        tissue_mask = max_prob > config.tissue_threshold

    # Step 2: Dense sampling within tissue boundary
    tissue_coords = grid_coords[tissue_mask]
    cell_positions = poisson_disk_sample(
        tissue_coords, min_distance=config.min_cell_distance,
        n_target=config.target_n_cells
    )

    # Step 3: Predict cell types and regions
    with torch.no_grad():
        output = model(cell_positions)
        ct_probs = F.softmax(
            output['annotation_logits']['cell_type'], dim=-1
        )
        cell_types = torch.multinomial(ct_probs, 1).squeeze()

        regions = None
        if 'region' in output['annotation_logits']:
            region_probs = F.softmax(
                output['annotation_logits']['region'], dim=-1
            )
            regions = region_probs.argmax(dim=-1)

    # Step 4: Generate expression profiles
    with torch.no_grad():
        expression = model.expression_decoder.sample(
            output['spatial_context'], cell_types, n_samples=1
        )

    # Step 5: Convert coordinates back to physical units
    physical_coords = (
        cell_positions.cpu().numpy() * norm_params['global_scale'] 
        + norm_params['center']
    )

    # Step 6: Assemble AnnData
    obs_dict = {
        'cell_type': [config.cell_type_names[i] for i in cell_types.cpu()],
        'section': f'virtual_z{z_query:.4f}',
        'is_virtual': True,
        'z_physical': z_query,
    }
    if regions is not None:
        obs_dict['region'] = [config.region_names[i] for i in regions.cpu()]
    
    adata_virtual = anndata.AnnData(
        X=expression.cpu().numpy(),
        obs=pd.DataFrame(obs_dict),
        obsm={
            'spatial': physical_coords[:, :2],
            'spatial_3d': physical_coords
        }
    )
    return adata_virtual
```

### 5.2 In Silico Sectioning at Arbitrary Angles

```python
def in_silico_section(model, plane_normal, plane_point, thickness, config):
    """
    Generate a section along any arbitrary plane through the 3D volume.
    No coordinate rotation needed — just query the continuous field.
    
    This is a native capability of the continuous field representation.
    SpatialZ requires a separate module with coordinate rotation.

    Args:
        plane_normal: (3,) unit normal vector defining the cutting plane
        plane_point: (3,) a point the plane passes through (physical units)
        thickness: scalar, thickness of the section (physical units)
        config: generation parameters

    Returns:
        adata: virtual section data
    """
    # Generate candidate points in 3D volume
    candidate_points = generate_3d_grid(
        config.volume_bounds, config.grid_spacing
    )

    # Select points within 'thickness' of the cutting plane
    distances = torch.abs(
        (candidate_points - plane_point).matmul(plane_normal)
    )
    in_slice = distances < thickness / 2
    slice_points = candidate_points[in_slice]

    # Normalize to model coordinates
    norm_params = config.coordinate_normalization
    slice_points_norm = (
        (slice_points - norm_params['center']) / norm_params['global_scale']
    )

    # Generate expression + annotations
    with torch.no_grad():
        output = model(slice_points_norm.to(config.device))
        # ... same as generate_virtual_slice Steps 3-5 ...
    
    return adata_section
```

### 5.3 Computing Spatial Expression Gradients (Unique to SpatialCPA)

```python
def compute_expression_gradient(model, coords_3d, cell_type_idx, gene_idx):
    """
    Compute the spatial gradient of a gene's expression at given positions.
    This is UNIQUE to our method — SpatialZ cannot do this because its
    representation is not differentiable.

    Returns: (batch, 3) gradient vectors [∂expr/∂x, ∂expr/∂y, ∂expr/∂z]
    
    Biological applications:
    - Gradient magnitude peaks at tissue boundaries
    - Gradient direction reveals molecular axes of organization
    - ∂expr/∂z specifically reveals between-layer transitions
    """
    coords_3d = coords_3d.clone().requires_grad_(True)
    output = model(coords_3d, cell_type_idx)
    expression_mean = output['expression_params']['mean'][:, gene_idx]

    gradient = torch.autograd.grad(
        outputs=expression_mean.sum(),
        inputs=coords_3d,
        create_graph=False
    )[0]  # (batch, 3)

    return gradient
```

### 5.4 Generating Dense 3D Atlas

```python
def generate_dense_atlas(model, z_start, z_end, z_step, xy_bounds, config):
    """
    Generate a complete dense 3D atlas by querying at regular z-intervals.
    
    Unlike SpatialZ, z_step can be ANY value — it doesn't need to be 
    related to the original section spacing. You can generate at 1 μm 
    resolution if desired.
    """
    z_values = np.arange(z_start, z_end + z_step, z_step)
    all_slices = []
    
    for z in z_values:
        adata_slice = generate_virtual_slice(model, z, xy_bounds, config)
        all_slices.append(adata_slice)
    
    # Concatenate into a single atlas
    atlas = anndata.concat(all_slices)
    return atlas
```

---

## 6. Evaluation Plan

### 6.1 Metrics (matching SpatialZ for fair comparison)

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| Gene-wise Pearson r | Per-gene correlation between predicted and real expression | Compute per gene across cells, report mean |
| Moran's I correlation | Spatial autocorrelation pattern agreement | Compute Moran's I per gene on real and virtual slices, correlate |
| Geary's C correlation | Local spatial pattern agreement | Same as above with Geary's C |
| Cell-type accuracy | Correct cell-type assignment rate | Compare predicted vs true labels |
| Region accuracy | Correct region assignment rate | Compare predicted vs true labels |
| SSIM | Pixel-level spatial expression fidelity | Rasterize expression to grid, compute structural similarity |
| Top SVG overlap | Agreement on spatially variable genes | Compare ranked lists from real vs virtual |

Additional metrics unique to SpatialCPA:

| Metric | What it measures |
|--------|-----------------|
| Gradient biological consistency | Do computed expression gradients align with known tissue boundaries? |
| Extrapolation quality | Performance on z-values outside the convex hull of observed slices |
| Cross-angle consistency | Does a coronal virtual slice agree with a sagittal virtual slice at their intersection? |
| **Gap-stratified performance** | **Performance broken down by gap size (NEW — see 6.3)** |

### 6.2 Evaluation Protocol

```
Experiment 1: Leave-one-out on STARmap 3D data (same as SpatialZ Fig. 2)
  - Partition 3D data into 7 consecutive sections
  - Hold out sections 2, 4, 6
  - Train on sections 1, 3, 5, 7
  - Predict held-out sections
  - Compare: SpatialZ vs SpatialCPA on all metrics above

Experiment 2: MERFISH hypothalamus (same as SpatialZ Extended Data Fig. 1)
  - Use 5 real sections with large z-gaps
  - Hold out 1 section at a time (5-fold)
  - Generate 3 virtual slices between each pair
  - Compare downstream clustering (STAGATE, BINARY)
  - Report ARI and NMI improvement

Experiment 3: BICCN brain atlas at scale (same as SpatialZ Fig. 3)
  - Use 129 MERFISH sections
  - Hold out every 10th section for validation
  - Train on remaining 116 sections
  - Generate virtual slices at held-out positions
  - Compare cell-type composition and marker gene patterns

Experiment 4: Arbitrary z-value querying (NEW — SpatialZ cannot do this)
  - Train on all available sections
  - Query at z-values that are NOT midpoints between adjacent sections
  - Show predictions remain biologically plausible
  - Validate with Allen Brain Atlas ISH reference images

Experiment 5: Arbitrary-angle sectioning (partially comparable to SpatialZ Fig. 4)
  - Generate sagittal and horizontal slices from coronal-trained model
  - Compare marker gene patterns with Allen Brain Atlas
  - Compute cross-angle consistency at intersections

Experiment 6: Expression gradient analysis (NEW — completely unique)
  - Compute ∂expression/∂z for layer markers (Cux2, Lamp5, Fezf2, etc.)
  - Show gradient magnitude peaks at cortical layer boundaries
  - Validate against known laminar organization
```

### 6.3 Gap-Stratified Evaluation (NEW)

A key advantage of SpatialCPA over SpatialZ is expected to be better performance
across variable gap sizes. We evaluate this explicitly:

```
Experiment 7: Gap-size robustness analysis (NEW)

  For datasets with many sections (DS1/DS2 with BICCN data):
  
  Step 1: Artificially create variable gap sizes
    - Start with all 129 sections
    - Create sparse subsets by removing sections at different rates:
      a) Remove every 2nd section (gaps ~200 μm)
      b) Remove every 3rd section (mixed gaps ~100-300 μm)
      c) Remove random sections (highly irregular gaps)
      d) Remove consecutive blocks (simulating damaged tissue regions)
  
  Step 2: Train SpatialCPA and run SpatialZ on each sparse subset
  
  Step 3: Evaluate on held-out sections, stratified by:
    - Small gap: held-out section is <50 μm from nearest training section
    - Medium gap: 50-150 μm
    - Large gap: 150-300 μm  
    - Very large gap: >300 μm
  
  Step 4: Report metrics per gap-size bin
    - Expected result: SpatialCPA degrades more gracefully than SpatialZ 
      as gaps increase, because SpatialZ only uses adjacent pairs while 
      SpatialCPA uses global spatial structure.
  
  Step 5: Plot "performance vs gap size" curves for both methods
    - This becomes a key figure in the paper showing our advantage.

Experiment 8: Section thickness sensitivity (NEW)

  Step 1: Using DS3 (which has z_plane), artificially vary the assumed 
          section thickness parameter in the slab module
    - Test: 5 μm, 10 μm, 20 μm, 50 μm
  
  Step 2: Compare prediction quality
    - Expected: performance is robust to moderate thickness 
      misspecification (±2x), but degrades if thickness is 
      grossly wrong (>5x error)
  
  Step 3: Validate learned thickness against known experimental value
    - If using learnable thickness parameter, check that it converges 
      to approximately the true value
```

---

## 7. Implementation Timeline

### Phase 1: Foundation (Months 1-3)

```
Month 1: Data harmonization + infrastructure
  Week 1-2:
    - Write harmonization pipeline for all 5 datasets (Section 2.3)
    - CRITICAL: verify with labmate:
      * How are spatial coordinates stored? (obsm vs obs columns)
      * What is the physical section thickness for each dataset?
      * What is the physical distance between sections?
      * Are DS1/DS2 registered to CCFv3? How to parse z from labels?
    - Implement physical z-coordinate registration (Section 2.4)
    - Implement gap-aware coordinate normalization (Section 2.5)
    - Create unified AnnData objects with all required metadata
  Week 3-4:
    - Set up training infrastructure (PyTorch, PyTorch Lightning)
    - Implement data loaders with multi-dataset sampling
    - Implement evaluation metrics (Moran's I, Geary's C, SSIM)
    - Create visualization utilities

Month 2: Core model implementation
  Week 1: AdaptiveFourierFeatures (Section 3.2) — calibrate to data
  Week 2: SpatialBackbone + BatchAwareSpatialBackbone (Section 3.3)
  Week 3: AnnotationHead + ExpressionDecoder (Sections 3.4, 3.5)
  Week 4: SliceAsSlabModule + full model assembly (Sections 3.6, 3.7)

Month 3: Training on smallest dataset first
  Week 1-2: Train on STARmap 3D data
    - Debug training loop with z-marginalization
    - Tune hyperparameters (frequencies, hidden dims, z_samples)
    - Sanity check: model can overfit a single slice
    - Verify slab module doesn't collapse to zero thickness
  Week 3-4: Implement gap-aware LOO training
    - Implement GapAwareLOOTrainer (Section 4.2)
    - Run first LOO evaluation
    - Compare with SpatialZ results
```

### Phase 2: Validation + Benchmarking (Months 4-6)

```
Month 4: MERFISH hypothalamus experiments
  - Train on hypothalamus data (larger z-gaps → tests gap handling)
  - Run Experiment 2 (LOO evaluation)
  - Benchmark against SpatialZ
  - This dataset is the BEST test of gap-aware design because it has 
    large, irregular spacing

Month 5: Scale to brain atlas
  - Implement mini-batch spatial stratification for BICCN data
  - Train on DS1/DS2
  - Run Experiment 3
  - Run Experiment 7 (gap-stratified evaluation) ← KEY EXPERIMENT
  - Profile computational costs

Month 6: Novel capabilities
  - Implement arbitrary z querying (Experiment 4)
  - Implement arbitrary-angle sectioning (Experiment 5)
  - Implement expression gradient computation (Experiment 6)
  - Run Experiment 8 (thickness sensitivity)
```

### Phase 3: Analysis + Paper (Months 7-10)

```
Month 7: Deep biological analysis
  - Cortical layer analysis using expression gradients
  - 3D spatial domain identification
  - Cross-angle consistency validation
  - Comparison with Allen Brain Atlas ISH data

Month 8: Robustness analysis
  - Vary number of training slices
  - Vary gene panel size
  - Uncertainty quantification
  - Gap-size robustness curves (key paper figure)

Month 9: Software packaging
  - Clean codebase, documentation, tutorials
  - pip-installable package
  - GitHub repository

Month 10: Paper writing
  - Target: Nature Methods or Genome Biology
  - Key framing: continuous neural fields + gap-aware design
  - Emphasize gap-stratified evaluation as differentiator
```

---

## 8. Computational Requirements

| Component | Estimated resource | Notes |
|-----------|-------------------|-------|
| Training (small, STARmap) | 1 GPU, ~2 hours | Quick iteration, debug |
| Training (MERFISH hypothalamus) | 1 GPU, ~8 hours | Medium scale |
| Training (full BICCN atlas) | 1-4 GPUs, ~48-96 hours | Largest experiment |
| Inference (single virtual slice) | 1 GPU, ~30 seconds | Real-time capable |
| Inference (full 3D atlas) | 1 GPU, ~2-4 hours | vs SpatialZ's 801 hours |
| Memory (training) | ~16-32 GB GPU RAM | Depends on batch size |
| Storage (processed data) | ~50-100 GB | All 5 datasets |

Note: z-marginalization adds ~n_z_samples× overhead per training step (default 5×),
but this is offset by the fact that the model converges faster due to better regularization.

---

## 9. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Oversmoothing of expression | High | Medium | ZINB sampling preserves stochasticity; increase frequency scale |
| Poor performance with large gaps | Medium | High | Gap-aware LOO training; global field uses all slices |
| Cell-type errors at boundaries | Medium | Medium | Soft probabilities; ensemble from nearby z-queries |
| Scalability to 38M cells | Medium | High | Mini-batch training; embarrassingly parallel inference |
| Batch effects between datasets | High | Medium | Per-dataset bias terms; careful normalization |
| z-marginalization too slow | Low | Medium | Reduce n_z_samples; only marginalize in Phase 1 |
| Learned thickness diverges | Low | Medium | Regularize toward known physical value; monitor |
| Fourier features wrong scale | Medium | Medium | Calibrate from data (Section 2.5); try learnable=True |
| Cannot reproduce SpatialZ results | Low | High | Use identical evaluation protocol and datasets |

---

## 10. Key Differences from SpatialZ (for paper framing)

| Aspect | SpatialZ | SpatialCPA (ours) |
|--------|----------|-------------------|
| Representation | Discrete pairwise interpolation | Continuous 3D neural field |
| z-coordinate handling | Sequential index or simple midpoint | Physical units with gap-aware design |
| Section thickness | Ignored (infinitely thin planes) | Modeled as slabs with z-marginalization |
| Non-uniform gaps | Each pair treated independently | Global field; gap-aware LOO training |
| Location generation | Wasserstein barycenter optimization | Density field prediction |
| Cell-type assignment | k-NN from adjacent slices | Learned spatial classifier |
| Gene expression | Copied from real cells (sampling) | Generated from learned ZINB distribution |
| Query flexibility | Fixed positions between slice pairs | Any (x, y, z) coordinate |
| In silico sectioning | Separate module with coordinate rotation | Native — query along any plane |
| Expression gradients | Not possible | Analytic via backpropagation |
| Joint optimization | No (3 independent stages) | Yes (end-to-end differentiable) |
| Extrapolation | Not supported | Possible (with caveats) |
| Training paradigm | No training (heuristic pipeline) | Supervised + self-supervised |
| Gap robustness | Degrades with gap size (only uses 2 slices) | Graceful degradation (uses all slices) |

---

## 11. Questions to Ask Your Labmate (Checklist)

Before starting implementation, confirm the following for each dataset:

```
For ALL datasets:
  □ How are spatial coordinates stored? (adata.obsm['spatial'] or obs columns?)
  □ What are the coordinate units? (μm? pixels? arbitrary?)
  □ What is the physical section thickness?
  □ What is the physical distance between consecutive sections?
  □ Are sections evenly spaced or irregularly spaced?
  □ Are any sections missing or damaged?

For DS1 / DS2 (MERFISH brain):
  □ Are sections registered to Allen CCFv3?
  □ What does the brain_section_label format mean exactly?
     (e.g., "C57BL/6J-1.050" — is 1.050 in mm from bregma?)
  □ What is the section thickness? (typical MERFISH: 10 μm)
  □ What is the gap between consecutive sections? (~100 μm for BICCN?)

For DS3 (sectioned tissue):
  □ Is z_plane in physical units or sequential indices?
  □ If indices, what is the spacing between consecutive z_planes?
  □ Are X_Scaled / Y_Scaled in μm or normalized?

For DS4 (integrated atlas):
  □ What technology was this generated with?
  □ Are sections from a single specimen or multiple?
  □ How are sections ordered? What is the spacing?

For DS5 (imaging-based):
  □ What are the units of x_pos, y_pos?
  □ What technology? (MERFISH, seqFISH, Visium, etc.)
  □ How many sections and what is the spacing?
```
