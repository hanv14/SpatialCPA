"""SpatialCPA-v15 — orchestration (Phase 4: inference).

The generator wires the phases together in the order the proposal specifies and
runs them on construction, so a caller only has to ask for a depth:

    gen = SpatialCPAv15(stack, gene_names=..., cell_type_names=...)
    vs  = gen.generate_virtual_slice(z=42.0)

Phase 4, step by step:

1. **Register** the sparse specimen into the common frame (Phase 1 machinery).
2. **Structure** — ``n_structure_seeds`` deterministic completions of the field
   at the requested depth; the voxelwise spread across seeds is the structural
   uncertainty map.
3. **Layout** — cell positions and types are drawn from the completed field as a
   marked spatial point process (sub-voxel positions, no raster round-trip).
4. **Context** — encode the context cells from the *real* bracketing slices.
5. **Expression** — joint-block latent diffusion conditioned on layout, context
   and ``dz``.
6. **Decode** the latents into counts through the frozen VAE decoder.
7. **Provenance (mandatory)** — every synthesized cell carries its distance to
   the nearest real slice and its cross-seed structural spread, and any cell in a
   region whose z-extent is below the Nyquist limit is flagged **unresolved**.
   Observed and generated slices are visually indistinguishable; that is the
   point of the method and its main hazard, so the flags travel with the data.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from . import runtime
from .config import SpatialCPAv15Config
from .data import SliceStack, VirtualSlice
from .phase1_frame import build_common_frame, build_neighbor_graph
from .phase1_types import (build_type_representation, joint_cell_typing,
                           section_qc)
from .phase2_field import (build_field_spec, fit_compressor, rasterize_stack)
from .phase2_layout import sample_layout
from .phase2_structure import complete_with_uncertainty, train_structure_model
from .phase3_expression import (build_cell_index, decode_latents,
                                expression_gate, sample_expression_latents,
                                train_expression_diffusion,
                                train_expression_vae)


class SpatialCPAv15:
    """The full pipeline of ``spatialcpa_v15_idea.md``, fitted on construction."""

    def __init__(self, stack: SliceStack, gene_names: Optional[List[str]] = None,
                 cell_type_names: Optional[List[str]] = None,
                 cfg: Optional[SpatialCPAv15Config] = None):
        self.cfg = cfg or SpatialCPAv15Config()
        self.gene_names = list(gene_names) if gene_names is not None else \
            [f"gene_{i}" for i in range(stack.n_genes)]
        self.diagnostics: Dict = {}
        t0 = time.time()

        # Device + GPU-sharing policy, decided once for the whole fit.  The GPU
        # is used automatically when present; see ``runtime.py`` for why the
        # allocator is configured to grow and shrink rather than hold its peak.
        self.device = runtime.configure(self.cfg.runtime)
        self.diagnostics["runtime"] = {
            "device": str(self.device),
            "description": runtime.describe(),
            "expandable_segments": self.cfg.runtime.expandable_segments,
            "release_cache_between_stages":
                self.cfg.runtime.release_cache_between_stages,
            "memory_fraction": self.cfg.runtime.memory_fraction,
        }
        self._log(f"  [runtime] {runtime.describe()}")

        # ── Phase 1.1 — QC (per section, so batch structure stays visible) ──
        self.stack, qc = section_qc(stack, self.cfg.qc)
        self.diagnostics["phase1_qc"] = qc
        self._log(f"  [phase1.1] {self.stack.n_slices} sections, "
                  f"{self.stack.n_cells} cells, {qc['n_dropped']} dropped by QC")

        # ── Phase 1.3 — joint typing across every section at once ───────────
        types = self.stack.all_types()
        if types is None or cell_type_names is None:
            types, cell_type_names = joint_cell_typing(
                self.stack.all_expression(), seed=self.cfg.seed)
            off = 0
            for s in self.stack.slices:
                s.cell_type_indices = types[off:off + s.n_spots]
                off += s.n_spots
            self._log(f"  [phase1.3] no labels supplied -> joint typing across "
                      f"all sections: {len(cell_type_names)} types")
        types = np.where(types < 0, 0, types)
        off = 0
        for s in self.stack.slices:
            s.cell_type_indices = types[off:off + s.n_spots]
            off += s.n_spots
        self.cell_type_names = list(cell_type_names)

        # ── Phase 1.2 — the continuous, hierarchical type embedding ─────────
        sections = np.concatenate([np.full(s.n_spots, s.section_id)
                                   for s in self.stack.slices])
        self.type_rep = build_type_representation(
            self.stack.all_expression(), types, sections, self.cell_type_names,
            self.gene_names, self.cfg.types, self.cfg.hierarchy)
        self.diagnostics["phase1_types"] = self.type_rep.diagnostics
        d = self.type_rep.diagnostics
        self._log(f"  [phase1.2] {d['n_types']} types ({d['label_kind']}), "
                  f"encoder={d['encoder']}, prior-vs-data mantel_r="
                  f"{d['prior_vs_data_mantel_r']:.3f} knn_overlap="
                  f"{d['prior_vs_data_knn_overlap']:.2f}, "
                  f"{d['n_rare_types_prior_dominated']} rare types prior-dominated")

        # ── Phase 1.4/1.5 — common frame + normalized anatomical coords ─────
        self.frame = build_common_frame(self.stack, self.cfg.frame)
        self.diagnostics["phase1_frame"] = {
            "stabilization_applied": self.frame.applied,
            "residual_drift_spacings": self.frame.drift,
            "median_cell_spacing": self.frame.median_spacing,
        }
        self._log(f"  [phase1.4] residual drift={self.frame.drift:.2f} spacings "
                  f"-> stabilization {'applied' if self.frame.applied else 'not needed'}")

        # ── Phase 1.6 — 3D kNN graph with explicit edge dz ──────────────────
        u_all, z_all = [], []
        for s in self.stack.slices:
            u, _ = self.frame.to_normalized(s.coords_xy, s.z_values, s.section_id)
            u_all.append(u)
            z_all.append(s.z_values)
        self.u_all = np.vstack(u_all)
        self.z_all = np.concatenate(z_all)
        self.spacing_norm = self.frame.median_spacing / self.frame.scale
        self.graph = build_neighbor_graph(
            self.u_all, self.z_all, 1.0 / max(self.frame.z_range, 1e-9), self.cfg.frame)
        self.diagnostics["phase1_graph"] = {
            "k": int(self.graph.idx.shape[1]),
            "median_edge_dz": float(np.median(self.graph.dz)),
        }

        # ── Phase 2.1/2.2 — the multi-channel structure field ───────────────
        self.spec = build_field_spec(self.u_all, self.type_rep,
                                     self.spacing_norm, self.cfg.field_)
        fields = rasterize_stack(self.stack, self.frame, self.spec)
        self.compressor = fit_compressor(fields, self.cfg.field_)
        fields = self.compressor.encode(fields).astype(np.float32)
        self.diagnostics["phase2_field"] = {
            "grid": [self.spec.grid_h, self.spec.grid_w],
            "n_channels": self.spec.n_channels,
            "n_group_channels": self.spec.n_groups,
            "n_embedding_channels": int(self.spec.type_embedding.shape[1]),
            "compressed": self.compressor.enabled,
        }
        self._log(f"  [phase2.1] field {self.spec.grid_h}x{self.spec.grid_w} x "
                  f"{self.spec.n_channels} channels "
                  f"({self.spec.n_groups} group + "
                  f"{self.spec.type_embedding.shape[1]} embedding)")

        # ── Phase 2.3/2.4/2.5 — query-based interpolation + gate ────────────
        self.structure = train_structure_model(
            fields, self.stack.z_positions, self.spec, self.cfg.structure,
            seed=self.cfg.seed, verbose=self.cfg.verbose,
            compressor=self.compressor, rt=self.cfg.runtime)
        self.diagnostics["phase2_structure"] = self.structure.diagnostics

        # ── Phase 3.1 — the expression VAE ──────────────────────────────────
        log_expr = self.stack.all_expression()
        counts = self.stack.all_counts()
        if counts is None:
            counts = np.expm1(np.clip(log_expr, 0, 30))
        self.space = train_expression_vae(counts, log_expr, self.cfg.vae,
                                          seed=self.cfg.seed,
                                          verbose=self.cfg.verbose,
                                          rt=self.cfg.runtime)
        self.diagnostics["phase3_vae"] = self.space.diagnostics

        # ── Phase 3.2 — conditional latent diffusion ────────────────────────
        self.index = build_cell_index(self.stack, self.frame,
                                      self.space.latents, self.space.log_lib)
        self.dz_unit = float(self.stack.median_spacing) or 1.0
        self.expr = train_expression_diffusion(
            self.index, self.type_rep, self.stack, self.space,
            self.cfg.diffusion, self.dz_unit, seed=self.cfg.seed,
            verbose=self.cfg.verbose, rt=self.cfg.runtime)
        self.diagnostics["phase3_diffusion"] = self.expr.diagnostics

        # ── Phase 3.3 — the gate (also selects the decoder readout) ─────────
        readout = self.cfg.diffusion.decoder_readout
        if readout == "auto":
            gate = expression_gate(self.expr, self.space, self.index,
                                   self.type_rep, self.stack, self.dz_unit,
                                   seed=self.cfg.seed, verbose=self.cfg.verbose)
            self.diagnostics["phase3_gate"] = gate
            readout = gate.get("chosen_readout", "mean") if gate.get("ran") else "mean"
        self.expr.readout = readout

        # Training is over: hand the cached GPU blocks back before inference, so
        # the card is free for other processes for as long as this run is idle.
        if self.cfg.runtime.release_cache_between_stages:
            runtime.release(self.device)
        peak = runtime.peak_memory_gib(self.device)
        if peak is not None:
            self.diagnostics["runtime"]["peak_reserved_gib"] = peak

        self.diagnostics["fit_seconds"] = time.time() - t0
        self._log(f"  [fit] {self.diagnostics['fit_seconds']:.1f}s "
                  f"(structure learned={self.structure.use_learned}, "
                  f"expression diffusion={self.expr.trained}, readout={readout}"
                  + (f", peak GPU {peak:.2f} GiB" if peak is not None else "") + ")")

    # ── Phase 4 ─────────────────────────────────────────────────────────────
    def generate_virtual_slice(self, z: float,
                               n_cells: Optional[int] = None) -> VirtualSlice:
        """Synthesize the section at depth ``z`` (Phase 4, steps 2-7)."""
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)

        # 2 — structure: N deterministic completions -> field + uncertainty.
        field, spread = complete_with_uncertainty(
            self.structure, float(z), cfg.inference.n_structure_seeds)

        # 3 — layout: a marked spatial point process on the completed field.
        layout = sample_layout(field, spread, self.spec, cfg.layout,
                               self.spacing_norm, rng, n_cells=n_cells)

        # 4 — context: the real bracketing slices only.
        a_out, a, b, b_out = self.stack.brackets(float(z))
        allowed = np.array(sorted({a_out, a, b, b_out}))
        z_norm = np.full(layout.u.shape[0], self.frame.z_to_normalized(float(z)))

        # library size: drawn per cell from the real flanking cells of the same
        # type — technical depth, so it is sampled, never modelled.
        log_lib = self._sample_library(layout.types, allowed, rng, u=layout.u)

        # 5/6 — expression: joint-block latent diffusion, then decode.
        if self.expr.trained:
            Z = sample_expression_latents(
                self.expr, self.index, self.type_rep, layout.u, z_norm,
                layout.types, log_lib, allowed, float(z), self.dz_unit,
                float(self.index.log_lib.mean()), seed=cfg.seed)
        else:
            Z = self._latent_fallback(layout, allowed, rng)
        Z = self._shrink_to_type_mean(Z, layout.types, allowed)
        X = decode_latents(self.space, Z, log_lib, self.expr.readout, rng)

        expr_spread = None
        if cfg.inference.n_expression_samples > 1:
            # Extra samples measure uncertainty; they must NOT be averaged into
            # the output.  Averaging in data space shrinks every cell toward the
            # conditional mean and flattens the gene-gene covariance that makes a
            # profile look like a real cell -- measured on a correlated-channel
            # panel, co-expression agreement falls 0.94 -> 0.86 at 4 samples and
            # 0.71 at 8.  This mirrors Phase 2.4, where each seed yields one
            # *coherent* volume and the spread across seeds is reported
            # separately rather than blended into one.
            samples = [X]
            for s in range(1, cfg.inference.n_expression_samples):
                Zs = sample_expression_latents(
                    self.expr, self.index, self.type_rep, layout.u, z_norm,
                    layout.types, log_lib, allowed, float(z), self.dz_unit,
                    float(self.index.log_lib.mean()), seed=cfg.seed + s) \
                    if self.expr.trained else Z
                samples.append(decode_latents(self.space, Zs, log_lib,
                                              self.expr.readout, rng))
            expr_spread = np.stack(samples).std(axis=0).mean(axis=1)
            # X stays the first (coherent) sample.

        if not cfg.inference.output_counts:
            tot = np.maximum(X.sum(axis=1, keepdims=True), 1e-9)
            X = np.log1p(X / tot * 1e4)

        # Back to the caller's frame.  The Phase 1.4 stabilization is per section,
        # so the shift to undo at an intermediate depth is the one interpolated
        # between the two inner brackets.
        shift = None
        if self.frame.applied:
            t_gap, _ = self.stack.interp_fraction(float(z))
            sa = self.frame.shifts.get(self.stack.slices[a].section_id, np.zeros(2))
            sb = self.frame.shifts.get(self.stack.slices[b].section_id, np.zeros(2))
            shift = (1.0 - t_gap) * sa + t_gap * sb
        xy = self.frame.from_normalized(layout.u, shift)
        coords = np.column_stack([xy, np.full(layout.u.shape[0], float(z))])

        # 7 — provenance, mandatory.
        dist = np.full(layout.u.shape[0],
                       self.stack.distance_to_nearest_section(float(z)))
        unresolved = None
        if cfg.inference.sub_nyquist_flag:
            unresolved = np.full(layout.u.shape[0],
                                 dist[0] > 0.5 * self.stack.median_spacing)

        names = np.array(self.cell_type_names, dtype=object)
        return VirtualSlice(
            coords=coords, expression=X.astype(np.float32),
            cell_type=names[np.clip(layout.types, 0, len(names) - 1)].astype(str),
            dist_to_real_slice=dist, structural_spread=layout.spread,
            expression_spread=expr_spread, unresolved=unresolved,
            diagnostics={
                "z": float(z),
                "n_cells": int(layout.u.shape[0]),
                "brackets": [self.stack.slices[i].section_id for i in allowed],
                "t_in_gap": self.stack.interp_fraction(float(z))[0],
                "dz_gap": self.stack.interp_fraction(float(z))[1],
                "dist_to_nearest_real_slice": float(dist[0]),
                "sub_nyquist": bool(unresolved[0]) if unresolved is not None else False,
                "mean_structural_spread": float(np.mean(layout.spread)),
                "structure_learned": bool(self.structure.use_learned),
                "expression_diffusion": bool(self.expr.trained),
                "readout": self.expr.readout,
                **layout.diagnostics,
            })

    # ── helpers ─────────────────────────────────────────────────────────────
    def _sample_library(self, types: np.ndarray, allowed: np.ndarray,
                        rng: np.random.Generator,
                        u: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-cell library size, taken from the real bracketing slices.

        Library size is technical depth, so it is *sampled* rather than modelled
        (Phase 3.1).  But "sampled" must not mean "spatially random": in a real
        section the per-cell total is strongly spatially autocorrelated — cell
        size, staining depth and tissue compaction all vary smoothly across the
        slice — and drawing a random same-type cell throws that structure away.
        Because the total multiplies every channel, destroying it dilutes the
        spatial autocorrelation of *every* gene at once; on real IMC this halves
        Moran's agreement (0.708 -> 0.307 measured with the layout held fixed).

        When positions are available the library is therefore taken from the
        **spatially nearest** cell of the same type in the bracketing slices,
        which keeps the field's own smooth variation. Without positions it falls
        back to the type-conditional random draw.
        """
        pool = np.where(np.isin(self.index.section, allowed))[0]
        if pool.size == 0:
            pool = np.arange(self.index.log_lib.shape[0])
        out = np.empty(types.shape[0])
        pool_types = self.index.types[pool]
        for c in np.unique(types):
            m = types == c
            same = pool[pool_types == c]
            src = same if same.size else pool
            if u is not None and self.cfg.inference.spatial_library and src.size:
                from scipy.spatial import cKDTree
                _, nn = cKDTree(self.index.u[src]).query(u[m], k=1)
                out[m] = self.index.log_lib[src[np.atleast_1d(nn)]]
            else:
                out[m] = self.index.log_lib[rng.choice(src, size=int(m.sum()),
                                                       replace=True)]
        return out

    def _shrink_to_type_mean(self, Z: np.ndarray, types: np.ndarray,
                             allowed: np.ndarray) -> np.ndarray:
        """Optionally pull generated latents toward their type-conditional mean.

        Identity at the default ``latent_shrinkage = 0``.  See the config note:
        this trades the within-type variation the generation metrics reward for
        the conditional-mean accuracy the cell-matched metric rewards.
        """
        s = float(self.cfg.inference.latent_shrinkage)
        if s <= 0.0:
            return Z
        pool = np.where(np.isin(self.index.section, allowed))[0]
        if pool.size == 0:
            pool = np.arange(self.index.latents.shape[0])
        out = Z.copy()
        pool_types = self.index.types[pool]
        for c in np.unique(types):
            m = types == c
            same = pool[pool_types == c]
            src = same if same.size else pool
            out[m] = (1.0 - s) * Z[m] + s * self.index.latents[src].mean(axis=0)
        return out

    def _latent_fallback(self, layout, allowed: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
        """Used only when the stack is too small to train Phase 3.2 (< 3 sections).

        The latent is the type-conditional mean of the bracketing cells plus the
        observed within-type latent scatter — i.e. still a draw in the VAE latent
        space, just without the diffusion prior.  The Phase 3.3 gate reports when
        this path is active.
        """
        pool = np.where(np.isin(self.index.section, allowed))[0]
        if pool.size == 0:
            pool = np.arange(self.index.latents.shape[0])
        Z = np.zeros((layout.u.shape[0], self.space.latent_dim))
        pool_types = self.index.types[pool]
        for c in np.unique(layout.types):
            m = layout.types == c
            same = pool[pool_types == c]
            src = same if same.size else pool
            Z[m] = self.index.latents[rng.choice(src, size=int(m.sum()),
                                                 replace=True)]
        return Z

    def _log(self, msg: str):
        if self.cfg.verbose:
            print(msg, flush=True)
