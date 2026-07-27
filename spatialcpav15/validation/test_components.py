"""Component tests for SpatialCPA-v15 — the properties the proposal insists on.

These are cheap, deterministic checks of the parts of the pipeline whose
correctness is a *stated requirement* rather than a matter of accuracy, so a
regression in any of them would silently change what the method claims to be:

* the type encoder is **order-invariant** over the marker set (Phase 1.2.1);
* the shrinkage blend really does hand rare types to the prior (Phase 1.2.2);
* the field rasterization **conserves mass** and its channels stay non-negative,
  and the Phase 2.2 compression round-trips (Phase 2.1/2.2);
* bracket selection returns **two slices per side** with correct edge handling
  (Phase 2.3);
* the deterministic sampler makes ``t -> I_t`` continuous under fixed noise
  (Phase 2.4);
* the layout point process reproduces the intensity it was given (Phase 4.3);
* generation is **reproducible** and carries provenance (Phase 4.7);
* the GPU-sharing policy is applied and every stage honours the device setting.

Run:  python spatialcpav15/validation/test_components.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from spatialcpav15 import (SpatialCPAv15, SpatialCPAv15Config, runtime,
                           build_stack_from_arrays)
from spatialcpav15.config import (FieldConfig, LayoutConfig, RuntimeConfig,
                                  TypeConfig)
from spatialcpav15.phase1_types import (classify_label_vocabulary, encode_types_prior,
                                        identify_markers, load_gene_encoder)
from spatialcpav15.phase2_field import fit_compressor, rasterize_section
from spatialcpav15.phase2_layout import _sample_positions

FAILURES = []


def check(name, cond, detail=""):
    status = "ok  " if cond else "FAIL"
    if not cond:
        FAILURES.append(name)
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")


def toy_stack(seed=0, n_sections=6, n_per=200, n_genes=24, n_types=4):
    rng = np.random.default_rng(seed)
    means = rng.gamma(2.0, 1.5, size=(n_types, n_genes))
    for t in range(n_types):
        means[t, t * 4:(t + 1) * 4] *= 5.0
    X, xyz, secs, ct = [], [], [], []
    for si in range(n_sections):
        r = np.sqrt(rng.uniform(0, 1, n_per)) * 30.0
        th = rng.uniform(0, 2 * np.pi, n_per)
        x, y = r * np.cos(th), r * np.sin(th)
        t = np.clip(np.digitize(r, np.linspace(0, 30, n_types + 1)) - 1, 0, n_types - 1)
        X.append(rng.poisson(means[t] * 30.0).astype(np.float32))
        xyz.append(np.column_stack([x, y, np.full(n_per, float(si))]))
        secs += [f"S{si}"] * n_per
        ct.append(t)
    counts = np.vstack(X)
    log = np.log1p(counts / np.maximum(counts.sum(1, keepdims=True), 1) * 1e4)
    return (build_stack_from_arrays(log, np.vstack(xyz), secs,
                                    np.concatenate(ct), counts),
            [f"g{i}" for i in range(n_genes)],
            [f"type_{i}" for i in range(n_types)])


# ── Phase 1.2 ────────────────────────────────────────────────────────────────

def test_label_kind():
    print("Phase 1.2.0 — label vocabulary classification")
    check("numbered clusters", classify_label_vocabulary(["0", "1", "leiden_7"]) == "numbered")
    check("text names", classify_label_vocabulary(["CA1 pyramidal", "astrocyte"]) == "text")
    check("mixed detected", classify_label_vocabulary(["0", "astrocyte"]) == "mixed")


def test_encoder_order_invariance():
    print("Phase 1.2.1 — the marker component is a SET, not a ranking")
    rng = np.random.default_rng(0)
    genes = [f"Gene{i}" for i in range(30)]
    X = rng.gamma(2.0, 1.0, size=(400, 30))
    types = rng.integers(0, 3, 400)
    X[types == 0, :5] *= 6.0
    X[types == 1, 5:10] *= 6.0
    cfg = TypeConfig()
    markers = identify_markers(X, types, 3, cfg)
    S, _ = load_gene_encoder(genes, cfg)
    E1 = encode_types_prior(markers, S, ["a", "b", "c"], "numbered", cfg)
    # Shuffle each marker set: a set encoder must be unmoved by this.
    shuffled = []
    for ms in markers:
        p = rng.permutation(ms.gene_idx.size)
        shuffled.append(type(ms)(ms.gene_idx[p], ms.effect[p]))
    E2 = encode_types_prior(shuffled, S, ["a", "b", "c"], "numbered", cfg)
    check("permuting the marker order changes nothing",
          np.allclose(E1, E2, atol=1e-10),
          f"max|dE| = {np.abs(E1 - E2).max():.2e}")


def test_shrinkage_blend():
    print("Phase 1.2.2 — rare types borrow the prior, well-sampled types do not")
    stack, genes, names = toy_stack()
    # Make type 3 rare by relabelling most of its cells to type 0.
    for s in stack.slices:
        m = s.cell_type_indices == 3
        drop = np.where(m)[0][: max(int(m.sum()) - 2, 0)]
        s.cell_type_indices[drop] = 0
    gen = SpatialCPAv15(stack, gene_names=genes, cell_type_names=names,
                        cfg=_fast_cfg())
    w = gen.type_rep.weights
    n_c = np.array([(stack.all_types() == c).sum() for c in range(len(names))])
    rare = int(np.argmin(n_c))
    check("weights increase with cell count",
          bool(np.all(np.diff(w[np.argsort(n_c)]) >= -1e-9)))
    check("the rarest type is prior-dominated", w[rare] < 0.5,
          f"w={w[rare]:.3f} at n={n_c[rare]}")
    check("agreement diagnostic reported",
          np.isfinite(gen.type_rep.diagnostics["prior_vs_data_mantel_r"]))


# ── Phase 2.1/2.2 ────────────────────────────────────────────────────────────

def test_field_properties():
    print("Phase 2.1/2.2 — field mass, non-negativity, compression round-trip")
    stack, genes, names = toy_stack()
    gen = SpatialCPAv15(stack, gene_names=genes, cell_type_names=names, cfg=_fast_cfg())
    u, _ = gen.frame.to_normalized(gen.stack.slices[0].coords_xy,
                                   gen.stack.slices[0].z_values,
                                   gen.stack.slices[0].section_id)
    F = rasterize_section(u, gen.stack.slices[0].cell_type_indices, gen.spec)
    n = gen.stack.slices[0].n_spots
    check("total-density channel integrates to the cell count",
          abs(F[0].sum() - n) < 0.02 * n, f"{F[0].sum():.1f} vs {n}")
    check("group channels sum to the total density",
          np.allclose(F[1:1 + gen.spec.n_groups].sum(axis=0), F[0], atol=1e-3))
    check("density channels are non-negative",
          bool((F[:1 + gen.spec.n_groups] >= -1e-6).all()))

    cfg = FieldConfig()
    cfg.compress_max_channels = max(2, F.shape[0] - 2)
    comp = fit_compressor(F[None], cfg)
    check("compression engages above the channel budget", comp.enabled)
    R = comp.decode(comp.encode(F[None]))[0]
    rel = np.abs(R - F).max() / max(np.abs(F).max(), 1e-9)
    check("compression round-trips within tolerance", rel < 0.05,
          f"max relative error {rel:.3f}")


def test_brackets():
    print("Phase 2.3 — two slices per side, with edge duplication")
    stack, _, _ = toy_stack(n_sections=6)
    check("interior query gets 4 distinct brackets",
          stack.brackets(2.5) == (1, 2, 3, 4))
    check("left edge duplicates the outer bracket",
          stack.brackets(0.5) == (0, 0, 1, 2))
    check("right edge duplicates the outer bracket",
          stack.brackets(4.5) == (3, 4, 5, 5))
    t, dz = stack.interp_fraction(2.5)
    check("t and dz are the gap fraction and gap width",
          abs(t - 0.5) < 1e-9 and abs(dz - 1.0) < 1e-9)


def test_deterministic_query_continuity():
    print("Phase 2.4 — fixed noise makes t -> I_t a continuous function")
    stack, genes, names = toy_stack(n_sections=6)
    cfg = _fast_cfg()
    cfg.structure.epochs = 40
    gen = SpatialCPAv15(stack, gene_names=genes, cell_type_names=names, cfg=cfg)
    m = gen.structure
    if not m.trained:
        check("structure model trained", False, "too few sections")
        return
    noise = m.fixed_noise(0)
    a = m.query_clipped(2.40, noise=noise)
    b = m.query_clipped(2.42, noise=noise)
    c = m.query_clipped(2.90, noise=noise)
    near = np.abs(a - b).mean()
    far = np.abs(a - c).mean()
    check("nearby queries give nearby fields under one seed", near <= far + 1e-9,
          f"|I(2.40)-I(2.42)|={near:.4f} <= |I(2.40)-I(2.90)|={far:.4f}")
    check("re-querying with the same noise is bit-identical",
          np.array_equal(a, m.query_clipped(2.40, noise=noise)))


def test_point_process():
    print("Phase 4.3 — the point process reproduces the intensity it was given")
    rng = np.random.default_rng(0)
    stack, genes, names = toy_stack()
    gen = SpatialCPAv15(stack, gene_names=genes, cell_type_names=names, cfg=_fast_cfg())
    spec = gen.spec
    yy, xx = np.mgrid[0:spec.grid_h, 0:spec.grid_w]
    intensity = np.exp(-((yy - spec.grid_h / 3) ** 2 + (xx - spec.grid_w / 2) ** 2) / 40.0)
    n = 4000
    u = _sample_positions(intensity, n, spec, rng, factor=LayoutConfig().intensity_upsample)
    check("exactly n points are produced", u.shape[0] == n, f"got {u.shape[0]}")
    yb, xb = spec.bin_of(u)
    hist = np.zeros((spec.grid_h, spec.grid_w))
    np.add.at(hist, (yb, xb), 1.0)
    r = np.corrcoef(hist.ravel(), intensity.ravel())[0, 1]
    check("sampled density correlates with the intensity", r > 0.97, f"r={r:.4f}")


def test_end_to_end_reproducibility():
    print("Phase 4 — reproducibility and mandatory provenance")
    stack, genes, names = toy_stack()
    cfg = _fast_cfg()
    g1 = SpatialCPAv15(stack, gene_names=genes, cell_type_names=names, cfg=cfg)
    v1 = g1.generate_virtual_slice(z=2.5)
    v2 = g1.generate_virtual_slice(z=2.5)
    check("repeated generation is identical",
          np.array_equal(v1.coords, v2.coords) and np.array_equal(v1.expression, v2.expression))
    stack2, _, _ = toy_stack()
    g2 = SpatialCPAv15(stack2, gene_names=genes, cell_type_names=names, cfg=cfg)
    v3 = g2.generate_virtual_slice(z=2.5)
    check("a fresh fit with the same seed reproduces the slice",
          np.allclose(v1.coords, v3.coords) and np.allclose(v1.expression, v3.expression, atol=1e-4))
    check("provenance: distance to nearest real slice attached",
          v1.dist_to_real_slice is not None and v1.dist_to_real_slice.shape[0] == v1.n_cells)
    check("provenance: structural spread attached",
          v1.structural_spread is not None and v1.structural_spread.shape[0] == v1.n_cells)
    check("provenance: sub-Nyquist flag attached",
          v1.unresolved is not None and v1.unresolved.shape[0] == v1.n_cells)
    check("expression is non-negative and finite",
          bool(np.isfinite(v1.expression).all() and (v1.expression >= 0).all()))
    check("cell count is emergent (not the training count)",
          0 < v1.n_cells < g1.stack.n_cells)
    check("both internal gates ran and were recorded",
          "phase2_structure" in g1.diagnostics and "phase3_gate" in g1.diagnostics)


def test_runtime_policy():
    print("Runtime — device selection and the GPU-sharing policy")
    import torch

    cpu = runtime.configure(RuntimeConfig(device="cpu"))
    check("device='cpu' is honoured", cpu.type == "cpu")
    auto = runtime.configure(RuntimeConfig(device="auto"))
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    check(f"device='auto' resolves to {expected}", auto.type == expected)
    check("release() is safe on any device", runtime.release(auto) is None)

    if torch.cuda.is_available():
        dev = runtime.configure(RuntimeConfig(device="cuda", memory_fraction=0.5))
        check("cuda requested and returned", dev.type == "cuda")
        check("expandable segments requested of the allocator",
              "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
              os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"))
        check("peak reserved memory is reported",
              runtime.peak_memory_gib(dev) is not None)
    else:
        # No GPU here: check the policy is *requested* correctly, and that asking
        # for CUDA explicitly fails loudly rather than silently running on CPU.
        cfg = RuntimeConfig(device="cuda")
        try:
            runtime.configure(cfg)
            check("explicit device='cuda' without a GPU raises", False)
        except RuntimeError:
            check("explicit device='cuda' without a GPU raises", True)
        print("       (no CUDA device here — the GPU branch is not exercised)")

    # Every trained stage must take its device from the one runtime config.
    stack, genes, names = toy_stack(n_sections=4, n_per=80)
    cfg = _fast_cfg()
    cfg.runtime.device = "cpu"
    gen = SpatialCPAv15(stack, gene_names=genes, cell_type_names=names, cfg=cfg)
    devs = {str(gen.device), str(gen.space.device)}
    if gen.structure.device is not None:
        devs.add(str(gen.structure.device))
    if gen.expr.device is not None:
        devs.add(str(gen.expr.device))
    check("all three trained stages share one device", devs == {"cpu"}, str(devs))
    check("runtime policy recorded in diagnostics",
          gen.diagnostics["runtime"]["expandable_segments"] is True
          and gen.diagnostics["runtime"]["release_cache_between_stages"] is True)


def _fast_cfg():
    cfg = SpatialCPAv15Config()
    cfg.verbose = False
    cfg.structure.epochs = 30
    cfg.structure.ddim_steps = 5
    cfg.vae.epochs = 40
    cfg.vae.min_passes = 2
    cfg.diffusion.epochs = 30
    cfg.diffusion.ddim_steps = 5
    cfg.inference.n_structure_seeds = 2
    return cfg


def main():
    for fn in (test_label_kind, test_encoder_order_invariance, test_shrinkage_blend,
               test_field_properties, test_brackets,
               test_deterministic_query_continuity, test_point_process,
               test_end_to_end_reproducibility, test_runtime_policy):
        fn()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        sys.exit(1)
    print("all component checks passed")


if __name__ == "__main__":
    main()
