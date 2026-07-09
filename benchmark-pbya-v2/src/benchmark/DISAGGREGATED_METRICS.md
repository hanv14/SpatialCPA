# Disaggregated metrics & plots — plain-language guide

This explains, in everyday terms, what
`evaluate_disaggregated.py` computes and what `plot_disaggregated.py` draws:
what each table column and each plotted metric **means**, **how it is
calculated**, which direction is "good", and what to watch out for.

The benchmark asks: *a method was given all the slices of a 3-D tissue except
one, and had to synthesize the missing slice. How close is the synthesized slice
to the real (held-out) one?* The standard evaluator answers with a single median
number per held-out slice. These disaggregated outputs keep the **full
distribution** behind that number — one value **per gene** or **per cell** — so
you can see spread, skew, and outliers, not just the average.

---

## 1. Machinery shared by several metrics

A few operations are reused, so it helps to define them once.

### Alignment (predicted slice → real slice)
Each method builds its slice in its own coordinate system, which can be rotated,
flipped, scaled, or shifted relative to the real slice. Before any spatial
comparison we rigidly move the predicted cells onto the real slice's frame:
we search over several rotations **and** a mirror flip, and keep the pose that
puts the most real cells close to a predicted cell (an "orientation-robust"
fit). This is done **only for scoring** — the method never sees it, so it is not
a form of leakage. Metrics that are marked "alignment-free" below don't depend on
this step at all.

### Cell matching (nearest neighbour within 50 µm)
Generation does not place cells on top of the real cells, so we build a
correspondence for the *matched* metrics: after alignment, for each **real**
cell we find the **nearest predicted** cell. If that nearest neighbour is within
**50 µm** (`NN_MATCH_THRESHOLD_UM`), the real cell is "matched". Matched metrics
are computed only over these pairs.

### Three ways expression is put on a common scale
Methods emit expression on different scales (raw counts vs. log-normalized vs.
…). Depending on the metric, one of three representations is used:

- **Raw** — the numbers exactly as the method output them.
- **Log-normalized** — divide each cell by its total counts, multiply by 10,000,
  then `log(1 + x)` (negatives clipped to 0 first). Puts cells on a comparable
  library size; used for the scale-sensitive mean/variance/error metrics.
- **Rank-normalized** — within each gene, replace values by their rank scaled to
  (0, 1). This is invariant to *any* monotonic rescaling, so a method that
  outputs raw counts and one that outputs log values get identical scores. Used
  for the primary "structure" metrics so the comparison is scale-fair.

---

## 2. The three data tables

`evaluate_disaggregated.py` writes three long-format CSVs (per method × dataset ×
held-out slice, plus concatenated masters under
`results/summary/disaggregated/`).

| File | One row per | Correspondence needed? |
|---|---|---|
| `per_gene_matched.csv` | gene | yes (uses matched cell pairs) |
| `per_gene_generation.csv` | (held-out section, gene) | no |
| `per_cell.csv` | held-out real cell | matched cells only, for the expression columns |

Identifier columns (`method`, `dataset`, `holdout_id`, and `section` where
relevant) are on every row so you can filter/group freely.

---

## 3. The nine plotted metrics

`plot_disaggregated.py` makes **one figure per metric**. Each figure is a grid of
panels, **one panel per dataset**, and inside a panel each method is drawn as a
**violin** (the full distribution over that dataset's genes or cells) with a slim
**box** (median line, interquartile box, whiskers). See §4 for how to read them.

Below, "↑ better" means larger is better; "↓ better" means smaller is better.

### From `per_gene_matched` (cell-matched — reference only)
These need the cell-to-cell correspondence that de-novo generation doesn't truly
produce, so — per the benchmark README — they are kept for reference and are
**not** the primary score for generation. Still useful as a sanity check.

1. **`pearson` — per-gene Pearson correlation (↑ better, −1…1)**
   For one gene, line up its expression in every matched predicted cell against
   the same gene in the paired real cell, and take the Pearson correlation across
   those pairs. One value per gene. Near +1 = the method reproduces that gene's
   cell-to-cell ups and downs; near 0 = no relationship. (Undefined, shown as
   missing, if either side is constant.) Pearson is unaffected by a method's
   overall scale.

2. **`spearman` — per-gene Spearman correlation (↑ better, −1…1)**
   Same as above but on ranks instead of raw values, so it captures any
   *monotonic* agreement and is robust to nonlinear scaling / outliers.

3. **`rmse` — per-gene root-mean-square error (↓ better, ≥ 0)**
   For one gene, the square root of the average squared difference between
   predicted and real expression over the matched pairs. Penalizes large
   per-cell misses heavily. Computed on **raw** expression, so it is
   scale-sensitive — compare methods with this cautiously.

4. **`mae` — per-gene mean absolute error (↓ better, ≥ 0)**
   Like RMSE but the average of the *absolute* differences; less dominated by a
   few big misses. Also on raw expression (scale-sensitive).

### From `per_gene_generation` (correspondence-free — the honest generation view)

5. **`field_pearson` — per-gene spatial-field agreement (↑ better, −1…1)**
   Do the two slices light this gene up in the *same places*? After alignment,
   lay a 20×20 grid over the tissue; in each grid cell take the average
   (rank-normalized) expression of the gene; then correlate the predicted grid
   against the real grid across all cells occupied in both. Near +1 = the spatial
   pattern matches; it needs no cell-to-cell correspondence, only a coarse
   alignment.

6. **`morans_i_pred` — spatial autocorrelation of the prediction (↑ = more
   structured, ≈ −0.1…1)**
   Moran's I measures whether a gene forms smooth spatial patches rather than
   salt-and-pepper noise, computed **within the predicted slice alone**. Build a
   10-nearest-neighbour graph on predicted cell positions; for each cell take its
   (mean-centered, rank-normalized) value times the average value of its
   neighbours, sum over cells, and divide by the total variance. ≈ +1 =
   strongly clustered/smooth, ≈ 0 = random, negative = checkerboard. This shows
   whether a method produces spatially structured genes at all. (The table also
   stores `morans_i_gt` for the real slice; it isn't plotted because it's the same
   regardless of method. The reduced score `gen_morans_agreement` correlates the
   predicted vs. real Moran's I across genes — here you see the raw predicted
   values per gene.)

### From `per_cell`

7. **`nn_dist_um` — distance to nearest predicted cell (↓ better, ≥ 0, in µm)**
   For each **real** held-out cell, the distance (after alignment) to the closest
   predicted cell. Small distances everywhere = the predicted cloud actually
   covers where cells are. This is the raw quantity behind the summary
   `matching_rate` (which is just the fraction of real cells with
   `nn_dist_um` ≤ 50 µm).

8. **`cell_pearson` — per-cell expression-profile similarity (↑ better, −1…1)**
   For each **matched** real cell (nearest predicted within 50 µm), correlate its
   whole log-normalized expression **profile across genes** against the nearest
   predicted cell's profile. Near +1 = the matched predicted cell "looks like"
   the real cell across the gene panel (right cell state); near 0 = unrelated
   profiles.

9. **`cell_mae` — per-cell expression error (↓ better, ≥ 0)**
   For each matched real cell, the average absolute difference across genes
   between its log-normalized profile and the nearest predicted cell's profile.
   Lower = the two profiles are numerically close, not just correlated.

> The `per_cell` table also stores cell-type columns (`gt_cell_type`,
> `nn_pred_cell_type`, `celltype_correct`) that disaggregate the summary
> `celltype_accuracy`; they aren't among the nine default plots but are in the CSV
> if you want them.

---

## 4. How to read a plot

- **Grid** — one **panel per dataset**; panel title is the dataset name.
- **Within a panel** — the x-axis lists the methods (order and display names are
  configurable, see §5); the y-axis is the metric.
- **Violin** — the shaded shape is the full distribution of the metric over that
  dataset's genes (metrics 1–6) or cells (metrics 7–9). A wide bulge means many
  genes/cells sit at that value.
- **Box inside the violin** — the white/《coloured》box is the interquartile range
  (25th–75th percentile), the horizontal line is the **median**, and the whiskers
  reach the rest of the bulk. The median here is exactly the number the standard
  `metrics.json` would report — so the plot shows that summary *and* everything it
  hides.
- **Dashed line at 0** — drawn for correlation-type metrics (1, 2, 5, 6, 8) as a
  reference: above the line is positive agreement, below is anti-correlation.
- **Axis ranges** — correlation metrics are fixed to −1…1 for comparability;
  error/distance metrics start at 0 and auto-scale.
- **Colour** — each method has a fixed colour (the Springer Nature / `ggsci` npg
  palette) that stays the same in every panel and every figure, so you can track
  a method by colour. With only the box (`--kind box`) or points (`--kind strip`)
  the encoding is the same.

**Reading it:** a good method sits **high** (metrics 1, 2, 5, 6, 8) or **low**
(metrics 3, 4, 7, 9), with a **tight** distribution (consistent across
genes/cells) and few bad outliers. A high median but a long tail toward bad
values means the method is good on average but fails on some genes/cells.

---

## 5. Choosing method order and names on the plots

- `--method-order` sets the left-to-right order, e.g.
  `--method-order spatialcpav8_gen spatialz feast isost`. Any method present in
  the data but not listed is appended afterwards (so nothing silently
  disappears).
- `--method-names` sets display labels, e.g.
  `--method-names spatialz=SpatialZ feast=FEAST spatialcpav8_gen='SpatialCPA v8'`.
  Ordering and colours still use the underlying method id, so renaming never
  changes a method's colour or position.

---

## 6. Typical workflow

```bash
# 1. run the benchmark so each run has a prediction.h5 (existing pipeline)
python -m src.benchmark.run_all --methods spatialcpav8_gen spatialz feast isost

# 2. write the per-gene / per-cell tables (this repo's evaluate_disaggregated.py)
python -m src.benchmark.evaluate_disaggregated --all

# 3. draw the nine metric figures (Springer Nature theme)
python -m src.benchmark.plot_disaggregated \
    --method-order spatialcpav8_gen spatialz feast isost \
    --method-names spatialcpav8_gen='SpatialCPA v8' spatialz=SpatialZ \
                   feast=FEAST isost=isoST
```

Tables land next to each `prediction.h5` and, concatenated, in
`results/summary/disaggregated/`; figures (PNG + editable PDF) in
`results/summary/figures/disaggregated/`.
