# The §9 thickness check — it cannot be run from the built input, and here is the proof

**Asked:** re-partition `merfish_thick_hypothalamus`'s 200 µm block into 14 slabs of ~13.5 µm, same
cells, thickness the only variable, and report the budget beside the 7-slab version.

**Answer: the built file cannot be re-partitioned, and no build of `resection.py` can change that.**
The depth it would be cut along is already gone. What *can* be run free is an exact substitute for
the 90° claim specifically, and it is below.

## 1. Why — from `loaders.py`, not from inspection of the file

```python
z_values = np.unique(z_per_cell[mask])
if z_values.size != 1:
    raise SchemaError(f"Section.z: section {label!r} spans {z_values.size} distinct depths ...")
```

`load_volume` **raises** unless every section spans exactly one depth — *"a section is one depth by
definition"*. The angle-budget sweep loaded `merfish_thick_hypothalamus` successfully. Therefore its
7 slabs are 7 discrete z values, and no cell retains its own depth within its slab.

Cutting that into 14 bins gives **7 populated slabs and 7 empty ones** — every cell of an original
slab shares one depth and so lands in one bin. That result would look like a measurement and be an
artefact.

## 2. `resection.py` is built anyway, and its refusal *is* the check

`specs/10` §9 owed this module and it did not exist. It is now built, and
`plan_partition` refuses the flattened case by counting distinct source depths:

```
plan_partition: the source has only 7 distinct depths along this normal, so it cannot be cut
into 14 slabs — at least 7 would be EMPTY and the rest would reproduce the partition the file
already has. ... Re-cut the volume at BUILD time (bench3's partition='z_width' with the slab
count you want), not from the built file.
```

It refuses the same-count case too (7 into 7 is not a re-cut), refuses slabs under bench3's own
50-cell floor by name, conserves every cell, folds the deepest cell back from `np.digitize`'s
phantom bin, and is deterministic. `tests/test_resection.py` covers all of it and needs no data.

**To actually run V4a**, bench3 must build the dataset with `partition="z_width"` and
`n_sections=14` from the raw block. That is a build, not a free read, and it is the same build
`specs/10` §8 already specifies.

## 3. The free substitute — and it is **exact** for the 90° claim, not an approximation

Run the budget on the shipped 7-slab build with the slab thickness halved:

```
python scripts/angle_budget.py --dataset merfish_thick_hypothalamus \
    --thickness 13.5 --out reports/angle_budget_thin.md
```

This halves the thickness and leaves the z-labelling at 7 slabs — so it is *not* V4a in general.
**But at 90° it is exactly V4a**, because at 90° the plane's normal is `(0, 1, 0)` and the band
selects on **y**: `Section.z` never enters the arithmetic, so the z-partition contributes nothing
and thickness is the whole of the manipulation.

That is asserted rather than argued: `test_at_90_degrees_the_band_is_an_IN_PLANE_cut_so_the_z_partition_cannot_matter`
moves every section's depth by an arbitrary offset and requires the 90° selection to be **bitwise
unchanged** — and requires the 0° selection to change, so the licence cannot be over-read to angles
where the partition is the whole selection.

**The claim rests on 90°.** So for the sentence the paper needs to write, the free substitute
settles it.

**Expected outcome, recorded before it runs.** Cells at 90° should fall roughly in proportion:
1988 × (13.5 / 27) ≈ **994**. G2 needs the largest type to hold ≥ 250 in the strip, and it halves
too. If the 7-slab largest type at 90° is above ~500 the thin partition still clears; below it, it
does not. **I do not know which**, and that is the point of running it.

## 4. How the claim is worded, decided in advance

- **If the thin substitute still clears 90°:** the claim is about *oblique generation*, and the
  thickness table is a robustness row.
- **If it does not:** the claim is about **thick-slab preparations specifically** — that oblique
  sections become generable and scorable when the specimen is a block cut into slabs rather than a
  stack of thin sections — and the paper says so in the abstract, with this table as the evidence.
  That is a narrower claim and a more defensible one, and stating it ourselves is worth more than
  having a reviewer state it.

Either way the budget table stands: no published method states what specimen geometry an oblique
evaluation needs.
