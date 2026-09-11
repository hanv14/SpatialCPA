# The oblique claim, split in two

*Step 6 of the six. Written **before** the cross-dataset budget runs, so the framing is not chosen
by its result. Both halves are stated now; the table decides only which angle each one gets.*

## Why it has to split

The method's contribution is that a continuous field makes section generation **well-defined** at an
arbitrary position and orientation. That is a property of the method: given a plane, there is an
answer, and no previous layout mode had one off-axis (`nearest-z` names a *z*-distance, which means
nothing for a plane that spans the stack).

Whether that answer can be **scored** is a property of the *specimen*. `paper_celltype_localization`
skips a type with fewer than `min_gt_cells` = 20 cells and subsamples every type to `max_n` = 250. A
plane tilted by θ through a slab of depth `D` and thickness `t` exits the thin dimension after
`(D + t) / sin θ`, so past some angle an "oblique section" is a sliver holding fewer cells than the
statistic's own design point. That angle is the specimen's, not the method's.

**Conflating the two is what would make the paper indefensible.** Two claims, separately evidenced:

| | claim | evidence | shown or scored |
|---|---|---|---|
| **A** | the method is **well-defined** at an arbitrary orientation | a coherent generated section at **45°**, beside the same plane's real cells | **shown**, and labelled as shown |
| **B** | it is **evaluable**, and how far | a score at the largest angle that clears the metric's own gates, with the cell count and the angle stated together | **scored** |

Claim A carries no number. Claim B carries a number and the budget table as the reason it is not a
larger angle. Neither borrows the other's standing.

## What the budget table contributes in its own right

`scripts/angle_budget.py --datasets all` measures, for every built specimen, what a tilted plane
actually cuts through and whether it clears the metric's own constants. **No published method states
what specimen geometry an oblique evaluation requires.** Whatever the numbers are, that table is a
contribution, and it is free — no fit, no model, no generation.

**If some specimen clears 30–45°**, claim B is made at that angle and the paper is materially
stronger: the oblique demonstration is *scored*, not shown. `allen_merfish_brain` (59 sections) and
the two `merfish_thick_*` volumes (13 µm and 27 µm slabs, the latter cut from a 200 µm block) are
the candidates. `specs/10` §5.4 excludes `allen_merfish_brain` from the campaign at 1.17 M cells —
that is a *fitting* cost and does not apply to reading a file's coordinates.

**If none clears**, the finding is stated as a finding about the field's data, not as a failure of
ours: 3D spatial transcriptomics is published as stacks of thin sections, and a stack of thin
sections does not contain an obliquely-cut section to score against at any useful angle. Tier-1 is
**21.6 : 1** in-plane to depth. GATE 2's synthetic fixture — the only geometry on which oblique
parity has ever been measured — was 7.5 : 1, and `reports/gate2.md`'s parity claim has already been
corrected to say so.

## What is ruled out

- Relabelling a 5° section as "oblique". A 5° tilt on a 21.6 : 1 slab is a coronal section.
- Quoting claim A's figure as if it carried claim B's number.
- Choosing the angle after seeing which one scores best. The gates are pre-registered and derived
  from `celltype_localization`'s own constants; the only number that is mine is the 60% of the
  coronal plane's scorable-type count in G1, and it is labelled as mine in every report.

## The evaluation set, still owed

`reports/oblique_layout_cost.md` §3c stands and is not resolved by any of this: at an oblique plane
the donors and the evaluation set are **the same real cells**, so the demonstration must exclude its
evaluation set from its donors (`cells_near_plane(..., exclude=...)`, which
`tests/test_layout.py` now asserts) **and** score against a baseline restricted to the same slab
under the same exclusion. That baseline is `flanking_copy` re-run under the exclusion, and it must
be pre-registered before the figure is run — not afterwards.
