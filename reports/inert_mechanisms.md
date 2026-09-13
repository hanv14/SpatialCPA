# The v18/v20/v21 rows ran with most of their mechanisms unreachable

**2026-09-13.** A code audit of `learn_spatialcpav18/20/21.py` against the way
`benchmark-pbya-v3` actually invokes them. Nothing here is a new measurement; it is a
statement about which branches the *existing* scored predictions executed. It was found
while scoping the `v21_<learner>` expression ablation, and it changes what that ablation
can be labelled.

Two independent causes, neither of them a deliberate setting.

---

## 1. `alpha` is exactly zero on every `paper_*` design — by construction, not by margin

`learn_spatialcpav21.py:1573-1580` (identically `learn_spatialcpav20.py:1364-1372`):

```python
zc_all  = self.stack.z_centers()                      # TRAINING sections only
med_gap = float(np.median(np.diff(np.sort(zc_all))))
this_gap = float(upper.z_center - lower.z_center)     # the flanking pair
alpha = float(np.clip((this_gap / med_gap - 1.0) /
                      max(cfg.gap_scale - 1.0, 1e-6), 0.0, 1.0))
```

`config.held_out_indices` (`benchmark-pbya-v3/src/bench3/config.py:887-903`) holds out
**every other** section — `tuple(range(2, n_sections, 2))`, and STARmap pins the same
`(2, 4, 6)`. The surviving training sections are therefore the odd ones, evenly spaced, so
`med_gap` *is* the spacing between consecutive training sections. `pick_flanking_slices`
(`learn_spatialcpav21.py:449`) returns the two training sections that bracket the target,
which under an alternating hold-out are always **consecutive** training sections. So

> `this_gap == med_gap` identically ⟹ the numerator is **0** ⟹ `alpha == 0`.

Worked on STARmap: training z-centres ≈ 19 / 41 / 63 / 85 planes, all gaps 22; held
section 2's flanks are sections 1 and 3, gap 22; `22/22 − 1 = 0`.

This is stronger than the explanation currently in the paper ("`paper_2_4_6` never reaches
that gap"), which reads as a threshold that happened not to be crossed. It is not a
threshold that was missed — it is a numerator that is structurally zero. No dataset, no
tissue, no `--gap-scale` value and no section thickness can make it nonzero while the
hold-out alternates. Only `--design wide` (a consecutive interior block, so the flanking
gap is `block+1` times the median) reaches `alpha > 0`.

Datasets partitioned by `z_width` or by real `sections` have slightly irregular section
centroids, so `this_gap/med_gap` can land a per-cent or two off 1.0. `alpha_tol = 1e-3`
(`:1580`) snaps the smallest of those to 0; anything surviving gives `w_other ≈ 0.25 %`
of genes. Immaterial either way.

**Gated off by `alpha == 0`:** v20's Bernoulli cross-mix (`:1738`), v20's `edit_gap_extra`
(`:1770`), v21's field-aligned re-grounding (`:1697`, via `field_align_gap_only=True`).

## 2. `edit_weight` defaults to 0.25, and four mechanisms are gated on it being 0.0

`METHODS["spatialcpav21_gen"]` carries **no `wrapper_args`**
(`config.py:1187-1215`), and `run_benchmark.run_single` passes only
`--input/--target-section/--target-z/--output/--seed` (`run_benchmark.py:236-249`). So the
wrapper's argparse defaults stand, and `--edit-weight` defaults to **0.25**
(`run_spatialcpav21.py:335`), mirroring `V14Config.edit_weight = 0.25`
(`learn_spatialcpav21.py:1038`). The same is true of the v18, v19, v20, v22 and v23
wrappers — all six declare `default=0.25`.

That makes the decode blend at `:1703` the **only** live expression mechanism, and it
switches four others off:

| line | mechanism | guard |
|---|---|---|
| `:1715` | v18 gene-mix (and v21's coherent/multi-partner version of it) | `gene_mix_frac > 0 and edit_weight == 0` |
| `:1770` | v20 `edit_gap_extra` | `edit_gap_extra > 0 and edit_weight == 0 and alpha > 0` |
| `:1778` | v21 per-gene field repair | `field_repair and edit_weight == 0` |
| `:1792` | v18 raw-output path | `raw_output and edit_weight == 0 and …` |

The gene-mix and raw-output lines are v18's own two headline mechanisms — the fixes for
the Sinkhorn losses and for the negative `gene_var_spearman` on EASI-FISH. **Neither has
ever run in a scored row.** Because `raw_ok` is False, the output tail is
`expm1(clip(expr, 0, 20))` (`:1800`) on every dataset, including the three EASI-FISH
volumes the raw path was written for.

## 3. What the scored v21 rows actually computed

Composing §1 and §2, `_generate` reduces on every `paper_*` row to:

```
expr = expm1(clip( 0.75 · X_log[donor] + 0.25 · PCA_decode(ê) , 0, 20))
```

with the donor index from layout inheritance (`_resample_layout`), minority flow
re-grounding (`_ground`), the type vote (`_vote_types`) and composition matching
(`_match_composition`). Note also that `margin_eff` ramps to `ground_margin_narrow = 5.0`
at `alpha = 0` (`:1666-1669`), a latent-distance threshold `_ground` rarely clears, so even
the minority re-grounding seldom fires.

Every mechanism v21 is *named* for — coherent mix fields, multi-partner per-gene draws,
field-aligned re-grounding, tail-rate-matched field repair, the mix-order fix — is
unreachable in the published numbers.

## 4. This replaces the "wide-gap only" reading of v18 ≡ v20

The byte-identity was established empirically on 2026-09-11 (commit `556d1cf`): on
`paper_2_4_6`, v18's and v20's `prediction.h5` match array by array across 12 403 cells and
344 361 non-zeros, with only `/uns` differing. The mechanism is now pinned, by a
symbol-level diff of the two module files:

* **79 shared top-level symbols; 75 byte-identical** after stripping comments. Only
  `V14Config`, `_generate`, `_phase_b` differ, plus `_pair_across_flanks` which exists only
  in v20.
* **`V14Config`: no shared default changed.** v20 only *adds* `gap_adapt`, `gap_scale`,
  `pair_interp_max`, `cross_mix`, `alpha_tol`, `edit_gap_extra`, `curriculum_flow`,
  `curriculum_p`.
* **`_phase_b`** differs only by the `curriculum_flow` branch. With the default
  `curriculum_flow=False`, `entry["far"]` is `None`, and the guard
  `if pl.get("far") is not None and rng.random() < cfg.curriculum_p` **short-circuits before
  the draw**. So training consumes an identical RNG stream and produces identical weights.
* **`_generate`** differs only by the alpha block (pure arithmetic, no RNG, no write to
  `expr`), the cross-mix block, the `edit_gap_extra` block, and one extra `and
  cfg.edit_gap_extra == 0.0` conjunct inside `raw_ok`. At `alpha == 0` the two stochastic
  blocks are skipped **without consuming any RNG**; the `raw_ok` conjunct is unreachable
  because `raw_ok` already requires `edit_weight == 0`. The gene-mix block is textually
  identical in both and skipped in both.

So the identity is not "the scores happened to coincide", nor "the gap was too narrow" — it
is that **v20 contains no reachable statement that v18 does not**, under this invocation.
Bitwise identity is the only possible outcome, and it would remain so at any seed.

The existing wording — *"v20's changes over v18 are expression-path only, and they fire only
when the section gap exceeds the volume's median spacing; `paper_2_4_6` never reaches that
gap"* — is right in direction and incomplete in two ways, both now corrected in
`paper/08_related_work.md` §8.2 and `progress/t10_benchmark.md`:

1. the gap is not *not reached*, it is identically equal, so `alpha` is **zero by
   construction** and no choice of protocol parameter within an alternating hold-out
   changes it;
2. "expression-path only" is true of the *difference* but understates the *state*: the
   `edit_weight` default independently disables v18's own gene-mix and raw-output paths in
   **both** versions, so neither version ran its own expression mechanisms either.

## 5. What separates v21 from v20 on `paper_*`, and why that row does differ

`specs/10` §13.2 lists different numbers for v20 and v21 on `paper_2_4_6`. That is
consistent with the above and locates the cause exactly. A v20↔v21 symbol diff shows 80
shared symbols, 70 identical; of the changed ones, only two are reachable at `alpha = 0`:

| knob | v20 | v21 at `alpha = 0` | effect |
|---|---|---|---|
| coherent-patch frequency (`:1538-1541`) | `coherent_freq = 4.0` | `layout_freq_narrow = 8.0` | finer layout granules — **changes the positions** |
| re-grounding swap margin (`:1666-1669`) | `ground_keep_margin = 1.0` | `ground_margin_narrow = 5.0` | far fewer exemplar swaps |

Both are v21's deliberate *narrow-gap* adaptations, which is the one part of v21 the
protocol does exercise. Everything else v21 added is in §1's or §2's dead set. A useful
consequence: v21 vs v20 on tier-1 is a **layout-and-donor-selection** comparison, not an
expression comparison, and should not be cited as evidence about expression synthesis.

## 6. Consequences

1. **Any claim of the form "v21's coherent mix / field repair / field alignment improves
   metric X" has no support from the `paper_*` tree**, because those branches never
   executed there. `specs/10` §13's directional readings are unaffected on v20 vs SpatialZ
   but must not be read as evidence about v21's named mechanisms.
2. **The EASI-FISH `gene_var_spearman` fix has never been exercised.** v18's raw-output path
   was written for exactly those three volumes and is off in every scored row on them.
3. **A one-line change would re-enable five mechanisms at once** — adding
   `"wrapper_args": ["--edit-weight", "0"]` to the v18/v20/v21 `METHODS` entries. That is a
   *different experiment*, not a fix: it would invalidate comparability with every existing
   row. If it is wanted, it belongs as new `METHODS` keys beside the old ones, scored into a
   separate tree, never as an edit to the pinned entries.
4. **The `v21_<learner>` ablation is labelled accordingly.** It contrasts donor copying with
   learned regression at the expression interface; it cannot be about cross-mix, which does
   not run on these rows. See `reports/` for that campaign when it lands.

## 7. How to verify, without re-running anything

```bash
# (a) the mechanism, from source — the symbol-level diff this report rests on
#     (75/79 shared symbols identical; only V14Config/_generate/_phase_b differ)
python - <<'PY'
import re
def blocks(p):
    lines=open(p).read().split("\n"); out={}; cur=None; st=[]
    def close(ind,end):
        while st and st[-1][0]>=ind:
            i,n,s=st.pop(); out[n]="\n".join(lines[s:end])
    for i,ln in enumerate(lines):
        m=re.match(r'^(\s*)(?:async\s+)?(def|class)\s+(\w+)',ln)
        if not m: continue
        ind,kind,name=len(m.group(1)),m.group(2),m.group(3); close(ind,i)
        if ind==0: cur=name if kind=="class" else None; q=name
        else: q=f"{cur}.{name}" if cur else name
        st.append((ind,q,i))
    close(0,len(lines)); return out
def norm(t): return "\n".join(s for s in (l.split("#")[0].rstrip() for l in t.split("\n")) if s.strip())
a,b=blocks("learn_spatialcpav18.py"),blocks("learn_spatialcpav20.py")
sh=sorted(set(a)&set(b))
print("shared",len(sh),"identical",sum(norm(a[k])==norm(b[k]) for k in sh))
print("differ:",[k for k in sh if norm(a[k])!=norm(b[k])])
PY

# (b) alpha, from the built dataset — prints this_gap / med_gap per held-out section
python - <<'PY'
import anndata as ad, numpy as np
a = ad.read_h5ad("benchmark-pbya-v3/data/processed/starmap_visual_cortex/data.h5ad")
sec = a.obs["section"].values.astype(str); xyz = np.asarray(a.obsm["spatial"])
z = {s: float(np.median(xyz[sec==s,2])) for s in np.unique(sec)}
held = ["section_2","section_4","section_6"]
train = sorted([s for s in z if s not in held], key=z.get)
tz = np.array([z[s] for s in train]); med = float(np.median(np.diff(tz)))
for h in held:
    lo = max([s for s in train if z[s] <= z[h]], key=z.get)
    hi = min([s for s in train if z[s] >  z[h]], key=z.get)
    print(h, "gap/med =", round((z[hi]-z[lo])/med, 6))
PY

# (c) what a scored row recorded — method_params is written per prediction
python -c "import h5py,json,sys; f=h5py.File(sys.argv[1]); print(json.loads(f['uns/method_params'][()]))" \
    "$BENCH_V3_RESULTS/spatialcpav21_gen/starmap_visual_cortex/paper_2_4_6/prediction.h5"
```

(c) is the decisive artifact check: `edit_weight` is recorded in `method_params`
(`run_spatialcpav21.py:488`), so every existing prediction states on its face which
configuration produced it. Expect `0.25`.

## 8. Line-number pin

Cited against the working tree at the commit that adds this file. `learn_spatialcpav21.py`
is 3 013 lines; `benchmark-pbya-v3/src/bench3/evaluate_paper.py` is 764 lines and hashes to
`7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992`, matching `specs/10` §0's
pin. Neither file is modified by this report or by the ablation it precedes.
