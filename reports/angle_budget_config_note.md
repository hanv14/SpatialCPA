# Why a geometry-only runner validates two fields it never reads

*Written because `angle_budget.py` — which reads coordinates and cell types and nothing else — was
stopped twice before its first coordinate, by `Config.region_key` and then by `Config.expr_pca_dim`.*

## The question

> angle_budget does not use `expr_pca_dim` at all. If a field the script never reads can block it,
> does the clamp belong in the runner, or should `validate_config_against_volume` only check fields
> the run will use?

## The answer: the clamp belongs in the runner, and the validator should stay as it is

Three reasons, then the cost I am conceding and what actually fixes it.

**1. "Fields the run will use" is not a set that exists at load time.** `load_volume` returns a
`Volume` and does not know what will be done with it. The same object is handed to `TrainingData.build`,
to `CTFFlow`, to the scorer, to `run_selection`, and to this geometry script. To validate only the
used fields, the caller would have to *declare* them — a new parameter, defaulted, that every new
runner gets wrong in its own way. That is the same defect class as the two crashes this note exists
for, reintroduced deliberately and with a wider blast radius.

**2. The clamp is a recording mechanism, not a convenience.** `clamp_config_to_input` does not
quietly fix the number; it emits `ConfigClampWarning` and says *"This changes the config's content
hash."* That matters because configs are persisted and re-loaded: `_starmap_run.py`'s own docstring
exists because "a config persisted by a selection run has to carry [these keys], or the next process
loads the volume under Config's defaults." A config that claims `expr_pca_dim=32` against a 28-gene
panel is **wrong as a record**, whether or not the process holding it happens to read the field. If
validation skipped unused fields, a geometry run could persist a config carrying 32 and the next
process to load it on a fitting path would silently get a different config out of the same file.
That is Convention 6's "no silent fallbacks" applied to the config itself.

**3. The check is cheap and the failure is loud.** It costs two integer comparisons and fails in
under a second, at the top of the run, with the field named. The alternative fails later, deeper,
and on a different machine.

## The cost I am conceding

It is a real ergonomic defect that a script asking about *angles* is stopped by a message about
*principal components*. The error said what was wrong and not what to do, and a new runner had no
way to learn the remedy except by reading nine other drivers. **That half I have fixed**: both
clampable errors in `validate_config_against_volume` now name `clamp_config_to_volume` /
`clamp_config_to_input` as the remedy and say why to use it rather than editing the number by hand.
The non-clampable one (`fourier_bands_z`) is left alone — there the value really is a choice.

## What actually caused both crashes

Not the validator. **The canonical form was two steps and only one of them was reachable from one
name.** `base_config` got the obs keys right and left the clamp to be remembered; nine drivers spell
the pair out longhand as `clamp_config_to_input(base_config(seed, **overrides), input_path)`, and the
tenth remembered half of it. Getting `region_key` right moved the crash exactly one line.

`_starmap_run.prepare_config(seed, input_path, **overrides)` is now that expression behind one name.
`base_config` stays public for the callers that genuinely have no input path yet — a config being
*built to persist* rather than to fit under.

## What the contract check now covers, and what it cannot

`specs/10` §4.2q's check verified **construction** and stopped there, which is why it passed a runner
that then died on the step after it. `_contract.bench3_clamp_discipline()` now scans the same 23
scripts for the clamp, by its sanctioned routes (`clamp_config_to_input`, `clamp_config_to_volume`,
`prepare_config`, `arm_config`, or a config splatted from a checkpoint subscript — already clamped by
whatever persisted it).

**It cannot check what the clamp narrows to.** That depends on the file's `n_vars` and `n_obs`, which
no source-level check can know. It checks only whether the runner *asks* — the same boundary §4.2q
already draws, one step further along the ritual.

It found one genuine holdout besides mine: `t10_r11_coupling.py` skipped the clamp and survived only
because its hardcoded `expr_pca_dim=16` happens to sit under tier-1's 28-gene panel. The clamp is a
**no-op** there, so its config, config hash and recorded numbers are unchanged — but "under the panel
width" was an accident of one constant rather than a property anyone had checked.

## Verified by reintroduction, not by argument

| defect put back into `angle_budget.py` | `--self-check` |
|---|---|
| none | **22/22** |
| `Config()` — wrong `region_key` | 19/22, naming the file and line |
| `base_config(seed=0)` — right keys, **no clamp** | 20/22, naming the file |
