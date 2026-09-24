"""The configuration the published v18 rows ran under, stated three ways, agree.

1. ``V18_ARGS`` in bench3/config.py — what run_benchmark actually appends.
2. The argparse defaults in run_spatialcpav18.py — what every flag NOT in
   V18_ARGS takes.
3. The "Effective configuration" table in README.md — what the reviewer reads.

Read with ``ast`` so the test needs neither torch nor anndata.
"""
from __future__ import annotations

import ast
import re

from conftest import REVIEW_ROOT, V18_WRAPPER, load_bench3_config

PUBLISHED_FLAGS = {
    "--edit-weight": "0.0", "--ground-blend-flow": "1.0", "--ground-k": "8",
    "--ground-temp": "0.25", "--ground-keep-margin": "1.0", "--type-mode": "vote",
    "--type-vote-k": "12", "--gene-mix-frac": "0.15",
}


def argparse_defaults(path):
    """{flag: default-as-string} for every add_argument in ``path``."""
    out = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            flag = node.args[0].value
            kw = {k.arg: k.value for k in node.keywords}
            if kw.get("action") is not None and ast.literal_eval(kw["action"]) == "store_true":
                out[flag] = "False"
            elif "default" in kw:
                out[flag] = str(ast.literal_eval(kw["default"]))
            else:
                out[flag] = "None"
    return out


def pairs(args):
    it = iter(args)
    return dict(zip(it, it))


def readme_table():
    text = (REVIEW_ROOT / "README.md").read_text()
    block = text.split("<!-- effective-config:begin -->")[1].split("<!-- effective-config:end -->")[0]
    rows = {}
    for line in block.splitlines():
        m = re.match(r"\|\s*`(--[a-z0-9-]+)`\s*\|\s*`([^`]*)`\s*\|\s*`([^`]*)`\s*\|", line)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(3))
    return rows


def effective():
    eff = argparse_defaults(V18_WRAPPER)
    eff.update(pairs(load_bench3_config().V18_ARGS))
    return eff


def test_v18_args_are_the_published_flags():
    assert pairs(load_bench3_config().V18_ARGS) == PUBLISHED_FLAGS


def test_v18_runs_under_v18_args():
    c = load_bench3_config()
    assert list(c.METHODS["spatialcpav18_gen"]["wrapper_args"]) == list(c.V18_ARGS)


def test_comparators_get_no_v18_flags():
    c = load_bench3_config()
    for name in ("spatialz", "feast", "isost"):
        assert not c.METHODS[name].get("wrapper_args"), name


def test_readme_table_is_the_effective_configuration():
    table = readme_table()
    defaults = argparse_defaults(V18_WRAPPER)
    eff = effective()
    method_flags = {f for f in defaults if f not in
                    ("--input", "--target-section", "--target-z", "--output", "--seed")}
    missing = method_flags - set(table)
    assert not missing, f"README effective-config table lacks {sorted(missing)}"
    for flag, (value, default) in table.items():
        if flag == "--seed":
            assert value == "42"
            continue
        assert default == defaults[flag], (flag, default, defaults[flag])
        assert value == eff[flag], (flag, value, eff[flag])
