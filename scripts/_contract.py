"""Cross-script contract checks that need no torch, no data and no fit.

``specs/10`` §4.2p: a check that verifies the criterion must also verify the wiring, and the wiring
check must not need the data. These read the **real source** with ``ast`` and run wherever the code
is edited rather than only where the data lives.

The check here is a different shape from a signature check, and was added because a signature check
could not have caught the defect it exists for: every call was well formed, and the fault was a
**default value the dataset cannot satisfy** — ``Config().region_key`` naming an ``obs`` column that
bench3's builds do not have, so ``load_volume`` raised a ``SchemaError`` before
``angle_budget.py`` read one coordinate. No source-level check can know what columns a file has
(``specs/10`` §4.2q).
What it *can* know is whether a new runner prepares its ``Config`` the way the runners that already
work do, and that is what this asserts.
"""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# The keys a bench3 volume needs named correctly. `region_key` is the one whose *default* is wrong
# for these builds, so it is the one a direct construction has to speak to.
BENCH3_SENSITIVE = "region_key"


def unsafe_config_calls(source: str) -> list[str]:
    """Every ``Config(...)`` in this source that cannot be trusted against a bench3 volume.

    A direct construction is trusted when it either splats a dict (``Config(**BENCH3_KEYS)``,
    ``Config(**checkpoint["config"])`` — both carry the keys from something that was fitted under
    them) or names ``region_key`` explicitly. A construction that does neither inherits
    ``Config``'s default, and that default is wrong for this dataset.

    Constructions reached through a helper (``base_config``, ``arm_config``) never appear here,
    because those helpers splat ``BENCH3_KEYS`` at the one place they build the object.
    """
    tree = ast.parse(source)
    repaired = _repaired_by_replace(tree)
    out: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "Config":
            continue
        splatted = any(kw.arg is None for kw in node.keywords)
        named = any(kw.arg == BENCH3_SENSITIVE for kw in node.keywords)
        if splatted or named or node in repaired:
            continue
        shown = ", ".join(kw.arg or "**" for kw in node.keywords)
        out.append(f"line {node.lineno}: Config({shown})")
    return out


def _repaired_by_replace(tree: ast.Module) -> set[ast.Call]:
    """``Config(...).replace(region_key=None)`` names the key too, one call later.

    Two runners that have been fitting bench3 volumes since R11 spell it that way, inlining what
    ``_starmap_run.BENCH3_KEYS`` holds. That is duplication, not the defect this check is for, and a
    checker that called them offenders would be reporting its own narrowness as a finding.
    """
    trusted: set[ast.Call] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "replace":
            continue
        if not any(kw.arg in (BENCH3_SENSITIVE, None) for kw in node.keywords):
            continue
        base = node.func.value
        while (  # walk back through any further chained .replace(...) to the construction
            isinstance(base, ast.Call)
            and isinstance(base.func, ast.Attribute)
            and base.func.attr == "replace"
        ):
            base = base.func.value
        if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
            if base.func.id == "Config":
                trusted.add(base)
    return trusted


def bench3_config_discipline() -> list[tuple[str, bool]]:
    """Every script that loads a bench3 volume must prepare its ``Config`` a sanctioned way.

    ``load_training_volume`` builds a ``Volume``, which validates every ``obs``/``obsm`` key the
    ``Config`` names. A bare ``Config()`` carries ``region_key="region"``, bench3's builds have
    no such column, and the load dies in ``loaders.py`` — deep enough that the runner's own code
    looks innocent.
    """
    offenders: list[str] = []
    scanned = 0
    for path in sorted(SCRIPTS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        text = path.read_text()
        if "load_training_volume" not in text:
            continue
        scanned += 1
        bad = unsafe_config_calls(text)
        if bad:
            offenders.append(f"{path.name} ({'; '.join(bad)})")
    return [
        (
            f"every script loading a bench3 volume prepares its Config a sanctioned way "
            f"({scanned} scanned)"
            + (f" — OFFENDERS: {', '.join(offenders)}" if offenders else ""),
            not offenders,
        ),
        ("and there were scripts to scan, so the check is not vacuous", scanned > 0),
        # Positive control: the check has to REJECT the construction that actually failed, or a
        # green result means nothing. This is the reintroduction test, written down instead of
        # done by hand once.
        (
            "and it still rejects the bare `Config()` that took angle_budget.py down",
            bool(unsafe_config_calls("from x import Config\ncfg = Config()\n")),
        ),
        (
            "and it still accepts a checkpoint-restored config, which is how the fitted runners "
            "get theirs",
            not unsafe_config_calls('cfg = Config(**ckpt["config"])'),
        ),
        (
            "and a chained `.replace(region_key=...)` counts, because it names the key too",
            not unsafe_config_calls("cfg = Config(seed=1).replace(region_key=None)"),
        ),
        (
            "but a chained replace that does NOT name it is still an offender",
            bool(unsafe_config_calls('cfg = Config(seed=1).replace(layout_mode="field")')),
        ),
    ]


def uses_shared_base_config(script: str) -> list[tuple[str, bool]]:
    """This particular script builds its Config from the shared builder. Scoped to one file.

    Stronger than :func:`bench3_config_discipline`: a runner with no checkpoint to restore from has
    only ``base_config`` available, and reaching for ``Config(...)`` directly is the defect that
    took ``angle_budget.py`` down on its first real run.
    """
    text = (SCRIPTS / script).read_text()
    calls = {
        n.func.id
        for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    return [
        (f"{script} builds its Config with the shared `base_config`", "base_config" in calls),
        (f"{script} constructs no Config of its own", not unsafe_config_calls(text)),
    ]
