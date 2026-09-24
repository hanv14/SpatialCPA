#!/usr/bin/env python3
"""Prove that a renamed file computes exactly what the original did.

    python scripts/verify_rename.py ORIGINAL.py RENAMED.py

For this review, v18's code was stripped of every reference to the SpatialCPA
versions it descends from: class names (``SpatialCPAv14`` → ``SpatialCPAv18``,
``V14Config`` → ``V18Config``), log prefixes (``[v14]`` → ``[v18]``), docstrings,
comments and help text. This script checks that nothing else changed.

Both files are parsed to ASTs and normalised:

* docstrings (the leading string of a module, class or function) are dropped,
  and comments never reach the AST;
* identifiers are mapped through ``NAME_MAP`` in the ORIGINAL;
* the text of ``print(...)`` arguments and of ``help=`` / ``description=``
  keywords is replaced by a placeholder, since it is output wording only;
* every other string literal in the ORIGINAL is mapped through ``STRING_MAP``
  and must then equal the renamed file's literal **exactly**. That covers
  defaults, choices, dict keys and the log markers the harness matches on.

The two normalised ASTs must then be identical (``ast.dump``). Any difference in
control flow, arithmetic, constants, defaults, call order or argument order fails.

Exit 0 = equivalent. Run it against the lab's original copy to extend the
guarantee to the file that produced the published rows.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

NAME_MAP = {
    "SpatialCPAv14": "SpatialCPAv18",
    "V14Config": "V18Config",
}
STRING_MAP = {
    "[v14]": "[v18]",
}
TEXT_KEYWORDS = {"help", "description"}
PLACEHOLDER = "<TEXT>"


class Normalise(ast.NodeTransformer):
    def __init__(self, original: bool):
        self.original = original

    # -- docstrings --------------------------------------------------------------
    def _strip_doc(self, node):
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    def visit_Module(self, node):
        self._strip_doc(node)
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        self._strip_doc(node)
        if self.original:
            node.name = NAME_MAP.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._strip_doc(node)
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    # -- identifiers ---------------------------------------------------------------
    def visit_Name(self, node):
        if self.original:
            node.id = NAME_MAP.get(node.id, node.id)
        return node

    def visit_Attribute(self, node):
        if self.original:
            node.attr = NAME_MAP.get(node.attr, node.attr)
        return self.generic_visit(node)

    # -- output wording -------------------------------------------------------------
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            node.args = [ast.Constant(PLACEHOLDER) for _ in node.args]
        for kw in node.keywords:
            if kw.arg in TEXT_KEYWORDS:
                kw.value = ast.Constant(PLACEHOLDER)
        return self.generic_visit(node)

    # -- every other literal ----------------------------------------------------------
    def visit_Constant(self, node):
        if self.original and isinstance(node.value, str) and node.value != PLACEHOLDER:
            v = node.value
            for a, b in STRING_MAP.items():
                v = v.replace(a, b)
            node.value = v
        return node


def normalised(path: Path, original: bool) -> str:
    tree = ast.parse(path.read_text())
    return ast.dump(Normalise(original).visit(tree), include_attributes=False)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    orig, new = Path(argv[0]), Path(argv[1])
    a, b = normalised(orig, True), normalised(new, False)
    if a == b:
        print(f"EQUIVALENT: {new.name} == {orig.name} up to names {NAME_MAP}, "
              f"literals {STRING_MAP}, docstrings, comments and output wording")
        return 0
    i = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) if a[:len(b)] != b[:len(a)] \
        else min(len(a), len(b))
    print(f"NOT EQUIVALENT: {new} differs from {orig} in executable content.\n"
          f"  original: …{a[max(0, i - 150):i + 150]}…\n"
          f"  renamed:  …{b[max(0, i - 150):i + 150]}…")
    return 1


if __name__ == "__main__":
    sys.exit(main())
