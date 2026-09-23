"""This project stands alone, and keeps standing alone.

It was split out of a larger one that also contained an application to a
particular physics. The split is only worth having if it holds, so it is a
test rather than an intention: nothing here may import the application, and
every block the CLI offers must be importable and runnable.
"""
from __future__ import annotations

import ast

import pytest

from lrdsr import paths

#: Names from the parent project that must not come back.
FORBIDDEN_MODULES = ("lrdsr.cml", "lrdsr.fluids", "experiments.cml_simulation",
                     "experiments.real_links", "experiments.fluids", "dataset",
                     "paper")


def _py_files():
    return [f for f in paths.ROOT.rglob("*.py")
            if "__pycache__" not in f.parts and ".venv" not in f.parts]


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: p.name)
def test_no_application_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        bad += [m for m in names
                if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)]
    assert not bad, f"{path.name} imports {bad}, which this project does not contain"


def test_every_cli_block_imports():
    import importlib

    from experiments.__main__ import SECTIONS
    for key, (module, _what) in SECTIONS.items():
        mod = importlib.import_module(module)
        assert hasattr(mod, "run"), f"{key} -> {module} has no run()"


def test_theory_blocks_are_listed_once():
    from lrdsr.theory import verification as V
    tags = [b[0] for b in V.BLOCKS]
    assert len(tags) == len(set(tags))
    for _tag, _what, fn in V.BLOCKS:
        assert callable(getattr(V, fn)), f"{fn} is listed but missing"
