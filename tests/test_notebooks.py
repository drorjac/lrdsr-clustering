"""The notebooks are generated, executed, and committed with their outputs.

Executing them here would take minutes, so this checks what can go stale
without running them: every source cell parses, every committed notebook
comes from a source (nobody hand-edited an orphan), and no committed output
is an error -- a notebook that failed when it was built is not a result.
"""
from __future__ import annotations

import ast

import nbformat
import pytest

from lrdsr import paths
from notebooks.sources import NOTEBOOKS

NB_DIR = paths.ROOT / "notebooks"


def _python(src: str) -> str:
    """Drop IPython magics, which are not Python."""
    return "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith(("%", "!")))


@pytest.mark.parametrize("stem", sorted(NOTEBOOKS))
def test_every_code_cell_parses(stem):
    for i, (kind, src) in enumerate(NOTEBOOKS[stem]):
        if kind == "code":
            try:
                ast.parse(_python(src))
            except SyntaxError as exc:            # pragma: no cover - message only
                raise AssertionError(f"{stem} cell {i}: {exc}") from exc
        else:
            assert kind == "md", f"{stem} cell {i}: unknown kind {kind!r}"


def test_no_orphan_notebooks():
    committed = {p.stem for p in NB_DIR.glob("*.ipynb")}
    assert committed <= set(NOTEBOOKS), (
        f"{sorted(committed - set(NOTEBOOKS))} have no source in notebooks/sources.py")


@pytest.mark.parametrize("path", sorted(NB_DIR.glob("*.ipynb")), ids=lambda p: p.stem)
def test_committed_notebook_has_no_error_output(path):
    nb = nbformat.read(path, as_version=4)
    errors = [o for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{path.name}: {errors[0].get('ename')}: {errors[0].get('evalue')}"
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    executed = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count"))
    assert executed == n_code, f"{path.name}: {n_code - executed} code cells never ran"
