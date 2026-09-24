"""Build and execute the notebooks from their source, here.

The notebooks are generated rather than hand-edited, for the same reason the
README is: a notebook edited by hand drifts from the code it demonstrates,
and a stale output cell is a claim nobody re-checked. Each notebook is a list
of cells in ``notebooks/sources.py``; this writes the ``.ipynb`` and runs it
top to bottom with the interpreter running this script, so every committed
output is one that code actually produced.

    python notebooks/build.py              # build + execute all
    python notebooks/build.py 01 04        # only these
    python notebooks/build.py --no-exec    # write without running

Each notebook is also exported to ``notebooks/html/<name>.html``: a single
self-contained page (figures and the animation embedded) that opens in any
browser, for when no Jupyter viewer is at hand.

The kernel is registered in a private directory under ``.cache/`` pointing at
``sys.executable``, so the build uses this environment and never touches the
user's Jupyter configuration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

KERNEL = "lrdsr-build"
HTML_DIR = HERE / "html"


def _private_kernel() -> None:
    """A kernelspec for ``sys.executable`` in ``.cache/jupyter`` (on JUPYTER_PATH)."""
    import os
    base = ROOT / ".cache" / "jupyter"
    spec = base / "kernels" / KERNEL
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "lrdsr (build)", "language": "python"}))
    os.environ["JUPYTER_PATH"] = str(base) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")


def make(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, src in cells:
        src = src.strip("\n")
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md"
                        else nbformat.v4.new_code_cell(src))
    return nb


def export_html(nb: nbformat.NotebookNode, stem: str) -> Path:
    """Write ``notebooks/html/<stem>.html``, a standalone copy with outputs."""
    from nbconvert import HTMLExporter

    out = HTML_DIR / f"{stem}.html"
    out.parent.mkdir(exist_ok=True)
    body, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    out.write_text(body, encoding="utf-8")
    return out


def build(only: list[str] | None = None, execute: bool = True, timeout: int = 1200) -> int:
    from notebooks.sources import NOTEBOOKS

    if execute:
        _private_kernel()
    failed = 0
    for stem, cells in NOTEBOOKS.items():
        if only and not any(stem.startswith(o) for o in only):
            continue
        nb = make(cells)
        path = HERE / f"{stem}.ipynb"
        t0 = time.time()
        if execute:
            client = NotebookClient(nb, timeout=timeout, kernel_name=KERNEL,
                                    resources={"metadata": {"path": str(HERE)}})
            try:
                client.execute()
            except Exception as exc:          # write what ran, then report
                failed += 1
                print(f"  [FAIL] {stem}: {type(exc).__name__}: {str(exc)[:400]}")
            nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                         "language": "python"}
        nbformat.write(nb, path)
        html = export_html(nb, stem)
        print(f"  [nb] notebooks/{path.name} + html/{html.name}  ({time.time() - t0:.0f}s)")
    return failed


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raise SystemExit(1 if build(args or None, execute="--no-exec" not in sys.argv) else 0)
