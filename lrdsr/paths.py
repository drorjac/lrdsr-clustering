"""Where things are on disk, decided once.

Twenty-four modules used to reach the repository root by counting ``..`` from
their own location -- ``Path(__file__).resolve().parents[3]`` and its cousins --
which is correct until a file moves one directory and silently starts resolving
somewhere else. This says it once.

``dataset/config.py`` deliberately keeps its own copy: ``dataset/`` imports
nothing from ``lrdsr`` so it can be lifted out whole. The duplicate is checked
against this one in ``tests/test_paths.py`` rather than trusted.

**This assumes an in-place checkout** -- the root is found by walking up from
this file. That holds for the editable install this project uses; a wheel
installed into ``site-packages`` has no ``results/`` to point at, and the
experiment drivers that need these paths are not packaged either.
"""
from __future__ import annotations

from pathlib import Path

#: The repository root: this file is ``<root>/lrdsr/paths.py``.
ROOT = Path(__file__).resolve().parents[1]

#: Committed artefacts. Every experiment writes under ``RESULTS/<name>/``.
RESULTS = ROOT / "results"

#: Intermediates: everything a run writes that is **not** a finding. Resume
#: checkpoints, per-window dumps, the preprocessed archive. All of it is
#: rebuildable from the archives and the code, none of it is committed, and
#: keeping it out of ``results/`` is what lets ``results/`` mean "a number
#: someone reads".
CACHE_ROOT = ROOT / ".cache"

#: The preprocessed archive: shared by every run and rebuildable at any time.
#: It is *not* a result -- rebuilding it per run would cost hours for
#: identical bytes. (Was ``results/cml_cache/`` until the artefact tiering.)
CACHE = CACHE_ROOT / "windows"

#: The manuscript source, and the figures rendered for print.
PAPER = ROOT / "paper"


def results_dir(name: str, figs: bool = True) -> Path:
    """``results/<name>/``, created, with its ``figs/`` subdirectory.

    The one artefact rule in one function: a figure never lives apart from the
    CSV it was drawn from. Returns the directory; ``<dir>/figs`` is created
    alongside unless ``figs=False``.
    """
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    if figs:
        (out / "figs").mkdir(parents=True, exist_ok=True)
    return out


def cache_dir(name: str) -> Path:
    """``.cache/<name>/``, created. The intermediate tier.

    Use this for anything a run writes that nobody reads as a result: resume
    checkpoints, per-window and per-fold dumps, anything a later stage reloads
    to avoid recomputing. It is gitignored, so it can be as large as it needs
    to be -- three of these dumps were 49 MB of the repository's 74 MB of
    committed artefacts before they were moved down here.

    The test is simple: if a reader would look at the file to learn something,
    it belongs in ``results_dir(name)``. If only the code reads it, it belongs
    here.
    """
    out = CACHE_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def require_cache(path: Path, how: str) -> Path:
    """Return ``path``, or explain how to rebuild it.

    Intermediates are not committed, so a fresh checkout reaches a stage whose
    input has never been built. A FileNotFoundError deep inside pandas does not
    say which command to run; this does.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{Path(path).relative_to(ROOT) if Path(path).is_absolute() else path} "
            f"is an intermediate and is not committed.\n"
            f"Rebuild it with:  {how}"
        )
    return Path(path)
