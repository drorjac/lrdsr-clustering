"""No results column may shadow a pandas attribute without being declared.

Twice in one session a column name that is also a ``DataFrame`` method made
a filter fail **silently**: ``meta.pipe`` returned the ``pipe`` method, and
``S.transform == "loglog"`` compared a bound method to a string, so every
downstream query came back empty instead of raising. Neither was caught by a
test, because both frames were otherwise correct.

This is the guard. A column whose name is a pandas attribute is allowed only
if it is listed in ``ALLOWED`` -- which is a promise that every access to it
uses ``df["name"]`` and never ``df.name``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from lrdsr import paths

#: Shadowing names that are present and known to be accessed by brackets.
#:
#: ``where`` -- ``results/estimator/shared_component*.csv``, the column
#: saying whether the shared term is inside or outside the library. Every
#: read is ``g["where"]`` or a ``groupby`` key.
ALLOWED = {"where"}

_RESERVED = {a for a in (set(dir(pd.DataFrame)) | set(dir(pd.Series)))
             if not a.startswith("_")}


def _csvs():
    return sorted(paths.RESULTS.rglob("*.csv"))


def test_results_exist():
    assert _csvs(), "no results CSVs found; run the experiments first"


@pytest.mark.parametrize("path", _csvs(), ids=lambda p: p.name)
def test_no_undeclared_pandas_shadowing_column(path):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    clash = sorted(set(cols) & _RESERVED - ALLOWED)
    assert not clash, (
        f"{path.relative_to(paths.ROOT)} has column(s) {clash} that shadow a "
        f"pandas attribute. `df.{clash[0]}` returns the method, so a filter "
        f"on it fails silently. Rename the column, or add it to ALLOWED and "
        f"use df[\"{clash[0]}\"] everywhere."
    )
