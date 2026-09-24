"""The problem zoo: its laws, its calibration, and its flags are what they say."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.problems.zoo import (
    PROBLEMS,
    gap_outside_library,
    in_library_residual,
    make_windows,
    oracle_labels,
    pairwise_gap_ms,
    sigma_for_rho,
)


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.name)
def test_laws_are_finite_and_distinct(problem):
    X = problem.sample(np.random.default_rng(0), (500,))
    assert X.shape == (500, problem.d)
    for f in problem.laws:
        assert np.all(np.isfinite(f(X)))
    G = pairwise_gap_ms(problem, n=20_000)
    assert np.all(G[~np.eye(problem.K, dtype=bool)] > 0)
    assert len(problem.truths) == problem.K


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.name)
@pytest.mark.parametrize("rho", [0.1, 1.0])
def test_sigma_hits_target_rho_for_the_closest_pair(problem, rho):
    sigma = sigma_for_rho(problem, rho)
    G = pairwise_gap_ms(problem, seed=7, n=200_000) / sigma ** 2
    got = G[~np.eye(problem.K, dtype=bool)].min()
    assert got == pytest.approx(rho, rel=0.05)


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.name)
def test_in_library_flag_is_true_exactly_when_the_laws_are_spanned(problem):
    exact = in_library_residual(problem) < 1e-12
    assert exact == problem.in_library
    if problem.in_library:
        assert gap_outside_library(problem) < 1e-12


def test_names_are_unique():
    names = [p.name for p in PROBLEMS]
    assert len(names) == len(set(names))


def test_oracle_is_nearly_perfect_at_high_rho():
    p = PROBLEMS[0]
    X, y, z = make_windows(p, sigma_for_rho(p, 4.0), 60, 32, seed=11)
    assert np.mean(oracle_labels(p, X, y) == z) > 0.97


def test_one_cell_end_to_end():
    from experiments.problems.run import run_cell, summarise
    p = next(q for q in PROBLEMS if q.name == "rescaled_copy")
    rows, laws = run_cell(p, 11, rho_grid=(1.0,), n_windows=60, window_len=24)
    import pandas as pd
    raw, lw = pd.DataFrame(rows), pd.DataFrame(laws)
    assert set(raw["method"]) >= {"oracle", "lrdsr", "soft_em", "geometry"}
    assert len(lw) == p.K and lw.law_nmse.notna().all()
    s = summarise(raw, lw)
    assert s.loc[0, "lrdsr"] <= 0.2
