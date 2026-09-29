"""The correlated-noise robustness block: its noise, its whitening, its theory."""
import numpy as np

from experiments.problems.zoo import PROBLEM_BY_NAME, sigma_for_rho
from experiments.robustness.run import (
    ar1,
    make_windows,
    oracle,
    predicted_errors,
    whiten,
)


def test_ar1_is_stationary_with_the_requested_correlation():
    u = np.random.default_rng(0).normal(size=(20000, 32))
    e = ar1(u, 0.7)
    assert abs(e.var() - 1.0) < 0.02
    assert abs(np.corrcoef(e[:, 10], e[:, 11])[0, 1] - 0.7) < 0.02


def test_whitening_undoes_ar1():
    u = np.random.default_rng(1).normal(size=(5000, 32))
    w = whiten(ar1(u, 0.8), 0.8)
    assert np.allclose(w[:, 1:], np.sqrt(1 - 0.8 ** 2) * u[:, 1:])


def test_phi_zero_makes_the_two_oracles_one():
    p = PROBLEM_BY_NAME["kink"]
    X, y, _ = make_windows(p, sigma_for_rho(p, 0.25), 0.0, n_windows=200, seed=3)
    assert np.array_equal(oracle(p, X, y, None), oracle(p, X, y, 0.0))


def test_exact_formulas_match_simulation():
    p = PROBLEM_BY_NAME["rescaled_copy"]
    s = sigma_for_rho(p, 0.25)
    X, y, z = make_windows(p, s, 0.8, n_windows=6000, seed=4)
    ols, gls = predicted_errors(p, X, s, 0.8)
    se = np.sqrt(0.25 / 6000)
    assert abs(np.mean(oracle(p, X, y, None) != z) - ols) < 4 * se
    assert abs(np.mean(oracle(p, X, y, 0.8) != z) - gls) < 4 * se
