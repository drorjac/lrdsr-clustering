"""The soft (EM) estimator: it must be a real EM, and it must find the laws."""
from __future__ import annotations

import numpy as np
import pytest

from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.soft import SoftLRDSR, library_terms, predict_soft


def _pair(rho=1.0, m=120, n=48, seed=0, df=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, (m, n))
    z = rng.integers(0, 2, m)
    sigma = np.sqrt(np.mean((2 * x) ** 2) / rho)
    clean = np.where(z[:, None] == 0, x ** 2 + x, x ** 2 - x)
    e = rng.normal(0, sigma, x.shape) if df is None else sigma * rng.standard_t(df, x.shape)
    return x[:, :, None], clean + e, z


def test_loglik_is_monotone_at_temperature_one():
    X, y, _ = _pair(rho=0.25)
    h = SoftLRDSR(2, feature_names=["x"], init="random", random_state=3).fit(X, y).history
    assert len(h) > 2
    assert np.all(np.diff(h["loglik"].to_numpy()) > -1e-6 * np.abs(h["loglik"]).max())


def test_recovers_partition_and_laws_at_large_separation():
    X, y, z = _pair(rho=1.0)
    res = SoftLRDSR(2, feature_names=["x"]).fit(X, y)
    assert aligned_accuracy(z, res.labels) > 0.97
    grid = np.linspace(-2, 2, 50)[:, None]
    Phi, _ = library_terms(grid, feature_names=["x"])
    fits = [Phi @ b for b in res.coef]
    truths = [grid[:, 0] ** 2 + grid[:, 0], grid[:, 0] ** 2 - grid[:, 0]]
    err = min(np.mean((fits[0] - truths[0]) ** 2) + np.mean((fits[1] - truths[1]) ** 2),
              np.mean((fits[0] - truths[1]) ** 2) + np.mean((fits[1] - truths[0]) ** 2))
    assert err < 0.1


def test_responsibilities_are_distributions_and_entropy_bounded():
    X, y, _ = _pair(rho=0.1)
    res = SoftLRDSR(2, feature_names=["x"]).fit(X, y)
    assert np.allclose(res.responsibilities.sum(axis=1), 1.0)
    assert np.all(res.entropy >= -1e-12) and np.all(res.entropy <= np.log(2) + 1e-9)
    assert np.allclose(predict_soft(res, X, y, feature_names=["x"]), res.responsibilities)


def test_student_t_learns_heavy_tails():
    X, y, z = _pair(rho=1.0, m=160, df=2.5, seed=4)
    res = SoftLRDSR(2, noise="student_t", feature_names=["x"]).fit(X, y)
    assert res.nu is not None and np.all(res.nu < 10)
    g = SoftLRDSR(2, noise="gaussian", feature_names=["x"]).fit(X, y)
    assert aligned_accuracy(z, res.labels) >= aligned_accuracy(z, g.labels) - 0.02


def test_annealing_runs_and_ends_at_temperature_one():
    X, y, _ = _pair(rho=0.25)
    h = SoftLRDSR(2, feature_names=["x"], temperature=4.0, anneal_iters=10).fit(X, y).history
    assert h["temperature"].iloc[0] == pytest.approx(4.0)
    assert h["temperature"].iloc[-1] == pytest.approx(1.0)


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        SoftLRDSR(2, noise="cauchy")
    with pytest.raises(ValueError):
        SoftLRDSR(2, temperature=0.5)
