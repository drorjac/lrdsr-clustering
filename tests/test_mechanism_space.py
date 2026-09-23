"""Invariants of mechanism space (``lrdsr.core.mechanism_space``).

These are the claims Proposition 2 of the paper makes, at test scale: the
separation identity, the two invariances, attainability of the oracle, and
the label-free ``rho``. Each is an equality the construction either has or
does not -- none of them is a tuned threshold.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from lrdsr.core.evaluation import aligned_accuracy
from lrdsr.core.mechanism_space import (
    TRIM,
    inliers,
    library_basis,
    mechanism_features,
    mechanism_init,
    mechanism_noise,
    rho_from_partition,
    separation_matrix,
    separation_unlabelled,
)

N, W = 64, 400
XLO, XHI = -2.0, 2.0


def _pair(rho, seed=3, shared=None, groups=None, gain=None):
    """Windows from f0 = -x, f1 = +x, plus an optional shared component."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(XLO, XHI, size=(W, N))
    z = rng.integers(0, 2, size=W)
    gap_ms = float(np.mean((2 * rng.uniform(XLO, XHI, 200_000)) ** 2))
    sigma = float(np.sqrt(gap_ms / rho))
    clean = np.where(z[:, None] == 1, x, -x)
    if shared is not None:
        clean = clean + shared(x)
    if gain is not None and groups is not None:
        clean = clean + gain[groups][:, None] * x ** 2
    y = clean + rng.normal(0.0, sigma, size=(W, N))
    return x[:, :, None], y, z, sigma


def test_separation_is_sqrt_n_rho():
    """The identity the whole construction exists for."""
    for rho in (0.25, 1.0):
        X, y, z, _ = _pair(rho)
        S = mechanism_features(X, y)
        d = separation_matrix(S, z, 2, sigma=mechanism_noise(X, y))[0, 1]
        assert d == pytest.approx(np.sqrt(N * rho), rel=0.10)


def test_kmeans_there_reaches_the_population_oracle():
    rho = 0.25
    X, y, z, _ = _pair(rho)
    err = 1.0 - aligned_accuracy(z, mechanism_init(X, y, 2, seed=3))
    assert err == pytest.approx(float(norm.sf(0.5 * np.sqrt(N * rho))), abs=0.02)


@pytest.mark.parametrize("c", [0.0, 1.0, 50.0])
def test_a_shared_component_in_the_library_changes_nothing(c):
    """Prop. 2(i): the oracle's invariance, inherited exactly."""
    base = mechanism_features(*_pair(0.25)[:2])
    X, y, _, _ = _pair(0.25, shared=lambda x: c * x ** 2)
    assert mechanism_features(X, y) == pytest.approx(base, rel=1e-8, abs=1e-8)


def test_a_per_group_term_is_removed_by_groups():
    """Prop. 2(i), the group half: a per-link coefficient leaves no trace."""
    groups = np.repeat(np.arange(8), W // 8)
    gain = np.linspace(1.0, 20.0, 8)
    base = mechanism_features(*_pair(0.25)[:2], groups=groups)
    X, y, _, _ = _pair(0.25, groups=groups, gain=gain)
    assert mechanism_features(X, y, groups=groups) == pytest.approx(
        base, rel=1e-6, abs=1e-6)


def test_rho_is_recoverable_without_labels():
    """Prop. 2(ii): rho from the trace, and from the estimated partition."""
    rho = 0.25
    X, y, _z, _ = _pair(rho)
    S, sig = mechanism_features(X, y), mechanism_noise(X, y)
    assert separation_unlabelled(S, sig, N)["rho"] == pytest.approx(rho, rel=0.35)
    lab = mechanism_init(X, y, 2, seed=3)
    assert rho_from_partition(S, lab, sig, N) == pytest.approx(rho, rel=0.25)


def test_a_free_per_window_nuisance_costs_the_collinear_part():
    """Prop. 2(iii): profiling x out per window removes the whole gap."""
    X, y, z, _ = _pair(1.0)
    lab = mechanism_init(X, y, 2, seed=3,
                         nuisance=lambda Xf: Xf[:, :1])
    assert 1.0 - aligned_accuracy(z, lab) > 0.3      # the gap IS the nuisance


def test_the_trim_protects_the_centres_from_one_bad_window():
    X, y, z, _ = _pair(1.0)
    y = y.copy()
    y[0] += 500.0                                     # one absurd window
    assert 1.0 - aligned_accuracy(z, mechanism_init(X, y, 2, seed=3)) < 0.05
    S = mechanism_features(X, y)
    keep = inliers(S)
    assert keep.mean() == pytest.approx(1.0 - TRIM, abs=1e-2)
    assert not keep[0]


def test_library_basis_is_the_backend_library_plus_an_intercept():
    from lrdsr.core.backends import FastSymbolicRegressor
    x = np.linspace(0.5, 3.0, 40)[:, None]
    B = library_basis(x)
    n_terms = len(FastSymbolicRegressor()._make_library(x))
    assert B.shape == (40, n_terms + 1)
    assert np.allclose(B[:, 0], 1.0)


def test_a_direction_the_library_cannot_resolve_is_dropped():
    """The eigen-truncation: a duplicated column must not add a dimension."""
    X, y, _z, _ = _pair(1.0)

    def basis(Xf):
        c = Xf[:, 0]
        return np.column_stack([np.ones_like(c), c, c ** 2])

    def basis_dup(Xf):
        B = basis(Xf)
        return np.column_stack([B, B[:, 1]])       # exactly collinear

    assert mechanism_features(X, y, basis=basis).shape[1] == 3
    assert mechanism_features(X, y, basis=basis_dup).shape[1] == 3
