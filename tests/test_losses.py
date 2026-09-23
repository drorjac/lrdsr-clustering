import numpy as np
import pytest

from lrdsr.core.losses import (
    aggregate_window_residual,
    huber_loss,
    joint_cost,
    normalize_cost_matrix,
    robust_scale,
)


def test_huber_quadratic_then_linear():
    r = np.array([0.5, 3.0])
    out = huber_loss(r, delta=1.5)
    assert out[0] == pytest.approx(0.5 * 0.25)
    assert out[1] == pytest.approx(1.5 * (3.0 - 0.75))


def test_robust_scale_positive_even_for_constant_input():
    assert robust_scale(np.zeros(50)) > 0
    assert robust_scale(np.random.default_rng(0).normal(size=200)) == pytest.approx(0.67, abs=0.15)


def test_normalize_cost_preserves_zero_and_order():
    cost = np.array([[0.0, 2.0], [4.0, 8.0]])
    norm = normalize_cost_matrix(cost)
    assert norm[0, 0] == 0.0
    assert np.all(np.argsort(cost, axis=None) == np.argsort(norm, axis=None))


def test_aggregate_window_residual_ranks_bad_fit_higher():
    rng = np.random.default_rng(1)
    y = rng.normal(size=40)
    good = aggregate_window_residual(y, y + rng.normal(0, 0.01, 40))
    bad = aggregate_window_residual(y, y + rng.normal(0, 5.0, 40))
    assert np.isfinite(good) and np.isfinite(bad)


def test_joint_cost_alpha_bounds_and_shape():
    g = np.abs(np.random.default_rng(2).normal(size=(10, 3)))
    e = np.abs(np.random.default_rng(3).normal(size=(10, 3)))
    out = joint_cost(g, e, alpha_geom=0.3)
    assert out.shape == (10, 3)
    with pytest.raises(ValueError):
        joint_cost(g, e, alpha_geom=1.5)


def test_joint_cost_alpha_one_is_pure_geometry():
    g = np.abs(np.random.default_rng(4).normal(size=(6, 2))) + 0.1
    e = np.abs(np.random.default_rng(5).normal(size=(6, 2))) + 0.1
    out = joint_cost(g, e, alpha_geom=1.0)
    assert np.allclose(out, normalize_cost_matrix(g))


def test_default_window_residual_is_scale_invariant():
    """Documents the historical behaviour: per-window standardization makes the
    score blind to the SIZE of the misfit (r and 10*r score identically)."""
    rng = np.random.default_rng(11)
    y = rng.normal(size=64)
    r = rng.normal(0, 1.0, size=64)
    small = aggregate_window_residual(y, y - r)
    large = aggregate_window_residual(y, y - 10.0 * r)
    assert small == pytest.approx(large, rel=1e-9)


def test_shared_scale_ranks_a_worse_fit_higher():
    """With a shared scale the score is monotone in the size of the misfit."""
    rng = np.random.default_rng(12)
    y = rng.normal(size=64)
    r = rng.normal(0, 1.0, size=64)
    good = aggregate_window_residual(y, y - 0.1 * r, scale=1.0)
    bad = aggregate_window_residual(y, y - 5.0 * r, scale=1.0)
    assert good < bad
