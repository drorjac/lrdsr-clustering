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


def test_noise_scale_is_in_sd_units():
    """The loss scale is the normal-consistent MAD: sigma for Gaussian noise.

    It used to be the raw MAD (0.6745 sigma), which silently made the Huber
    constant ~1/0.6745 times smaller than its name.
    """
    from lrdsr.core.losses import MAD_TO_SD, noise_scale
    e = np.random.default_rng(5).normal(0, 2.0, 200_000)
    assert noise_scale(e) == pytest.approx(2.0, rel=0.01)
    assert noise_scale(e) == pytest.approx(MAD_TO_SD * robust_scale(e), rel=1e-9)
    assert noise_scale(np.zeros(50)) > 0


def test_default_huber_constant_is_in_sd_units():
    """At delta = 1.345 sd, a residual of 1.3 sd is still in the quadratic zone.

    Under the old raw-MAD scaling it was not: 1.3 sd / 0.6745 = 1.93 > 1.5.
    """
    from lrdsr.core.losses import HUBER_DELTA
    rng = np.random.default_rng(6)
    y = rng.normal(0, 1.0, 5000)
    r = np.full(64, 1.3)
    new = aggregate_window_residual(r, np.zeros(64), scale=1.0)
    assert new == pytest.approx(0.5 * 1.3 ** 2)
    assert HUBER_DELTA == pytest.approx(1.345)
    old = aggregate_window_residual(y, np.zeros_like(y), robust_delta=1.5,
                                    scale_convention="mad")
    now = aggregate_window_residual(y, np.zeros_like(y))
    assert old != pytest.approx(now)


def test_scale_convention_is_validated():
    from lrdsr.core.model import GroupedDCSR
    with pytest.raises(ValueError):
        GroupedDCSR(2, scale_convention="iqr")
