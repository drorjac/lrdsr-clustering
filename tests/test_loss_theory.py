"""The loss theory (V8) and the learned loss keep their defining properties."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lrdsr.core.losses import learn_loss
from lrdsr.core.model import GroupedDCSR
from lrdsr.theory.losses import (
    LOSS_GRID,
    efficiency,
    lrt_efficiency,
    noise_families,
    predicted_error,
    simulate_error,
)

FAMS = noise_families()


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parametrize("noise", list(FAMS))
def test_squared_loss_has_unit_efficiency_under_every_noise(noise):
    """rho is defined against the variance, so squared loss sees only it."""
    assert efficiency("squared", FAMS[noise]) == pytest.approx(1.0, rel=1e-6)


def test_gaussian_lrt_is_one_and_textbook_huber():
    g = FAMS["gaussian"]
    assert lrt_efficiency(g) == pytest.approx(1.0, rel=1e-6)
    assert efficiency("huber", g, delta=1.345) == pytest.approx(0.95, abs=0.002)
    assert efficiency("absolute", g) == pytest.approx(2 / np.pi, rel=1e-4)


@pytest.mark.parametrize("noise", list(FAMS))
def test_lrt_bounds_every_loss(noise):
    """Cramer-Rao: no loss is more efficient than the likelihood ratio."""
    top = lrt_efficiency(FAMS[noise])
    for loss, params in LOSS_GRID.items():
        assert efficiency(loss, FAMS[noise], **params) <= top * (1 + 1e-6), loss


def test_closed_form_fisher_information():
    # t_nu: Var * I = nu/(nu-2) * (nu+1)/(nu+3) = 2 at nu = 3; Laplace: 2
    assert lrt_efficiency(FAMS["student_t3"]) == pytest.approx(2.0, rel=1e-5)
    assert lrt_efficiency(FAMS["laplace"]) == pytest.approx(2.0, rel=1e-9)


def test_simulation_matches_formula_for_gaussian_squared():
    """Exact in this case: the statistic is Gaussian given the design."""
    out = simulate_error(FAMS["gaussian"], 32, 4.0 / 32, seed=11, n_windows=20_000,
                         losses={"squared": {}})
    sim, pred, eta = out["squared"]
    assert eta == pytest.approx(1.0)
    assert abs(sim - pred) < 4 * np.sqrt(pred * (1 - pred) / 20_000)
    assert predicted_error(32, 4.0 / 32, 1.0) == pytest.approx(0.1587, abs=1e-3)


def test_robust_loss_beats_squared_under_contamination():
    out = simulate_error(FAMS["contaminated_10"], 64, 1.0 / 64, seed=23,
                         n_windows=10_000, losses={"squared": {}, "huber": {}})
    assert out["huber"][0] < 0.5 * out["squared"][0]


def test_learn_loss_recovers_student_t_nu():
    r = np.random.default_rng(3).standard_t(3.0, 20_000) * 2.5
    got = learn_loss(r)
    assert got.loss == "student_t"
    assert got.params["nu"] == pytest.approx(3.0, rel=0.15)
    assert got.scale == pytest.approx(2.5, rel=0.05)


def test_learn_loss_names_laplace():
    assert learn_loss(np.random.default_rng(4).laplace(0, 1, 5_000)).loss == "absolute"


def test_grouped_dcsr_learned_loss_records_nu():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, (40, 32))
    z = rng.integers(0, 2, 40)
    y = np.where(z[:, None] == 0, x ** 2 + x, x ** 2 - x) + rng.standard_t(3, x.shape)
    Z = np.column_stack([y.mean(1), y.std(1)])
    res = GroupedDCSR(n_clusters=2, alpha_geom=0.0, loss="learned", max_iter=3,
                      init="mechanism", random_state=0).fit(x[:, :, None], y, Z,
                                                            feature_names=["x"])
    h = res.history
    assert "loss" in h and "loss_nu" in h
    assert h["loss"].iloc[-1] == "student_t"
    assert 1.5 < h["loss_nu"].iloc[-1] < 8


def test_invalid_loss_is_refused():
    with pytest.raises(ValueError):
        GroupedDCSR(n_clusters=2, loss="nope")
    with pytest.raises(ValueError):
        GroupedDCSR(n_clusters=2, loss="learned", residual_scale="per_window")
