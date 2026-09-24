"""Real-time clustering: what the streaming estimators and their theory must keep."""
from __future__ import annotations

import numpy as np
import pytest

from lrdsr.core.online import CusumSegmenter, OnlineLRDSR
from lrdsr.theory import sequential as V9

LAWS = (lambda x: x ** 2 + x, lambda x: x ** 2 - x, lambda x: x ** 2 + 6 * np.sin(2 * x))


def _windows(z, sigma, rng, n=32):
    x = rng.uniform(-2, 2, size=(len(z), n))
    y = np.empty_like(x)
    for k in np.unique(z):
        y[z == k] = LAWS[k](x[z == k])
    return x[:, :, None], y + rng.normal(0, sigma, x.shape)


def _err(truth, pred):
    from experiments.online.streams import matched_error
    return matched_error(truth, pred)


def test_warm_start_then_stream_is_accurate_at_large_rho():
    rng = np.random.default_rng(0)
    sigma = np.sqrt(16 / 3 / 2.0)                       # rho = 2
    Xh, yh = _windows(rng.integers(0, 2, 60), sigma, rng)
    z = rng.integers(0, 2, 150)
    X, y = _windows(z, sigma, rng)
    on = OnlineLRDSR(feature_names=["x"]).warm_start(Xh, yh, n_clusters=2)
    tl = on.fit_stream(X, y, true_labels_for_eval=z)
    assert _err(z, tl["label"].to_numpy()) < 0.02
    assert on.n_clusters == 2
    assert "truth" in tl and len(tl) == 150


def test_a_new_law_is_born_and_a_stationary_stream_births_nothing():
    rng = np.random.default_rng(1)
    sigma = np.sqrt(16 / 3)                             # rho = 1 for the pair
    Xh, yh = _windows(rng.integers(0, 2, 80), sigma, rng)
    z = np.r_[rng.integers(0, 2, 60), rng.integers(0, 3, 120)]
    X, y = _windows(z, sigma, rng)
    on = OnlineLRDSR(feature_names=["x"], novelty_alpha=1e-3,
                     novelty_patience=4).warm_start(Xh, yh, n_clusters=2)
    tl = on.fit_stream(X, y)
    births = tl.loc[tl["spawned"].notna(), "t"].to_numpy()
    assert len(births) == 1 and births[0] >= 60
    assert on.n_clusters == 3

    zc = rng.integers(0, 2, 180)
    Xc, yc = _windows(zc, sigma, rng)
    ctrl = OnlineLRDSR(feature_names=["x"], novelty_alpha=1e-3,
                       novelty_patience=4).warm_start(Xh, yh, n_clusters=2)
    ctrl.fit_stream(Xc, yc)
    assert ctrl.n_clusters == 2


def test_forgetting_is_validated_and_novelty_can_be_disabled():
    with pytest.raises(ValueError):
        OnlineLRDSR(forgetting=0.0)
    with pytest.raises(ValueError):
        OnlineLRDSR(forgetting=1.5)
    with pytest.raises(RuntimeError):
        OnlineLRDSR().partial_fit(np.zeros((4, 1)), np.zeros(4))
    rng = np.random.default_rng(2)
    Xh, yh = _windows(rng.integers(0, 2, 30), 1.0, rng)
    on = OnlineLRDSR(feature_names=["x"], novelty_alpha=0.0).warm_start(
        Xh, yh, labels=np.r_[np.zeros(15, int), np.ones(15, int)])
    X, y = _windows(np.full(20, 2), 1.0, rng)             # an unseen law
    tl = on.fit_stream(X, y)
    assert (tl["label"] >= 0).all() and on.n_clusters == 2


def test_vectorised_v9_cusum_equals_the_segmenter():
    """``sequential.first_alarm`` must be CusumSegmenter, sample for sample."""
    rho, h = 0.5, 4.0
    sigma = np.sqrt(V9.GAP_MS / rho)
    rng = np.random.default_rng(5)
    t_vec = V9.first_alarm(1, rho, h, reps=1, cap=5000, rng=rng)[0]
    # replay the identical draws
    rng = np.random.default_rng(5)
    x = rng.uniform(*V9.X_RANGE, size=(1, 4096))[0]
    y = V9.F1(x) + rng.normal(0, sigma, size=(1, 4096))[0]
    seg = CusumSegmenter([V9.F0, V9.F1], sigma, threshold=h, start=0).run(x, y)
    t_seg = int(np.flatnonzero(seg["alarm"].to_numpy())[0]) + 1
    assert t_vec == t_seg


def test_cusum_delay_matches_theory_at_one_setting():
    rho, h = 1.0, 6.0
    d = V9.first_alarm(1, rho, h, reps=1500, cap=10_000, rng=np.random.default_rng(3))
    pred = V9.predicted_delay(h, rho, V9.KAPPA)
    assert abs(d.mean() / pred - 1) < 0.15
    # and the first-order law is within a factor, as an asymptote should be
    assert 0.5 < d.mean() / V9.predicted_delay(h, rho, method="first_order") < 2.0


def test_segmenter_finds_a_single_switch():
    rng = np.random.default_rng(4)
    sigma = 1.0
    x = rng.uniform(-2, 2, 2000)
    truth = np.r_[np.zeros(1000, int), np.ones(1000, int)]
    y = np.where(truth == 0, LAWS[0](x), LAWS[1](x)) + rng.normal(0, sigma, 2000)
    out = CusumSegmenter(LAWS[:2], sigma, threshold=8.0).run(x, y)
    st = out["state"].to_numpy()
    assert np.mean(st == truth) > 0.98
    first_1 = int(np.argmax(st[1000:] == 1))
    assert first_1 < 5 * V9.predicted_delay(8.0, 16 / 3 / sigma ** 2, V9.KAPPA)


def test_formula_sanity():
    assert V9.kl_per_sample(0.5) == pytest.approx(0.25)
    assert V9.predicted_delay(6, 1.0, method="first_order") == pytest.approx(12.0)
    assert V9.predicted_arl(6, 1.0, method="first_order") == pytest.approx(np.exp(6))
    # delay grows with h, falls with rho; the ARL exceeds Lorden's bound
    assert V9.predicted_delay(8, 1.0) > V9.predicted_delay(4, 1.0)
    assert V9.predicted_delay(6, 4.0) < V9.predicted_delay(6, 1.0)
    for rho in (0.1, 1.0, 4.0):
        assert V9.predicted_arl(6, rho) > np.exp(6)
    # with a constant gap, the tilted and Brownian forms coincide
    assert V9.predicted_delay(5, 0.3, 0.0, "tilted") == pytest.approx(
        V9.predicted_delay(5, 0.3, 0.0, "brownian"), rel=1e-9)
    with pytest.raises(ValueError):
        V9.predicted_delay(5, 1.0, method="nope")
