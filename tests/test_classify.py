"""The law classifier, the feature map, and the V10 formula's limits."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.classify.ucr import parse_ts, subsample, to_grid, to_windows
from lrdsr.core.classify import LawClassifier, MechanismFeatures
from lrdsr.core.kernel import CosineBasis
from lrdsr.core.mechanism_space import mechanism_features
from lrdsr.theory.classification import (
    Q,
    labelled_windows_needed,
    plugin_error,
    plugin_error_first_order,
    plugin_error_random_design,
)


def _two_laws(W=80, n=32, sigma=0.3, seed=0, fixed=True, level=False):
    rng = np.random.default_rng(seed)
    x = (np.broadcast_to((np.arange(n) + 0.5) / n, (W, n)) if fixed
         else rng.uniform(0, 1, (W, n)))
    z = rng.integers(0, 2, W)
    f = np.where(z[:, None] == 1, np.cos(np.pi * x), np.cos(2 * np.pi * x))
    y = f + rng.normal(0, sigma, (W, n))
    if level:
        y = y + rng.normal(0, 5, (W, 1))       # a per-window level: nuisance
    return x[..., None].copy(), y, z


def test_separates_clean_laws():
    X, y, z = _two_laws(sigma=0.05)
    clf = LawClassifier(basis=CosineBasis(4, lo=0, hi=1), nuisance=None).fit(X, y, z)
    assert np.mean(clf.predict(X, y) == z) == 1.0


def test_level_nuisance_is_profiled_out():
    X, y, z = _two_laws(level=True, seed=1)
    Xt, yt, zt = _two_laws(level=True, seed=2)
    with_n = LawClassifier(basis=CosineBasis(4, lo=0, hi=1), nuisance="intercept")
    without = LawClassifier(basis=CosineBasis(4, lo=0, hi=1), nuisance=None)
    acc_n = np.mean(with_n.fit(X, y, z).predict(Xt, yt) == zt)
    acc_0 = np.mean(without.fit(X, y, z).predict(Xt, yt) == zt)
    assert acc_n > 0.95 and acc_n > acc_0


def test_ragged_equals_array():
    X, y, z = _two_laws(fixed=False, seed=3)
    b = CosineBasis(5, lo=0, hi=1)
    a = LawClassifier(basis=b).fit(X, y, z).predict_proba(X, y)
    Xl = [X[w] for w in range(len(X))]
    yl = [y[w] for w in range(len(y))]
    r = LawClassifier(basis=CosineBasis(5, lo=0, hi=1)).fit(Xl, yl, z).predict_proba(Xl, yl)
    np.testing.assert_allclose(a, r, atol=1e-10)


def test_windows_of_different_lengths_are_scored_as_they_are():
    X, y, z = _two_laws(seed=4)
    clf = LawClassifier(basis=CosineBasis(4, lo=0, hi=1)).fit(X, y, z)
    keep = [np.sort(np.random.default_rng(w).choice(32, 8 + w % 9, replace=False))
            for w in range(len(X))]
    Xr = [X[w][k] for w, k in enumerate(keep)]
    yr = [y[w][k] for w, k in enumerate(keep)]
    assert np.mean(clf.predict(Xr, yr) == z) > 0.85


def test_fixed_design_fast_path_matches_general_path():
    X, y, z = _two_laws(seed=5)
    clf = LawClassifier(basis=CosineBasis(5, lo=0, hi=1), laws_per_class=2).fit(X, y, z)
    Xl, yl = [X[w] for w in range(10)], [y[w] for w in range(10)]
    fast = clf.law_rss(Xl, yl)
    prof = clf.design_.profile_all(Xl, yl)
    G = np.stack([p[0].T @ p[0] for p in prof])
    G[0] *= 1.0 + 1e-6                      # break the equality: general path
    a = np.stack([p[0].T @ p[1] for p in prof])
    yy = np.array([p[1] @ p[1] for p in prof])
    slow = clf._rss_stats(G, a, yy, clf.coef_)
    np.testing.assert_allclose(fast[1:], slow[1:], rtol=1e-8)


def test_a_class_that_is_two_laws_needs_two():
    rng = np.random.default_rng(6)
    n, W = 32, 120
    x = np.broadcast_to((np.arange(n) + 0.5) / n, (W, n))[..., None].copy()
    sub = rng.integers(0, 2, W)
    cls = rng.integers(0, 2, W)
    # class 0 is +/- cos(pi x), class 1 is +/- cos(2 pi x): both classes have
    # the SAME mean law (zero), so one law per class cannot tell them apart
    sign = np.where(sub == 1, 1.0, -1.0)[:, None]
    y = sign * np.cos((1 + cls)[:, None] * np.pi * x[..., 0])
    y = y + rng.normal(0, 0.3, y.shape)
    b = CosineBasis(6, lo=0, hi=1)
    one = LawClassifier(basis=b, laws_per_class=1).fit(x, y, cls)
    two = LawClassifier(basis=CosineBasis(6, lo=0, hi=1), laws_per_class=2).fit(x, y, cls)
    assert np.mean(two.predict(x, y) == cls) > np.mean(one.predict(x, y) == cls) + 0.2
    assert two.n_laws == 4


def test_all_mode_is_nearest_law():
    X, y, z = _two_laws(W=30, seed=7)
    clf = LawClassifier(basis=CosineBasis(4, lo=0, hi=1), laws_per_class="all")
    clf.fit(X, y, z)
    assert clf.n_laws == 30 and np.all(np.isfinite(clf.sigma_))


def test_symbolic_law_names_each_class():
    X, y, z = _two_laws(sigma=0.05, seed=8)
    clf = LawClassifier(basis=None, feature_names=["x"], law="symbolic",
                        nuisance=None).fit(X * np.pi, y, z)
    assert len(clf.expressions()) == 2
    assert np.mean(clf.predict(X * np.pi, y) == z) > 0.95


def test_feature_map_train_test_consistent_and_matches_transductive():
    X, y, _ = _two_laws(seed=9)
    fm = MechanismFeatures(basis=CosineBasis(5, lo=0, hi=1), nuisance=None)
    S = fm.fit_transform(X, y)
    np.testing.assert_allclose(S, fm.transform(X, y))
    ref = mechanism_features(X, y, basis=CosineBasis(5, lo=0, hi=1))
    # the same statistic up to an orthogonal change of eigenbasis sign/order
    np.testing.assert_allclose(S @ S.T, ref @ ref.T, rtol=1e-6, atol=1e-8)


def test_v10_limits():
    assert plugin_error(9.0, 5, 1e6) == pytest.approx(Q(1.5), abs=1e-3)
    assert plugin_error_first_order(9.0, 5, 1e9) == pytest.approx(Q(1.5), abs=1e-6)
    # the price of the basis is p / m: more dimensions, worse; more windows, better
    assert plugin_error(9.0, 20, 4) > plugin_error(9.0, 5, 4)
    assert plugin_error(9.0, 5, 16) < plugin_error(9.0, 5, 4)
    assert labelled_windows_needed(9.0, 40) > labelled_windows_needed(9.0, 10)
    assert plugin_error_random_design(9.0, 31, 0.5, 48) == 0.5


def test_ucr_parsing_and_views():
    s, y = parse_ts("#c\n@problemName t\n@data\n1,2,?,4:a\n1,2,3,4:b\n")
    assert np.isnan(s[0][2]) and list(y) == ["a", "b"]
    _, v = to_windows(s)
    assert len(v[0]) == 3 and len(v[1]) == 4
    G = to_grid(s)
    assert G[0, 2] == pytest.approx(3.0)       # linear interpolation of the gap
    sub = subsample([np.arange(20.0)], 0.1, seed=0)[0]
    assert np.isfinite(sub).sum() == 3         # never fewer than three samples


def _shifted_classes(W=60, n=40, seed=10):
    """Two pulse shapes played at random times: a law up to a shift."""
    rng = np.random.default_rng(seed)
    x = np.broadcast_to((np.arange(n) + 0.5) / n, (W, n))[..., None].copy()
    z = rng.integers(0, 2, W)
    t0 = rng.uniform(0.3, 0.7, W)[:, None]
    u = x[..., 0] - t0
    # the same sign and similar size: the classes differ only in the pulse's
    # WIDTH, so a law that ignores when the pulse happens averages it away
    y = np.where(z[:, None] == 1, np.exp(-(u / 0.04) ** 2), np.exp(-(u / 0.08) ** 2))
    return x, y + rng.normal(0, 0.1, y.shape), z


def test_a_shift_nuisance_repairs_classes_played_early_or_late():
    X, y, z = _shifted_classes()
    Xt, yt, zt = _shifted_classes(seed=11)
    plain = LawClassifier(basis=CosineBasis(20, lo=0, hi=1), nuisance=None).fit(X, y, z)
    warp = LawClassifier(basis=CosineBasis(20, lo=0, hi=1), nuisance=None,
                         shift_grid=np.arange(-0.25, 0.26, 0.0125)).fit(X, y, z)
    acc_p = np.mean(plain.predict(Xt, yt) == zt)
    acc_w = np.mean(warp.predict(Xt, yt) == zt)
    assert acc_w >= 0.9 and acc_w >= acc_p + 0.05
    assert np.abs(warp.train_shift_).max() > 0         # the class laws were aligned


def test_no_shift_grid_is_the_plain_classifier():
    X, y, z = _two_laws(seed=12)
    a = LawClassifier(basis=CosineBasis(5, lo=0, hi=1)).fit(X, y, z).predict_proba(X, y)
    b = LawClassifier(basis=CosineBasis(5, lo=0, hi=1),
                      shift_grid=[0.0]).fit(X, y, z).predict_proba(X, y)
    np.testing.assert_allclose(a, b)


def test_law_stack_trains_on_cross_fitted_evidence():
    from lrdsr.core.classify import LawStack
    X, y, z = _two_laws(W=120, sigma=0.4, seed=13)
    Xt, yt, zt = _two_laws(W=200, sigma=0.4, seed=14)
    st = LawStack(lambda: LawClassifier(basis=CosineBasis(5, lo=0, hi=1),
                                        laws_per_class="all")).fit(X, y, z)
    assert np.mean(st.predict(Xt, yt) == zt) > 0.9
